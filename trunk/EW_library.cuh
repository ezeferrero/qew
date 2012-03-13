/*
 * EW_Langevin_CUDA_library
 *
 *  Created on: Feb 26, 2012
 *      Author: eze
 */

#ifndef EW_LIBRARY_CUH_
#define EW_LIBRARY_CUH_

//#include "../common/common.c"
//#include <stdint.h> /* uint8_t */
//#include "../common/nrutil.h"

//#include <thrust/device_ptr.h> 
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/transform_reduce.h>
//#include <thrust/reduce.h>
//using namespace thrust;

using namespace std;

//texture<float,1,cudaReadModeElementType> Etexture;

__device__ float rand_MWC_normal(unsigned long long* x, const unsigned int* a)
{
  // Use Box-Muller algorithm
  // Get normal (Gaussian) random sample with mean 0 and standard deviation 1
  float u1 = rand_MWC_co(x, a);
  float u2 = rand_MWC_co(x, a);
  float r = __fsqrt_rz( -2.0*__logf(u1) );
  float theta = 2.0*M_PI*u2;
  return r*__sinf(theta);
}

__global__ void CudaKernel_SetDisorder(float *d_disorder, int nx, int ny){
	const unsigned int jAux = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iAux = blockIdx.y*TILE_Y + threadIdx.y;
	const unsigned int tid = iAux*FRAME2D + jAux;
	// move thread RNG state to registers. 
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];
	
	unsigned int i;	
	for (unsigned int iFrame=0; iFrame<(nx/FRAME2D); iFrame++) {
		i = iAux + FRAME2D*iFrame;
		unsigned int j;
		for (unsigned int jFrame=0; jFrame<(ny/FRAME2D); jFrame++) {
			j = jAux + FRAME2D*jFrame;	
			// compute index with i and j, the location of the element in the original L*M array 
			int index = i*ny + j;
			d_disorder[index] = C_EPSILON*(rand_MWC_normal(&rng_state, &rng_const));
		}
	}	
	d_x[tid] = rng_state; // store RNG state into global again		
}

__global__ void CudaKernel_R2RScaled(cufftReal *a, int nx, int ny, float scale){
	// compute idx and idy, the location of the element in the original L*M array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx && idy < ny){	
		int index = idx*ny + idy;
		a[index]= scale*a[index];
	}	
}

__global__ void CudaKernel_C2CScaled(cufftComplex *a, unsigned int nx, unsigned int ny, float scale){
	// compute idx and idy, the location of the element in the original L*M array 
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx && idy < ny){	
		unsigned int index = idx*ny + idy;
		a[index].x= scale*a[index].x;
		a[index].y= scale*a[index].y;
	}	
}

__global__ void CudaKernel_RecalculateForce(const float *a, float *b, float *d, float uniform_field, int nx){
	const unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
	// move thread RNG state to registers. 
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];

	unsigned int index;
	for (unsigned int iFrame=0; iFrame<(nx/FRAME1D); iFrame++) {
		index = tid + FRAME1D*iFrame;
		
		const float pinning_force = C_EPSILON*d[index];
		const float elastic_force = C_BETA*(a[(index-1+nx)%nx] -2*a[index] + a[(index+1)%nx]);
//		const float epsilon = tex1Dfetch(Etexture, index);
		//b[index] =  elastic_force + xtemp*(rand_MWC_co(&rng_state, &rng_const)-0.5) + uniform_field;
		b[index] =  elastic_force + pinning_force + uniform_field + xtemp*(rand_MWC_co(&rng_state, &rng_const)-0.5);
	}
	d_x[tid] = rng_state; // store RNG state into global again		
}

__global__ void CudaKernel_EulerStep(const float *a, float *b, int nx){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < nx ){
		b[idx] += a[idx]*DT;
		//b[idx] = (b[idx]+M)%M; //TODO: Maybe this pbc is missing
	}	
}

__global__ void CudaKernel_CastFloatToCufftReal(const float *a, cufftComplex *b, int nx){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < nx ){
		b[idx].x = a[idx];
		b[idx].y = 0;
	}
}

__global__ void CudaKernel_CalculateStructureFactor(const cufftComplex *a, float *b, int nx){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < nx ){
		b[idx] = a[idx].x * a[idx].x + a[idx].y * a[idx].y ;
	}
}

__global__ void CudaKernel_AcumulateStructureFactor(const float *a, double *b, int nx, int i){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < nx ){
		b[i*nx + idx] += a[idx] ;
	}
}


__global__ void CudaKernel_GetSplints(const float *u, const float *disorder, const float *second_deriv, float *y, int nx, int ny){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if ( i < nx ){
	    // from Num. Recepies. Chap 3, sec. 3
	    float u_aux = u[i];
	    int j = int(u_aux);
	    int index = i*ny + j;
	    int index_p1 = i*ny + (j+1)%ny;
	    int jp1 = j+1;
	    float A = (jp1-u_aux)/(jp1-j); 
	    float B = 1-A;
	    float C = (A*A*A-A)*(jp1-j)*(jp1-j)/6.;
	    float D = (B*B*B-B)*(jp1-j)*(jp1-j)/6.;
	    y[i] = A*disorder[index] + B*disorder[index_p1] + C*second_deriv[index] + D*second_deriv[index_p1];

	}
}

//////////////////////////////////////////////////////////////////////
class elastic_system
{
	private:
	// cuFFT plan
	cufftHandle Sr2c; 

	public:
	//real arrays to hold u(z,t) on host and device (line)
	float *d_u, *h_u;

	//complex array to hold F[u] on device (line)
	cufftComplex *d_u_complex;

	//float arrays to hold the disorder (matrix)
	float *d_disorder, *h_disorder;
	
	//float arrays for the disorder spline (matrix)
	float *d_spline, *h_spline;

	//float array for the interpolated (local) disorder
	float *d_splint;

	//real array for the force
	float *d_force;
	
	//TODO: create arrays for the physical quantities
	//real array to hold the Structure Factor 
	float *d_Sk, *h_Sk;
	double *d_Sk_acum, *h_Sk_acum;
	//float h_wall[3][T_RUN/T_DATA+1];
	
	// intializing the class
	elastic_system(){
      
		//allocating memory for the system array in host and device (line)
		h_u = (float *)malloc(sizeof(float)*L);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_u, sizeof(float)*L));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_u_complex, sizeof(cufftComplex)*L));		
		//allocating memory for the disorder matrix in host and device 
		h_disorder = (float *)malloc(sizeof(float)*L*M);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_disorder, sizeof(float)*L*M));
		//allocating memory for the disorder spline (matrix)
		h_spline = (float *)malloc(sizeof(float)*L*M);		
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_spline, sizeof(float)*L*M));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_splint, sizeof(float)*L));
		//allocating memory for the force (line)
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_force, sizeof(float)*L));
		//allocating memory for the Structure Factor
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_Sk, sizeof(float)*(L/2+1)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_Sk_acum, sizeof(double)*(T_RUN/T_DATA)*(L/2+1)));
		h_Sk = (float *)malloc(sizeof(float)*(L/2+1));
		h_Sk_acum = (double *)malloc(sizeof(double)*(T_RUN/T_DATA)*(L/2+1));
		
		/*Fill everithing with 0's*/
		CUDA_SAFE_CALL(cudaMemset(d_u, 0, L*sizeof(float)));
		CUDA_SAFE_CALL(cudaMemset(d_u_complex, 0, L*sizeof(cufftComplex)));

		CUDA_SAFE_CALL(cudaMemset(d_disorder, 0, L*M*sizeof(float)));
		CUDA_SAFE_CALL(cudaMemset(d_spline, 0, L*M*sizeof(float)));
		
		CUDA_SAFE_CALL(cudaMemset(d_splint, 0, L*sizeof(float)));
		
		CUDA_SAFE_CALL(cudaMemset(d_force, 0, L*sizeof(float)));

		CUDA_SAFE_CALL(cudaMemset(d_Sk, 0, (L/2+1)*sizeof(float)));
		CUDA_SAFE_CALL(cudaMemset(d_Sk_acum, 0, (T_RUN/T_DATA)*(L/2+1)*sizeof(double)));

	
		for (unsigned int k=0; k<L; k++){
			h_u[k] = 0.;
		}
		for (unsigned int k=0; k<L*M; k++){
			h_disorder[k] = 0.;
			h_spline[k] = 0.;		  
		}
		for (unsigned int k=0; k<(L/2+1); k++){
			h_Sk[k] = 0.;
		}
		for (unsigned int k=0; k<(T_RUN/T_DATA)*(L/2+1); k++){
			h_Sk_acum[k] = 0.;
		}	

		// plan R2C
		cufftPlan1d(&Sr2c, L, CUFFT_R2C, BATCH);
	};


	//////////// Calculate and print functions ////////////

	void print_input(ofstream &fstr){
		for(unsigned int i=0;i<L;i++)
		{
		fstr << i << "  " << setprecision (4) << h_u[i] << endl;
		}
		fstr << endl;
	};

	void print_structure_factor(ofstream &fstr){
		CUDA_SAFE_CALL(cudaMemcpy(h_Sk, d_Sk, sizeof(float)*(L/2+1), cudaMemcpyDeviceToHost));

		for(unsigned int i=0;i<(L/2+1);i++)
		{
		fstr << 2*sin(M_PI*i/(float) L) << "  " << setprecision (6) << h_Sk[i] << endl;
		}
		fstr << endl;
	};

	void print_structure_factor_acum(ofstream &fstr){
		CUDA_SAFE_CALL(cudaMemcpy(h_Sk_acum, d_Sk_acum, sizeof(double)*(T_RUN/T_DATA)*(L/2+1), cudaMemcpyDeviceToHost));

		for(int t=0; t<(T_RUN/T_DATA); t++){
			for(unsigned int i=0;i<(L/2+1);i++)
			{
			fstr << 2*sin(M_PI*i/(float) L) << "  " << setprecision (6) << h_Sk_acum[t*(L/2+1) + i]/(double) SAMPLES << endl;
			}
			fstr << endl;
		}
		fstr << endl;

	};


	void CalculateStructureFactor(){
		dim3 dimBlock(TILE);
		dim3 dimGrid(L/TILE);
		assert(L%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);

		//CudaKernel_CastFloatToCufftReal<<<dimGrid, dimBlock>>>(d_u, d_u_complex, L);	  
		//CUFFT_SAFE_CALL(cufftExecR2C(Sr2c, (cufftReal*)d_u_complex, d_u_complex));
		CUFFT_SAFE_CALL(cufftExecR2C(Sr2c, d_u, d_u_complex));

		//CUDA_SAFE_CALL(cudaDeviceSynchronize());	

		CudaKernel_CalculateStructureFactor<<<dimGrid, dimBlock>>>(d_u_complex, d_Sk, L/2+1);
	};


	void AcumulateStructureFactor(unsigned int time){
		dim3 dimBlock(TILE);
		dim3 dimGrid(L/TILE);
		assert(L%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);

		CudaKernel_AcumulateStructureFactor<<<dimGrid, dimBlock>>>(d_Sk, d_Sk_acum, L/2+1, time/T_DATA);
	};	  


	
/*	
	// apply scaling ( an FFT followed by iFFT will give you back the same array times the length of the transform)
	void normalize(void){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(L/TILE_X, M/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		//CudaKernel_C2CScaled<<<dimGrid, dimBlock>>>(d_phi, L, M, 1.f / ((float) L * (float) M));
		CudaKernel_R2RScaled<<<dimGrid, dimBlock>>>(d_phi_r, L, M, 1.f / ((float) L * (float) M));
	}
*/

	void PrintCenterOfMass(int i){
		//thrust::device_ptr<float> dev_ptr(d_u);		  	  
		float cm = 0.5;  
		//float cm = thrust::reduce(dev_ptr, dev_ptr + L);
		//return thrust::reduce(dev_ptr,dev_ptr+L, (cufftReal) 0., thrust::plus<cufftReal>())/float(L);
		printf("%i %f\n",i, cm);
	};


	////////////////////////////////////////////////

	///////// Setup and Update Functions ///////////

	void InitFlat(){ //present by default
		CUDA_SAFE_CALL(cudaMemset(d_u, 0, L*sizeof(float)));
	};
	
	void InitParticular(int nx){
		for(int i=0;i<nx;i++){
		
		//a sinusoidal line
		h_u[i]= 10 + sin(2*M_PI*i/float(nx));
		  
		//a flat line
		//h_u[i]=10; 
		}
	};

	/* transfer the system from CPU to GPU memory */
	void CpyHostToDevice(){
		CUDA_SAFE_CALL(cudaMemcpy(d_u, h_u, sizeof(float)*L, cudaMemcpyHostToDevice));
	};
	
	/* transfer the system from GPU to CPU memory */
	void CpyDeviceToHost(){	
		CUDA_SAFE_CALL(cudaMemcpy(h_u, d_u, sizeof(float)*L, cudaMemcpyDeviceToHost));
	};


	void SetDisorder(){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(FRAME2D/TILE_X, FRAME2D/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		
		CudaKernel_SetDisorder<<<dimGrid, dimBlock>>>(d_disorder,L,M);
		CUDA_SAFE_CALL(cudaMemcpy(h_disorder, d_disorder, sizeof(float)*L*M, cudaMemcpyDeviceToHost));
	};

	void CreateSpline(){
		//float x[M],y[M], y2[M];
		float *x, *y, *y2;
		x = (float *)malloc(sizeof(float)*M);
		y = (float *)malloc(sizeof(float)*M);
		y2 = (float *)malloc(sizeof(float)*M);
	  
		//We need to createn spline for each line separately
		//Creating arrays x[] and y[] for each row and writing the result of NR "spline" function in h_spline 
		for (unsigned int j=0; j<M; j++){
			x[j] = (float) j;	//check if that +1 is needed
		}
	  
		for (unsigned int i=0; i<L; i++){
			for (unsigned int j=0; j<M; j++){
				y[j]= h_disorder[i*M + j];
			}
			y[M-1]= y[0]; //avoid pbc problems, NOT sure about this.
	    
			spline(x, y, M, 0., 0., y2);
	    
			for (unsigned int j=0; j<M; j++){
				h_spline[i*M + j] = y2[j];
			}
	  
		}
	  
		CUDA_SAFE_CALL(cudaMemcpy(d_spline,h_spline,sizeof(float)*L*M, cudaMemcpyHostToDevice));
	  
		free(x);
		free(y);	
		free(y2);		

		//TODO: Move d_spline to texture memory	    
		//CUDA_SAFE_CALL(cudaBindTexture(NULL, Etexture, d_disorder, L*M*sizeof(float)));
	  
		// testing
		/*for (int k=1; k<1000; k++){
		  printf("%i %f\n",k, h_spline[k]);
		}
		printf("%f\n",h_spline[(L-1)*M + (M-1)]);
		*/
	};
	
/*	void init_random(){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(FRAME/TILE_X, FRAME/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		
		kernel_init_random<<<dimGrid, dimBlock>>>(d_u,L);
	};
*/
	
	void GetSplints(){
		dim3 dimBlock(TILE);
		dim3 dimGrid(L/TILE);
		assert(L%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);
		CudaKernel_GetSplints<<<dimGrid, dimBlock>>>(d_u, d_disorder, d_spline, d_splint, L, M);
	};
	
	void RecalculateForce(float uniform_field, unsigned int t){
		dim3 dimBlock(TILE);
		dim3 dimGrid(FRAME1D/TILE);
		assert(FRAME1D%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);
		
		CudaKernel_RecalculateForce<<<dimGrid, dimBlock>>>(d_u, d_force, d_splint, uniform_field, L);
	};
	
	void EulerStep(unsigned int t){
		dim3 dimBlock(TILE);
		dim3 dimGrid(L/TILE);
		assert(L%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);
	  
		CudaKernel_EulerStep<<<dimGrid, dimBlock>>>(d_force, d_u, L);
	};
	
	/*FROM NUMERICAL RECEPIES*/
	void spline(float *x, float *y, int n, float yp1, float ypn, float *y2)
	/*Given arrays x[1..n] and y[1..n] containing a tabulated function, i.e., yi = f(xi), with
	x1 < x2 < . . . < xN, and given values yp1 and ypn for the first derivative of the interpolating
	function at points 1 and n, respectively, this routine returns an array y2[1..n] that contains
	the second derivatives of the interpolating function at the tabulated points xi. If yp1 and/or
	ypn are equal to 1 × 1030 or larger, the routine is signaled to set the corresponding boundary
	condition for a natural spline, with zero second derivative on that boundary.*/
	{
		int i,k;
		float p,qn,sig,un,*u;
		//u=vector(1,n-1); // I replaced this with the malloc bellow, CHECK!!
		u = (float *)malloc(sizeof(float)*(n-1));
		if (yp1 > 0.99e30){ //The lower boundary condition is set either to be “natural”
			y2[0]=u[0]=0.0;
		} else { 		//or else to have a specified first derivative.
			y2[0] = -0.5;
			u[0]=(3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
		}

		for (i=1;i<n-1;i++) { //modified to fit C vectors indexing
		//for (i=2;i<=n-1;i++) { //This is the decomposition loop of the tridiagonal algorithm. 
		// y2 and u are used for temporary storage of the decomposed factors.
			sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
			p=sig*y2[i-1]+2.0;
			y2[i]=(sig-1.0)/p;
			u[i]=(y[i+1]-y[i])/(x[i+1]-x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1]);
			u[i]=(6.0*u[i]/(x[i+1]-x[i-1])-sig*u[i-1])/p;
		}
		if (ypn > 0.99e30){ //The upper boundary condition is set either to be “natural”
			qn=un=0.0; 
		} else { 	//or else to have a specified first derivative.
			qn=0.5;
			un=(3.0/(x[n-1]-x[n-2]))*(ypn-(y[n-1]-y[n-2])/(x[n-1]-x[n-2]));
		}
		y2[n-1]=(un-qn*u[n-2])/(qn*y2[n-2]+1.0);
		for (k=n-2;k>-1;k--){ //modified to fit C vectors indexing 
		//for (k=n-1;k>=1;k--) //This is the backsubstitution loop of the tridiagonal algorithm.
			y2[k]=y2[k]*y2[k+1]+u[k]; 
			//free_vector(u,1,n-1); // I replaced this with the "free" bellow
		}
		free(u);
	};
	
	/////////////// Clean Memory ////////////////
	~elastic_system(){
		cufftDestroy(Sr2c);
		cudaFree(d_u);
		cudaFree(d_force);
		cudaFree(d_disorder);
		cudaFree(d_spline);
		free(h_u);
	
	};

};

#endif /* EW_LIBRARY_CUH_ */
