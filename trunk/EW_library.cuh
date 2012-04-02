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

#include <thrust/device_ptr.h> 
#include <thrust/reduce.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/transform_reduce.h>

using namespace thrust;
using namespace std;

//texture<float,1,cudaReadModeElementType> DisorderTexture;
//texture<float,1,cudaReadModeElementType> SplineTexture;

//////////////////// DEVICE FUNCTIONS ////////////////////

__device__ float rand_MWC_normal(unsigned long long* x, unsigned int* a)
{
  // Use Box-Muller algorithm
  // Get normal (Gaussian) random sample with mean 0 and standard deviation 1
  float u1 = rand_MWC_oc(x, a);
  float u2 = rand_MWC_oc(x, a);
  float r = __fsqrt_rz( -2.0*__logf(u1) );
  float theta = 2.0*M_PI*u2;
  return r*__sinf(theta);
}

////////// Initializing Functions //////////

__global__ void CudaKernel_SetDisorder(float *d_disorder, int nx, int ny){
	const unsigned int jAux = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iAux = blockIdx.y*TILE_Y + threadIdx.y;
	const unsigned int tid = iAux*FRAME2D + jAux;
	// move thread RNG state to registers. 
	unsigned long long rng_state = d_x[tid];
	unsigned int rng_const = d_a[tid];
	
	unsigned int i;	
	for (unsigned int iFrame=0; iFrame<(nx/FRAME2D); iFrame++) {
		i = iAux + FRAME2D*iFrame;
		unsigned int j;
		for (unsigned int jFrame=0; jFrame<(ny/FRAME2D); jFrame++) {
			j = jAux + FRAME2D*jFrame;	
			// compute index with i and j, the location of the element in the original LL*MM array 
			int index = i*ny + j;
			d_disorder[index] = C_EPSILON*(rand_MWC_normal(&rng_state, &rng_const));
			// cut-off for the gaussian random numbers, avoid problems 
			d_disorder[index] = (d_disorder[index] < 0.99e30 && d_disorder[index] > -0.99e30)*d_disorder[index]; 
		}
	}	
	d_x[tid] = rng_state; // store RNG state into global again		
}



////////// Evolution functions //////////

/*
__global__ void CudaKernel_InterpolateDisorderPotential(const float *u, const float *disorder, const float *second_deriv, float *y, int nx, int ny){
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if ( i < nx ){
		// from Num. Recepies. Chap 3, sec. 3
		const float u_aux = fmod(u[i],ny);
		const unsigned int j = int(u_aux);
		const unsigned int index = i*ny + j;
		const unsigned int index_p1 = i*ny + (j+1 != ny)*(j+1); //index_p1 = i*ny + (j+1)%ny;
		const unsigned int jp1 = j+1;

		const float A = (jp1-u_aux);	// A = (jp1-u_aux)/(jp1-j); 
		const float B = 1-A; 		// B = 1-A;
		const float C = (A*A*A-A)/6.; 	// C = (A*A*A-A)*(jp1-j)*(jp1-j)/6.;
		const float D = (B*B*B-B)/6.; 	// D = (B*B*B-B)*(jp1-j)*(jp1-j)/6.;

		y[i] = A*disorder[index] + B*disorder[index_p1] + C*second_deriv[index] + D*second_deriv[index_p1];

		//TODO: Test Texture Memory usage
		//DONE: Tested and result slower --> TODO: Guess Why
		//const float disorderA = tex1Dfetch(DisorderTexture, index);
		//const float disorderB = tex1Dfetch(DisorderTexture, index_p1);
		//const float derivC = tex1Dfetch(SplineTexture, index);
		//const float derivD = tex1Dfetch(SplineTexture, index_p1);
		//y[i] = A*disorderA + B*disorderB + C*derivC + D*derivD;

	}
}
*/

__global__ void CudaKernel_InterpolateDisorderSpline(const float *u, const float *disorder, const float *second_deriv, float *y, int nx, int ny){
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if ( tid < nx ){
		// from Num. Recepies. Chap 3, sec. 3
		const float u_aux = u[tid] - floor(u[tid]/ny)*ny;
		//const float u_aux = fmod(u[tid],ny);
		const unsigned int j = int(u_aux);
		const unsigned int index = tid*ny + j;
		const unsigned int index_p1 = tid*ny + (j+1 != ny)*(j+1); //index_p1 = i*ny + (j+1)%ny;
		const unsigned int jp1 = j+1;

		const float A = (jp1-u_aux);	// A = (jp1-u_aux)/(jp1-j); 
		const float B = 1-A; 		// B = 1-A;
	
		y[tid] = -(disorder[index_p1] - disorder[index] - ((3*A*A-1)/6.)* second_deriv[index] + ((3*B*B-1)/6.)* second_deriv[index_p1]) ;

	}
}


__global__ void CudaKernel_RecalculateForce(const float *a, float *b, const float *d, const float uniform_field, int nx){
	const unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
	// move thread RNG state to registers. 
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];

	unsigned int index;
	for (unsigned int iFrame=0; iFrame<(nx/FRAME1D); iFrame++) {
		index = tid + FRAME1D*iFrame;

		unsigned int index_m1 = (index==0)*(nx-1)+(index!=0)*(index-1);
		unsigned int index_p1 = (index!=(nx-1))*(index+1);
		
		//const float elastic_force = C_BETA*(a[(index-1+nx)%nx] -2*a[index] + a[(index+1)%nx]);
		//const float elastic_force = C_BETA*(a[(index==0)*(nx-1)+(index!=0)*(index-1)] -2*a[index] + a[(index==(nx-1))*0+(index!=(nx-1))*(index+1)]); //better
		//const float elastic_force = C_BETA*(a[((index==0)?(nx-1):(index-1))] -2*a[index] + a[((index==(nx-1))?(0):(index+1))]); 
		const float elastic_force = C_BETA*(a[index_m1] -2*a[index] + a[index_p1]); //even better

		const float pinning_force = C_EPSILON*d[index];

		const float thermal_force = xtemp*(rand_MWC_co(&rng_state, &rng_const)-0.5);

		b[index] =  elastic_force + pinning_force + thermal_force + uniform_field;
	}
	d_x[tid] = rng_state; // store RNG state into global again		
}

/* // using Shared Memory // does not improve performance as it is now 
__global__ void CudaKernel_RecalculateForce(const float *a, float *b, const float *d, const float uniform_field, int nx){
	const unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
	const unsigned int tidx = threadIdx.x;
	// move thread RNG state to registers. 
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];

	unsigned int index;
	for (unsigned int iFrame=0; iFrame<(nx/FRAME1D); iFrame++) {
		index = tid + FRAME1D*iFrame;

		unsigned int index_m1 = (index==0)*(nx-1)+(index!=0)*(index-1);
		unsigned int index_p1 = (index!=(nx-1))*(index+1);

		__shared__ float as[TILE]; 
		__shared__ float ds[TILE]; 
		__shared__ float bs[TILE];
		as[tidx] = a[index]; 
		ds[tidx] = d[index]; 

		__syncthreads();		

		const float right = ((tidx<(blockDim.x-1)) ? as[tidx+1]:a[index_p1]); 
 		const float left = ((tidx > 0) ? as[tidx-1]:a[index_m1]); 
		
		const float elastic_force = C_BETA*(left -2*as[tidx] + right); 

		const float pinning_force = C_EPSILON*ds[tidx]; 

		const float thermal_force = xtemp*(rand_MWC_co(&rng_state, &rng_const)-0.5);

		bs[tidx] =  elastic_force + pinning_force + thermal_force + uniform_field; 

		__syncthreads(); 

		b[index] = bs[tidx];
	}
	d_x[tid] = rng_state; // store RNG state into global again		
}
*/

__global__ void CudaKernel_EulerStep(const float *a, float *b, int nx){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < nx ){
		b[idx] += a[idx]*DT;
	}	
}

////////// Rescaling/Casting functions //////////

__global__ void CudaKernel_R2RScaled(cufftReal *a, int nx, int ny, float scale){
	// compute idx and idy, the location of the element in the original LL*MM array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx && idy < ny){	
		int index = idx*ny + idy;
		a[index]= scale*a[index];
	}	
}

__global__ void CudaKernel_C2CScaled(cufftComplex *a, unsigned int nx, unsigned int ny, float scale){
	// compute idx and idy, the location of the element in the original LL*MM array 
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx && idy < ny){	
		unsigned int index = idx*ny + idy;
		a[index].x= scale*a[index].x;
		a[index].y= scale*a[index].y;
	}	
}

__global__ void CudaKernel_CastFloatToCufftReal(const float *a, cufftComplex *b, int nx){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < nx ){
		b[idx].x = a[idx];
		b[idx].y = 0;
	}
}

////////// Calculate/Accumulate functions //////////

__global__ void CudaKernel_CalculateStructureFactor(const cufftComplex *a, float *b, int nx){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < nx ){
		b[idx] = a[idx].x * a[idx].x + a[idx].y * a[idx].y ;
	}
}

__global__ void CudaKernel_AccumulateStructureFactor(const float *a, double *b, int nx, int i){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < nx){
		b[i*nx + idx] += a[idx] ;
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
	
	//Arrays for the physical quantities
	//float array to hold the Structure Factor 
	float *d_structure_factor, *h_structure_factor;
	//double array to accumulate the Structure Factor 
	double *d_structure_factor_acum, *h_structure_factor_acum;
	//double array to accumulate the Center of Mass Position 
	double *h_center_mass; //*d_center_mass; 

	// intializing the class
	elastic_system(){
      
		//allocating memory for the system array in host and device (vector)
		h_u = (float *)malloc(sizeof(float)*LL);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_u, sizeof(float)*LL));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_u_complex, sizeof(cufftComplex)*(LL/2+1)*BATCH));		
		//allocating memory for the disorder matrix in host and device 
		h_disorder = (float *)malloc(sizeof(float)*LL*MM);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_disorder, sizeof(float)*LL*MM));
		//allocating memory for the disorder spline (matrix)
		h_spline = (float *)malloc(sizeof(float)*LL*MM);		
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_spline, sizeof(float)*LL*MM));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_splint, sizeof(float)*LL));
		//allocating memory for the force (vector)
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_force, sizeof(float)*LL));
		//allocating memory for the Structure Factor
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_structure_factor, sizeof(float)*(LL/2+1)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_structure_factor_acum, sizeof(double)*(T_RUN/T_DATA+1)*(LL/2+1)));
		h_structure_factor = (float *)malloc(sizeof(float)*(LL/2+1));
		h_structure_factor_acum = (double *)malloc(sizeof(double)*(T_RUN/T_DATA+1)*(LL/2+1));
		//allocating memory for the Center of Mass
		//CUDA_SAFE_CALL(cudaMalloc((void**)&d_center_mass, sizeof(double)*(T_RUN/T_DATA+1)));
		h_center_mass = (double *)malloc(sizeof(double)*(T_RUN/T_DATA+1));		


		/* Fill everything with 0s */
		CUDA_SAFE_CALL(cudaMemset(d_u, 0, LL*sizeof(float)));
		CUDA_SAFE_CALL(cudaMemset(d_u_complex, 0, (LL/2+1)*sizeof(cufftComplex)));
		if ( cudaGetLastError ( ) != cudaSuccess ) {
		fprintf (stderr , "Cuda error : Failed to allocate \n" ) ;
		return ;
		};

		CUDA_SAFE_CALL(cudaMemset(d_disorder, 0, LL*MM*sizeof(float)));
		CUDA_SAFE_CALL(cudaMemset(d_spline, 0, LL*MM*sizeof(float)));
		
		CUDA_SAFE_CALL(cudaMemset(d_splint, 0, LL*sizeof(float)));
		
		CUDA_SAFE_CALL(cudaMemset(d_force, 0, LL*sizeof(float)));

		CUDA_SAFE_CALL(cudaMemset(d_structure_factor, 0, (LL/2+1)*sizeof(float)));
		CUDA_SAFE_CALL(cudaMemset(d_structure_factor_acum, 0, (T_RUN/T_DATA+1)*(LL/2+1)*sizeof(double)));

		//CUDA_SAFE_CALL(cudaMemset(d_center_mass, 0, (T_RUN/T_DATA+1)*sizeof(double)));
	
		for (unsigned int k=0; k<LL; k++){
			h_u[k] = 0.;
		}
		for (unsigned int k=0; k<LL*MM; k++){
			h_disorder[k] = 0.;
			h_spline[k] = 0.;		  
		}
		for (unsigned int k=0; k<(LL/2+1); k++){
			h_structure_factor[k] = 0.;
		}
		for (unsigned int k=0; k<(T_RUN/T_DATA+1)*(LL/2+1); k++){
			h_structure_factor_acum[k] = 0.;
		}	
		for (unsigned int k=0; k<(T_RUN/T_DATA+1); k++){
			h_center_mass[k] = 0.;
		}

		// plan R2C
		if (cufftPlan1d(&Sr2c, LL, CUFFT_R2C, BATCH) != CUFFT_SUCCESS ){
		fprintf ( stderr , "CUFFT error : Plan creation failed " ) ;
		return ;
		};

	}

	//////////// Print functions ////////////

	void PrintLine(ofstream &fstr){
		for(unsigned int i=0;i<LL;i++)
		{
		fstr << i << "  " << h_u[i] << endl;
		}
		fstr << endl;
	};

	void PrintStructureFactor(ofstream &fstr){
		CUDA_SAFE_CALL(cudaMemcpy(h_structure_factor, d_structure_factor, sizeof(float)*(LL/2+1), cudaMemcpyDeviceToHost));
		for(unsigned int i=1;i<(LL/2+1);i++)
		{
		fstr << 2*sin(M_PI*i/(double) LL) << "  " << h_structure_factor[i] << endl;
		}
		fstr << endl;
	};

	void PrintStructureFactorAcum(ofstream &fstr){
		CUDA_SAFE_CALL(cudaMemcpy(h_structure_factor_acum, d_structure_factor_acum, sizeof(double)*(T_RUN/T_DATA+1)*(LL/2+1), cudaMemcpyDeviceToHost));
		for(int t=0; t<(T_RUN/T_DATA+1); t++){
			for(unsigned int i=1;i<(LL/2+1);i++)
			{
			fstr << 2*sin(M_PI*i/(double) LL) << "  " << h_structure_factor_acum[t*(LL/2+1) + i]/(double) SAMPLES << endl;
			}
			fstr << endl;
		}
		fstr << endl;

	};

	void PrintCenterOfMassAcum(ofstream &fstr){
		//CUDA_SAFE_CALL(cudaMemcpy(h_center_mass, d_center_mass, sizeof(double)*(T_RUN/T_DATA+1), cudaMemcpyDeviceToHost));
		for(int t=0; t<(T_RUN/T_DATA+1); t++){
			fstr << t*T_DATA << "  " << h_center_mass[t]/(double) SAMPLES << endl;
		}
		fstr << endl;

	};

	void PrintDisorder(ofstream &fstr){
		for(int i=0;i<LL;i++)
		{
			for(int j=0;j<MM;j++)
			{
				int k=i*MM+j;
				fstr << h_disorder[k] << endl; //<< " " ;
			}
			//fstr << endl;
		}
		fstr << endl;
	}

	//////////// Calculate/Accumulate functions ////////////

	void CalculateStructureFactor(){
		dim3 dimBlock(TILE);
		dim3 dimGrid(LL/TILE);
		assert(LL%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);

		//CudaKernel_CastFloatToCufftReal<<<dimGrid, dimBlock>>>(d_u, d_u_complex, LL);	  
		//CUFFT_SAFE_CALL(cufftExecD2Z(Sr2c, (cufftDoubleReal*)d_u_complex, d_u_complex));
		//if (cufftExecD2Z(Sr2c, (cufftDoubleReal*)d_u_complex, d_u_complex) != CUFFT_SUCCESS){
		//fprintf ( stderr , "CUFFT error : ExecC2C Forward failed " ) ;
		//return ;
		//};		

		CUFFT_SAFE_CALL(cufftExecR2C(Sr2c, d_u, d_u_complex));
		//TODO: check if normalization is needed
		
		//CUDA_SAFE_CALL(cudaDeviceSynchronize()); // Not needed. Here only for debug purposes.	
		CudaKernel_CalculateStructureFactor<<<dimGrid, dimBlock>>>(d_u_complex, d_structure_factor, LL/2+1);
	};


	void AccumulateStructureFactor(unsigned int time){
		dim3 dimBlock(TILE);
		dim3 dimGrid(LL/TILE);
		assert(LL%TILE==0);
		assert(time%T_DATA==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);

		CudaKernel_AccumulateStructureFactor<<<dimGrid, dimBlock>>>(d_structure_factor, d_structure_factor_acum, LL/2+1, time/T_DATA);
	};	  

	
/*	
	// apply scaling ( an FFT followed by iFFT will give you back the same array times the length of the transform)
	void normalize(void){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(LL/TILE_X, MM/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		//CudaKernel_C2CScaled<<<dimGrid, dimBlock>>>(d_phi, LL, MM, 1.f / ((float) LL * (float) MM));
		CudaKernel_R2RScaled<<<dimGrid, dimBlock>>>(d_phi_r, LL, MM, 1.f / ((float) LL * (float) MM));
	}
*/

	void CalculateAccumulateCenterOfMass(unsigned int time){
		assert(time%T_DATA==0);
		int index = time/T_DATA;
		thrust::device_ptr<float> dev_ptr(d_u);
		double cm = thrust::reduce(dev_ptr, dev_ptr + LL) / (double) LL;
		h_center_mass[index] += cm;
	};

	double CalculateCenterOfMass(){
		thrust::device_ptr<float> dev_ptr(d_u);
		double cm = thrust::reduce(dev_ptr, dev_ptr + LL) / (double) LL;
		return cm;
	};

	///////// Setup and Update Functions ///////////

	void InitFlat(float offset){ // value=0 is default
		CUDA_SAFE_CALL(cudaMemset(d_u, offset, LL*sizeof(float)));
	};
	
	void InitParticular(unsigned int nx, int shape, float offset){
		for(unsigned int i=0;i<nx;i++){
			//a flat line
			if (shape==0) h_u[i]= offset; 
			//a sinusoidal line
			if (shape==1) h_u[i]= offset + sin(2*M_PI*i/float(nx));
		}
	 	CpyHostToDevice();
	};

	/* transfer the system from CPU to GPU memory */
	void CpyHostToDevice(){
		CUDA_SAFE_CALL(cudaMemcpy(d_u, h_u, sizeof(float)*LL, cudaMemcpyHostToDevice));
	};
	
	/* transfer the system from GPU to CPU memory */
	void CpyDeviceToHost(){	
		CUDA_SAFE_CALL(cudaMemcpy(h_u, d_u, sizeof(float)*LL, cudaMemcpyDeviceToHost));
	};

	void SetDisorder(){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(FRAME2D/TILE_X, FRAME2D/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		
		CudaKernel_SetDisorder<<<dimGrid, dimBlock>>>(d_disorder,LL,MM);
		CUDA_SAFE_CALL(cudaMemcpy(h_disorder, d_disorder, sizeof(float)*LL*MM, cudaMemcpyDeviceToHost));

		//TODO: Move d_disorder to texture memory	    
		//CUDA_SAFE_CALL(cudaUnbindTexture(DisorderTexture));
		//CUDA_SAFE_CALL(cudaBindTexture(NULL, DisorderTexture, d_disorder, LL*MM*sizeof(float)));
	};

	void CreateSpline(){
		float *x, *y, *y2;
		x = (float *)malloc(sizeof(float)*MM);
		y = (float *)malloc(sizeof(float)*MM);
		y2 = (float *)malloc(sizeof(float)*MM);

		for (unsigned int j=0; j<MM; j++){
			y2[j] = 0.;
		}
	  
		//We need to createn spline for each line separately
		//Creating arrays x[] and y[] for each row and writing the result of NR "spline" function in h_spline 
		for (unsigned int j=0; j<MM; j++){
			x[j] = (float) j;
		}
	  
		for (unsigned int i=0; i<LL; i++){
			for (unsigned int j=0; j<MM; j++){
				y[j]= h_disorder[i*MM + j];
			}
			//spline(x, y, MM, 0., 0., y2); // poner 
			//y2[MM-1]= y2[0]; //avoid pbc problems, TODO: NOT sure about this.
			periodic_spline(y,MM,y2);
			for (unsigned int j=0; j<MM; j++){
				h_spline[i*MM + j] = y2[j];
			}
		}
	  
		CUDA_SAFE_CALL(cudaMemcpy(d_spline,h_spline,sizeof(float)*LL*MM, cudaMemcpyHostToDevice));
		free(x);
		free(y);	
		free(y2);		

		//TODO: Move d_spline to texture memory	    
		//CUDA_SAFE_CALL(cudaUnbindTexture(SplineTexture));
		//CUDA_SAFE_CALL(cudaBindTexture(NULL, SplineTexture, d_spline, LL*MM*sizeof(float)));
	};
	
/*	void init_random(){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(FRAME/TILE_X, FRAME/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		
		kernel_init_random<<<dimGrid, dimBlock>>>(d_u,LL);
	};
*/
	
	void InterpolateDisorderSpline(){
		dim3 dimBlock(TILE);
		dim3 dimGrid(LL/TILE);
		assert(LL%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);
		CudaKernel_InterpolateDisorderSpline<<<dimGrid, dimBlock>>>(d_u, d_disorder, d_spline, d_splint, LL, MM);
	};
	
	void RecalculateForce(float uniform_field, unsigned int t){
		dim3 dimBlock(TILE);
		dim3 dimGrid(FRAME1D/TILE);
		assert(FRAME1D%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);
		
		CudaKernel_RecalculateForce<<<dimGrid, dimBlock>>>(d_u, d_force, d_splint, uniform_field, LL);
		//CUDA_SAFE_CALL(cudaDeviceSynchronize()); // Not needed. Here only for debug purposes.	
	};
	
	void EulerStep(unsigned int t){
		dim3 dimBlock(TILE);
		dim3 dimGrid(LL/TILE);
		assert(LL%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);
	  
		CudaKernel_EulerStep<<<dimGrid, dimBlock>>>(d_force, d_u, LL);
	};
	

	void cyclic(float *a, float *b, float *c, float alpha, float beta, float *r, float *x, unsigned int n)
	//Solves for a vector x[1..n] the \cyclic" set of linear equations given by equation (2.7.9). a, b, c, and 
	//r are input vectors, all dimensioned as [1..n], while alpha and beta are the corner
	//entries in the matrix. The input is not modified.
	{
		//void tridag(float *a, float *b, float *c, float *r, float *u, unsigned int n);
		unsigned int i;
		float fact,gamma,*bb,*u,*z;

		//if (n <= 2) nrerror("n too small in cyclic");
		bb = (float *)malloc(sizeof(float)*(n));
		u = (float *)malloc(sizeof(float)*(n));
		z = (float *)malloc(sizeof(float)*(n));
		gamma = -b[0]; 							//Avoid subtraction error in forming bb[1].
		bb[0]=b[0]-gamma; 						//Set up the diagonal of the modified tridiagonal system.
		bb[n-1]=b[n-1]-alpha*beta/gamma; 
		for (i=1;i<n-1;i++) bb[i]=b[i];
		tridag(a,bb,c,r,x,n);			 			//Solve A . x = r.
		u[0]=gamma; 							//Set up the vector u.
		u[n-1]=alpha;
		for (i=1;i<n-1;i++) u[i]=0.0;
		tridag(a,bb,c,u,z,n); 						//Solve A . z = u.
		fact=(x[0]+beta*x[n-1]/gamma)/(1.0+z[0]+beta*z[n-1]/gamma);		//Form v . x=(1 + v . z).
		for (i=0;i<=n-1;i++) x[i] -= fact*z[i]; 				//Now get the solution vector x.
		free(z);
		free(u);
		free(bb);
	}


	void tridag(const float *a, const float *b, const float *c, const float *r, float *u, unsigned int n)
		//Solves for a vector u[1..n] the tridiagonal linear set given by equation (2.4.1). a[1..n],
		//b[1..n], c[1..n], and r[1..n] are input vectors and are not modified.
		{
		int j;
		float bet,*gam;
		gam = (float *)malloc(sizeof(float)*(n)); //One vector of workspace, gam is needed.
		//if (b[1] == 0.0) nrerror("Error 1 in tridag");
		//If this happens then you should rewrite your equations as a set of order N − 1, with u2 trivially eliminated.
		u[0]=r[0]/(bet=b[0]);
		for (j=1;j<n;j++) { 						//Decomposition and forward substitution.
			gam[j]=c[j-1]/bet;
			bet=b[j]-a[j]*gam[j];
			//if (bet == 0.0) nrerror("Error 2 in tridag"); 	//Algorithm fails; see below.
			u[j]=(r[j]-a[j]*u[j-1])/bet;
		}
		for (j=(n-2);j>-1;j--){
			u[j] -= gam[j+1]*u[j+1]; 		//Backsubstitution.
		}
		free(gam);
	}


	void periodic_spline(float *y, unsigned int n, float *y2){

		float *a, *b, *c, *r;		
		a = (float *)malloc(sizeof(float)*(n));
		b = (float *)malloc(sizeof(float)*(n));
		c = (float *)malloc(sizeof(float)*(n));
		r = (float *)malloc(sizeof(float)*(n));
		
		unsigned int i;
		for(i=0;i<n;i++){a[i]= 1/6.;}
		for(i=0;i<n;i++){b[i]= 2/3.;}
		for(i=0;i<n;i++){c[i]= 1/6.;}

		r[0] = y[1] -2*y[0] + y[n-1];
		for(i=1;i<n-1;i++) r[i] = y[i+1] -2*y[i] +y[i-1];
		r[n-1] = y[0] -2*y[n-1] +y[n-2];

		float alpha = c[0];
		float beta = a[0];
		cyclic(a,b,c,alpha,beta,r,y2,n);

		free(a); free(b); free(c); free(r);
	}

	/*FROM NUMERICAL RECEPIES*/
	void spline(float *x, float *y, int n, float yp1, float ypn, float *y2)
	/*Given arrays x[1..n] and y[1..n] containing a tabulated function, i.e., yi = f(xi), with
	x1 < x2 < . . . < xN, and given values yp1 and ypn for the first derivative of the interpolating
	function at points 1 and n, respectively, this routine returns an array y2[1..n] that contains
	the second derivatives of the interpolating function at the tabulated points xi. If yp1 and/or
	ypn are equal to 1 × 10^30 or larger, the routine is signaled to set the corresponding boundary
	condition for a natural spline, with zero second derivative on that boundary.*/
	// NOTE: x[1..n] logic used by NR was translated to x[0..n-1] in all cases. 
	{
		int i,k;
		float p,qn,sig,un,*u;
		u = (float *)malloc(sizeof(float)*(n-1));
		if (yp1 > 0.99e30){ //The lower boundary condition is set either to be “natural”
			y2[0]=u[0]=0.0;
		} else { 		//or else to have a specified first derivative.
			y2[0] = -0.5;
			u[0]=(3.0/(x[1]-x[0]))*((y[1]-y[0])/(x[1]-x[0])-yp1);
		}

		for (i=1;i<n-1;i++) { //This is the decomposition loop of the tridiagonal algorithm. 
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
		for (k=n-2;k>-1;k--){ //This is the backsubstitution loop of the tridiagonal algorithm.
			y2[k]=y2[k]*y2[k+1]+u[k]; 
		}
		free(u);
	};
	
	/////////////// Clean Memory ////////////////
	~elastic_system(){
		cufftDestroy(Sr2c);
		cudaFree(d_u); cudaFree(d_u_complex); 
		cudaFree(d_disorder); cudaFree(d_spline); cudaFree(d_splint); cudaFree(d_force);
		cudaFree(d_structure_factor); cudaFree(d_structure_factor_acum);
		free(h_u); 
		free(h_disorder); free(h_spline); 
		free(h_structure_factor); free(h_structure_factor_acum); free(h_center_mass);
	};

};

#endif /* EW_LIBRARY_CUH_ */
