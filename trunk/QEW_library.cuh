/*
 * QEW_CUDA_library
 *
 *  Started on: Feb 26, 2012
 *      Author: eze
 */

#ifndef QEW_LIBRARY_CUH_
#define QEW_LIBRARY_CUH_

#include "../common/mean_and_variance.cuh"
#include "../common/histo.h"

#include <thrust/device_ptr.h> 
#include <thrust/reduce.h>
#include <thrust/copy.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/transform_reduce.h>

using namespace thrust;
using namespace std;

//texture<FLOAT,1,cudaReadModeElementType> DisorderTexture;
//texture<FLOAT,1,cudaReadModeElementType> SplineTexture;

//////////////////// DEVICE FUNCTIONS ////////////////////

__device__ FLOAT rand_MWC_normal(unsigned long long* x, unsigned int* a)
{
  // Use Box-Muller algorithm
  // Get normal (Gaussian) random sample with mean 0 and standard deviation 1
  FLOAT u1 = rand_MWC_oc(x, a);
  FLOAT u2 = rand_MWC_oc(x, a);
#ifdef DOUBLE_PRECISION
  FLOAT r = sqrt( -2.0*log(u1) );
  FLOAT theta = 2.0*M_PI*u2;
  return r*sin(theta);
#else
  FLOAT r = sqrtf( -2.0*logf(u1) );
  FLOAT theta = 2.0*M_PI*u2;
  return r*sinf(theta);
#endif
}

////////// Initializing Functions //////////

__global__ void CudaKernel_SetDisorder(FLOAT *d_disorder, int nx, int ny){
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
			#ifndef UNIFORM_DISORDER
			d_disorder[index] = C_EPSILON*(rand_MWC_normal(&rng_state, &rng_const));
			#else
			d_disorder[index] = C_EPSILON*3.46410162*(rand_MWC_oc(&rng_state, &rng_const)-0.5);
			#endif
		}
	}	
	d_x[tid] = rng_state; // store RNG state into global again
}



////////// Evolution functions //////////

#ifndef LINEAR_SPLINE

__global__ void CudaKernel_InterpolateDisorderSpline(const FLOAT *u, const FLOAT *disorder, const FLOAT *second_deriv, FLOAT *y, int nx, int ny){
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if ( tid < nx ){
		// from Num. Recepies. Chap 3, sec. 3
		const FLOAT u_aux = u[tid] - floor(u[tid]/ny)*ny;
		//const FLOAT u_aux = fmod(u[tid],ny);
		const unsigned int j = int(u_aux);
		const unsigned int index = tid*ny + j;
		const unsigned int index_p1 = tid*ny + (j+1 != ny)*(j+1); //index_p1 = i*ny + (j+1)%ny;
		const unsigned int jp1 = j+1;

		const FLOAT A = (jp1-u_aux);	// A = (jp1-u_aux)/(jp1-j); 
		const FLOAT B = 1-A; 		// B = 1-A;
	
		y[tid] = -(disorder[index_p1] - disorder[index] - ((3*A*A-1)/6.)* second_deriv[index] + ((3*B*B-1)/6.)* second_deriv[index_p1]) ;
		//y[tid] = (disorder[index_p1] - disorder[index])*B + disorder[index]; //lineal!!!

	}
}

#else

__global__ void CudaKernel_InterpolateDisorderLinearSpline(const FLOAT *u, const FLOAT *disorder, FLOAT *y, int nx, int ny){
	int tid = blockIdx.x*blockDim.x+threadIdx.x;
	if ( tid < nx ){
		// from Num. Recepies. Chap 3, sec. 3
		const FLOAT u_aux = u[tid] - floor(u[tid]/ny)*ny;
		//const FLOAT u_aux = fmod(u[tid],ny);
		const unsigned int j = int(u_aux);
		const unsigned int index = tid*ny + j;
		const unsigned int index_p1 = tid*ny + (j+1 != ny)*(j+1); //index_p1 = i*ny + (j+1)%ny;
		const unsigned int jp1 = j+1;

		//const FLOAT A = (jp1-u_aux);	// A = (jp1-u_aux)/(jp1-j); 
		//const FLOAT B = 1-A; 		// B = 1-A;

		#ifdef RF
		y[tid] = disorder[index_p1];
		#else
		y[tid] = (disorder[index_p1] - disorder[index]);
		#endif

	}
}

#endif


__global__ void CudaKernel_RecalculateForce(const FLOAT *a, FLOAT *b, const FLOAT *d, const FLOAT uniform_field, int nx){
	const unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
	// move thread RNG state to registers.
	#ifndef ZEROTEMP
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];
	#endif

	unsigned int index;
	for (unsigned int iFrame=0; iFrame<(nx/FRAME1D); iFrame++) {
		index = tid + FRAME1D*iFrame;

		unsigned int index_m1 = (index==0)*(nx-1)+(index!=0)*(index-1);
		unsigned int index_p1 = (index!=(nx-1))*(index+1);
		
		#ifdef ANHARMONIC
		const FLOAT elastic_force = C_BETA*(a[index_m1] -2*a[index] + a[index_p1]) + \
		C_OMEGA*((a[index_p1]-a[index])*(a[index_p1]-a[index])*(a[index_p1]-a[index])+\
		(a[index_m1]-a[index])*(a[index_m1]-a[index])*(a[index_m1]-a[index]));
		#else
		const FLOAT elastic_force = C_BETA*(a[index_m1] -2*a[index] + a[index_p1]); //even better
		#endif
		
		const FLOAT pinning_force = C_EPSILON*d[index];

		#ifndef ZEROTEMP
		const FLOAT thermal_force = XTEMP*(rand_MWC_co(&rng_state, &rng_const)-0.5);
		b[index] =  elastic_force + pinning_force + thermal_force + uniform_field;
		#else
		b[index] =  elastic_force + pinning_force + uniform_field;
		#endif

	}
	#ifndef ZEROTEMP
	d_x[tid] = rng_state; // store RNG state into global again
	#endif
}


/*
// using Shared Memory // does not improve performance as it is now 
__global__ void CudaKernel_RecalculateForce(const FLOAT *a, FLOAT *b, const FLOAT *d, const FLOAT uniform_field, int nx){
	const unsigned int tid = blockIdx.x*blockDim.x+threadIdx.x;
	const unsigned int tidx = threadIdx.x;
	// move thread RNG state to registers. 
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];

	unsigned int index;
	for (unsigned int iFrame=0; iFrame<(nx/FRAME1D); iFrame++) {
		index = tid + FRAME1D*iFrame;

		__shared__ FLOAT as[TILE+2]; 
		__shared__ FLOAT ds[TILE+2]; 
		__shared__ FLOAT bs[TILE+2];
		as[tidx+1] = a[index]; 
		ds[tidx+1] = d[index]; 
		if(tidx==0){ 
			unsigned int index_m1 = (index==0)*(nx-1)+(index!=0)*(index-1);
			as[tidx] = a[index_m1]; 
		}
		if(tidx==TILE-1){
			unsigned int index_p1 = (index!=(nx-1))*(index+1);
			as[tidx+2] = a[index_p1];
		}

		__syncthreads();

		const FLOAT right = as[tidx+2];
 		const FLOAT left = as[tidx]; 
		
		const FLOAT elastic_force = C_BETA*(left -2*as[tidx+1] + right); 

		const FLOAT pinning_force = C_EPSILON*ds[tidx+1]; 

		const FLOAT thermal_force = XTEMP*(rand_MWC_co(&rng_state, &rng_const)-0.5);

		bs[tidx+1] = elastic_force + pinning_force + thermal_force + uniform_field; 

		__syncthreads(); 

		b[index] = bs[tidx+1];
	}
	d_x[tid] = rng_state; // store RNG state into global again
}
*/

__global__ void CudaKernel_EulerStep(const FLOAT *a, FLOAT *b, int nx){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < nx ){
		b[idx] += a[idx]*DT;
	}	
}

////////// Rescaling/Casting functions //////////

__global__ void CudaKernel_R2RScaled(cufftReal *a, int nx, int ny, FLOAT scale){
	// compute idx and idy, the location of the element in the original LL*MM array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx && idy < ny){
		int index = idx*ny + idy;
		a[index]= scale*a[index];
	}	
}

__global__ void CudaKernel_C2CScaled(COMPLEX *a, unsigned int nx, unsigned int ny, FLOAT scale){
	// compute idx and idy, the location of the element in the original LL*MM array 
	unsigned int idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx && idy < ny){
		unsigned int index = idx*ny + idy;
		a[index].x= scale*a[index].x;
		a[index].y= scale*a[index].y;
	}	
}

__global__ void CudaKernel_CastFloatToCufftReal(const FLOAT *a, COMPLEX *b, int nx){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < nx ){
		b[idx].x = a[idx];
		b[idx].y = 0;
	}
}

////////// Calculate/Accumulate functions //////////

__global__ void CudaKernel_CalculateStructureFactor(const COMPLEX *a, FLOAT *b, int nx){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	if ( idx < nx ){
		b[idx] = a[idx].x * a[idx].x + a[idx].y * a[idx].y ;
	}
}

__global__ void CudaKernel_AccumulateStructureFactor(const FLOAT *a, FLOAT *b, int nx, int i){
	int idx = blockIdx.x*blockDim.x+threadIdx.x;

	if (idx < nx){
		b[i*nx + idx] += a[idx]/(FLOAT) nx ;
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
	FLOAT *d_u, *h_u;

	//complex array to hold F[u] on device (line)
	COMPLEX *d_u_complex;

	//FLOAT arrays to hold the disorder (matrix)
	FLOAT *d_disorder, *h_disorder;
	
	//FLOAT arrays for the disorder spline (matrix)
	FLOAT *d_spline, *h_spline;

	//FLOAT array for the interpolated (local) disorder
	FLOAT *d_splint;

	//real array for the force
	FLOAT *d_force;
	
	//Arrays for the physical quantities
	unsigned int *tabulatedtime;
	//FLOAT array to hold the Structure Factor 
	FLOAT *d_structure_factor, *h_structure_factor;
	//FLOAT array to accumulate the Structure Factor 
	FLOAT *d_structure_factor_accum, *h_structure_factor_accum;
	//FLOAT array to accumulate the Center of Mass Position 
	FLOAT *h_center_mass_accum, *h_width_accum; 
	FLOAT *h_velocity_accum, *h_velocityvariance_accum;

	typedef FLOAT TT;
	summary_stats_data<TT> result;

	thrust::device_vector<FLOAT> histogram_accum;

	// intializing the class
	elastic_system(){

		//allocating memory for the system array in host and device (vector)
		h_u = (FLOAT *)malloc(sizeof(FLOAT)*LL);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_u, sizeof(FLOAT)*LL));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_u_complex, sizeof(COMPLEX)*(LL/2+1)*BATCH));
		//allocating memory for the disorder matrix in host and device 
		h_disorder = (FLOAT *)malloc(sizeof(FLOAT)*LL*MM);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_disorder, sizeof(FLOAT)*LL*MM));
		//allocating memory for the disorder spline (matrix)
		h_spline = (FLOAT *)malloc(sizeof(FLOAT)*LL*MM);
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_spline, sizeof(FLOAT)*LL*MM));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_splint, sizeof(FLOAT)*LL));
		//allocating memory for the force (vector)
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_force, sizeof(FLOAT)*LL));
		//allocating memory for the Structure Factor
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_structure_factor, sizeof(FLOAT)*(LL/2+1)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_structure_factor_accum, sizeof(FLOAT)*(NPOINTS*(LL/2+1))));
		h_structure_factor = (FLOAT *)malloc(sizeof(FLOAT)*(LL/2+1));
		h_structure_factor_accum = (FLOAT *)malloc(sizeof(FLOAT)*NPOINTS*(LL/2+1));
		//allocating memory for the Center of Mass
		//CUDA_SAFE_CALL(cudaMalloc((void**)&d_center_mass_accum, sizedata_cm));
		h_center_mass_accum = (FLOAT *)malloc(sizeof(FLOAT)*NPOINTS);
		h_width_accum = (FLOAT *)malloc(sizeof(FLOAT)*NPOINTS);
		h_velocity_accum = (FLOAT *)malloc(sizeof(FLOAT)*NPOINTS);
		h_velocityvariance_accum = (FLOAT *)malloc(sizeof(FLOAT)*NPOINTS);
		tabulatedtime = (unsigned int *)malloc(NPOINTS*sizeof(unsigned int));

		histogram_accum.resize(NBINS*(T_RUN_EXP+1));

		/* Fill everything with 0s */
		CUDA_SAFE_CALL(cudaMemset(d_u, 0, LL*sizeof(FLOAT)));
		CUDA_SAFE_CALL(cudaMemset(d_u_complex, 0, (LL/2+1)*sizeof(COMPLEX)));
		if ( cudaGetLastError ( ) != cudaSuccess ) {
		fprintf (stderr , "Cuda error : Failed to allocate \n" ) ;
		return ;
		};

		CUDA_SAFE_CALL(cudaMemset(d_disorder, 0, LL*MM*sizeof(FLOAT)));
		CUDA_SAFE_CALL(cudaMemset(d_spline, 0, LL*MM*sizeof(FLOAT)));
		
		CUDA_SAFE_CALL(cudaMemset(d_splint, 0, LL*sizeof(FLOAT)));
		
		CUDA_SAFE_CALL(cudaMemset(d_force, 0, LL*sizeof(FLOAT)));

		CUDA_SAFE_CALL(cudaMemset(d_structure_factor, 0, (LL/2+1)*sizeof(FLOAT)));
		CUDA_SAFE_CALL(cudaMemset(d_structure_factor_accum, 0, sizeof(FLOAT)*NPOINTS*(LL/2+1)));

		thrust::fill(histogram_accum.begin(),histogram_accum.end(),0);

		//CUDA_SAFE_CALL(cudaMemset(d_center_mass, 0, (T_RUN/T_DATA+1)*sizeof(FLOAT)));
	
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
		for (unsigned int k=0; k<NPOINTS*(LL/2+1); k++){
			h_structure_factor_accum[k] = 0.;
		}	
		for (unsigned int k=0; k<NPOINTS; k++){
			h_width_accum[k] = 0.;
			h_center_mass_accum[k] = 0.;
			h_velocity_accum[k] = 0.;
			h_velocityvariance_accum[k] = 0.;
		}

		// plan R2C
		#ifdef DOUBLE_PRECISION
		if (cufftPlan1d(&Sr2c, LL, CUFFT_D2Z, BATCH) != CUFFT_SUCCESS ){
		fprintf ( stderr , "CUFFT error : Plan creation failed " ) ;
		return ;
		};
		#else
		if (cufftPlan1d(&Sr2c, LL, CUFFT_R2C, BATCH) != CUFFT_SUCCESS ){
		fprintf ( stderr , "CUFFT error : Plan creation failed " ) ;
		return ;
		};
		#endif

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
		CUDA_SAFE_CALL(cudaMemcpy(h_structure_factor, d_structure_factor, sizeof(FLOAT)*(LL/2+1), cudaMemcpyDeviceToHost));
		for(unsigned int i=1;i<(LL/2+1);i++)
		{
		//with lattice effect standard correction
		//fstr << 2*sin(M_PI*i/(FLOAT) LL) << "  " << h_structure_factor[i] << endl;
		//pure mode q
		fstr << 2*M_PI*i/(FLOAT) LL << "  " << h_structure_factor[i] << endl;
		}
		fstr << endl;
	};

	void PrintStructureFactorAccum(ofstream &fstr, int s){
		CUDA_SAFE_CALL(cudaMemcpy(h_structure_factor_accum, d_structure_factor_accum, sizeof(FLOAT)*NPOINTS*(LL/2+1), cudaMemcpyDeviceToHost));
		fstr << "#Structure Factor for" << s << "samples " << endl;
		for(int index=0; index<NPOINTS; index++){
			fstr << "#Structure Factor for time " << tabulatedtime[index] << endl;
			for(unsigned int i=1;i<(LL/2+1);i++)
			{
			//with lattice effect standard correction
			//fstr << 2*sin(M_PI*i/(FLOAT) LL) << "  " << h_structure_factor_accum[index*(LL/2+1) + i]/(FLOAT) s << endl;
			//pure mode q
			fstr << 2*M_PI*i/(FLOAT) LL << "  " << h_structure_factor_accum[index*(LL/2+1) + i]/(FLOAT) s << endl;
			}
			fstr << endl  << endl;
		}
		fstr << endl;

	};

	void PrintVelocityHistogramAccum(ofstream &fstr, int s, FLOAT leftlimit, FLOAT rightlimit){
		//thrust::host_vector<int> H = histogram_accum;
		FLOAT increment= (rightlimit-leftlimit)/NBINS;

		fstr << "#Velocity histogram for" << s << "samples " << endl;
		for(int index=0; index<(T_RUN_EXP+1); index++){
			fstr << "#Velocity histogram for two to the" << index << endl;
			for(int i=0;i<NBINS;i++)
			{
			fstr << leftlimit+(i)*increment << " " << histogram_accum[index*NBINS+i] / FLOAT(s) << endl;
			}
			fstr << endl << endl;
		}
		fstr << endl;
	};


	void PrintCenterOfMassAccum(ofstream &fstr, int s){
		fstr << "#Center of Mass possition for" << s << "samples " << endl;
		for(int index=0; index<NPOINTS; index++){
			fstr << tabulatedtime[index] << "  " << h_center_mass_accum[index]/(FLOAT) s \
			<< "  " << h_width_accum[index]/(FLOAT) s << endl;
		}
		fstr << endl;
	};

	void PrintVelocityAccum(ofstream &fstr,  int s){
		fstr << "#Center of Mass velocity for" << s << "samples " << endl;
		for(int index=0; index<NPOINTS; index++){
			FLOAT sigma = sqrt(h_velocity_accum[index]*h_velocity_accum[index] /(FLOAT) (s*s) - h_velocityvariance_accum[index] /(FLOAT) (s*s));
			fstr << tabulatedtime[index] << "  " << h_velocity_accum[index]/(FLOAT) s << "  " << sigma << endl;
		}
		fstr << endl;
	};

	void PrintDisorder(ofstream &fstr){
		for(int i=0;i<LL;i++)
		{
			for(int j=0;j<MM;j++)
			{
				int k=i*MM+j;
				fstr << setprecision(10) << h_disorder[k] << endl; //<< " " ;
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

		#ifdef DOUBLE_PRECISION
		CUFFT_SAFE_CALL(cufftExecD2Z(Sr2c, d_u, d_u_complex));
		#else
		CUFFT_SAFE_CALL(cufftExecR2C(Sr2c, d_u, d_u_complex));
		#endif
		//TODO: check if normalization is needed
		
		//CUDA_SAFE_CALL(cudaDeviceSynchronize()); // Not needed. Here only for debug purposes.	
		CudaKernel_CalculateStructureFactor<<<dimGrid, dimBlock>>>(d_u_complex, d_structure_factor, LL/2+1);
	};


/*	
	// apply scaling ( an FFT followed by iFFT will give you back the same array times the length of the transform)
	void Normalize(void){
		dim3 dimBlock(TILE);
		dim3 dimGrid(LL/TILE);
		assert(LL%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);
		CudaKernel_C2CScaled<<<dimGrid, dimBlock>>>(d_u_complex, LL, 1.f / (FLOAT) LL));
	}
*/


	void TabulateTime(unsigned int index, unsigned int time){
		tabulatedtime[index]=time;
	}


	void AccumulateStructureFactor(unsigned int index, unsigned int time){
		assert(tabulatedtime[index]==time);
		dim3 dimBlock(TILE);
		dim3 dimGrid(LL/TILE);
		assert(LL%TILE==0);
		assert(dimBlock.x<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID);

		CudaKernel_AccumulateStructureFactor<<<dimGrid, dimBlock>>>(d_structure_factor, d_structure_factor_accum, LL/2+1, index);
	};

	void CalculateAccumulateCenterOfMass(unsigned int index, unsigned int time){
		assert(tabulatedtime[index]==time);
		//thrust::device_ptr<FLOAT> dev_ptr(d_u);
		//FLOAT cm = thrust::reduce(dev_ptr, dev_ptr + LL) / (FLOAT) LL;
		//h_center_mass_accum[index] += cm;

		MeanVariance(d_u, LL, result);
		h_center_mass_accum[index] += result.mean;
		h_width_accum[index] += result.variance();
	};

	void CalculateAccumulateVelocity(unsigned int index, unsigned int time){
		assert(tabulatedtime[index]==time);
		thrust::device_ptr<FLOAT> dev_ptr(d_force);

		FLOAT velocity = thrust::reduce(dev_ptr, dev_ptr + LL) / (FLOAT) LL;
		h_velocity_accum[index] += velocity;
		h_velocityvariance_accum[index] += velocity*velocity;

		};

	void CalculateAccumulateVelocityHistogram(unsigned int index, unsigned int time, FLOAT leftlimit, FLOAT rightlimit, int numberofbins){
		thrust::device_ptr<FLOAT> dev_ptr(d_force);
		thrust::device_vector<FLOAT> input;
		input.resize(LL);
		thrust::copy(dev_ptr, dev_ptr + LL, input.begin());

		thrust::device_vector<FLOAT> histogram(numberofbins);
		dense_histogram_data_on_device(input, histogram, leftlimit, rightlimit);

		thrust::transform(histogram.begin(), histogram.end(),\
				 histogram_accum.begin()+index*numberofbins, histogram_accum.begin()+index*numberofbins, thrust::plus<FLOAT>());
	}

	FLOAT CalculateCenterOfMass(){
		thrust::device_ptr<FLOAT> dev_ptr(d_u);
		FLOAT cm = thrust::reduce(dev_ptr, dev_ptr + LL) / (FLOAT) LL;
		return cm;
	};

	///////// Setup and Update Functions ///////////

	void InitFlat(FLOAT offset){ // value=0 is default
		CUDA_SAFE_CALL(cudaMemset(d_u, offset, LL*sizeof(FLOAT)));
	};
	
	void InitParticular(unsigned int nx, int shape, FLOAT offset){
		for(unsigned int i=0;i<nx;i++){
			//a flat line
			if (shape==0) h_u[i]= offset; 
			//a sinusoidal line
			if (shape==1) h_u[i]= offset + sin(2*M_PI*i/FLOAT(nx));
		}
	 	CpyHostToDevice();
	};

	/* transfer the system from CPU to GPU memory */
	void CpyHostToDevice(){
		CUDA_SAFE_CALL(cudaMemcpy(d_u, h_u, sizeof(FLOAT)*LL, cudaMemcpyHostToDevice));
	};
	
	/* transfer the system from GPU to CPU memory */
	void CpyDeviceToHost(){	
		CUDA_SAFE_CALL(cudaMemcpy(h_u, d_u, sizeof(FLOAT)*LL, cudaMemcpyDeviceToHost));
	};

	void SetDisorder(){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(FRAME2D/TILE_X, FRAME2D/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		
		CudaKernel_SetDisorder<<<dimGrid, dimBlock>>>(d_disorder,LL,MM);
		CUDA_SAFE_CALL(cudaMemcpy(h_disorder, d_disorder, sizeof(FLOAT)*LL*MM, cudaMemcpyDeviceToHost));

		//TODO: Move d_disorder to texture memory
		//CUDA_SAFE_CALL(cudaUnbindTexture(DisorderTexture));
		//CUDA_SAFE_CALL(cudaBindTexture(NULL, DisorderTexture, d_disorder, LL*MM*sizeof(FLOAT)));
	};

	void CreateSpline(){
		FLOAT *y, *y2;
		y = (FLOAT *)malloc(sizeof(FLOAT)*MM);
		y2 = (FLOAT *)malloc(sizeof(FLOAT)*MM);

		for (unsigned int j=0; j<MM; j++){
			y2[j] = 0.;
		}
	
		//We need to createn spline for each line separately
		//Creating array y[] for each row and writing the result of NR "spline" function in h_spline 

		for (unsigned int i=0; i<LL; i++){
			for (unsigned int j=0; j<MM; j++){
				y[j]= h_disorder[i*MM + j];
			}
			periodic_spline(y,MM,y2);
			for (unsigned int j=0; j<MM; j++){
				h_spline[i*MM + j] = y2[j];
			}
		}
	
		CUDA_SAFE_CALL(cudaMemcpy(d_spline,h_spline,sizeof(FLOAT)*LL*MM, cudaMemcpyHostToDevice));
		free(y);
		free(y2);

		//TODO: Move d_spline to texture memory
		//CUDA_SAFE_CALL(cudaUnbindTexture(SplineTexture));
		//CUDA_SAFE_CALL(cudaBindTexture(NULL, SplineTexture, d_spline, LL*MM*sizeof(FLOAT)));
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
	
	void RecalculateForce(FLOAT uniform_field, unsigned int t){
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
	

	void cyclic(FLOAT *a, FLOAT *b, FLOAT *c, FLOAT alpha, FLOAT beta, FLOAT *r, FLOAT *x, unsigned int n)
	//Solves for a vector x[1..n] the \cyclic" set of linear equations given by equation (2.7.9). a, b, c, and 
	//r are input vectors, all dimensioned as [1..n], while alpha and beta are the corner
	//entries in the matrix. The input is not modified.
	{
		//void tridag(FLOAT *a, FLOAT *b, FLOAT *c, FLOAT *r, FLOAT *u, unsigned int n);
		unsigned int i;
		FLOAT fact,gamma,*bb,*u,*z;

		//if (n <= 2) nrerror("n too small in cyclic");
		bb = (FLOAT *)malloc(sizeof(FLOAT)*(n));
		u = (FLOAT *)malloc(sizeof(FLOAT)*(n));
		z = (FLOAT *)malloc(sizeof(FLOAT)*(n));
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


	void tridag(const FLOAT *a, const FLOAT *b, const FLOAT *c, const FLOAT *r, FLOAT *u, unsigned int n)
		//Solves for a vector u[1..n] the tridiagonal linear set given by equation (2.4.1). a[1..n],
		//b[1..n], c[1..n], and r[1..n] are input vectors and are not modified.
		{
		int j;
		FLOAT bet,*gam;
		gam = (FLOAT *)malloc(sizeof(FLOAT)*(n)); //One vector of workspace, gam is needed.
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


	void periodic_spline(FLOAT *y, unsigned int n, FLOAT *y2){

		FLOAT *a, *b, *c, *r;
		a = (FLOAT *)malloc(sizeof(FLOAT)*(n));
		b = (FLOAT *)malloc(sizeof(FLOAT)*(n));
		c = (FLOAT *)malloc(sizeof(FLOAT)*(n));
		r = (FLOAT *)malloc(sizeof(FLOAT)*(n));
		
		unsigned int i;
		for(i=0;i<n;i++){a[i]= 1/6.;}
		for(i=0;i<n;i++){b[i]= 2/3.;}
		for(i=0;i<n;i++){c[i]= 1/6.;}

		r[0] = y[1] -2*y[0] + y[n-1];
		for(i=1;i<n-1;i++) r[i] = y[i+1] -2*y[i] +y[i-1];
		r[n-1] = y[0] -2*y[n-1] +y[n-2];

		FLOAT alpha = c[0];
		FLOAT beta = a[0];
		cyclic(a,b,c,alpha,beta,r,y2,n);

		free(a); free(b); free(c); free(r);
	}

	/////////////// Clean Memory ////////////////
	~elastic_system(){
		cufftDestroy(Sr2c);
		cudaFree(d_u); cudaFree(d_u_complex); 
		cudaFree(d_disorder); cudaFree(d_spline); cudaFree(d_splint); cudaFree(d_force);
		cudaFree(d_structure_factor); cudaFree(d_structure_factor_accum);
		free(h_u); 
		free(h_disorder); free(h_spline); 
		free(h_structure_factor); free(h_structure_factor_accum); free(h_center_mass_accum);
	};

};

#endif /* QEW_LIBRARY_CUH_ */
