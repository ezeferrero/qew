/*
 * EW_Langevin_CUDA
 *
 *  Simulates the 1-dimensional elastic line in a disordered 2-dimensional media.	
 *  Disorder potential is choosen from a Gaussian distribution	
 *  Elastic restoring force is given by a discretized Laplacian	(elastic limit of small distortions)
 *  An uniform external field can be applied
 *  Langevin noise is simulated with a uniform random distribution
 *
 *  Created on: Feb 26, 2012
 *      Author: eze
*/

// TODO: Not sure how many of the followin libraries are necessary
#include <cmath>
#include <stdio.h> 	/* printf */
#include <stdlib.h>
#include <iomanip> 	/* print stuff*/
#include <fstream> 	/* print stuff*/
#include <assert.h>	/* assertions */
#include <sys/time.h> 	/* gettimeofday */

#include <cuda.h>
#include <cuda_runtime.h>
#include "cutil.h"	/* CUDA_SAFE_CALL, CUT_CHECK_ERROR */
#include <cufft.h>	/* CUDA Fast Fourier Transform Library */

/*Parameter Definitions*/
#ifndef LL
#define LL 512		/* System's longitudinal size (line size) */
#endif

#ifndef MM
#define MM 1024		/* System's transversal size*/
#endif

#ifndef SAMPLES
#define SAMPLES 10	/* Numer of averaged samples */
#endif

#ifndef T_RUN
#define	T_RUN 1000	/* Running time */
#endif

#ifndef T_DATA
#define T_DATA 100	/* Data acquisition time */
#endif

#ifndef C_BETA
#define C_BETA 1.0f	/* Laplacian prefactor */
#endif

#ifndef C_EPSILON
#define C_EPSILON 1.0f	/* Intensity of the disorder potential */
#endif

#ifndef TEMP
#define TEMP 0.1f	/* Heat Bath temperature*/
#endif
#define	xtemp sqrt(24.*TEMP/DT) 
//#define xtemp sqrt(24.*TEMP*DT)

#ifndef H_START
#define H_START 0.0f	/* Uniform external field */
#endif

#ifndef DHDT
#define DHDT 0.f	/* External field changing rate */
#endif

#ifndef T_START_FIELD
#define T_START_FIELD 50 /* Time to start changing the field */
#endif

#ifndef DT
#define DT 0.1f		/* Integration step time */
#endif

#ifndef CUDA_DEVICE
#define CUDA_DEVICE 0	/* GPU to be used ID */
#endif

// Some Functions (just in case)
#define MAX(a,b) (((a)<(b))?(b):(a))	/* maximum */
#define MIN(a,b) (((a)<(b))?(a):(b))	/* minimum */ 
#define ABS(a) (((a)< 0)?(-a):(a))	/* absolute value */

// Hardware parameters for GTX 470/480 (GF100)
#define SHARED_PER_BLOCK 49152
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 65535

// Used for cuFFT-1D purposes
#define BATCH 1

// RNG: Multiply With Carry
#include "../common/CUDAMCMLrng.cu"

// Parameters to frame the system according to the maximum number of independent MWC RNGs
#define FRAME1D MIN(LL,131072)	// the whole thing should be framed for the RNG if LL>2^17
#define FRAME2D 256		// the whole thing is framed for the RNG
#define TILE 128		// each block of threads is a tile
#define TILE_X 16		// each block of threads is a tile
#define TILE_Y 32		// each block of threads is a tile
#define NUM_THREADS_2D (FRAME2D*FRAME2D) 
#define NUM_THREADS_1D (FRAME1D)
#define NUM_THREADS MAX(NUM_THREADS_1D, NUM_THREADS_2D) 
#define MICROSEC (1E-6)
#define SEED time(NULL)

// Safe-Primes file used by the MWC RNG
#define SAFE_PRIMES_FILENAME "../common/safeprimes_base32.txt"
//#define SAFE_PRIMES_FILENAME "../common/safeprimes_base32_plus.txt"

// State of the random number generator, last number (x_n), last carry (c_n) packed in 64 bits
__device__ static unsigned long long d_x[NUM_THREADS];
// multipliers (constants)
__device__ static unsigned int d_a[NUM_THREADS];

// Including our functions
#include "EW_library.cuh"

int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y);

static int ConfigureRandomNumbers(void) {
	/* Allocate memory for RNG's*/
	unsigned long long h_x[NUM_THREADS];
	unsigned int h_a[NUM_THREADS];
	unsigned long long seed = (unsigned long long) SEED;

	/* Init RNG's*/
	int error = init_RNG(h_x, h_a, NUM_THREADS, SAFE_PRIMES_FILENAME, seed);

	if (!error) {
		size_t size_x = NUM_THREADS * sizeof(unsigned long long);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_x, h_x, size_x));

		size_t size_a = NUM_THREADS * sizeof(unsigned int);
		assert(size_a<size_x);
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(d_a, h_a, size_a));
	}

	return error;
}

/////////////////////////////////////////////////////////
int main(){
	assert(TILE%2==0);
	assert(TILE_X%2==0);
	assert(TILE_Y%2==0);
	assert(FRAME1D%TILE==0);	
	assert(FRAME2D%TILE_X==0);
	assert(FRAME2D%TILE_Y==0);
	assert(LL%FRAME1D==0);
	assert(LL%FRAME2D==0);
	assert(MM%FRAME2D==0);
	assert(T_RUN%T_DATA==0);
	assert(DT<=0.2);	//We've assumed a small temporal step in the integration scheme
	
	// Set the GPGPU computing device
	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE));

	// Choosing less "shared memory", more cache.
	CUDA_SAFE_CALL(cudaThreadSetCacheConfig(cudaFuncCachePreferL1)); 
	
	double secs = 0.0;
	struct timeval start = {0L,0L}, end = {0L,0L}, elapsed = {0L,0L};

	// Start timer
	gettimeofday(&start, NULL);	

	// Print header
	printf("# L: %i\n", LL);
	printf("# M: %i\n", MM);
	printf("# Number of Samples: %i\n", SAMPLES);
	printf("# Running time (in Euler steps): %i\n", T_RUN);
	printf("# Data Acquiring Step: %i\n", T_DATA);
	printf("# beta: %f\n", C_BETA);
	printf("# epsilon: %f\n", C_EPSILON);
	printf("# Dt for Euler step: %f\n", DT);
	printf("# Temperature: %f\n", TEMP);
	printf("# External field: %f\n", H_START);
	printf("# SEED used: %i\n", SEED);

	// Initialize the RNGs
	if (ConfigureRandomNumbers()) {
		return 1;
	}

	// main{} variables
	char filename [200];

	// Our class
	elastic_system T;

	for(unsigned int s=0; s < SAMPLES; s++){

		/* Choose a particular initial condition for the line (made on the host)*/
	 	T.InitParticular(LL, 0, 10.); // InitParticular(L, shape, offset) shape: 0:flat; 1:sinusoidal, ..... more to come

		/* Flat initial condition for the line (made on the device)*/
	 	//T.InitFlat(1.); // InitFlat(offset)

		/*Initializes the quenched disorder and creates the cubic spline*/
		T.SetDisorder();
		T.CreateSpline();

		float h_field = H_START; 

		// Time loop, from 0 to T_RUN
		for(unsigned int time=0;time<T_RUN+1; time++){
			/* Get data each T_DATA steps*/
			if(time%T_DATA==0){
				// Uncomment visualization is desired
				/*if(s==0){
					T.CpyDeviceToHost(); 
					sprintf(filename, "sample%dframe%d.dat", s, 100000000+time);
					ofstream file0(filename);
					T.print_input(file0);
				}*/
			      
				/* Calculate all external forces*/
				T.InterpolateDisorderPotential();
				T.RecalculateForce(h_field, time);

				/* Integration step*/
				T.EulerStep(time);

				/* Calculate some quantities*/	
				T.CalculateAccumulateCenterOfMass(time);
				T.CalculateStructureFactor();
				T.AccumulateStructureFactor(time);

			}else{
				/* Calculate all external forces*/
				T.InterpolateDisorderPotential();
				T.RecalculateForce(h_field, time);

				/* Integration step*/
				T.EulerStep(time);
			}
		    
			/* Choose external field protocol */

			//h_field=(i*DT>3000)?(0.01-0.0000005*(i*DT-3000)):(0.01);
			//h_field=(i*DT>T_START_FIELD)?(H_START+DHDT*(i*DT-T_START_FIELD)):(H_START);

			//if (i<58000) h_field=(i*DT>T_START_FIELD)?(H_START+DHDT*(i*DT-T_START_FIELD)):(H_START);
			//if (i>=58000) h_field=-0.018;
		    
			/*Increment it at T_START_FIELD*/
			//if (time>=T_START_FIELD) h_field = H_START + DHDT;
		    
			/* field pulses */
			//h_field=(i*DT>3000)?(0.01-0.0000005*(i*DT-3000)):(0.01);
			//h_field=(i*DT>T_START_FIELD)?(H_START+DHDT*((i/400)%2)):(H_START);
		    
			/*constant field*/
			// Do nothing, it stays in H_START

			//CUDA_SAFE_CALL(cudaUnbindTexture(SplineTexture));	
		}
		/* Print Structure Factor in a temporal file after each sample finishes*/
		sprintf(filename, "Sk_L%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.2fdt%.2fgpu%i.tmp",LL,MM,SAMPLES,T_RUN,T_DATA,TEMP,C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
		ofstream file2(filename);
		T.print_structure_factor_acum(file2);
		printf("# unnormalized Structure Factor for %i samples in %s\n", s, filename);

	}
	/* Print final averages of Structure Factor and Center of Mass position*/

	sprintf(filename, "Sk_L%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.2fdt%.2fgpu%i.dat",LL,MM,SAMPLES,T_RUN,T_DATA,TEMP,C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
	ofstream file2(filename);
	T.print_structure_factor_acum(file2);
	printf("# Structure Factor averaged over %i samples in %s\n", SAMPLES, filename);

	sprintf(filename, "CM_L%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.2fdt%.2fgpu%i.dat",LL,MM,SAMPLES,T_RUN,T_DATA,TEMP,C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
	ofstream file3(filename);
	T.print_center_of_mass_acum(file3);
	printf("# Center of Mass position averaged over %i samples in %s\n", SAMPLES, filename);

	// Stop timer
	gettimeofday(&end,NULL);
	timeval_subtract(&elapsed, &end, &start);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("# Sistem size / code execution time (in secs) %i %i %lf \n", LL, MM, secs);
	
	return 0;
}

/*
 * http://www.gnu.org/software/libtool/manual/libc/Elapsed-Time.html
 * Subtract the `struct timeval' values X and Y,
 * storing the result in RESULT.
 * Return 1 if the difference is negative, otherwise 0.
 */

int timeval_subtract (struct timeval *result, struct timeval *x, struct timeval *y) {
	/* Perform the carry for the later subtraction by updating y. */
	if (x->tv_usec < y->tv_usec) {
		int nsec = (y->tv_usec - x->tv_usec) / 1000000 + 1;
		y->tv_usec -= 1000000 * nsec;
		y->tv_sec += nsec;
	}
	if (x->tv_usec - y->tv_usec > 1000000) {
		int nsec = (x->tv_usec - y->tv_usec) / 1000000;
		y->tv_usec += 1000000 * nsec;
		y->tv_sec -= nsec;
	}

	/* Compute the time remaining to wait. tv_usec is certainly positive. */
	result->tv_sec = x->tv_sec - y->tv_sec;
	result->tv_usec = x->tv_usec - y->tv_usec;

	/* Return 1 if result is negative. */
	return x->tv_sec < y->tv_sec;
}
