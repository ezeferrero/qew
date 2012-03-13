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

// NOTE: Not sure how many of the followin libraries are necessary
#include <cmath>
#include <stdio.h> /* printf */
#include <stdlib.h>
#include <iostream> /*para los printf*/
#include <iomanip> /*para los printf*/
#include <fstream> /*para los printf*/
#include <assert.h>
#include <sys/time.h> /* gettimeofday */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "cutil.h" // CUDA_SAFE_CALL, CUT_CHECK_ERROR
/////////////////////////////////

/*Parameter Definitions*/
#ifndef L
#define L 512
#endif

#ifndef M
#define M 256
#endif

#ifndef T_RUN
#define	T_RUN 1000
#endif

#ifndef T_DATA
#define T_DATA 100
#endif

#ifndef SAMPLES
#define SAMPLES 10
#endif

#ifndef DT
#define DT 0.01f
#endif

#ifndef C_EPSILON
#define C_EPSILON 0.5f	/*magnitud del desorden que se le suma a Calpha, si es 0.f no hay desorden*/
#endif

#ifndef C_BETA
#define C_BETA 1.0f	/*factor termino Laplaciano phi*/
#endif

#ifndef TEMP
#define TEMP 0.1f
#endif
//#define	xtemp sqrt(24.*TEMP*DT)  
#define	xtemp sqrt(24.*TEMP/DT)

#ifndef H_START
#define H_START 0.0f
#endif

#ifndef DHDT
#define DHDT 0.1f
#endif

#ifndef T_START_FIELD
#define T_START_FIELD 50
#endif

// Functions (in case they're needed)
// maximum
#define MAX(a,b) (((a)<(b))?(b):(a))
// minimum
#define MIN(a,b) (((a)<(b))?(a):(b))
//absolute value
#define ABS(a) (((a)< 0)?(-a):(a))

// Hardware parameters for GTX 470/480 (GF100)
#define CUDA_DEVICE 0	// card number
#define SHARED_PER_BLOCK 49152
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 65535

// Used for cuFFT-1D porpouses
#define BATCH 1

// RNG: multiply with carry
#include "../common/CUDAMCMLrng.cu"

/*for the MWC RNG*/
#define FRAME1D MIN(L,131072)	// the whole thing should be framed for the RNG if L>2^17
#define FRAME2D 256		// the whole thing is framed for the RNG
#define TILE 128		// each block of threads is a tile
#define TILE_X 16		// each block of threads is a tile
#define TILE_Y 32		// each block of threads is a tile

//TODO: FRAME_X, FRAME_Y and change the following:
#define NUM_THREADS_2D (FRAME2D*FRAME2D) 
#define NUM_THREADS_1D (FRAME1D)
#define NUM_THREADS MAX(NUM_THREADS_1D, NUM_THREADS_2D) 
#define MICROSEC (1E-6)
#define SEED time(NULL)

//Good mults files
#define SAFE_PRIMES_FILENAME "../common/safeprimes_base32.txt"
//#define SAFE_PRIMES_FILENAME "../common/safeprimes_base32_plus.txt"

// state of the random number generator, last number (x_n), last carry (c_n) packed in 64 bits
__device__ static unsigned long long d_x[NUM_THREADS];
// multipliers (constants)
__device__ static unsigned int d_a[NUM_THREADS];

#include "EW_library.cuh"
//#include "../common/nrutil.h"

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
	assert(L%FRAME1D==0);
	assert(L%FRAME2D==0);
	assert(M%FRAME2D==0);
	assert(T_RUN%T_DATA==0);
	assert(DT<=0.2);	//We've assumed a small temporal step in the integration scheme
	
	// set the GPGPU computing device
	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE));

	CUDA_SAFE_CALL(cudaThreadSetCacheConfig(cudaFuncCachePreferL1)); // Less Shared Memory, more cache.
	
	double secs = 0.0;
	struct timeval start = {0L,0L}, end = {0L,0L}, elapsed = {0L,0L};

	//char picturename [100];
	char filename [200];

	// Start timer
	gettimeofday(&start, NULL);	

	// Initialize the RNGs
	if (ConfigureRandomNumbers()) {
		return 1;
	}

	//Our class
	elastic_system T;

	for(int s=0; s < SAMPLES; s++){
		/*choose a particular initial condition for the line (made on the host)*/
	 	T.InitParticular(L);
	 	T.CpyHostToDevice();

		/* choose a random initial condition for the line (made on the device)*/
	 	//T.init_random(); // Not a good idea!!

		/*Initializes the quenched disorder and creates the cubic spline*/
		T.SetDisorder();
		T.CreateSpline();

		float h_field = H_START; 

		// Time loop, from 0 to T_RUN
		for(unsigned int time=0;time<T_RUN; time++){
		    /* get data each T_DATA steps*/
		    if(time%T_DATA==0) {
		      
		      //T.CpyDeviceToHost(); /*we'll use it in a moment*/
		      
		      T.GetSplints();
		      T.RecalculateForce(h_field, time); /*Aca le estoy pasando el parametro del campo que va variando, */

		      /*incremento temporal*/
		      T.EulerStep(time);

		      //Do some calculation and print

		      //sprintf(filename, "sample%dframe%d.dat", s, 100000000+time);
		      //ofstream file0(filename);
		      //T.print_input(file0);
		      
		      //T.PrintCenterOfMass(time);
		      
		      T.CalculateStructureFactor();

		      //sprintf(filename, "sample%dSk%d.dat", s, 100000000+time);
		      //ofstream file1(filename);
		      //T.print_structure_factor(file1);

		      T.AcumulateStructureFactor(time);

		    }else{

	      	      T.GetSplints();
		      T.RecalculateForce(h_field, time); /*Aca le estoy pasando el parametro del campo que va variando*/

		      /*incremento temporal*/
		      T.EulerStep(time);
		    }
		    
		    /* different field protocols */
		    //h_field=(i*DT>3000)?(0.01-0.0000005*(i*DT-3000)):(0.01);
		    //h_field=(i*DT>T_START_FIELD)?(H_START+DHDT*(i*DT-T_START_FIELD)):(H_START);

		    //if (i<58000) h_field=(i*DT>T_START_FIELD)?(H_START+DHDT*(i*DT-T_START_FIELD)):(H_START);
		    //if (i>=58000) h_field=-0.018;
		    
		    /*tur it ON at T_START_FIELD*/
		    if (time>=T_START_FIELD) h_field = H_START + DHDT;
		    
		    /* field pulses */
		    //h_field=(i*DT>3000)?(0.01-0.0000005*(i*DT-3000)):(0.01);
		    //h_field=(i*DT>T_START_FIELD)?(H_START+DHDT*((i/400)%2)):(H_START);
		    
		    /*constant field*/
		    // Do nothing, it stays in H_START
		}

	      sprintf(filename, "StructureFactor.dat");
	      ofstream file2(filename);
	      T.print_structure_factor_acum(file2);

	}


	// Stop timer
	gettimeofday(&end,NULL);
	timeval_subtract(&elapsed, &end, &start);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("# Tamaño/Tiempo de ejecucion CUDA %i %i %lf \n", L, M, secs);
	
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

