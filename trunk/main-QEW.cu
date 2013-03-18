/*
 * QEW_CUDA
*/

/* main-QEW.cu together with its library QEW_library.cuh, is a CUDA implementation of the
 * 1D Quenched Edwards-Wilkinson (QEW) model.
 * The program simulates a 1-dimensional elastic line in a 2-dimensional disorder medium
 * The disorder potential is choosen from a Gaussian or Uniform distribution
 * Elastic restoring force is given by a discretized Laplacian	(elastic limit of small distortions)
 * An uniform external field can be applied
 * Langevin noise with a uniform random distribution can be optonally turned on.
.*
 * Started on: Feb 26, 2012 by ezeferrero
 * Copyright (C) 2013 Ezequiel E. Ferrero, Alejandro B. Kolton and Sebastián Bustingorry.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * This code was originally implemented for: "Nonsteady relaxation and critical exponents at the depinning transition",
 * E.E. Ferrero, A.B. Kolton and S. Bustingorry
 * http://arxiv.org/abs/1211.7275
 * Phys. Rev. E 87, 032122 (2013)
 * http://link.aps.org/doi/10.1103/PhysRevE.87.032122
 * DOI:	10.1103/PhysRevE.87.032122
 * The implementation is justified and benchmarked in this Supplemental Material
 * http://pre.aps.org/supplemental/PRE/v87/i3/e032122
 * Please cite when appropriate.
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
#define LL 1024		/* System's longitudinal size (line size) */
#endif

#ifndef MM
#define MM 2048		/* System's transversal size*/
#endif

#ifndef SAMPLES
#define SAMPLES 10	/* Numer of averaged samples */
#endif

#ifndef T_RUN_EXP
#define T_RUN_EXP 10	/* Running time */
#endif

#ifndef T_RUN
#define T_RUN 1024	/* Running time */
#endif

#ifndef T_DATA
#define T_DATA 16	/* Data acquisition time */
#endif

#ifndef C_BETA
#define C_BETA 1.0f	/* Laplacian prefactor */
#endif

#ifndef C_OMEGA
#define C_OMEGA 1.0f	/* Anharmonic term prefactor */
#endif

#ifndef C_EPSILON
#define C_EPSILON 1.0f	/* Intensity of the disorder potential */
#endif

#ifndef TEMP
#define TEMP 0.1f	/* Heat Bath temperature*/
#endif
#define	XTEMP sqrt(24.*TEMP/DT)
//#define XTEMP sqrt(24.*TEMP*DT)

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

#ifdef LINEAR
#define NPOINTS (int) (T_RUN/(float) T_DATA + 1.1)
#endif

#ifdef LOGARITHMIC
#define NPOINTS ((T_RUN_EXP*5)-8)
#endif

#define NBINS 200
#define LEFTLIMIT -0.5
#define RIGHTLIMIT 1.0

// Some Functions (just in case)
#define MAX(a,b) (((a)<(b))?(b):(a))	/* maximum */
#define MIN(a,b) (((a)<(b))?(a):(b))	/* minimum */ 
#define ABS(a) (((a)< 0)?(-a):(a))	/* absolute value */
#define ROUND_DOWN_POWER_OF_2(x) (( ( (x)>>0 | (x)>>1 | (x)>>2 | (x)>>3 | (x)>>4 | (x)>>5 | (x)>>6 | (x)>>7 | (x)>>8 | \
(x)>>9 | (x)>>10 | (x)>>11 | (x)>>12 | (x)>>13 | (x)>>14 | (x)>>15 | (x)>>16 | (x)>>17 | (x)>>18 | (x)>>19 | (x)>>20 | \
(x)>>21 | (x)>>22 | (x)>>23 | (x)>>24 | (x)>>25 | (x)>>26) +1 ) >> 1)

// Hardware parameters for GTX 470/480 (GF100)
#define SHARED_PER_BLOCK 49152
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 65535

// Used for cuFFT-1D purposes
#define BATCH 1

// RNG: Multiply With Carry
#include "../common/CUDAMCMLrng.cu"
#include "../common/timers_cpu.hpp"

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
//#define SEED time(NULL)
#define SEED 1332277027

// Safe-Primes file used by the MWC RNG
#define SAFE_PRIMES_FILENAME "../common/safeprimes_base32.txt"
//#define SAFE_PRIMES_FILENAME "../common/safeprimes_base32_plus.txt"

#ifdef DOUBLE_PRECISION
typedef double FLOAT;
typedef cufftDoubleComplex COMPLEX;
#else
typedef float FLOAT;
typedef cufftComplex COMPLEX;
#endif

// State of the random number generator, last number (x_n), last carry (c_n) packed in 64 bits
__device__ static unsigned long long d_x[NUM_THREADS];
// multipliers (constants)
__device__ static unsigned int d_a[NUM_THREADS];

// Including our functions
#include "QEW_library.cuh"

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

void RunSample(int s, elastic_system &T){

	// variables
	//char filename [200];
	float h_field;

	h_field = H_START; 

	//cpu_timer reloj;
	//reloj.tic();

	/* Choose a particular initial condition for the line (made on the host)*/
	T.InitParticular(LL, 0, 0.); // InitParticular(L, shape, offset) shape: 0:flat; 1:sinusoidal, ..... more to come

	/* Flat initial condition for the line (made on the device)*/
	//T.InitFlat(1.); // InitFlat(offset)

	/*Initializes the quenched disorder and creates the cubic spline*/
	T.SetDisorder();
	T.CreateSpline();

	//ofstream disorderfile("disorder.dat");
	//T.PrintDisorder(disorderfile);

	//reloj.tac();
	//reloj.print();
	//reloj.tic();

	unsigned int index=0;
	//unsigned int index1=0;
	// Time loop, from 0 to T_RUN (included)
	for(unsigned int time=0;time<T_RUN+1; time++){

		#ifdef LINEAR
		/* Get data each T_DATA steps*/
		if(time%T_DATA==0){
		#endif

		#ifdef LOGARITHMIC
		/* Get data logarithmically spaced*/
		int aux = ROUND_DOWN_POWER_OF_2(time);
		if( time==aux || ((time-aux)*2) == aux || ((time-aux)*4) == aux || ((time-aux)*8) == aux || \
		((time-aux)*16) == aux ){
		#endif
			/* Print Configurations if desired*/
			/*if(s==0){
				T.CpyDeviceToHost(); 
				sprintf(filename, "U_frame%dL%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.2fdt%.2fgpu%i.dat", \
				10000000+time,LL,MM,SAMPLES,T_RUN,T_DATA, TEMP,C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
				ofstream file0(filename);
				T.PrintLine(file0);
			}*/
		
			/* Calculate some quantities*/
			if (s==0) T.TabulateTime(index,time);
			T.CalculateAccumulateCenterOfMass(index, time);
			#ifndef IGNORE_SF
			T.CalculateStructureFactor();
			T.AccumulateStructureFactor(index, time);
			#endif

			/* Print Structure Factor if desired*/
			/*if(s==0){
				sprintf(filename, "Sk_frame%dL%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.2fdt%.2fgpu%i.dat", \
				10000000+time,LL,MM,SAMPLES,T_RUN,T_DATA,TEMP,C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
				ofstream file2(filename);
				T.PrintStructureFactor(file2);
			}*/

			/* Calculate all external forces*/
			T.InterpolateDisorderSpline();
			T.RecalculateForce(h_field, time);
			T.CalculateAccumulateVelocity(index, time);

			/* Integration step*/
			T.EulerStep(time);

			index++;

		}else{
			/* Calculate all external forces*/
			T.InterpolateDisorderSpline();
			T.RecalculateForce(h_field, time);

			/* Integration step*/
			T.EulerStep(time);
		}

		//CUDA_SAFE_CALL(cudaUnbindTexture(SplineTexture));
	}

	//reloj.tac();
	//reloj.print();
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
	#ifdef LINEAR_SPLINE
	assert(DT<=0.05);	//If linear spline is used, the integration is even more 
	#endif


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
	printf("# GPU used (device ID): %i\n", CUDA_DEVICE);
	printf("# SEED used: %i\n", SEED);
	#ifdef RF
	printf("# RANDOM_FIELD ON");
	#endif
	#ifdef LINEAR_SPLINE
	printf("# LINEAR_SPLINE ON");
	#endif
	#ifdef UNIFORM_DISORDER
	printf("# UNIFORM_DISORDER ON");
	#endif
	#ifdef ANHARMONIC
	printf("# ANHARMONIC ON");
	#endif

	// Initialize the RNGs
	if (ConfigureRandomNumbers()) {
		return 1;
	}

	// main{} variables
	char filename [240];

	// Our class
	elastic_system T;

	for(unsigned int s=0; s < SAMPLES; s++){
		RunSample(s,T);

		if ((s+1)%100==0){
			/* Print partial averages of Structure Factor and Center of Mass position*/

			#ifndef IGNORE_SF
			sprintf(filename, "Sk_L%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.3fdt%.2fgpu%i.tmp",\
			LL,MM,SAMPLES,T_RUN,T_DATA,TEMP,C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
			ofstream file2(filename);
			T.PrintStructureFactorAccum(file2, s);
			#endif

			sprintf(filename, "CM_L%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.3fdt%.2fgpu%i.tmp",\
			LL,MM,SAMPLES,T_RUN,T_DATA,TEMP,C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
			ofstream file3(filename);
			T.PrintCenterOfMassAccum(file3, s);

			sprintf(filename, "V_L%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.3fdt%.2fgpu%i.tmp",\
			LL,MM,SAMPLES,T_RUN,T_DATA,TEMP,C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
			ofstream file4(filename);
			T.PrintVelocityAccum(file4, s);
		}

	}

	/* Print final averages of Structure Factor and Center of Mass position*/

	#ifndef IGNORE_SF
	sprintf(filename, "Sk_L%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.3fdt%.2fgpu%i.dat",LL,MM,SAMPLES,T_RUN,T_DATA,TEMP,\
	C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
	ofstream file2(filename);
	T.PrintStructureFactorAccum(file2, SAMPLES);
	printf("# Structure Factor averaged over %i samples in %s\n", SAMPLES, filename);
	#endif

	sprintf(filename, "CM_L%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.3fdt%.2fgpu%i.dat",LL,MM,SAMPLES,T_RUN,T_DATA,TEMP,\
	C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
	ofstream file3(filename);
	T.PrintCenterOfMassAccum(file3, SAMPLES);
	printf("# Center of Mass position averaged over %i samples in %s\n", SAMPLES, filename);

	sprintf(filename, "V_L%iM%iS%iTr%iTd%iT%.2fb%.2fe%.2fh%.3fdt%.2fgpu%i.dat",LL,MM,SAMPLES,T_RUN,T_DATA,TEMP,\
	C_BETA,C_EPSILON,H_START,DT,CUDA_DEVICE);
	ofstream file4(filename);
	T.PrintVelocityAccum(file4, SAMPLES);
	printf("# Center of Mass velocity aveaged over %i samples in %s\n", SAMPLES, filename);

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
