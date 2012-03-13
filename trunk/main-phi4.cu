// main
#include<cmath>
#include<stdio.h> /* printf */
#include<stdlib.h>
#include<iostream> /*para los printf*/
#include <iomanip> /*para los printf*/
#include<fstream> /*para los printf*/
#include <assert.h>
#include <sys/time.h> /* gettimeofday */

//#include<fftw3.h>
//#include<vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cutil.h" // CUDA_SAFE_CALL, CUT_CHECK_ERROR
#include <cufft.h>

#ifndef Lx
#define Lx 512
#endif
#ifndef Ly
#define Ly 512
#endif

#ifndef Trun
#define	Trun 1000
#endif

#ifndef Tframe
#define Tframe 100
#endif

#ifndef Dt
#define Dt 0.5f
#endif

#ifndef Calpha
#define Calpha 0.8f 	/*factor termino phi-phi^3*/
#endif

#ifndef Cepsilon
#define Cepsilon 0.1f	/*magnitud del desorden que se le suma a Calpha, si es 0.f no hay desorden*/
#endif

#ifndef Cbeta
#define Cbeta 2.0f	/*factor termino Laplaciano phi*/
#endif

#ifndef Cgamma
#define Cgamma 0.f	/*factor termino Dipolar*/
#endif

#ifndef TEMP
#define TEMP 0.f
#endif
//#define	xtemp sqrt(24.*TEMP*Dt)  
#define	xtemp sqrt(24.*TEMP/Dt)

#ifndef h_start
#define h_start 0.1f
#endif

#ifndef dhdt
#define dhdt 0.f
#endif

#ifndef t_start
#define t_start 0
#endif

#define n_umbral -0.5f
#define p_umbral 0.5f

//#define h_0 0.01d //0.01f  //Campo externo

//#define PI 3.1415926535897932f
//#define PI 3.1415926f

// Functions
// maximum
#define MAX(a,b) (((a)<(b))?(b):(a))
// minimum
#define MIN(a,b) (((a)<(b))?(a):(b))
//valor absoluto
#define ABS(a) (((a)< 0)?(-a):(a))

// Hardware parameters for GTX 470/480 (GF100)
#define SHARED_PER_BLOCK 49152
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 1024
#define BLOCKS_PER_GRID 65535

/*para el RNG*/
#define FRAME 256	// the whole thing is framed for the RNG
#define TILE_X 8	// each block of threads is a tile
#define TILE_Y 32	// each block of threads is a tile

#define SAFE_PRIMES_FILENAME "../common/safeprimes_base32.txt"
//#define SAFE_PRIMES_FILENAME "../common/safeprimes_base32_plus.txt"
// RNG: multiply with carry
#include "../common/CUDAMCMLrng.cu"
#define NUM_THREADS (FRAME*FRAME)
// state of the random number generator, last number (x_n), last carry (c_n) packed in 64 bits
__device__ static unsigned long long d_x[NUM_THREADS];
// multipliers (constants)
__device__ static unsigned int d_a[NUM_THREADS];

#define MICROSEC (1E-6)
#define SEED time(NULL)

//#define facx 2*M_PI/nx
//#define facy 2*M_PI/ny

//#include "cufftw_wrap-phi4-singleMany.cuh"
#include "cufftw_wrap-phi4-singleR2C.cuh"
//#include "cufftw_wrap-phi4-single.cuh"

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
	assert(TILE_X%2==0);
	assert(TILE_Y%2==0);	
	assert(FRAME%TILE_X==0);
	assert(FRAME%TILE_Y==0);
	assert(Lx%FRAME==0);
	assert(Ly%FRAME==0);

	CUDA_SAFE_CALL(cudaThreadSetCacheConfig(cudaFuncCachePreferL1));
	
	double secs = 0.0;
	struct timeval start = {0L,0L}, end = {0L,0L}, elapsed = {0L,0L};

	// Start timer
	gettimeofday(&start, NULL);	

	if (ConfigureRandomNumbers()) {
		return 1;
	}

	cufft_complex_to_complex_2d T;

	/*condicion inicial a manopla (en el host) (calcula tambien U)*/
 	T.init_particular(Lx,Ly);
 	T.Cpy_Host_to_Device();

	/*condicion inicial flat  (directamente en el device) (calcula tambien U)*/
// 	T.init_flat();	
	
	/* condicion inicial random (directamente en el device) (calcula tambien U)*/
 	//T.init_random();

	//ofstream resultado1("movie.dat");
	/* imprimo condicion inicial, no hace falta si ya esta contemplado en el for Trun*/
	//T.Cpy_Device_to_Host(); //necesario si inicializamos en el device
	//T.print_input(resultado1);

	/*calculo los Qs*/
	T.set_qx_qy();

	/*calculo el desorden*/
	T.set_epsilon();

	char picturename [100];
	char datamatrixname [100];
	
	float h_field=h_start; //0.01;
	for(int i=0;i<Trun; i++){

		/* imprimo cada Tframe*/	
		if(i%Tframe==0) {
			T.Cpy_Device_to_Host(); //lo voy a usar enseguida para imprimir pero primero largo algun kernel para aprovechar asincronicidad
			
			T.recalcularRealForce(h_field); /*Aca le estoy pasando el parametro del campo que va variando, */
			
			T.transformar(); /*vuelvo a Fourier transformando phi y U*/
			
			/*incremento temporal*/
			//T.euler();
			T.euler_semiimplicit();

// 			sprintf(datamatrixname, "frame%d.dat", 100000000+i);
// 			ofstream file0(datamatrixname);
// 			T.print_input(file0);
			
 			sprintf(picturename, "frame%d.ppm", 100000000+i);
 			ifstream file(picturename);
 			T.print_dibujito(picturename,i);
			
			//T.medir_ancho(i);
		
			/*vuelvo al espacio real para recalcular U y cuando toque imprimir*/
			T.antitransformar();
			T.normalize();
		}
		else{ //lo mismo sin medir nada 

			T.recalcularRealForce(h_field); /*Aca le estoy pasando el parametro del campo que va variando*/ 

			T.transformar(); /*vuelvo a Fourier transformando phi y U*/
			
			/*incremento temporal*/
			//T.euler();
			T.euler_semiimplicit();
			
			/*vuelvo al espacio real para recalcular U*/
			T.antitransformar();
			T.normalize();
		}	
		
		/* rampa de campo fig1 Jagla */
		//h_field=(i*Dt>3000)?(0.01-0.0000005*(i*Dt-3000)):(0.01);
		//h_field=(i*Dt>t_start)?(h_start+dhdt*(i*Dt-t_start)):(h_start);

		/*rampa-->constante figuras 2 y 6 Jagla*/
		//if (i<58000) h_field=(i*Dt>t_start)?(h_start+dhdt*(i*Dt-t_start)):(h_start);
		//if (i>=58000) h_field=-0.018;
		if (i>=t_start) h_field=dhdt;

		/* pulsos de campo a lo Metaxas */
		//h_field=(i*Dt>3000)?(0.01-0.0000005*(i*Dt-3000)):(0.01);
// 		h_field=(i*Dt>t_start)?(h_start+dhdt*((i/400)%2)):(h_start);
		
		/*campo constante*/
		// no hacer nada, queda en h_start
	}

	// Stop timer
	gettimeofday(&end,NULL);
	timeval_subtract(&elapsed, &end, &start);
	secs = (double)elapsed.tv_sec + ((double)elapsed.tv_usec*MICROSEC);
	printf("# Tamaño/Tiempo de ejecucion CUDA %i %i %lf \n", Lx, Ly, secs);
	
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

