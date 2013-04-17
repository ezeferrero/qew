#include "include/timer.h"
#include<cmath>
#include<fstream>
#include <iostream>

/* counter-based random numbers */
// http://www.thesalmons.org/john/random123/releases/1.06/docs/
#include <Random123/philox.h> // philox headers
#include <Random123/u01.h>    // to get uniform deviates [0,1]
typedef r123::Philox2x32 RNG; // particular counter-based RNG


/* includes from thrust library */
// https://github.com/thrust/thrust/wiki/Documentation
#include<thrust/transform.h>
#include<thrust/copy.h>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include<thrust/functional.h>
#include<thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#ifdef CUFFT
#include "qew_cufft.h"
#endif

/* problem parameters */
#define L	1024   // number of particles/monomers
#define F0	0.125       // uniform driving force
#define Dt	0.1       // time step
#define TRUN	1000000    // number of temporal iterations 
#define TEMP	0.000     // temperature	
#define D0	1.0	  // disorder intensity
#define SEED 	12345678 // global seed RNG (quenched noise)
#define SEED2 	87654321 // global seed#2 RNG (thermal noise)


using namespace thrust;

typedef double 	REAL;

// to generate gaussian random numbers from uniform random numbers
// http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
__device__
REAL box_muller(RNG::ctr_type r_philox)
{
 	REAL u1 = u01_closed_closed_32_53(r_philox[0]);
  	REAL u2 = u01_closed_closed_32_53(r_philox[1]);
  	REAL r = sqrt( -2.0*log(u1) );
  	REAL theta = 2.0*M_PI*u2;
	return r*sin(theta);    			
}


void print_configuration(device_vector<REAL> &u, REAL color, int size)
{
	long i=0;
	int every = int(L/128.0);
	for(i=1;i<u.size()-1;i+=every)
	{
		std::cout << u[i] << " " << color << std::endl;
	}
	std::cout << "\n\n";
}

// functor used to compute the roughness
struct roughtor: public thrust::unary_function<REAL,REAL>
{
    REAL u0;	
    roughtor(REAL _u0):u0(_u0){};	
    __device__
    REAL operator()(REAL u)
    {	
	return (u-u0)*(u-u0);
    }
};	



/////////////////////////////////////////////////////////////////////////////
// FUNCTORS used in TRANSFORM algorithms

struct fuerza
{
    RNG rng;       // random number generator
    long tiempo;   // parameter
    REAL noiseamp; // parameter
		
    fuerza(long _t):tiempo(_t)
    {
	noiseamp=sqrt(TEMP/Dt);
    }; 

    __device__
    REAL operator()(tuple<long,REAL,REAL,REAL> tt)
    {	
	// thread/particle id
	uint32_t tid=get<0>(tt); 

	// keys and counters 
    	RNG::ctr_type c={{}};
    	RNG::key_type k={{}};
	RNG::ctr_type r;

	k[0]=tid; 	  //  KEY = {threadID} 
	c[1]=SEED; 	  // COUNTER[1] = {to set later, GLOBAL SEED}

	// LAPLACIAN
	REAL um1=get<1>(tt);
	REAL u=get<2>(tt);
	REAL up1=get<3>(tt);
	REAL laplaciano = um1 + up1 - 2.*u;

	// DISORDER
	REAL firstRN, secondRN;
	int U=int(u);
	c[0]=uint32_t(U); // COUNTER={U,GLOBAL SEED}
	r = rng(c, k);
	firstRN=(u01_closed_closed_32_53(r[0])-0.5); // alternative: box_muller(r);
	c[0]=uint32_t(U+1); // COUNTER={U+1,GLOBAL SEED}
	r = rng(c, k);
	secondRN=(u01_closed_closed_32_53(r[0])-0.5); // alternative: box_muller(r);
	REAL quenched_noise = D0*(firstRN - (firstRN-secondRN)*(u-U));// linearly interpolated force

	// THERMAL NOISE
	c[0]=tiempo; // COUNTER={tiempo, GLOBAL SEED #2}
	c[1]=SEED2; // to avoid correlations between thermal and quenched noise
	r = rng(c, k);
	REAL thermal_noise = noiseamp*box_muller(r);    			

	// Total force on the given monomer
	return (laplaciano+quenched_noise+thermal_noise+F0);
  }
};


// Explicit forward Euler step
struct euler
{
    __device__
    REAL operator()(thrust::tuple<REAL,REAL> tt)
    {	
	return REAL(thrust::get<0>(tt)+thrust::get<1>(tt)*Dt);
    }
};	

/////////////////////////////////////////////////////////////////////////////

int main(){

	/* arrays */

	// Monomers positions: 
	// Note that two additional elements are alocated to be used as a "halo"
	// in order to force, for instance, periodic boundary conditions: u[0]=u[L]; u[L+1]=u[1];
	device_vector<REAL> u(L+2);
	device_vector<REAL>::iterator u_it0 = u.begin()+1;
	device_vector<REAL>::iterator u_it1 = u.end()-1;

	// If you need a raw pointer to pass to a kernel de CUDA C/C++:
	REAL * u_raw_ptr = raw_pointer_cast(&u[1]);

	// total force: 
	device_vector<REAL> Ftot(L); 
	device_vector<REAL>::iterator Ftot_it0 = Ftot.begin();
	device_vector<REAL>::iterator Ftot_it1 = Ftot.end();

	// some flat initial condition
	fill(u_it0,u_it1,10.0); 

	// simple (GPU/CPU) timer
	timer t;
	double timer_fuerzas_elapsed=0.0;
	double timer_euler_elapsed=0.0;
	double timer_props_elapsed=0.0;


	// output file for some time dependent properties
	std::ofstream propsout("someprops.dat");

	#ifdef CUFFT
	// instantaneous and acumulated structure factor object
	Structure_Factor Str(L);	
	#endif


	// temporal loop
	for(long n=0;n<TRUN;n++)
	{
		// Imposes PBC in the "halo"		
		u[0]=u[L];u[L+1]=u[1];

		// Force on each monomer, concurrently calculated in the GPU: 
		// Ftot(X)= laplacia + disorder + thermal_noise + F0, X=0,..,L-1 
		t.restart();
		transform(
			make_zip_iterator(make_tuple(
			make_counting_iterator<long>(0),u_it0-1,u_it0,u_it0+1
			)),
			make_zip_iterator(make_tuple(
			make_counting_iterator<long>(L),u_it1-1,u_it1,u_it1+1
			)),
			Ftot_it0,
			fuerza(n)
		);
		timer_fuerzas_elapsed+=t.elapsed();

		// Explicit forward Euler step, concurrently done in the GPU: 
		// u(X,n) += Ftot(X,n) Dt, X=0,...,L-1
		t.restart();
		transform(
			make_zip_iterator(make_tuple(u_it0,Ftot_it0)),
			make_zip_iterator(make_tuple(u_it1,Ftot_it1)),	
			u_it0, 
			euler()
		);
		timer_euler_elapsed+=t.elapsed();

		// some properties of interest
		if(n%100==0){		
			t.restart();
			REAL velocity = reduce(Ftot_it0,Ftot_it1,REAL(0.0))/REAL(L); //center of mass velocity
			REAL center_of_mass = reduce(u_it0,u_it1,REAL(0.0))/REAL(L); // center of mass position
  			REAL roughness = reduce
			(
			make_transform_iterator(u_it0, roughtor(center_of_mass)),
			make_transform_iterator(u_it1, roughtor(center_of_mass)),
			REAL(0.0)
			); // mean square width
	
			propsout << velocity << " " << center_of_mass << " " << roughness << std::endl;
			
			print_configuration(u,velocity,128);// prints a 128 points configuration...

			#ifdef CUFFT
			Str.calculate(u); // calculates the instantaneous and acumulated structure factor
			Str.print_acum(n); // prints the instantaneous and acumulated structure factor
			#endif
			timer_props_elapsed+=t.elapsed();
		}

		// visualization
		// ....WELCOME!
	}

	// timer results
	std::cout << "Each update, on average:\n"; 
	std::cout << "Forces calculation -> " << 1e3 * timer_fuerzas_elapsed/TRUN << " miliseconds" << std::endl;
	std::cout << "Euler step -> " << 1e3 * timer_euler_elapsed/TRUN << " miliseconds" << std::endl;
	std::cout << "Properties -> " << 1e3 * timer_props_elapsed/TRUN << " miliseconds" << std::endl;
	std::cout << "Total -> " << 1e3 * (timer_fuerzas_elapsed+timer_euler_elapsed+timer_props_elapsed)/TRUN << " miliseconds" << std::endl;

	return 0;
}

