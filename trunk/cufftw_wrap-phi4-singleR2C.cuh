/*
 * cufftw_wrap-phi4.hpp
 *
 *  Created on: Jul 11, 2011
 *      Author: eze, ale
 */

#ifndef FFTW_WRAP_HPP_
#define FFTW_WRAP_HPP_

//#include "../common/common.c"
#include <stdint.h> /* uint8_t */

using namespace std;

texture<float,1,cudaReadModeElementType> Qtexture;
texture<float,1,cudaReadModeElementType> Etexture;

//__global__ void kernel_init_random(cufftComplex *a, int nx, int ny){
__global__ void kernel_init_random(cufftReal *a, int nx, int ny){
	const unsigned int jAux = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iAux = blockIdx.y*TILE_Y + threadIdx.y;
	const unsigned int tid = iAux*FRAME + jAux;
	// move thread RNG state to registers. 
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];

	unsigned int i;	
	for (unsigned int iFrame=0; iFrame<(nx/FRAME); iFrame++) {
		i = iAux + FRAME*iFrame;
		unsigned int j;
		for (unsigned int jFrame=0; jFrame<(ny/FRAME); jFrame++) {
			j = jAux + FRAME*jFrame;	
			// compute index with i and j, the location of the element in the original Lx*Ly array 
			int index = i*ny + j;
//			a[index].x = 2.f*(rand_MWC_co(&rng_state, &rng_const)-0.5);
//			a[index].y = 0.f;
			a[index] = 2.f*(rand_MWC_co(&rng_state, &rng_const)-0.5);
		}
	}	
	d_x[tid] = rng_state; // store RNG state into global again		
}


__global__ void kernel_set_epsilon(float *d_epsilon, int nx, int ny){
	const unsigned int jAux = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iAux = blockIdx.y*TILE_Y + threadIdx.y;
	const unsigned int tid = iAux*FRAME + jAux;
	// move thread RNG state to registers. 
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];

	unsigned int i;	
	for (unsigned int iFrame=0; iFrame<(nx/FRAME); iFrame++) {
		i = iAux + FRAME*iFrame;
		unsigned int j;
		for (unsigned int jFrame=0; jFrame<(ny/FRAME); jFrame++) {
			j = jAux + FRAME*jFrame;	
			// compute index with i and j, the location of the element in the original Lx*Ly array 
			int index = i*ny + j;
			d_epsilon[index] = Calpha + Cepsilon*2.f*(rand_MWC_co(&rng_state, &rng_const)-0.5);
		}
	}	
	d_x[tid] = rng_state; // store RNG state into global again		
}


__global__ void kernel_set_qx_qy(float *qqmodule, int nx, int ny){
	// compute idx and idy, the location of the element in the original Lx*Ly array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx && idy < ny){
		//int nxs2=(nx/2); int nys2=(ny/2);
		// cuando usamos los cosenos no es necesario meter ningún módulo, las cc quedan determinadas por la periodicidad de la funcion
		float qx=2.f*M_PI/(float)Lx * idx;
		float qy=2.f*M_PI/(float)Ly * idy;
		//float qx=2*M_PI/nx*((idx+nxs2)-nx);
		//float qy=2*M_PI/ny*((idy+nys2)-ny);
		//int index = ((idx+nxs2)%nx)*ny + ((idy+nys2)%ny); 
		int index = idx*ny + idy;
		//qqmodule[index]=qx*qx+qy*qy;
		qqmodule[index]=-2.f*(cosf(qx)+cosf(qy)-2.f);
	}
}

__global__ void Real2Real_scaled(cufftReal *a, int nx, int ny, float scale){
	// compute idx and idy, the location of the element in the original Lx*Ly array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx && idy < ny){	
		int index = idx*ny + idy;
		a[index]= scale*a[index];
	}	
}

__global__ void complex2complex_scaled(cufftComplex *a, int nx, int ny, float scale){
	// compute idx and idy, the location of the element in the original Lx*Ly array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx && idy < ny){	
		int index = idx*ny + idy;
		a[index].x= scale*a[index].x;
		a[index].y= scale*a[index].y;
	}	
}


//__global__ void recalculatingRealForce(const float *e, const cufftComplex *a, cufftComplex *b, int nx, int ny, float uniform_field){
__global__ void recalculatingRealForce(const float *e, const cufftReal *a, cufftReal *b, int nx, int ny, float uniform_field){
	const unsigned int jAux = blockIdx.x*TILE_X + threadIdx.x;
	const unsigned int iAux = blockIdx.y*TILE_Y + threadIdx.y;
	const unsigned int tid = iAux*FRAME + jAux;
	// move thread RNG state to registers. 
	unsigned long long rng_state = d_x[tid];
	const unsigned int rng_const = d_a[tid];

	unsigned int i;	
	for (unsigned int iFrame=0; iFrame<(nx/FRAME); iFrame++) {
		i = iAux + FRAME*iFrame;
		unsigned int j;
		for (unsigned int jFrame=0; jFrame<(ny/FRAME); jFrame++) {
			j = jAux + FRAME*jFrame;
			// compute index with i and j, the location of the element in the original Lx*Ly array 
			int index = i*ny + j;
// 			const float epsilon = e[index];
			const float epsilon = tex1Dfetch(Etexture, index);
//			b[index].x = epsilon*(a[index].x-a[index].x*a[index].x*a[index].x) + xtemp*(rand_MWC_co(&rng_state, &rng_const)-0.5) + uniform_field;
//			b[index].y = 0.f;
			b[index] = epsilon*(a[index]-a[index]*a[index]*a[index]) + xtemp*(rand_MWC_co(&rng_state, &rng_const)-0.5) + uniform_field;
		}	
	}
	d_x[tid] = rng_state; // store RNG state into global again		
}

__global__ void euler_step(cufftComplex *a, const cufftComplex *b, const float *qqmodule, int nx, int ny){
	// compute idx and idy, the location of the element in the original NxN array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx && idy < ny){
		int index = idx*ny + idy;
		//const float fac=-(qqmodule[index]);
		const float fac=-(tex1Dfetch(Qtexture, index));
		const float dip = -(9.05f-2.f*M_PI*sqrtf(-1.f*fac));
		a[index].x = (1.f+Cbeta*fac*Dt+Cgamma*dip*Dt)*a[index].x + Dt*b[index].x;
		a[index].y = (1.f+Cbeta*fac*Dt+Cgamma*dip*Dt)*a[index].y + Dt*b[index].y;
	}	
}

__global__ void euler_semiimplicit_step(cufftComplex *a, const cufftComplex *b, const float *qqmodule, int nx, int ny){
	// compute idx and idy, the location of the element in the original NxN array 
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	int idy = blockIdx.y*blockDim.y+threadIdx.y;
	if ( idx < nx  && idy < ny){
		int index = idx*ny + idy;
		//const float fac=-(qqmodule[index]);
		const float fac=-(tex1Dfetch(Qtexture, index));
		const float dip = -(9.05f-2.f*M_PI*sqrtf(-1.f*fac));
		a[index].x = (a[index].x*(1.f+Cgamma*dip*Dt) + Dt*b[index].x)/(1.f-fac*Cbeta*Dt);
		a[index].y = (a[index].y*(1.f+Cgamma*dip*Dt) + Dt*b[index].y)/(1.f-fac*Cbeta*Dt);
		//a[index].x = (a[index].x + Dt*b[index].x)/(1.-fac*Cbeta*Dt);
		//a[index].y = (a[index].y + Dt*b[index].y)/(1.-fac*Cbeta*Dt);
	}
}

/* RGB formulae.
 * http://gnuplot-tricks.blogspot.com/2009/06/comment-on-phonged-surfaces-in-gnuplot.html
 * TODO: cannot find what to do with over/under flows, currently saturate in [0,255]
 *
 * there are 37 available rgb color mapping formulae:
 * 0: 0               1: 0.5             2: 1
 * 3: x               4: x^2             5: x^3
 * 6: x^4             7: sqrt(x)         8: sqrt(sqrt(x))
 * 9: sin(90x)        10: cos(90x)       11: |x-0.5|
 * 12: (2x-1)^2       13: sin(180x)      14: |cos(180x)|
 * 15: sin(360x)      16: cos(360x)      17: |sin(360x)|
 * 18: |cos(360x)|    19: |sin(720x)|    20: |cos(720x)|
 * 21: 3x             22: 3x-1           23: 3x-2
 * 24: |3x-1|         25: |3x-2|         26: (3x-1)/2
 * 27: (3x-2)/2       28: |(3x-1)/2|     29: |(3x-2)/2|
 * 30: x/0.32-0.78125 31: 2*x-0.84       32: 4x;1;-2x+1.84;x/0.08-11.5
 * 33: |2*x - 0.5|    34: 2*x            35: 2*x - 0.5
 * 36: 2*x - 1
 * 
 * Some nice schemes in RGB color space
 * 7,5,15   ... traditional pm3d (black-blue-red-yellow)
 * 3,11,6   ... green-red-violet
 * 23,28,3  ... ocean (green-blue-white); try also all other permutations
 * 21,22,23 ... hot (black-red-yellow-white)
 * 30,31,32 ... color printable on gray (black-blue-violet-yellow-white)
 * 33,13,10 ... rainbow (blue-green-yellow-red)
 * 34,35,36 ... AFM hot (black-red-yellow-white)
 */

#define RGBf05 (x*x*x)
#define RGBf07 (sqrt(x))
#define RGBf15 (sin(2*M_PI*x))
#define RGBf21 (3*x)
#define RGBf22 (3*x-1)
#define RGBf23 (3*x-2)
#define RGBf30 (x/0.32-0.78125)
#define RGBf31 (2*x-0.84)
#define RGBf32 (x/0.08-11.5)

#define MAX_COMPONENT_VALUE 255
//void writePPMbinaryImage(const char * filename, const cufftComplex * vector)
void writePPMbinaryImage(const char * filename, const cufftReal * vector)
{
	FILE *f = NULL;
	unsigned int i = 0, j = 0;
	uint8_t RGB[3] = {0,0,0}; /* use 3 first bytes for BGR */ 
	f = fopen(filename, "w");
	assert(f);
	fprintf(f, "P6\n"); /* Portable colormap, binary */
	fprintf(f, "%d %d\n", Ly, Lx); /* Image size */
	fprintf(f, "%d\n", MAX_COMPONENT_VALUE); /* Max component value */
	for (i=0; i<Lx; i++){
		for (j=0; j<Ly; j++){
			/* ColorMaps
			 * http://mainline.brynmawr.edu/Courses/cs120/spring2008/Labs/ColorMaps/colorMap.py
			 */
			//float x = vector[i*Ly+j].x; //HOT; /* [0,1] */
			float x = vector[i*Ly+j]; //HOT; /* [0,1] */			
			RGB[0] = (uint8_t) MIN(255, MAX(0, 256*RGBf21));
			RGB[1] = (uint8_t) MIN(255, MAX(0, 256*RGBf22));
			RGB[2] = (uint8_t) MIN(255, MAX(0, 256*RGBf23));
			//assert(0<=RGB[0] && RGB[0]<256);
			//assert(0<=RGB[1] && RGB[1]<256);
			//assert(0<=RGB[2] && RGB[2]<256);
			fwrite((void *)RGB, 3, 1, f);
		}
	}
	fclose(f); f = NULL;
	return;
}

//////////////////////////////////////////////////////////////////////
class cufft_complex_to_complex_2d
{
	private:
	// plan
	cufftHandle Sp, Sr2c, Sc2r; 

	public:
	// arrays complejos para poner el campo escalar y la fuerza
//	cufftComplex *h_phi, *h_U; 
	cufftComplex *d_phi, *d_U; 
	cufftReal *h_phi_r, *h_U_r;
	cufftReal *d_phi_r, *d_U_r;
	
	//los qs
	float *d_qmodule; 
	
	//el desorden
	float *d_epsilon;
	
	//la posicion, el ancho
	float h_pared[3][Trun/Tframe+1];
	
	
    // inicializacion de la transformada
    cufft_complex_to_complex_2d(){

		//h_phi = (cufftComplex *)malloc(sizeof(cufftComplex)*Lx*Ly);
		//h_U = (cufftComplex *)malloc(sizeof(cufftComplex)*Lx*Ly);

		h_phi_r = (cufftReal *)malloc(sizeof(cufftReal)*Lx*Ly);
		h_U_r = (cufftReal *)malloc(sizeof(cufftReal)*Lx*Ly);
	
		//h_pared = (int *)malloc(sizeof(int)*(Trun/Tframe+1)*3); 
		
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_phi, sizeof(cufftComplex)*Lx*(Ly/2+1)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_U, sizeof(cufftComplex)*Lx*(Ly/2+1)));

		CUDA_SAFE_CALL(cudaMalloc((void**)&d_phi_r, sizeof(cufftReal)*Lx*Ly));
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_U_r, sizeof(cufftReal)*Lx*Ly));
		
		CUDA_SAFE_CALL(cudaMalloc((void**)&d_qmodule, sizeof(float)*Lx*(Ly/2+1))); 

		CUDA_SAFE_CALL(cudaMalloc((void**)&d_epsilon, sizeof(float)*Lx*Ly)); 
		
		// plan: in-place complex to complex 2d
//		CUFFT_SAFE_CALL(cufftPlan2d(&Sp,Lx,Ly,CUFFT_C2C)); 

		// plans R2C and C2R
		CUFFT_SAFE_CALL(cufftPlan2d(&Sr2c,Lx,Ly,CUFFT_R2C));
		CUFFT_SAFE_CALL(cufftPlan2d(&Sc2r,Lx,Ly,CUFFT_C2R));
		
	//	cufftSetCompatibilityMode(Sr2c, CUFFT_COMPATIBILITY_NATIVE);
	//	cufftSetCompatibilityMode(Sc2r, CUFFT_COMPATIBILITY_NATIVE);
		
    }


	void set_qx_qy(){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(Lx/TILE_X, Ly/TILE_Y/2+1);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		
		kernel_set_qx_qy<<<dimGrid, dimBlock>>>(d_qmodule,Lx,Ly/2+1);
		CUDA_SAFE_CALL(cudaBindTexture(NULL, Qtexture, d_qmodule, Lx*(Ly/2+1)*sizeof(float)));
		
	};

	void set_epsilon(){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(FRAME/TILE_X, FRAME/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		
		kernel_set_epsilon<<<dimGrid, dimBlock>>>(d_epsilon,Lx,Ly);
		CUDA_SAFE_CALL(cudaBindTexture(NULL, Etexture, d_epsilon, Lx*Ly*sizeof(float)));		
	};

	void init_random(){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(FRAME/TILE_X, FRAME/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		
//		kernel_init_random<<<dimGrid, dimBlock>>>(d_phi,Lx,Ly);
		kernel_init_random<<<dimGrid, dimBlock>>>(d_phi_r,Lx,Ly);
	};

	void init_flat(){
	       CUDA_SAFE_CALL(cudaMemset(d_phi, 0, Lx*(Ly/2+1)*sizeof(cufftComplex)));
	       CUDA_SAFE_CALL(cudaMemset(d_U, 0, Lx*(Ly/2+1)*sizeof(cufftComplex)));
	       
      	       CUDA_SAFE_CALL(cudaMemset(d_phi_r, 0, Lx*Ly*sizeof(cufftReal)));
	       CUDA_SAFE_CALL(cudaMemset(d_U_r, 0, Lx*Ly*sizeof(cufftReal))); 
	};	


	void init_particular(int nx, int ny){
		for(int i=0;i<nx;i++){
			for(int j=0;j<ny;j++){
				int k=i*ny+j;
				// un cuadrado hueco
				//h_phi[k].x= (nx*0.35 < i && i < nx*0.65 && ny*0.35 < j && j < ny*0.65)?(1.0):(0.0);
				//h_phi[k].x= (nx*0.45 < i && i < nx*0.55 && ny*0.45 < j && j < ny*0.55)?(0.0):(h_phi[k].x);

				//circulo al diome
// 				h_phi_r[k]= ((nx/2-i)*(nx/2-i)+(ny/2-j)*(ny/2-j) < 10000.)?(1.0):(-1.0);				

				//linea
// 				h_phi[k].x= (0 <= j && j < ny*0.05)?(1.0):(-1.0);
				
				//banda
				//h_phi_r[k]= (ny*0.4 < j && j < ny*0.6)?(1.0):(-1.0);
				//h_phi_r[k]= (ny*0.45 < j && j < ny*0.55)?(1.0):(-1.0);
//				h_phi[k].x= (nx*0.45 < i && i < nx*0.55)?(1.0):(-1.0);				
//				h_phi[k].x= (i+j<ny*0.5 || (nx-i)+(ny-j)<ny*0.5)?(1.0):(-1.0);
				//h_phi[k].y = 0.0;

				//varios stripes paralelos
				h_phi_r[k]= (0.f < sin((j+1.33*i)*50*2*M_PI/ny))?(1.0):(-1.0);
			}
		}
	}

	/* transfer from CPU to GPU memory */
	void Cpy_Host_to_Device(){
//		CUDA_SAFE_CALL(cudaMemcpy(d_phi,h_phi, sizeof(cufftComplex)*Lx*Ly, cudaMemcpyHostToDevice));
//		CUDA_SAFE_CALL(cudaMemcpy(d_U,h_U, sizeof(cufftComplex)*Lx*Ly, cudaMemcpyHostToDevice));
		
		CUDA_SAFE_CALL(cudaMemcpy(d_phi_r,h_phi_r, sizeof(cufftReal)*Lx*Ly, cudaMemcpyHostToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(d_U_r,h_U_r, sizeof(cufftReal)*Lx*Ly, cudaMemcpyHostToDevice));			
	}	

	/* transfer from GPU to CPU memory */
	void Cpy_Device_to_Host(){	
//		CUDA_SAFE_CALL(cudaMemcpy(h_phi, d_phi, sizeof(cufftComplex)*Lx*Ly, cudaMemcpyDeviceToHost));

		CUDA_SAFE_CALL(cudaMemcpy(h_phi_r, d_phi_r, sizeof(cufftReal)*Lx*Ly, cudaMemcpyDeviceToHost));	
	}

	void transformar()
	{
//		CUFFT_SAFE_CALL(cufftExecC2C(Sp, d_phi, d_phi, CUFFT_FORWARD));
//		CUFFT_SAFE_CALL(cufftExecC2C(Sp, d_U, d_U, CUFFT_FORWARD));

		CUFFT_SAFE_CALL(cufftExecR2C(Sr2c, d_phi_r, d_phi));
		CUFFT_SAFE_CALL(cufftExecR2C(Sr2c, d_U_r, d_U));
	};

	void antitransformar()
	{
//		CUFFT_SAFE_CALL(cufftExecC2C(Sp, d_phi, d_phi, CUFFT_INVERSE)); 

		CUFFT_SAFE_CALL(cufftExecC2R(Sc2r, d_phi, d_phi_r)); 	
	};	
	
	/* apply scaling ( an FFT followed by iFFT will give you back the same array times the length of the transform) */
	void normalize(void){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(Lx/TILE_X, Ly/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		
//		complex2complex_scaled<<<dimGrid, dimBlock>>>(d_phi, Lx, Ly, 1.f / ((float) Lx * (float) Ly));
		Real2Real_scaled<<<dimGrid, dimBlock>>>(d_phi_r, Lx, Ly, 1.f / ((float) Lx * (float) Ly));
	}

	void recalcularRealForce(float uniform_field){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(FRAME/TILE_X, FRAME/TILE_Y);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);		

//		recalculatingRealForce<<<dimGrid, dimBlock>>>(d_epsilon,d_phi,d_U,Lx,Ly,uniform_field);	
		recalculatingRealForce<<<dimGrid, dimBlock>>>(d_epsilon,d_phi_r,d_U_r,Lx,Ly,uniform_field);	
	}	


	void print_input(ofstream &fstr)
	{
		//fstr.seekp(ios_base::beg);

		for(int i=0;i<Lx;i++)
		{
			for(int j=0;j<Ly;j++)
			{
				int k=i*Ly+j;
//				fstr << setprecision (4) << h_phi[k].x << " " ;
				fstr << setprecision (4) << h_phi_r[k] << " " ;
			}
			fstr << endl;
		}
		fstr << endl << endl;

	};
	
	void print_dibujito(char * picturename, int x){
	char cmd[200];	char cmd2[200];
//	writePPMbinaryImage(picturename, h_phi);
	writePPMbinaryImage(picturename, h_phi_r);	
	sprintf(cmd, "convert frame%d.ppm frame%d.jpg", 100000000+x, 100000000+x);
	system(cmd);
	sprintf(cmd2, "rm frame%d.ppm", 100000000+x);
	system(cmd2);	
	};
	
	void medir_ancho(int t){
		int w_init[Lx]={0};
		int w_end[Lx]={0};
		int ancho[Lx]={0};
		//int posicion[Lx]={0};
		int ancho_m=0; int posicion_m=0;		
		for(int i=0;i<Lx;i++){
			for(int j=0;j<Ly;j++){
			int k=i*Ly+j;
			if (w_init[i]==0 && h_phi_r[k] >= n_umbral) w_init[i]= j;
			if (w_end[i]==0 && h_phi_r[k] >= p_umbral) w_end[i]= j;
			}
			ancho_m += w_end[i]-w_init[i];
			posicion_m += w_init[i] + ancho[i]/2;
			/*if (i==0){ 
				cout << "t=" << t << ", posicion=" << posicion[i] << ", ancho=" << ancho[i] << endl;
			}*/
		}	
		h_pared[0][t/Tframe]= (float) t;
		h_pared[1][t/Tframe]= (float) posicion_m/Lx;
		h_pared[2][t/Tframe]= (float) ancho_m/Lx;
		cout << h_pared[0][t/Tframe] << " " << h_pared[1][t/Tframe] << " " << h_pared[2][t/Tframe] << endl;				
	};

	
	void euler(){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(Lx/TILE_X, Ly/TILE_Y/2+1);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);		
		euler_step<<<dimGrid, dimBlock>>>(d_phi, d_U, d_qmodule, Lx, (Ly/2+1));
	};	
	
	void euler_semiimplicit(){
		dim3 dimBlock(TILE_X, TILE_Y);
		dim3 dimGrid(Lx/TILE_X, Ly/TILE_Y/2+1);
		assert(dimBlock.x*dimBlock.y<=THREADS_PER_BLOCK);
		assert(dimGrid.x<=BLOCKS_PER_GRID && dimGrid.y<=BLOCKS_PER_GRID);
		euler_semiimplicit_step<<<dimGrid, dimBlock>>>(d_phi, d_U, d_qmodule, Lx, (Ly/2+1));
	};
	
	~cufft_complex_to_complex_2d(){
		cufftDestroy(Sp);
		cudaFree(d_phi);
		//cudaFree(h_phi);
		cudaFree(d_U);
		//cudaFree(h_U);
		cudaFree(d_phi_r);
		cudaFree(h_phi_r);
		cudaFree(d_U_r);
		cudaFree(h_U_r);
	
	};

};


#endif /* FFTW_WRAP_HPP_ */
