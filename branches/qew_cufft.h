#include <cufft.h>
#include "cutil.h"	// CUDA_SAFE_CALL, CUT_CHECK_ERROR

#ifndef SINGLE_PRECISION
typedef cufftDoubleReal Real;
typedef cufftDoubleComplex COMPLEX;
#else
typedef cufftReal Real;
typedef cufftComplex COMPLEX;
#endif

struct plussquared
{
        __host__ __device__
        Real operator()(const Real& x, const Real& y) const {
            return y+x*x;
        }
};

struct CalculateModulo
{
    __host__ __device__
        Real operator()(const COMPLEX& Z) const {
            return Z.x*Z.x+Z.y*Z.y;
        }
};
struct Acumulador
{
	Real a;
	Acumulador(Real _a):a(_a){};
        __host__ __device__
        Real operator()(const Real& x, const Real& y) const {
            return (x - y)/a + y;
        }
};

class Structure_Factor
{
	public:
	int counter;
	int N;
	thrust::device_vector<Real> Str;
	thrust::device_vector<Real> Str_acum;
	thrust::device_vector<Real> Str_acum2;
	thrust::device_vector<COMPLEX> uq;
	std::ofstream strout;

	cufftHandle planfft;

	Structure_Factor(int _N):N(_N){
		Str.resize(N/2+1);
		Str_acum.resize(N/2+1);
		Str_acum2.resize(N/2+1);
		uq.resize(N/2+1);
		counter=0;
		thrust::fill(Str.begin(),Str.end(), 0.0f);	
		thrust::fill(Str_acum.begin(),Str_acum.end(), 0.0f);	
		thrust::fill(Str_acum2.begin(),Str_acum2.end(), 0.0f);	
		COMPLEX zeroc={0.0,0.0};
		thrust::fill(uq.begin(),uq.end(),zeroc);	
		#ifndef SINGLE_PRECISION
		CUFFT_SAFE_CALL(cufftPlan1d(&planfft,N,CUFFT_D2Z,1));
		#else
		CUFFT_SAFE_CALL(cufftPlan1d(&planfft,N,CUFFT_R2C,1));
		#endif

		//char nom[50]; sprintf(nom,"str%d.dat",t);
		char nom[50]; sprintf(nom,"str.dat");
		strout.open(nom);
	};
	void calculate(thrust::device_vector<Real> &u){
		thrust::device_ptr<Real> itu=u.data(); 
		thrust::device_ptr<Real> itup1=(itu++);
		Real * d_input_raw = thrust::raw_pointer_cast(itup1);
		COMPLEX * d_output_raw = thrust::raw_pointer_cast(&uq[0]);
		//cout << "calculando Str\n";

		#ifndef SINGLE_PRECISION
		CUFFT_SAFE_CALL(cufftExecD2Z(planfft, d_input_raw, d_output_raw));
		#else
		//cufftExecR2C(plan, d_input_raw, d_output_raw);
		CUFFT_SAFE_CALL(cufftExecR2C(planfft, d_input_raw, d_output_raw));
		#endif
		thrust::transform(uq.begin(),uq.end(),Str.begin(),CalculateModulo());
		thrust::transform(Str.begin(),Str.end(),Str_acum.begin(),Str_acum.begin(),thrust::plus<Real>());
		thrust::transform(Str.begin(),Str.end(),Str_acum2.begin(),Str_acum2.begin(),plussquared());
		counter++;
	}
	void print_acum(int t)
	{
		int j=1;
		for(int i=1;i<Str_acum.size();i+=j){
			strout << (i*2*M_PI/N) << " " << Str[i] << " " << Str_acum[i]/counter << " " << Str_acum2[i]/counter << std::endl;
		}
		strout << "\n\n";

	}
};

