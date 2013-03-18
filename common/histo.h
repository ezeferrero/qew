#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/count.h>

#include <iostream>
#include <iomanip>
#include <iterator>
#include <fstream>

// functor que multiplica y shiftea
struct xhisto : public thrust::unary_function<float,float>
{

    const float a,s;

    xhisto(float _a, float _s) : a(_a), s(_s) {};

    __host__ __device__
    float operator()(float x) const
    {
      return a + s*x;
    }
};


// simple routine to print contents of an histogram
template <typename Vector>
void print_histograma(const std::string& name, const Vector& v, float a, float b, unsigned long size, std::ofstream &fout)
{
  typedef typename Vector::value_type T;  
  int num_bins = v.size();
  float s= (b-a)/num_bins;

  for(int i=0;i<num_bins;i++)
  {
	fout << a+(i)*s << " " << v[i] << std::endl; 
  }
}


// dense histogram using binary search
// Especializacion:
// * float data ya esta en device
// * num_bins esta en el argumento
// * valores histogrameados a<=x<=b
template <typename Vector1,
          typename Vector2>
void dense_histogram_data_on_device(Vector1& data,
                           Vector2& histogram, float a, float b)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());
    
  // find the end of each bin of values (histogram becomes the cumulative)
  thrust::counting_iterator<IndexType> search_begin(0);

  
  unsigned long num_bins = histogram.size();
  float s= (b-a)/num_bins;
  thrust::upper_bound(data.begin(), data.end(),
		      thrust::make_transform_iterator(search_begin, xhisto(a,s)),
		      thrust::make_transform_iterator(search_begin+num_bins, xhisto(a,s)),
                      histogram.begin());

  float fac = (float)data.size()*s;
  thrust::constant_iterator<float> iter(fac);
  thrust::transform(histogram.begin(), histogram.end(), iter, 
		    histogram.begin(),thrust::divides<float>());
  
  // compute the histogram by taking differences of the cumulative histogram
  thrust::adjacent_difference(histogram.begin(), histogram.end(), histogram.begin());
}

