#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <cmath>
#include <limits>

// This example computes several statistical properties of a data
// series in a single reduction.  The algorithm is described in detail here:
// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
//
// Thanks to Joseph Rhoads for contributing this example


// structure used to accumulate the moments and other 
// statistical properties encountered so far.
template <typename T>
struct summary_stats_data
{
    T n;
    T mean;
    T M2;
    
    // initialize to the identity element
    void initialize()
    {
      n = mean = M2 = 0;
    }

    T variance()   { return M2 / (n - 1); }
    T variance_n() { return M2 / n; }
};

// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
template <typename T>
struct summary_stats_unary_op
{
    __host__ __device__
    summary_stats_data<T> operator()(const T& x) const
    {
         summary_stats_data<T> result;
         result.n    = 1;
         result.mean = x;
         result.M2   = 0;

         return result;
    }
};

// summary_stats_binary_op is a functor that accepts two summary_stats_data 
// structs and returns a new summary_stats_data which are an
// approximation to the summary_stats for 
// all values that have been agregated so far
template <typename T>
struct summary_stats_binary_op 
    : public thrust::binary_function<const summary_stats_data<T>&, 
                                     const summary_stats_data<T>&,
                                           summary_stats_data<T> >
{
    __host__ __device__
    summary_stats_data<T> operator()(const summary_stats_data<T>& x, const summary_stats_data <T>& y) const
    {
        summary_stats_data<T> result;
        
        // precompute some common subexpressions
        T n  = x.n + y.n;

        T delta  = y.mean - x.mean;
        T delta2 = delta  * delta;
        
        //Basic number of samples (n), min, and max
        result.n   = n;

        result.mean = x.mean + delta * y.n / n;

        result.M2  = x.M2 + y.M2;
        result.M2 += delta2 * x.n * y.n / n;

        return result;
    }
};

template <typename Iterator>
void print_range(const std::string& name, Iterator first, Iterator last)
{
    typedef typename std::iterator_traits<Iterator>::value_type T;

    std::cout << name << ": ";
    thrust::copy(first, last, std::ostream_iterator<T>(std::cout, " "));  
    std::cout << "\n";
}


template <typename T>
void MeanVariance(thrust::device_vector<T> &d_x, summary_stats_data<T> &result)
{
    // setup arguments
    summary_stats_unary_op<T>  unary_op;
    summary_stats_binary_op<T> binary_op;
    summary_stats_data<T>      init;

    init.initialize();

    // compute summary statistics
    result = thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op);
}


#include <thrust/device_ptr.h>
#include <cuda.h>

template <typename T>
void MeanVariance(T * raw_ptr, size_t N, summary_stats_data<T> &result)
{
    // setup arguments
    summary_stats_unary_op<T>  unary_op;
    summary_stats_binary_op<T> binary_op;
    summary_stats_data<T>      init;

    init.initialize();

    thrust::device_ptr<T> dev_ptr(raw_ptr);

    // compute summary statistics
    result = thrust::transform_reduce(dev_ptr, dev_ptr+N, unary_op, init, binary_op);
}

