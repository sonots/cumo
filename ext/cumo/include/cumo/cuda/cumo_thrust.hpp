#ifndef CUMO_CUDA_THRUST_H
#define CUMO_CUDA_THRUST_H

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system_error.h>
#include <thrust/transform_reduce.h>

// Scratch for an algorithm that needs it, taken from the same pool as array
// data so that a loop calling one per row reuses a free-list entry instead of
// going to cudaMalloc every time. Declared here rather than including
// cuda/memory_pool.h, which needs ruby.h.
extern "C" char* cumo_cuda_runtime_malloc(size_t size);
extern "C" void cumo_cuda_runtime_free(char *ptr);

struct cumo_thrust_pool_allocator
{
    typedef char value_type;
    char *allocate(std::ptrdiff_t n) { return cumo_cuda_runtime_malloc((size_t)n); }
    void deallocate(char *p, size_t) { cumo_cuda_runtime_free(p); }
};

// this example illustrates how to make strided access to a range of values
// examples:
//   strided_range([0, 1, 2, 3, 4, 5, 6], 1) -> [0, 1, 2, 3, 4, 5, 6]
//   strided_range([0, 1, 2, 3, 4, 5, 6], 2) -> [0, 2, 4, 6]
//   strided_range([0, 1, 2, 3, 4, 5, 6], 3) -> [0, 3, 6]
//   ...
// ref. https://github.com/thrust/thrust/blob/master/examples/strided_range.cu (Apache License)

template <typename Iterator>
class cumo_thrust_strided_range
{
    public:

    typedef typename thrust::iterator_traits<Iterator>::difference_type difference_type;

    struct stride_functor
    {
        using argument_type = difference_type;
        using result_type   = difference_type;
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        {
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    cumo_thrust_strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), stride(stride), count(((last - first) + (stride - 1)) / stride) {}

    // A stride of 0 repeats one element and a negative stride walks backwards, so
    // neither can be counted from (last - first); those callers pass the count.
    cumo_thrust_strided_range(Iterator first, difference_type stride, difference_type count)
        : first(first), stride(stride), count(count) {}

    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + count;
    }

    protected:
    Iterator first;
    difference_type stride;
    difference_type count;
};


// compute minimum and maximum values in a single reduction
// ref. https://github.com/thrust/thrust/blob/master/examples/minmax.cu (Apache License)

// cumo_thrust_minmax_pair stores the minimum and maximum
// values that have been encountered so far
template <typename T>
struct cumo_thrust_minmax_pair
{
    T min_val;
    T max_val;
};

// cumo_thrust_minmax_unary_op is a functor that takes in a value x and
// returns a cumo_thrust_minmax_pair whose minimum and maximum values
// are initialized to x.
template <typename T>
struct cumo_thrust_minmax_unary_op
{
    using argument_type = T;
    using result_type   = cumo_thrust_minmax_pair<T>;
    __host__ __device__ cumo_thrust_minmax_pair<T> operator()(const T& x) const
    {
        cumo_thrust_minmax_pair<T> result;
        result.min_val = x;
        result.max_val = x;
        return result;
    }
};

// cumo_thrust_minmax_binary_op is a functor that accepts two cumo_thrust_minmax_pair
// structs and returns a new cumo_thrust_minmax_pair whose minimum and
// maximum values are the min() and max() respectively of
// the minimums and maximums of the input pairs
template <typename T>
struct cumo_thrust_minmax_binary_op
{
    using first_argument_type  = cumo_thrust_minmax_pair<T>;
    using second_argument_type = cumo_thrust_minmax_pair<T>;
    using result_type          = cumo_thrust_minmax_pair<T>;
    __host__ __device__ cumo_thrust_minmax_pair<T> operator()(const cumo_thrust_minmax_pair<T>& x, const cumo_thrust_minmax_pair<T>& y) const
    {
        cumo_thrust_minmax_pair<T> result;
        result.min_val = thrust::min(x.min_val, y.min_val);
        result.max_val = thrust::max(x.max_val, y.max_val);
        return result;
    }
};

// ref. https://github.com/thrust/thrust/blob/master/examples/summary_statistics.cu

// structure used to accumulate the moments and other
// statistical properties encountered so far.
template <typename T>
struct cumo_thrust_variance_data
{
    T n;
    T mean;
    T M2;

    // initialize to the identity element
    void initialize()
    {
        n = mean = M2 = 0;
    }

    __host__ __device__ T variance()   { return M2 / (n - 1); }
    __host__ __device__ T variance_n() { return M2 / n; }
};

// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
template <typename T>
struct cumo_thrust_variance_unary_op
{
    __host__ __device__
    cumo_thrust_variance_data<T> operator()(const T& x) const
    {
         cumo_thrust_variance_data<T> result;
         result.n    = 1;
         result.mean = x;
         result.M2   = 0;

         return result;
    }
};

// cumo_thrust_variance_binary_op is a functor that accepts two cumo_thrust_variance_data
// structs and returns a new cumo_thrust_variance_data which are an
// approximation to the cumo_thrust_variance for
// all values that have been agregated so far
template <typename T>
struct cumo_thrust_variance_binary_op
{
    using first_argument_type  = const cumo_thrust_variance_data<T>&;
    using second_argument_type = const cumo_thrust_variance_data<T>&;
    using result_type          = cumo_thrust_variance_data<T>;
    __host__ __device__
    cumo_thrust_variance_data<T> operator()(const cumo_thrust_variance_data<T>& x, const cumo_thrust_variance_data <T>& y) const
    {
        cumo_thrust_variance_data<T> result;

        // precompute some common subexpressions
        T n  = x.n + y.n;

        T delta  = y.mean - x.mean;
        T delta2 = delta  * delta;

        //Basic number of samples (n)
        result.n   = n;

        result.mean = x.mean + delta * y.n / n;

        result.M2  = x.M2 + y.M2;
        result.M2 += delta2 * x.n * y.n / n;

        return result;
    }
};

#endif /* ifndef CUMO_CUDA_THRUST_H */
