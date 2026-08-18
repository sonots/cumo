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


#endif /* ifndef CUMO_CUDA_THRUST_H */
