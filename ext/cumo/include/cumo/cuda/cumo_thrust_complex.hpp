#ifndef CUMO_CUDA_THRUST_COMPLEX_H
#define CUMO_CUDA_THRUST_COMPLEX_H

#include "cumo/types/complex_kernel.h"
#include "cumo/cuda/cumo_thrust.hpp"

// ref. https://github.com/thrust/thrust/blob/master/examples/summary_statistics.cu

// structure used to accumulate the moments and other
// statistical properties encountered so far.
template <typename T, typename R>
struct cumo_thrust_complex_variance_data
{
    R n;
    T mean;
    R M2;

    // initialize to the identity element
    void initialize()
    {
        n = M2 = 0;
        mean = c_zero();
    }

    __host__ __device__ R variance()   { return M2 / (n - 1); }
    __host__ __device__ R variance_n() { return M2 / n; }
};

// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
template <typename T, typename R>
struct cumo_thrust_complex_variance_unary_op
{
    __host__ __device__
    cumo_thrust_complex_variance_data<T,R> operator()(const T& x) const
    {
         cumo_thrust_complex_variance_data<T,R> result;
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
template <typename T, typename R>
struct cumo_thrust_complex_variance_binary_op
{
    using first_argument_type  = const cumo_thrust_complex_variance_data<T,R>&;
    using second_argument_type = const cumo_thrust_complex_variance_data<T,R>&;
    using result_type          = cumo_thrust_complex_variance_data<T,R>;
    __host__ __device__
    cumo_thrust_complex_variance_data<T,R> operator()(const cumo_thrust_complex_variance_data<T,R>& x, const cumo_thrust_complex_variance_data<T,R>& y) const
    {
        cumo_thrust_complex_variance_data<T,R> result;

        // precompute some common subexpressions
        R n  = x.n + y.n;

        T delta = c_sub(y.mean, x.mean);
        R delta2 = c_abs_square(delta);

        //Basic number of samples (n)
        result.n = n;

        result.mean = c_add(x.mean, c_mul_r(delta, y.n / n));

        result.M2 = x.M2 + y.M2;
        result.M2 += delta2 * x.n * y.n / n;

        return result;
    }
};

#endif /* ifndef CUMO_CUDA_THRUST_COMPLEX_H */
