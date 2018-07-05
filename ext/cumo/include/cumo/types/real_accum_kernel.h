#ifndef CUMO_REAL_ACCUM_KERNEL_H
#define CUMO_REAL_ACCUM_KERNEL_H

#define not_nan(x) ((x)==(x))

#define m_mulsum(x,y,z) {z = m_add(m_mul(x,y),z);}
#define m_mulsum_nan(x,y,z) {          \
        if(not_nan(x) && not_nan(y)) { \
            z = m_add(m_mul(x,y),z);   \
        }}

#define m_cumsum(x,y) {(x)=m_add(x,y);}
#define m_cumsum_nan(x,y) {      \
        if (!not_nan(x)) {       \
            (x) = (y);           \
        } else if (not_nan(y)) { \
            (x) = m_add(x,y);    \
        }}

#define m_cumprod(x,y) {(x)=m_mul(x,y);}
#define m_cumprod_nan(x,y) {     \
        if (!not_nan(x)) {       \
            (x) = (y);           \
        } else if (not_nan(y)) { \
            (x) = m_mul(x,y);    \
        }}

__host__ __device__ static inline dtype f_maximum(dtype x, dtype y)
{
    if (m_ge(x,y)) {
        return x;
    }
    if (not_nan(y)) {
        return y;
    }
    return x;
}

__host__ __device__ static inline dtype f_maximum_nan(dtype x, dtype y)
{
    if (m_ge(x,y)) {
        return x;
    }
    if (!not_nan(x)) {
        return x;
    }
    return y;
}

__host__ __device__ static inline dtype f_minimum(dtype x, dtype y)
{
    if (m_le(x,y)) {
        return x;
    }
    if (not_nan(y)) {
        return y;
    }
    return x;
}

__host__ __device__ static inline dtype f_minimum_nan(dtype x, dtype y)
{
    if (m_le(x,y)) {
        return x;
    }
    if (!not_nan(x)) {
        return x;
    }
    return y;
}

/* --------- thrust ----------------- */
#include "cumo/cuda/cumo_thrust.hpp"

struct cumo_thrust_plus : public thrust::binary_function<dtype, dtype, dtype>
{
    __host__ __device__ dtype operator()(dtype x, dtype y) { return m_add(x,y); }
};

struct cumo_thrust_multiplies : public thrust::binary_function<dtype, dtype, dtype>
{
    __host__ __device__ dtype operator()(dtype x, dtype y) { return m_mul(x,y); }
};

struct cumo_thrust_multiplies_mulsum_nan : public thrust::binary_function<dtype, dtype, dtype>
{
    __host__ __device__ dtype operator()(dtype x, dtype y) {
        if (not_nan(x) && not_nan(y)) {
            return m_mul(x, y);
        } else {
            return m_zero;
        }
    }
};

struct cumo_thrust_square : public thrust::unary_function<dtype, dtype>
{
    __host__ __device__ rtype operator()(const dtype& x) const { return m_square(x); }
};

#endif // CUMO_REAL_ACCUM_KERNEL_H
