#ifndef CUMO_XINT_MACRO_KERNEL_H
#define CUMO_XINT_MACRO_KERNEL_H

#define m_zero 0
#define m_one  1

#define m_from_double(x) (x)
#define m_from_real(x) (x)

#define m_add(x,y) ((x)+(y))
#define m_sub(x,y) ((x)-(y))
#define m_mul(x,y) ((x)*(y))
#define m_div(x,y) ((x)/(y))
#define m_mod(x,y) ((x)%(y))
#define m_divmod(x,y,a,b) {a=(x)/(y); b=m_mod(x,y);}
#define m_pow(x,y) pow_int(x,y)
#define m_pow_int(x,y) pow_int(x,y)

#define m_bit_and(x,y) ((x)&(y))
#define m_bit_or(x,y)  ((x)|(y))
#define m_bit_xor(x,y) ((x)^(y))
#define m_bit_not(x)   (~(x))

#define m_minus(x)   (-(x))
#define m_reciprocal(x) int_reciprocal(x)
#define m_square(x)  ((x)*(x))

#define m_eq(x,y) ((x)==(y))
#define m_ne(x,y) ((x)!=(y))
#define m_gt(x,y) ((x)>(y))
#define m_ge(x,y) ((x)>=(y))
#define m_lt(x,y) ((x)<(y))
#define m_le(x,y) ((x)<=(y))
#define m_left_shift(x,y) ((x)<<(y))
#define m_right_shift(x,y) ((x)>>(y))

#define m_isnan(x) 0

#define m_mulsum(x,y,z) {z += x*y;}
#define m_mulsum_init 0
#define m_gemm_init 0
#define m_gemv_init 0
//#define m_cumsum(x,y) {x += y;}
//#define m_cumprod(x,y) {x *= y;}

__host__ __device__ static inline double f_seq(double x, double y, double c)
{
    return x + y * c;
}

/* --------- thrust ----------------- */
#include "cumo/cuda/cumo_thrust.hpp"

struct thrust_plus : public thrust::binary_function<dtype, dtype, dtype>
{
    __host__ __device__ dtype operator()(dtype x, dtype y) { return m_add(x,y); }
};

struct thrust_multiplies : public thrust::binary_function<dtype, dtype, dtype>
{
    __host__ __device__ dtype operator()(dtype x, dtype y) { return m_mul(x,y); }
};

struct thrust_square : public thrust::unary_function<dtype, dtype>
{
    __host__ __device__ rtype operator()(const dtype& x) const { return m_square(x); }
};

#endif // CUMO_XINT_MACRO_KERNEL_H
