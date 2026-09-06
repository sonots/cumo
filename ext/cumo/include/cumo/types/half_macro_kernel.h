#ifndef CUMO_HALF_MACRO_KERNEL_H
#define CUMO_HALF_MACRO_KERNEL_H

#include "float_def_kernel.h"
#include "half_def_kernel.h"

#define CUMO_HALF_EPSILON 9.765625e-04f

#define m_zero cumo_float2half(0.0f)
#define m_one  cumo_float2half(1.0f)

#define m_from_double(x) cumo_double2half((double)(x))
#define m_from_real(x)   cumo_double2half((double)(x))
#define m_from_sint(x)   cumo_double2half((double)(x))
#define m_from_int32(x)  cumo_double2half((double)(x))
#define m_from_int64(x)  cumo_double2half((double)(x))
#define m_from_uint32(x) cumo_double2half((double)(x))
#define m_from_uint64(x) cumo_double2half((double)(x))
#define m_from_half(x)   (x)

#define m_add(x,y) cumo_float2half(cumo_half2float(x)+cumo_half2float(y))
#define m_sub(x,y) cumo_float2half(cumo_half2float(x)-cumo_half2float(y))
#define m_mul(x,y) cumo_float2half(cumo_half2float(x)*cumo_half2float(y))
#define m_div(x,y) cumo_float2half(cumo_half2float(x)/cumo_half2float(y))
#define m_div_check(x,y) (cumo_half2float(y)==0)

__host__ __device__ static inline float cumo_half_floored_mod(float x, float y)
{
    float r = fmodf(x,y);
    if (r != 0 && ((r < 0) != (y < 0))) {
        r += y;
    }
    return r;
}

#define m_mod(x,y) cumo_float2half(cumo_half_floored_mod(cumo_half2float(x),cumo_half2float(y)))
#define m_divmod(x,y,a,b)                                               \
    {float cumo_h_x = cumo_half2float(x), cumo_h_y = cumo_half2float(y); \
     float cumo_h_b = fmodf(cumo_h_x,cumo_h_y);                         \
     float cumo_h_a = roundf((cumo_h_x-cumo_h_b)/cumo_h_y);             \
     if (cumo_h_b != 0 && ((cumo_h_b < 0) != (cumo_h_y < 0))) {         \
         cumo_h_b += cumo_h_y; cumo_h_a -= 1;                           \
     }                                                                  \
     a = cumo_float2half(cumo_h_a); b = cumo_float2half(cumo_h_b);}
#define m_pow(x,y) cumo_float2half(powf(cumo_half2float(x),cumo_half2float(y)))
#define m_pow_int(x,y) cumo_float2half(cumo_half_pow_int(cumo_half2float(x),(int)(y)))

#define m_abs(x)     cumo_float2half(fabsf(cumo_half2float(x)))
#define m_minus(x)   cumo_float2half(-cumo_half2float(x))
#define m_reciprocal(x) cumo_float2half(1.0f/cumo_half2float(x))
#define m_square(x)  cumo_float2half(cumo_half2float(x)*cumo_half2float(x))
#define m_floor(x)   cumo_float2half(floorf(cumo_half2float(x)))
#define m_round(x)   cumo_float2half(roundf(cumo_half2float(x)))
#define m_ceil(x)    cumo_float2half(ceilf(cumo_half2float(x)))
#define m_trunc(x)   cumo_float2half(truncf(cumo_half2float(x)))
#define m_rint(x)    cumo_float2half(rintf(cumo_half2float(x)))
#define m_sign(x)    cumo_float2half(cumo_half_sign(cumo_half2float(x)))
#define m_copysign(x,y) cumo_float2half(copysignf(cumo_half2float(x),cumo_half2float(y)))
#define m_signbit(x) signbit(cumo_half2float(x))
#define m_modf(x,y,z) {float cumo_h_i; float cumo_h_f = modff(cumo_half2float(x),&cumo_h_i); y = cumo_float2half(cumo_h_f); z = cumo_float2half(cumo_h_i);}

#define m_eq(x,y) (cumo_half2float(x)==cumo_half2float(y))
#define m_ne(x,y) (cumo_half2float(x)!=cumo_half2float(y))
#define m_gt(x,y) (cumo_half2float(x)>cumo_half2float(y))
#define m_ge(x,y) (cumo_half2float(x)>=cumo_half2float(y))
#define m_lt(x,y) (cumo_half2float(x)<cumo_half2float(y))
#define m_le(x,y) (cumo_half2float(x)<=cumo_half2float(y))

#define m_isnan(x) isnan(cumo_half2float(x))
#define m_isinf(x) isinf(cumo_half2float(x))
#define m_isposinf(x) (isinf(cumo_half2float(x)) && signbit(cumo_half2float(x))==0)
#define m_isneginf(x) (isinf(cumo_half2float(x)) && signbit(cumo_half2float(x)))
#define m_isfinite(x) isfinite(cumo_half2float(x))

#define not_nan(x) (cumo_half2float(x)==cumo_half2float(x))

#define m_mulsum_init m_zero

#define m_sprintf(s,x) sprintf(s,"%g",cumo_half2float(x))

#define m_sqrt(x)    cumo_float2half(sqrtf(cumo_half2float(x)))

__host__ __device__ static inline float cumo_half_pow_positive_int(float x, unsigned int p)
{
    float r = 1.0f;
    switch (p) {
    case 0: return 1.0f;
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    case 4: x = x*x; return x*x;
    }
    if (p > 64) return powf(x, (float)p);
    while (p) {
        if (p & 1) r *= x;
        x *= x;
        p >>= 1;
    }
    return r;
}

__host__ __device__ static inline float cumo_half_pow_int(float x, int p)
{
    if (p < 0) return 1.0f / cumo_half_pow_positive_int(x, -(unsigned int)p);
    return cumo_half_pow_positive_int(x, (unsigned int)p);
}

__host__ __device__ static inline float cumo_half_sign(float x)
{
    return (x==0) ? 0.0f : ((x>0) ? 1.0f : ((x<0) ? -1.0f : x));
}

__host__ __device__ static inline cumo_half f_seq(cumo_half x, cumo_half y, double c)
{
    return cumo_float2half(cumo_half2float(x) + cumo_half2float(y) * (float)c);
}

#include "real_accum_kernel.h"

#endif // CUMO_HALF_MACRO_KERNEL_H
