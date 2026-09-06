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

// Overloaded rather than substituted, so that a scan or a reduction can apply
// the same rule to its own wider accumulator.
__host__ __device__ static inline float cumo_half_add(float x, float y) { return x + y; }
__host__ __device__ static inline float cumo_half_sub(float x, float y) { return x - y; }
__host__ __device__ static inline float cumo_half_mul(float x, float y) { return x * y; }
__host__ __device__ static inline float cumo_half_div(float x, float y) { return x / y; }
__host__ __device__ static inline bool cumo_half_not_nan(float x) { return x == x; }
__host__ __device__ static inline cumo_half cumo_half_add(cumo_half x, cumo_half y) { return cumo_float2half(cumo_half2float(x) + cumo_half2float(y)); }
__host__ __device__ static inline cumo_half cumo_half_sub(cumo_half x, cumo_half y) { return cumo_float2half(cumo_half2float(x) - cumo_half2float(y)); }
__host__ __device__ static inline cumo_half cumo_half_mul(cumo_half x, cumo_half y) { return cumo_float2half(cumo_half2float(x) * cumo_half2float(y)); }
__host__ __device__ static inline cumo_half cumo_half_div(cumo_half x, cumo_half y) { return cumo_float2half(cumo_half2float(x) / cumo_half2float(y)); }
__host__ __device__ static inline bool cumo_half_not_nan(cumo_half x) { return cumo_half2float(x) == cumo_half2float(x); }

#define m_add(x,y) cumo_half_add(x,y)
#define m_sub(x,y) cumo_half_sub(x,y)
#define m_mul(x,y) cumo_half_mul(x,y)
#define m_div(x,y) cumo_half_div(x,y)
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

#define not_nan(x) cumo_half_not_nan(x)

#define m_mulsum_init m_zero

#define m_sprintf(s,x) sprintf(s,"%g",cumo_half2float(x))

#define m_sqrt(x)    cumo_float2half(sqrtf(cumo_half2float(x)))
#define m_cbrt(x)    cumo_float2half(cbrtf(cumo_half2float(x)))
#define m_log(x)     cumo_float2half(logf(cumo_half2float(x)))
#define m_log2(x)    cumo_float2half(log2f(cumo_half2float(x)))
#define m_log10(x)   cumo_float2half(log10f(cumo_half2float(x)))
#define m_exp(x)     cumo_float2half(expf(cumo_half2float(x)))
#define m_exp2(x)    cumo_float2half(exp2f(cumo_half2float(x)))
#ifdef HAVE_EXP10
#define m_exp10(x)   cumo_float2half(exp10f(cumo_half2float(x)))
#else
#define m_exp10(x)   cumo_float2half(powf(10.0f,cumo_half2float(x)))
#endif
#define m_expm1(x)   cumo_float2half(expm1f(cumo_half2float(x)))
#define m_log1p(x)   cumo_float2half(log1pf(cumo_half2float(x)))

#define m_sin(x)     cumo_float2half(sinf(cumo_half2float(x)))
#define m_cos(x)     cumo_float2half(cosf(cumo_half2float(x)))
#define m_tan(x)     cumo_float2half(tanf(cumo_half2float(x)))
#define m_asin(x)    cumo_float2half(asinf(cumo_half2float(x)))
#define m_acos(x)    cumo_float2half(acosf(cumo_half2float(x)))
#define m_atan(x)    cumo_float2half(atanf(cumo_half2float(x)))
#define m_sinh(x)    cumo_float2half(sinhf(cumo_half2float(x)))
#define m_cosh(x)    cumo_float2half(coshf(cumo_half2float(x)))
#define m_tanh(x)    cumo_float2half(tanhf(cumo_half2float(x)))
#define m_asinh(x)   cumo_float2half(asinhf(cumo_half2float(x)))
#define m_acosh(x)   cumo_float2half(acoshf(cumo_half2float(x)))
#define m_atanh(x)   cumo_float2half(atanhf(cumo_half2float(x)))
#define m_atan2(x,y) cumo_float2half(atan2f(cumo_half2float(x),cumo_half2float(y)))
#define m_hypot(x,y) cumo_float2half(hypotf(cumo_half2float(x),cumo_half2float(y)))
#define m_sinc(x)    cumo_float2half(cumo_half_sinc(cumo_half2float(x)))

#define m_erf(x)     cumo_float2half(erff(cumo_half2float(x)))
#define m_erfc(x)    cumo_float2half(erfcf(cumo_half2float(x)))
#define m_ldexp(x,y) cumo_float2half(cumo_half_ldexp(cumo_half2float(x),cumo_half2float(y)))
#define m_frexp(x,exp) cumo_float2half(frexpf(cumo_half2float(x),exp))


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

__host__ __device__ static inline cumo_half cumo_half_step_down(cumo_half h)
{
    unsigned short b = __half_as_ushort(h);
    if (b == 0x0000u) return __ushort_as_half(0x8001u);
    return __ushort_as_half((unsigned short)((b & 0x8000u) ? (b + 1) : (b - 1)));
}

__host__ __device__ static inline float cumo_half_ldexp(float x, float y)
{
    if (!(y == y)) return y;
    if (y > 64.0f) y = 64.0f;
    if (y < -64.0f) y = -64.0f;
    return ldexpf(x, (int)y);
}

__host__ __device__ static inline float cumo_half_sinc(float x)
{
    return (x==0) ? 1.0f : (sinf(x)/x);
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
