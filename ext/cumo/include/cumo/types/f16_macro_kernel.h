/* The body both 16 bit types share.
 *
 * half and bfloat16 compute in float and differ only in how they convert, so
 * the includer names the conversions and this file is written in terms of them.
 * cumo/types/half_macro_kernel.h and bf16_macro_kernel.h are the two includers.
 */

#include <string.h>

#define m_zero CUMO_F_TO_F16(0.0f)
#define m_one  CUMO_F_TO_F16(1.0f)

#define m_from_double(x) CUMO_D_TO_F16((double)(x))
#define m_from_real(x)   CUMO_D_TO_F16((double)(x))
#define m_from_sint(x)   CUMO_D_TO_F16((double)(x))
#define m_from_int32(x)  CUMO_D_TO_F16((double)(x))
#define m_from_int64(x)  CUMO_D_TO_F16((double)(x))
#define m_from_uint32(x) CUMO_D_TO_F16((double)(x))
#define m_from_uint64(x) CUMO_D_TO_F16((double)(x))

// Overloaded rather than substituted, so that a scan or a reduction can apply
// the same rule to its own wider accumulator.
__host__ __device__ static inline float cumo_f16_add(float x, float y) { return x + y; }
__host__ __device__ static inline float cumo_f16_sub(float x, float y) { return x - y; }
__host__ __device__ static inline float cumo_f16_mul(float x, float y) { return x * y; }
__host__ __device__ static inline float cumo_f16_div(float x, float y) { return x / y; }
__host__ __device__ static inline bool cumo_f16_not_nan(float x) { return x == x; }
__host__ __device__ static inline dtype cumo_f16_add(dtype x, dtype y) { return CUMO_F_TO_F16(CUMO_F16_TO_F(x) + CUMO_F16_TO_F(y)); }
__host__ __device__ static inline dtype cumo_f16_sub(dtype x, dtype y) { return CUMO_F_TO_F16(CUMO_F16_TO_F(x) - CUMO_F16_TO_F(y)); }
__host__ __device__ static inline dtype cumo_f16_mul(dtype x, dtype y) { return CUMO_F_TO_F16(CUMO_F16_TO_F(x) * CUMO_F16_TO_F(y)); }
__host__ __device__ static inline dtype cumo_f16_div(dtype x, dtype y) { return CUMO_F_TO_F16(CUMO_F16_TO_F(x) / CUMO_F16_TO_F(y)); }
__host__ __device__ static inline bool cumo_f16_not_nan(dtype x) { return CUMO_F16_TO_F(x) == CUMO_F16_TO_F(x); }

#define m_add(x,y) cumo_f16_add(x,y)
#define m_sub(x,y) cumo_f16_sub(x,y)
#define m_mul(x,y) cumo_f16_mul(x,y)
#define m_div(x,y) cumo_f16_div(x,y)
#define m_div_check(x,y) (CUMO_F16_TO_F(y)==0)

__host__ __device__ static inline float cumo_f16_floored_mod(float x, float y)
{
    float r = fmodf(x,y);
    if (r != 0 && ((r < 0) != (y < 0))) {
        r += y;
    }
    return r;
}

#define m_mod(x,y) CUMO_F_TO_F16(cumo_f16_floored_mod(CUMO_F16_TO_F(x),CUMO_F16_TO_F(y)))
#define m_divmod(x,y,a,b)                                               \
    {float cumo_f16_x = CUMO_F16_TO_F(x), cumo_f16_y = CUMO_F16_TO_F(y); \
     float cumo_f16_b = fmodf(cumo_f16_x,cumo_f16_y);                         \
     float cumo_f16_a = roundf((cumo_f16_x-cumo_f16_b)/cumo_f16_y);             \
     if (cumo_f16_b != 0 && ((cumo_f16_b < 0) != (cumo_f16_y < 0))) {         \
         cumo_f16_b += cumo_f16_y; cumo_f16_a -= 1;                           \
     }                                                                  \
     a = CUMO_F_TO_F16(cumo_f16_a); b = CUMO_F_TO_F16(cumo_f16_b);}
#define m_pow(x,y) CUMO_F_TO_F16(powf(CUMO_F16_TO_F(x),CUMO_F16_TO_F(y)))
#define m_pow_int(x,y) CUMO_F_TO_F16(cumo_f16_pow_int(CUMO_F16_TO_F(x),(int)(y)))

#define m_abs(x)     CUMO_F_TO_F16(fabsf(CUMO_F16_TO_F(x)))
#define m_minus(x)   CUMO_F_TO_F16(-CUMO_F16_TO_F(x))
#define m_reciprocal(x) CUMO_F_TO_F16(1.0f/CUMO_F16_TO_F(x))
#define m_square(x)  CUMO_F_TO_F16(CUMO_F16_TO_F(x)*CUMO_F16_TO_F(x))
#define m_floor(x)   CUMO_F_TO_F16(floorf(CUMO_F16_TO_F(x)))
#define m_round(x)   CUMO_F_TO_F16(roundf(CUMO_F16_TO_F(x)))
#define m_ceil(x)    CUMO_F_TO_F16(ceilf(CUMO_F16_TO_F(x)))
#define m_trunc(x)   CUMO_F_TO_F16(truncf(CUMO_F16_TO_F(x)))
#define m_rint(x)    CUMO_F_TO_F16(rintf(CUMO_F16_TO_F(x)))
#define m_sign(x)    CUMO_F_TO_F16(cumo_f16_sign(CUMO_F16_TO_F(x)))
#define m_copysign(x,y) CUMO_F_TO_F16(copysignf(CUMO_F16_TO_F(x),CUMO_F16_TO_F(y)))
#define m_signbit(x) signbit(CUMO_F16_TO_F(x))
#define m_modf(x,y,z) {float cumo_f16_i; float cumo_f16_f = modff(CUMO_F16_TO_F(x),&cumo_f16_i); y = CUMO_F_TO_F16(cumo_f16_f); z = CUMO_F_TO_F16(cumo_f16_i);}

#define m_eq(x,y) (CUMO_F16_TO_F(x)==CUMO_F16_TO_F(y))
#define m_ne(x,y) (CUMO_F16_TO_F(x)!=CUMO_F16_TO_F(y))
#define m_gt(x,y) (CUMO_F16_TO_F(x)>CUMO_F16_TO_F(y))
#define m_ge(x,y) (CUMO_F16_TO_F(x)>=CUMO_F16_TO_F(y))
#define m_lt(x,y) (CUMO_F16_TO_F(x)<CUMO_F16_TO_F(y))
#define m_le(x,y) (CUMO_F16_TO_F(x)<=CUMO_F16_TO_F(y))

#define m_isnan(x) isnan(CUMO_F16_TO_F(x))
#define m_isinf(x) isinf(CUMO_F16_TO_F(x))
#define m_isposinf(x) (isinf(CUMO_F16_TO_F(x)) && signbit(CUMO_F16_TO_F(x))==0)
#define m_isneginf(x) (isinf(CUMO_F16_TO_F(x)) && signbit(CUMO_F16_TO_F(x)))
#define m_isfinite(x) isfinite(CUMO_F16_TO_F(x))

#define not_nan(x) cumo_f16_not_nan(x)

#define m_mulsum_init m_zero

#define m_sprintf(s,x) sprintf(s,"%g",CUMO_F16_TO_F(x))

#define m_sqrt(x)    CUMO_F_TO_F16(sqrtf(CUMO_F16_TO_F(x)))
#define m_cbrt(x)    CUMO_F_TO_F16(cbrtf(CUMO_F16_TO_F(x)))
#define m_log(x)     CUMO_F_TO_F16(logf(CUMO_F16_TO_F(x)))
#define m_log2(x)    CUMO_F_TO_F16(log2f(CUMO_F16_TO_F(x)))
#define m_log10(x)   CUMO_F_TO_F16(log10f(CUMO_F16_TO_F(x)))
#define m_exp(x)     CUMO_F_TO_F16(expf(CUMO_F16_TO_F(x)))
#define m_exp2(x)    CUMO_F_TO_F16(exp2f(CUMO_F16_TO_F(x)))
#ifdef HAVE_EXP10
#define m_exp10(x)   CUMO_F_TO_F16(exp10f(CUMO_F16_TO_F(x)))
#else
#define m_exp10(x)   CUMO_F_TO_F16(powf(10.0f,CUMO_F16_TO_F(x)))
#endif
#define m_expm1(x)   CUMO_F_TO_F16(expm1f(CUMO_F16_TO_F(x)))
#define m_log1p(x)   CUMO_F_TO_F16(log1pf(CUMO_F16_TO_F(x)))

#define m_sin(x)     CUMO_F_TO_F16(sinf(CUMO_F16_TO_F(x)))
#define m_cos(x)     CUMO_F_TO_F16(cosf(CUMO_F16_TO_F(x)))
#define m_tan(x)     CUMO_F_TO_F16(tanf(CUMO_F16_TO_F(x)))
#define m_asin(x)    CUMO_F_TO_F16(asinf(CUMO_F16_TO_F(x)))
#define m_acos(x)    CUMO_F_TO_F16(acosf(CUMO_F16_TO_F(x)))
#define m_atan(x)    CUMO_F_TO_F16(atanf(CUMO_F16_TO_F(x)))
#define m_sinh(x)    CUMO_F_TO_F16(sinhf(CUMO_F16_TO_F(x)))
#define m_cosh(x)    CUMO_F_TO_F16(coshf(CUMO_F16_TO_F(x)))
#define m_tanh(x)    CUMO_F_TO_F16(tanhf(CUMO_F16_TO_F(x)))
#define m_asinh(x)   CUMO_F_TO_F16(asinhf(CUMO_F16_TO_F(x)))
#define m_acosh(x)   CUMO_F_TO_F16(acoshf(CUMO_F16_TO_F(x)))
#define m_atanh(x)   CUMO_F_TO_F16(atanhf(CUMO_F16_TO_F(x)))
#define m_atan2(x,y) CUMO_F_TO_F16(atan2f(CUMO_F16_TO_F(x),CUMO_F16_TO_F(y)))
#define m_hypot(x,y) CUMO_F_TO_F16(hypotf(CUMO_F16_TO_F(x),CUMO_F16_TO_F(y)))
#define m_sinc(x)    CUMO_F_TO_F16(cumo_f16_sinc(CUMO_F16_TO_F(x)))

#define m_erf(x)     CUMO_F_TO_F16(erff(CUMO_F16_TO_F(x)))
#define m_erfc(x)    CUMO_F_TO_F16(erfcf(CUMO_F16_TO_F(x)))
#define m_gelu(x)      CUMO_F_TO_F16(cumo_f16_gelu(CUMO_F16_TO_F(x)))
#define m_gelu_tanh(x) CUMO_F_TO_F16(cumo_f16_gelu_tanh(CUMO_F16_TO_F(x)))
#define m_silu(x)      CUMO_F_TO_F16(cumo_f16_silu(CUMO_F16_TO_F(x)))
#define m_softplus(x)  CUMO_F_TO_F16(cumo_f16_softplus(CUMO_F16_TO_F(x)))
#define m_ldexp(x,y) CUMO_F_TO_F16(cumo_f16_ldexp(CUMO_F16_TO_F(x),CUMO_F16_TO_F(y)))
#define m_frexp(x,exp) CUMO_F_TO_F16(frexpf(CUMO_F16_TO_F(x),exp))

__host__ __device__ static inline float cumo_f16_pow_positive_int(float x, unsigned int p)
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

__host__ __device__ static inline float cumo_f16_pow_int(float x, int p)
{
    if (p < 0) return 1.0f / cumo_f16_pow_positive_int(x, -(unsigned int)p);
    return cumo_f16_pow_positive_int(x, (unsigned int)p);
}

// The bit-pattern intrinsics are device-only before CUDA 12, so the bits are
// taken the way a host caller can take them too.
__host__ __device__ static inline dtype cumo_f16_step_down(dtype h)
{
    unsigned short b;
    dtype r;
    memcpy(&b, &h, sizeof(b));
    if (b == 0x0000u) {
        b = 0x8001u;
    } else {
        b = (unsigned short)((b & 0x8000u) ? (b + 1) : (b - 1));
    }
    memcpy(&r, &b, sizeof(b));
    return r;
}

__host__ __device__ static inline float cumo_f16_ldexp(float x, float y)
{
    if (!(y == y)) return y;
    if (y > 64.0f) y = 64.0f;
    if (y < -64.0f) y = -64.0f;
    return ldexpf(x, (int)y);
}

__host__ __device__ static inline float cumo_f16_sinc(float x)
{
    return (x==0) ? 1.0f : (sinf(x)/x);
}

__host__ __device__ static inline float cumo_f16_sign(float x)
{
    return (x==0) ? 0.0f : ((x>0) ? 1.0f : ((x<0) ? -1.0f : x));
}

__host__ __device__ static inline dtype f_seq(dtype x, dtype y, double c)
{
    return CUMO_F_TO_F16(CUMO_F16_TO_F(x) + CUMO_F16_TO_F(y) * (float)c);
}

#include "real_accum_kernel.h"

__host__ __device__ static inline float cumo_f16_gelu(float x)
{
    return 0.5f * x * (1.0f + erff(x * (float)CUMO_M_SQRT1_2));
}

// The approximation GPT-2 and the transformers after it were trained with.
__host__ __device__ static inline float cumo_f16_gelu_tanh(float x)
{
    return 0.5f * x * (1.0f + tanhf((float)CUMO_M_SQRT_2_OVER_PI * (x + (float)CUMO_GELU_TANH_CUBIC * x * x * x)));
}

// x * sigmoid(x) as one division, cumo having no sigmoid to call.
__host__ __device__ static inline float cumo_f16_silu(float x)
{
    return x / (1.0f + expf(-x));
}

// Not log1p(exp(x)): the sum is taken first here, which is a different number
// and the one the implementations this follows answer. Taken that way it
// reaches an infinity once exp does, where softplus is x to the last bit.
__host__ __device__ static inline float cumo_f16_softplus(float x)
{
    float e = expf(x);
    return isinf(e) ? x : logf(1.0f + e);
}
