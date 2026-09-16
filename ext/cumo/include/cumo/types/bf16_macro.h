#include "float_def.h"
#include "half_def.h"
#include "bf16_def.h"

#define CUMO_BFLOAT_EPSILON 7.8125e-03f

#define m_zero cumo_float2bfloat(0.0f)
#define m_one  cumo_float2bfloat(1.0f)

#define m_num_to_data(x) cumo_na_ruby_num_to_bfloat(x)
#define m_data_to_num(x) rb_float_new(cumo_bfloat2float(x))

#define m_from_double(x) cumo_double2bfloat((double)(x))
#define m_from_real(x)   cumo_double2bfloat((double)(x))
#define m_from_sint(x)   cumo_double2bfloat((double)(x))
#define m_from_int32(x)  cumo_double2bfloat((double)(x))
#define m_from_int64(x)  cumo_double2bfloat((double)(x))
#define m_from_uint32(x) cumo_double2bfloat((double)(x))
#define m_from_uint64(x) cumo_double2bfloat((double)(x))
#define m_from_bfloat(x)   (x)
#define m_from_half(x) cumo_float2bfloat(cumo_half2float(x))

#define m_add(x,y) cumo_float2bfloat(cumo_bfloat2float(x)+cumo_bfloat2float(y))
#define m_sub(x,y) cumo_float2bfloat(cumo_bfloat2float(x)-cumo_bfloat2float(y))
#define m_mul(x,y) cumo_float2bfloat(cumo_bfloat2float(x)*cumo_bfloat2float(y))
#define m_div(x,y) cumo_float2bfloat(cumo_bfloat2float(x)/cumo_bfloat2float(y))
#define m_div_check(x,y) (cumo_bfloat2float(y)==0)

static inline float cumo_bfloat_floored_mod(float x, float y)
{
    float r = fmodf(x,y);
    if (r != 0 && ((r < 0) != (y < 0))) {
        r += y;
    }
    return r;
}

#define m_mod(x,y) cumo_float2bfloat(cumo_bfloat_floored_mod(cumo_bfloat2float(x),cumo_bfloat2float(y)))
#define m_divmod(x,y,a,b)                                               \
    {float cumo_b_x = cumo_bfloat2float(x), cumo_b_y = cumo_bfloat2float(y); \
     float cumo_b_b = fmodf(cumo_b_x,cumo_b_y);                         \
     float cumo_b_a = roundf((cumo_b_x-cumo_b_b)/cumo_b_y);             \
     if (cumo_b_b != 0 && ((cumo_b_b < 0) != (cumo_b_y < 0))) {         \
         cumo_b_b += cumo_b_y; cumo_b_a -= 1;                           \
     }                                                                  \
     a = cumo_float2bfloat(cumo_b_a); b = cumo_float2bfloat(cumo_b_b);}
#define m_pow(x,y) cumo_float2bfloat(powf(cumo_bfloat2float(x),cumo_bfloat2float(y)))
#define m_pow_int(x,y) cumo_float2bfloat(cumo_bfloat_pow_int(cumo_bfloat2float(x),(int)(y)))

#define m_abs(x)     cumo_float2bfloat(fabsf(cumo_bfloat2float(x)))
#define m_minus(x)   cumo_float2bfloat(-cumo_bfloat2float(x))
#define m_reciprocal(x) cumo_float2bfloat(1.0f/cumo_bfloat2float(x))
#define m_square(x)  cumo_float2bfloat(cumo_bfloat2float(x)*cumo_bfloat2float(x))
#define m_floor(x)   cumo_float2bfloat(floorf(cumo_bfloat2float(x)))
#define m_round(x)   cumo_float2bfloat(roundf(cumo_bfloat2float(x)))
#define m_ceil(x)    cumo_float2bfloat(ceilf(cumo_bfloat2float(x)))
#define m_trunc(x)   cumo_float2bfloat(truncf(cumo_bfloat2float(x)))
#define m_rint(x)    cumo_float2bfloat(rintf(cumo_bfloat2float(x)))
#define m_sign(x)    cumo_float2bfloat(cumo_bfloat_sign(cumo_bfloat2float(x)))
#define m_copysign(x,y) cumo_float2bfloat(copysignf(cumo_bfloat2float(x),cumo_bfloat2float(y)))
#define m_signbit(x) signbit(cumo_bfloat2float(x))
#define m_modf(x,y,z) {float cumo_b_i; float cumo_b_f = modff(cumo_bfloat2float(x),&cumo_b_i); y = cumo_float2bfloat(cumo_b_f); z = cumo_float2bfloat(cumo_b_i);}

#define m_eq(x,y) (cumo_bfloat2float(x)==cumo_bfloat2float(y))
#define m_ne(x,y) (cumo_bfloat2float(x)!=cumo_bfloat2float(y))
#define m_gt(x,y) (cumo_bfloat2float(x)>cumo_bfloat2float(y))
#define m_ge(x,y) (cumo_bfloat2float(x)>=cumo_bfloat2float(y))
#define m_lt(x,y) (cumo_bfloat2float(x)<cumo_bfloat2float(y))
#define m_le(x,y) (cumo_bfloat2float(x)<=cumo_bfloat2float(y))

#define m_isnan(x) isnan(cumo_bfloat2float(x))
#define m_isinf(x) isinf(cumo_bfloat2float(x))
#define m_isposinf(x) (isinf(cumo_bfloat2float(x)) && signbit(cumo_bfloat2float(x))==0)
#define m_isneginf(x) (isinf(cumo_bfloat2float(x)) && signbit(cumo_bfloat2float(x)))
#define m_isfinite(x) isfinite(cumo_bfloat2float(x))

#define not_nan(x) (cumo_bfloat2float(x)==cumo_bfloat2float(x))

#define m_mulsum_init INT2FIX(0)

#define m_sprintf(s,x) sprintf(s,"%g",cumo_bfloat2float(x))

#define m_sqrt(x)    cumo_float2bfloat(sqrtf(cumo_bfloat2float(x)))
#define m_cbrt(x)    cumo_float2bfloat(cbrtf(cumo_bfloat2float(x)))
#define m_log(x)     cumo_float2bfloat(logf(cumo_bfloat2float(x)))
#define m_log2(x)    cumo_float2bfloat(log2f(cumo_bfloat2float(x)))
#define m_log10(x)   cumo_float2bfloat(log10f(cumo_bfloat2float(x)))
#define m_exp(x)     cumo_float2bfloat(expf(cumo_bfloat2float(x)))
#define m_exp2(x)    cumo_float2bfloat(exp2f(cumo_bfloat2float(x)))
#ifdef HAVE_EXP10
#define m_exp10(x)   cumo_float2bfloat(exp10f(cumo_bfloat2float(x)))
#else
#define m_exp10(x)   cumo_float2bfloat(powf(10.0f,cumo_bfloat2float(x)))
#endif
#define m_expm1(x)   cumo_float2bfloat(expm1f(cumo_bfloat2float(x)))
#define m_log1p(x)   cumo_float2bfloat(log1pf(cumo_bfloat2float(x)))

#define m_sin(x)     cumo_float2bfloat(sinf(cumo_bfloat2float(x)))
#define m_cos(x)     cumo_float2bfloat(cosf(cumo_bfloat2float(x)))
#define m_tan(x)     cumo_float2bfloat(tanf(cumo_bfloat2float(x)))
#define m_asin(x)    cumo_float2bfloat(asinf(cumo_bfloat2float(x)))
#define m_acos(x)    cumo_float2bfloat(acosf(cumo_bfloat2float(x)))
#define m_atan(x)    cumo_float2bfloat(atanf(cumo_bfloat2float(x)))
#define m_sinh(x)    cumo_float2bfloat(sinhf(cumo_bfloat2float(x)))
#define m_cosh(x)    cumo_float2bfloat(coshf(cumo_bfloat2float(x)))
#define m_tanh(x)    cumo_float2bfloat(tanhf(cumo_bfloat2float(x)))
#define m_asinh(x)   cumo_float2bfloat(asinhf(cumo_bfloat2float(x)))
#define m_acosh(x)   cumo_float2bfloat(acoshf(cumo_bfloat2float(x)))
#define m_atanh(x)   cumo_float2bfloat(atanhf(cumo_bfloat2float(x)))
#define m_atan2(x,y) cumo_float2bfloat(atan2f(cumo_bfloat2float(x),cumo_bfloat2float(y)))
#define m_hypot(x,y) cumo_float2bfloat(hypotf(cumo_bfloat2float(x),cumo_bfloat2float(y)))
#define m_sinc(x)    cumo_float2bfloat(cumo_bfloat_sinc(cumo_bfloat2float(x)))

#define m_erf(x)     cumo_float2bfloat(erff(cumo_bfloat2float(x)))
#define m_erfc(x)    cumo_float2bfloat(erfcf(cumo_bfloat2float(x)))
#define m_gelu(x)      cumo_float2bfloat(cumo_bfloat_gelu(cumo_bfloat2float(x)))
#define m_gelu_tanh(x) cumo_float2bfloat(cumo_bfloat_gelu_tanh(cumo_bfloat2float(x)))
#define m_silu(x)      cumo_float2bfloat(cumo_bfloat_silu(cumo_bfloat2float(x)))
#define m_ldexp(x,y) cumo_float2bfloat(cumo_bfloat_ldexp(cumo_bfloat2float(x),cumo_bfloat2float(y)))
#define m_frexp(x,exp) cumo_float2bfloat(frexpf(cumo_bfloat2float(x),exp))


static inline float cumo_bfloat_pow_positive_int(float x, unsigned int p)
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

static inline float cumo_bfloat_pow_int(float x, int p)
{
    if (p < 0) return 1.0f / cumo_bfloat_pow_positive_int(x, -(unsigned int)p);
    return cumo_bfloat_pow_positive_int(x, (unsigned int)p);
}

static inline float cumo_bfloat_ldexp(float x, float y)
{
    if (!(y == y)) return y;
    if (y > 64.0f) y = 64.0f;
    if (y < -64.0f) y = -64.0f;
    return ldexpf(x, (int)y);
}

static inline float cumo_bfloat_sinc(float x)
{
    return (x==0) ? 1.0f : (sinf(x)/x);
}

static inline float cumo_bfloat_sign(float x)
{
    return (x==0) ? 0.0f : ((x>0) ? 1.0f : ((x<0) ? -1.0f : x));
}

static inline cumo_bfloat cumo_na_ruby_num_to_bfloat(VALUE x)
{
    return cumo_double2bfloat(NIL_P(x) ? nan("") : NUM2DBL(x));
}

static inline cumo_bfloat f_seq(cumo_bfloat x, cumo_bfloat y, double c)
{
    return cumo_float2bfloat(cumo_bfloat2float(x) + cumo_bfloat2float(y) * (float)c);
}

#include "real_accum.h"

static inline float cumo_bfloat_gelu(float x)
{
    return 0.5f * x * (1.0f + erff(x * (float)CUMO_M_SQRT1_2));
}

// The approximation GPT-2 and the transformers after it were trained with.
static inline float cumo_bfloat_gelu_tanh(float x)
{
    return 0.5f * x * (1.0f + tanhf((float)CUMO_M_SQRT_2_OVER_PI * (x + (float)CUMO_GELU_TANH_CUBIC * x * x * x)));
}

// x * sigmoid(x) as one division, cumo having no sigmoid to call.
static inline float cumo_bfloat_silu(float x)
{
    return x / (1.0f + expf(-x));
}
