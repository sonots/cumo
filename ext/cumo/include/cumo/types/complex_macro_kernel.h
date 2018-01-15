#ifndef CUMO_COMPLEX_MACRO_KERNEL_H
#define CUMO_COMPLEX_MACRO_KERNEL_H

#include "float_def_kernel.h"

extern double round(double);
extern double log2(double);
extern double exp2(double);
extern double exp10(double);

#define r_abs(x)   fabs(x)
#define r_sqrt(x)  sqrt(x)
#define r_exp(x)   exp(x)
#define r_log(x)   log(x)
#define r_sin(x)   sin(x)
#define r_cos(x)   cos(x)
#define r_sinh(x)  sinh(x)
#define r_cosh(x)  cosh(x)
#define r_tanh(x)  tanh(x)
#define r_atan2(y,x)  atan2(y,x)
#define r_hypot(x,y)  hypot(x,y)

#include "complex_kernel.h"

__host__ __device__ static inline dtype c_from_scomplex(scomplex x) {
    dtype z;
    REAL(z) = REAL(x);
    IMAG(z) = IMAG(x);
    return z;
}

__host__ __device__ static inline dtype c_from_dcomplex(dcomplex x) {
    dtype z;
    REAL(z) = REAL(x);
    IMAG(z) = IMAG(x);
    return z;
}

/* --------------------------- */

#define m_zero c_zero()
#define m_one  c_one()

//#define m_num_to_data(x) NUM2COMP(x)
//#define m_data_to_num(x) COMP2NUM(x)

#define m_from_double(x) c_new(x,0)
#define m_from_real(x)   c_new(x,0)
#define m_from_scomplex(x) c_from_scomplex(x)
#define m_from_dcomplex(x) c_from_dcomplex(x)

//#define m_extract(x) COMP2NUM(*(dtype*)x)

#define m_real(x)  REAL(x)
#define m_imag(x)  IMAG(x)
#define m_set_real(x,y)  c_set_real(x,y)
#define m_set_imag(x,y)  c_set_imag(x,y)

#define m_add(x,y) c_add(x,y)
#define m_sub(x,y) c_sub(x,y)
#define m_mul(x,y) c_mul(x,y)
#define m_div(x,y) c_div(x,y)
#define m_mod(x,y) c_mod(x,y)
#define m_pow(x,y) c_pow(x,y)
#define m_pow_int(x,y) c_pow_int(x,y)

#define m_abs(x)   c_abs(x)
#define m_minus(x) c_minus(x)
#define m_reciprocal(x) c_reciprocal(x)
#define m_square(x) c_square(x)
#define m_floor(x) c_new(floor(REAL(x)),floor(IMAG(x)))
#define m_round(x) c_new(round(REAL(x)),round(IMAG(x)))
#define m_ceil(x)  c_new(ceil(REAL(x)),ceil(IMAG(x)))
#define m_trunc(x) c_new(trunc(REAL(x)),trunc(IMAG(x)))
#define m_rint(x)  c_new(rint(REAL(x)),rint(IMAG(x)))
#define m_sign(x)  c_new( \
 ((REAL(x)==0) ? 0.0:((REAL(x)>0) ? 1.0:((REAL(x)<0) ? -1.0:REAL(x)))), \
 ((IMAG(x)==0) ? 0.0:((IMAG(x)>0) ? 1.0:((IMAG(x)<0) ? -1.0:IMAG(x)))))
#define m_copysign(x,y) c_new(copysign(REAL(x),REAL(y)),copysign(IMAG(x),IMAG(y)))

#define m_im(x)    c_im(x)
#define m_conj(x)  c_new(REAL(x),-IMAG(x))
#define m_arg(x)   atan2(IMAG(x),REAL(x))

#define m_eq(x,y) c_eq(x,y)
#define m_ne(x,y) c_ne(x,y)
#define m_nearly_eq(x,y) c_nearly_eq(x,y)

#define m_isnan(x)    c_isnan(x)
#define m_isinf(x)    c_isinf(x)
#define m_isposinf(x) c_isposinf(x)
#define m_isneginf(x) c_isneginf(x)
#define m_isfinite(x) c_isfinite(x)

#define m_sprintf(s,x) sprintf(s,"%g%+gi",REAL(x),IMAG(x))

#define m_sqrt(x)    c_sqrt(x)
#define m_cbrt(x)    c_cbrt(x)
#define m_log(x)     c_log(x)
#define m_log2(x)    c_log2(x)
#define m_log10(x)   c_log10(x)
#define m_exp(x)     c_exp(x)
#define m_exp2(x)    c_exp2(x)
#define m_exp10(x)   c_exp10(x)
#define m_sin(x)     c_sin(x)
#define m_cos(x)     c_cos(x)
#define m_tan(x)     c_tan(x)
#define m_asin(x)    c_asin(x)
#define m_acos(x)    c_acos(x)
#define m_atan(x)    c_atan(x)
#define m_sinh(x)    c_sinh(x)
#define m_cosh(x)    c_cosh(x)
#define m_tanh(x)    c_tanh(x)
#define m_asinh(x)   c_asinh(x)
#define m_acosh(x)   c_acosh(x)
#define m_atanh(x)   c_atanh(x)
#define m_hypot(x,y) c_hypot(x,y)
#define m_sinc(x)    c_div(c_sin(x),x)

#define m_sum_init 0
#define m_mulsum_init 0

#define not_nan(x) (REAL(x)==REAL(x) && IMAG(x)==IMAG(x))

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

#endif // CUMO_COMPLEX_MACRO_KERNEL_H
