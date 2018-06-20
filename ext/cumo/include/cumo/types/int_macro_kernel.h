#ifndef CUMO_INT_MACRO_KERNEL_H
#define CUMO_INT_MACRO_KERNEL_H

#include "xint_macro_kernel.h"

#define m_sign(x)    (((x)==0) ? 0 : (((x)>0) ? 1 : -1))

__host__ __device__ static inline dtype m_abs(dtype x) {
    // TODO(sonots): How to handle in CUDA kernel?
    // if (x==DATA_MIN) {
    //     rb_raise(na_eValueError, "cannot convert the minimum integer");
    // }
    return (x<0)?-x:x;
}

__host__ __device__ static inline dtype int_reciprocal(dtype x) {
    switch (x) {
    case 1:
        return 1;
    case -1:
        return -1;
    case 0:
        return 0; // as CUDA kernel 1/0 results in 0.
        //rb_raise(rb_eZeroDivError, "divided by 0");
    default:
        return 0;
    }
}

__device__ static dtype pow_int(dtype x, int p)
{
    dtype r = m_one;
    switch(p) {
    case 0: return 1;
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    }
    if (p<0) return 0;
    while (p) {
        if (p&1) r *= x;
        x *= x;
        p >>= 1;
    }
    return r;
}

#endif // CUMO_INT_MACRO_KERNEL_H
