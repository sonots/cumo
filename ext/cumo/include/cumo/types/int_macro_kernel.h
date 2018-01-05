#ifndef CUMO_INT_MACRO_KERNEL_H
#define CUMO_INT_MACRO_KERNEL_H

#include "xint_macro_kernel.h"

#define m_sign(x)    (((x)==0) ? 0 : (((x)>0) ? 1 : -1))

__device__ static inline dtype int_reciprocal(dtype x) {
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

#endif // CUMO_INT_MACRO_KERNEL_H
