#ifndef CUMO_INT_MACRO_KERNEL_H
#define CUMO_INT_MACRO_KERNEL_H

#include "xint_macro_kernel.h"

#define m_sign(x)    (((x)==0) ? 0 : (((x)>0) ? 1 : -1))

// Ruby floors the quotient and gives the remainder the divisor's sign; C
// truncates toward zero, so the two part company once the signs differ. This
// is for the signed types only: with both operands non-negative the two agree,
// and comparing an unsigned value with zero warns.
__host__ __device__ static inline dtype m_floored_mod(dtype x, dtype y) {
    dtype r = x % y;
    if (r != 0 && ((r < 0) != (y < 0))) {
        r += y;
    }
    return r;
}

#undef m_mod
#undef m_divmod
#define m_mod(x,y) m_floored_mod(x,y)
// The quotient is corrected rather than worked out from the remainder: x-b
// overflows when x is near the low end of the type, and it costs a second
// division on top.
#define m_divmod(x,y,a,b) {a=(x)/(y); b=(x)%(y); if (b != 0 && ((b < 0) != ((y) < 0))) {b+=(y); a-=1;}}

__host__ __device__ static inline dtype m_abs(dtype x) {
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
