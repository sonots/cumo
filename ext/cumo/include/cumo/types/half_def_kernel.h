#ifndef CUMO_HALF_DEF_KERNEL_H
#define CUMO_HALF_DEF_KERNEL_H

#include <cuda_fp16.h>

typedef __half cumo_half;

// The arithmetic instructions on __half start at sm_53, but the conversions
// have no such floor, so every operation is computed in float. For + - * / and
// sqrt that answers what the native half instruction would: a float carries
// more than twice the bits of a half, so the second rounding cannot land on a
// tie the first one moved.
__host__ __device__ static inline float cumo_half2float(cumo_half h)
{
    return __half2float(h);
}

__host__ __device__ static inline cumo_half cumo_float2half(float f)
{
    return __float2half(f);
}

#endif // CUMO_HALF_DEF_KERNEL_H
