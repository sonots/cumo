#ifndef CUMO_BF16_DEF_KERNEL_H
#define CUMO_BF16_DEF_KERNEL_H

#include <cuda_bf16.h>

typedef __nv_bfloat16 cumo_bfloat;

// The arithmetic on __nv_bfloat16 starts at sm_80, but the conversions have no
// such floor, so every operation is computed in float. For + - * / and sqrt
// that answers what the native instruction would: a float carries three times
// the mantissa bits of a bfloat16, so the second rounding cannot land on a tie
// the first one moved.
__host__ __device__ static inline float cumo_bfloat2float(cumo_bfloat b)
{
    return __bfloat162float(b);
}

__host__ __device__ static inline cumo_bfloat cumo_float2bfloat(float f)
{
    return __float2bfloat16(f);
}

__host__ __device__ static inline cumo_bfloat cumo_double2bfloat(double d)
{
    return __double2bfloat16(d);
}

#endif // CUMO_BF16_DEF_KERNEL_H
