#ifndef CUMO_HFLOAT_KERNEL_H
#define CUMO_HFLOAT_KERNEL_H

#include "half_def_kernel.h"

typedef cumo_half dtype;
typedef cumo_half rtype;

#include "half_macro_kernel.h"

#define m_nearly_eq(x,y) (fabsf(cumo_half2float(x)-cumo_half2float(y))<=(fabsf(cumo_half2float(x))+fabsf(cumo_half2float(y)))*CUMO_HALF_EPSILON*2)

#define DATA_MIN cumo_float2half(-65504.0f)
#define DATA_MAX cumo_float2half(65504.0f)

#endif // CUMO_HFLOAT_KERNEL_H
