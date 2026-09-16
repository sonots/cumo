#ifndef CUMO_BFLOAT_KERNEL_H
#define CUMO_BFLOAT_KERNEL_H

#include "bf16_def_kernel.h"

typedef cumo_bfloat dtype;
typedef cumo_bfloat rtype;

#include "bf16_macro_kernel.h"

#define m_nearly_eq(x,y) (fabsf(cumo_bfloat2float(x)-cumo_bfloat2float(y))<=(fabsf(cumo_bfloat2float(x))+fabsf(cumo_bfloat2float(y)))*CUMO_BFLOAT_EPSILON*2)

#define DATA_MIN cumo_float2bfloat(-3.38953138925153547590e+38f)
#define DATA_MAX cumo_float2bfloat(3.38953138925153547590e+38f)

#endif // CUMO_BFLOAT_KERNEL_H
