#ifndef CUMO_HALF_MACRO_KERNEL_H
#define CUMO_HALF_MACRO_KERNEL_H

#include "float_def_kernel.h"
#include "bf16_def_kernel.h"
#include "half_def_kernel.h"

#define CUMO_HALF_EPSILON 9.765625e-04f

#define CUMO_F16_TO_F(x) cumo_half2float(x)
#define CUMO_F_TO_F16(x) cumo_float2half(x)
#define CUMO_D_TO_F16(x) cumo_double2half(x)

#define m_from_half(x)   (x)
#define m_from_bfloat(x) cumo_float2half(cumo_bfloat2float(x))

#include "f16_macro_kernel.h"

#endif // CUMO_HALF_MACRO_KERNEL_H
