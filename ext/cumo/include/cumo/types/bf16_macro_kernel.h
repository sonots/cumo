#ifndef CUMO_BFLOAT_MACRO_KERNEL_H
#define CUMO_BFLOAT_MACRO_KERNEL_H

#include "float_def_kernel.h"
#include "half_def_kernel.h"
#include "bf16_def_kernel.h"

#define CUMO_BFLOAT_EPSILON 7.8125e-03f

#define CUMO_F16_TO_F(x) cumo_bfloat2float(x)
#define CUMO_F_TO_F16(x) cumo_float2bfloat(x)
#define CUMO_D_TO_F16(x) cumo_double2bfloat(x)

#define m_from_bfloat(x) (x)
#define m_from_half(x)   cumo_float2bfloat(cumo_half2float(x))

#include "f16_macro_kernel.h"

#endif // CUMO_BFLOAT_MACRO_KERNEL_H
