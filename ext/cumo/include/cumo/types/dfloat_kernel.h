#ifndef CUMO_DFLOAT_KERNEL_H
#define CUMO_DFLOAT_KERNEL_H

typedef double dtype;
typedef double rtype;

#include "float_macro_kernel.h"

#define m_nearly_eq(x,y) (fabs(x-y)<=(fabs(x)+fabs(y))*DBL_EPSILON*2)

#endif // CUMO_DFLOAT_KERNEL_H
