#ifndef CUMO_UINT16_KERNEL_H
#define CUMO_UINT16_KERNEL_H

typedef u_int16_t dtype;
typedef u_int16_t rtype;

#ifndef UINT16_MAX
#define UINT16_MAX (65535)
#endif

#define DATA_MIN UINT16_MIN
#define DATA_MAX UINT16_MAX

#include "uint_macro_kernel.h"

#endif // CUMO_UINT16_KERNEL_H
