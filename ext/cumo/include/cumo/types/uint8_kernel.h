#ifndef CUMO_UINT8_KERNEL_H
#define CUMO_UINT8_KERNEL_H

typedef u_int8_t dtype;
typedef u_int8_t rtype;

#ifndef UINT8_MAX
#define UINT8_MAX (255)
#endif

#define DATA_MIN UINT8_MIN
#define DATA_MAX UINT8_MAX

#include "uint_macro_kernel.h"

#endif // CUMO_UINT8_KERNEL_H
