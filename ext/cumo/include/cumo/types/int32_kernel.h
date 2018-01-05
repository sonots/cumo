#ifndef CUMO_INT32_KERNEL_H
#define CUMO_INT32_KERNEL_H

typedef int32_t dtype;
typedef int32_t rtype;

#ifndef INT32_MIN
#define INT32_MIN (-2147483647-1)
#endif
#ifndef INT32_MAX
#define INT32_MAX (2147483647)
#endif

#define DATA_MIN INT32_MIN
#define DATA_MAX INT32_MAX

#include "int_macro_kernel.h"

#endif // CUMO_INT32_KERNEL_H
