#ifndef CUMO_UINT32_KERNEL_H
#define CUMO_UINT32_KERNEL_H

typedef u_int32_t dtype;
typedef u_int32_t rtype;

#ifndef UINT32_MIN
#define UINT32_MIN (0)
#endif

#ifndef UINT32_MAX
#define UINT32_MAX (4294967295u)
#endif

#define DATA_MIN UINT32_MIN
#define DATA_MAX UINT32_MAX

#include "uint_macro_kernel.h"

#endif // CUMO_UINT32_KERNEL_H
