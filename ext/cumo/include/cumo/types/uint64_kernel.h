#ifndef CUMO_UINT64_KERNEL_H
#define CUMO_UINT64_KERNEL_H

typedef u_int64_t dtype;
typedef u_int64_t rtype;

#ifndef UINT64_MIN
#define UINT64_MIN (0)
#endif

#ifndef UINT64_MAX
#define UINT64_MAX (18446744073709551615ul)
#endif

#define DATA_MIN UINT64_MIN
#define DATA_MAX UINT64_MAX

#include "uint_macro_kernel.h"

#endif // CUMO_UINT64_KERNEL_H
