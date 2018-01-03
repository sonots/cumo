typedef dcomplex dtype;
typedef double rtype;

#include "complex_macro_kernel.h"

__device__ static inline bool c_nearly_eq(dtype x, dtype y) {
    return c_abs(c_sub(x,y)) <= (c_abs(x)+c_abs(y))*DBL_EPSILON*2;
}
