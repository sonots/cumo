typedef scomplex dtype;
typedef float rtype;

#include "complex_macro_kernel.h"

static inline bool c_nearly_eq(dtype x, dtype y) {
    return c_abs(c_sub(x,y)) <= (c_abs(x)+c_abs(y))*FLT_EPSILON*2;
}