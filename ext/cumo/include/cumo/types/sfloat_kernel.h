typedef float dtype;
typedef float rtype;

#include "float_macro_kernel.h"

#define m_nearly_eq(x,y) (fabs(x-y)<=(fabs(x)+fabs(y))*FLT_EPSILON*2)
