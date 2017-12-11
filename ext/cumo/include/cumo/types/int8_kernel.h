typedef int8_t dtype;
typedef int8_t rtype;

#ifndef INT8_MIN
#define INT8_MIN (-127-1)
#endif
#ifndef INT8_MAX
#define INT8_MAX (127)
#endif

#define DATA_MIN INT8_MIN
#define DATA_MAX INT8_MAX

#include "int_macro_kernel.h"

