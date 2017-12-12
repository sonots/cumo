typedef int64_t dtype;
typedef int64_t rtype;

#ifndef INT64_MIN
#define INT64_MIN (-9223372036854775807l-1)
#endif
#ifndef INT64_MAX
#define INT64_MAX (9223372036854775807l)
#endif

#define DATA_MIN INT64_MIN
#define DATA_MAX INT64_MAX

#include "int_macro_kernel.h"

