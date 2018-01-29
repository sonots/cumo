#ifndef CUMO_INT16_KERNEL_H
#define CUMO_INT16_KERNEL_H

typedef int16_t dtype;
typedef int16_t rtype;
#define cT  cumo_cInt16
#define cRT cT

#define m_sprintf(s,x)   sprintf(s,"%d",(int)(x))

#ifndef INT16_MIN
#define INT16_MIN (-32767-1)
#endif
#ifndef INT16_MAX
#define INT16_MAX (32767)
#endif

#define DATA_MIN INT16_MIN
#define DATA_MAX INT16_MAX

#include "int_macro_kernel.h"

#endif // CUMO_INT16_KERNEL_H
