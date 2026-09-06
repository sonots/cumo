#include "half_def.h"

typedef cumo_half dtype;
typedef cumo_half rtype;
#define cT  cumo_cHFloat
#define cRT cumo_cHFloat

#include "half_macro.h"

#define m_extract(x) rb_float_new(cumo_half2float(*(cumo_half*)x))
#define m_nearly_eq(x,y) (fabsf(cumo_half2float(x)-cumo_half2float(y))<=(fabsf(cumo_half2float(x))+fabsf(cumo_half2float(y)))*CUMO_HALF_EPSILON*2)

#define M_EPSILON rb_float_new(9.765625e-04)
#define M_MIN     rb_float_new(6.103515625e-05)
#define M_MAX     rb_float_new(65504.0)

#define DATA_MIN cumo_float2half(-65504.0f)
#define DATA_MAX cumo_float2half(65504.0f)
