#include "bf16_def.h"

typedef cumo_bfloat dtype;
typedef cumo_bfloat rtype;
#define cT  cumo_cBFloat
#define cRT cumo_cBFloat
#define mTM cumo_mBFloatMath

#include "bf16_macro.h"
#include "cublas_v2.h"
#include "cumo/cuda/cublas.h"
#include "cumo/cuda/cudnn.h"

#define m_extract(x) rb_float_new(cumo_bfloat2float(*(cumo_bfloat*)x))
#define m_nearly_eq(x,y) (fabsf(cumo_bfloat2float(x)-cumo_bfloat2float(y))<=(fabsf(cumo_bfloat2float(x))+fabsf(cumo_bfloat2float(y)))*CUMO_BFLOAT_EPSILON*2)

#define M_EPSILON rb_float_new(7.8125e-03)
#define M_MIN     rb_float_new(1.17549435082228750797e-38)
#define M_MAX     rb_float_new(3.38953138925153547590e+38)

#define DATA_MIN cumo_float2bfloat(-3.38953138925153547590e+38f)
#define DATA_MAX cumo_float2bfloat(3.38953138925153547590e+38f)
