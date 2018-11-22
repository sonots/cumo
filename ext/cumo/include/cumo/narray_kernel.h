#ifndef CUMO_NARRAY_KERNEL_H
#define CUMO_NARRAY_KERNEL_H

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

#include <math.h>
//#include "cumo/compat.h"
#include "cumo/template_kernel.h"
//#include "cumo/extconf.h"

#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#endif

#ifdef HAVE_STDINT_H
# include <stdint.h>
#endif

#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif

#ifndef HAVE_U_INT8_T
# ifdef HAVE_UINT8_T
    typedef uint8_t u_int8_t;
# endif
#endif

#ifndef HAVE_U_INT16_T
# ifdef HAVE_UINT16_T
    typedef uint16_t u_int16_t;
# endif
#endif

#ifndef HAVE_U_INT32_T
# ifdef HAVE_UINT32_T
    typedef uint32_t u_int32_t;
# endif
#endif

#ifndef HAVE_U_INT64_T
# ifdef HAVE_UINT64_T
    typedef uint64_t u_int64_t;
# endif
#endif

#ifndef SZF
#define SZF PRI_SIZE_PREFIX // defined in ruby.h
#endif

#if   SIZEOF_LONG==8
# ifndef NUM2INT64
#  define NUM2INT64(x) NUM2LONG(x)
# endif
# ifndef INT642NUM
#  define INT642NUM(x) LONG2NUM(x)
# endif
# ifndef NUM2UINT64
#  define NUM2UINT64(x) NUM2ULONG(x)
# endif
# ifndef UINT642NUM
#  define UINT642NUM(x) ULONG2NUM(x)
# endif
# ifndef PRId64
#  define PRId64 "ld"
# endif
# ifndef PRIu64
#  define PRIu64 "lu"
# endif
#elif SIZEOF_LONG_LONG==8
# ifndef NUM2INT64
#  define NUM2INT64(x) NUM2LL(x)
# endif
# ifndef INT642NUM
#  define INT642NUM(x) LL2NUM(x)
# endif
# ifndef NUM2UINT64
#  define NUM2UINT64(x) NUM2ULL(x)
# endif
# ifndef UINT642NUM
#  define UINT642NUM(x) ULL2NUM(x)
# endif
# ifndef PRId64
#  define PRId64 "lld"
# endif
# ifndef PRIu64
#  define PRIu64 "llu"
# endif
#endif

#if   SIZEOF_LONG==4
# ifndef NUM2INT32
#  define NUM2INT32(x) NUM2LONG(x)
# endif
# ifndef INT322NUM
#  define INT322NUM(x) LONG2NUM(x)
# endif
# ifndef NUM2UINT32
#  define NUM2UINT32(x) NUM2ULONG(x)
# endif
# ifndef UINT322NUM
#  define UINT322NUM(x) ULONG2NUM(x)
# endif
# ifndef PRId32
#  define PRId32 "ld"
# endif
# ifndef PRIu32
#  define PRIu32 "lu"
# endif
#elif SIZEOF_INT==4
# ifndef NUM2INT32
#  define NUM2INT32(x) NUM2INT(x)
# endif
# ifndef INT322NUM
#  define INT322NUM(x) INT2NUM(x)
# endif
# ifndef NUM2UINT32
#  define NUM2UINT32(x) NUM2UINT(x)
# endif
# ifndef UINT322NUM
#  define UINT322NUM(x) UINT2NUM(x)
# endif
# ifndef PRId32
#  define PRId32 "d"
# endif
# ifndef PRIu32
#  define PRIu32 "u"
# endif
#endif

#ifndef HAVE_TYPE_BOOL
  typedef int bool;
#endif
#ifndef FALSE                   /* in case these macros already exist */
# define FALSE   0              /* values of bool */
#endif
#ifndef TRUE
# define TRUE    1
#endif

typedef struct { float dat[2]; }  cumo_scomplex;
typedef struct { double dat[2]; } cumo_dcomplex;

#define CUMO_REAL(x) ((x).dat[0])
#define CUMO_IMAG(x) ((x).dat[1])

extern int cumo_na_debug_flag;

#define CUMO_NARRAY_DATA_T     0x1
#define CUMO_NARRAY_VIEW_T     0x2
#define CUMO_NARRAY_FILEMAP_T  0x3

//#define CUMO_NA_MAX_DIMENSION (int)(sizeof(VALUE)*8-2)
#define CUMO_NA_MAX_DIMENSION 12
#define CUMO_NA_MAX_ELMSZ     65535

typedef unsigned int CUMO_BIT_DIGIT;
#define CUMO_BYTE_BIT_DIGIT sizeof(CUMO_BIT_DIGIT)
#define CUMO_NB     (sizeof(CUMO_BIT_DIGIT)*8)
#define CUMO_BALL   (~(CUMO_BIT_DIGIT)0)
#define CUMO_SLB(n) (((n)==CUMO_NB)?~(CUMO_BIT_DIGIT)0:(~(~(CUMO_BIT_DIGIT)0<<(n))))

typedef union {
    ssize_t stride;
    size_t *index;
} cumo_stridx_t;

#define CUMO_SDX_IS_STRIDE(x) ((x).stride&0x1)
#define CUMO_SDX_IS_INDEX(x)  (!CUMO_SDX_IS_STRIDE(x))
#define CUMO_SDX_GET_STRIDE(x) ((x).stride>>1)
#define CUMO_SDX_GET_INDEX(x)  ((x).index)

#include "cumo/indexer.h"
#include "cumo/intern_kernel.h"

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_NARRAY_KERNEL_H */
