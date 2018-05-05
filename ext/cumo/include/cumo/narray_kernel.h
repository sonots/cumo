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

#define SZF PRI_SIZE_PREFIX // defined in ruby.h

#if   SIZEOF_LONG==8
# define NUM2INT64(x) NUM2LONG(x)
# define INT642NUM(x) LONG2NUM(x)
# define NUM2UINT64(x) NUM2ULONG(x)
# define UINT642NUM(x) ULONG2NUM(x)
# ifndef PRId64
#  define PRId64 "ld"
# endif
# ifndef PRIu64
#  define PRIu64 "lu"
# endif
#elif SIZEOF_LONG_LONG==8
# define NUM2INT64(x) NUM2LL(x)
# define INT642NUM(x) LL2NUM(x)
# define NUM2UINT64(x) NUM2ULL(x)
# define UINT642NUM(x) ULL2NUM(x)
# ifndef PRId64
#  define PRId64 "lld"
# endif
# ifndef PRIu64
#  define PRIu64 "llu"
# endif
#endif

#if   SIZEOF_LONG==4
# define NUM2INT32(x) NUM2LONG(x)
# define INT322NUM(x) LONG2NUM(x)
# define NUM2UINT32(x) NUM2ULONG(x)
# define UINT322NUM(x) ULONG2NUM(x)
# ifndef PRId32
#  define PRId32 "ld"
# endif
# ifndef PRIu32
#  define PRIu32 "lu"
# endif
#elif SIZEOF_INT==4
# define NUM2INT32(x) NUM2INT(x)
# define INT322NUM(x) INT2NUM(x)
# define NUM2UINT32(x) NUM2UINT(x)
# define UINT322NUM(x) UINT2NUM(x)
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

typedef struct { float dat[2]; }  scomplex;
typedef struct { double dat[2]; } dcomplex;
typedef int fortran_integer;

#define REAL(x) ((x).dat[0])
#define IMAG(x) ((x).dat[1])

extern int na_debug_flag;

#define NARRAY_DATA_T     0x1
#define NARRAY_VIEW_T     0x2
#define NARRAY_FILEMAP_T  0x3

//#define NA_MAX_DIMENSION (int)(sizeof(VALUE)*8-2)
#define NA_MAX_DIMENSION 8
#define NA_MAX_ELMSZ     65535

typedef unsigned int BIT_DIGIT;
#define BYTE_BIT_DIGIT sizeof(BIT_DIGIT)
#define NB     (sizeof(BIT_DIGIT)*8)
#define BALL   (~(BIT_DIGIT)0)
#define SLB(n) (((n)==NB)?~(BIT_DIGIT)0:(~(~(BIT_DIGIT)0<<(n))))

#define ELEMENT_BIT_SIZE  "ELEMENT_BIT_SIZE"
#define ELEMENT_BYTE_SIZE "ELEMENT_BYTE_SIZE"
#define CONTIGUOUS_STRIDE "CONTIGUOUS_STRIDE"

#include "cumo/indexer.h"
#include "cumo/intern_kernel.h"

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_NARRAY_KERNEL_H */
