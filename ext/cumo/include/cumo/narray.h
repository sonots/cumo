#ifndef CUMO_NARRAY_H
#define CUMO_NARRAY_H

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

#include <math.h>
#include "cumo/compat.h"
#include "cumo/template.h"
#include "cumo/extconf.h"

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

#ifndef IS_INTEGER_CLASS
#ifdef RUBY_INTEGER_UNIFICATION
#define IS_INTEGER_CLASS(c) ((c)==rb_cInteger)
#else
#define IS_INTEGER_CLASS(c) ((c)==rb_cFixnum||(c)==rb_cBignum)
#endif
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

#if SIZEOF_VALUE > 4
# undef INT322NUM
# undef UINT322NUM
# define INT322NUM(x) INT2FIX(x)
# define UINT322NUM(x) INT2FIX(x)
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

#define mCumo rb_mCumo
extern VALUE rb_mCumo;
#define cNArray cumo_cNArray
extern VALUE cumo_cNArray;
extern VALUE cumo_na_eCastError;
extern VALUE cumo_na_eShapeError;
extern VALUE cumo_na_eOperationError;
extern VALUE cumo_na_eDimensionError;
extern VALUE cumo_na_eValueError;
extern const rb_data_type_t cumo_na_data_type;

//EXTERN const int cumo_na_sizeof[CUMO_NA_NTYPES+1];

//#define cumo_na_upcast(x,y) cumo_na_upcast(x,y)

/* global variables within this module */
extern VALUE cumo_cBit;
extern VALUE cumo_cDFloat;
extern VALUE cumo_cSFloat;
extern VALUE cumo_cDComplex;
extern VALUE cumo_cSComplex;
extern VALUE cumo_cInt64;
extern VALUE cumo_cInt32;
extern VALUE cumo_cInt16;
extern VALUE cumo_cInt8;
extern VALUE cumo_cUInt64;
extern VALUE cumo_cUInt32;
extern VALUE cumo_cUInt16;
extern VALUE cumo_cUInt8;
extern VALUE cumo_cRObject;
#ifndef HAVE_RB_CCOMPLEX
extern VALUE rb_cComplex;
#endif
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
extern VALUE rb_cArithSeq;
#endif

extern VALUE cumo_sym_reduce;
extern VALUE cumo_sym_option;
extern VALUE cumo_sym_loop_opt;
extern VALUE cumo_sym_init;

#define CUMO_NARRAY_DATA_T     0x1
#define CUMO_NARRAY_VIEW_T     0x2
#define CUMO_NARRAY_FILEMAP_T  0x3

typedef struct {
    unsigned char ndim;     // # of dimensions
    unsigned char type;
    unsigned char flag[2];  // flags
    unsigned short elmsz;    // element size
    size_t   size;          // # of total elements
    size_t  *shape;         // # of elements for each dimension
    VALUE    reduce;
} cumo_narray_t;


typedef struct {
    cumo_narray_t base;
    char    *ptr;
} cumo_narray_data_t;


typedef union {
    ssize_t stride;
    size_t *index;
} cumo_stridx_t;

typedef struct {
    cumo_narray_t base;
    VALUE    data;       // data object
    size_t   offset;     // offset of start point from data pointer
                         // :in units of elm.unit_bits
                         // address_unit  pointer_unit access_unit data_unit
                         // elm.step_unit = elm.bit_size / elm.access_unit
                         // elm.step_unit = elm.size_bits / elm.unit_bits
    cumo_stridx_t *stridx;    // stride or indices of data pointer for each dimension
} cumo_narray_view_t;


// filemap is unimplemented
typedef struct {
    cumo_narray_t base;
    char    *ptr;
#ifdef WIN32
    HANDLE hFile;
    HANDLE hMap;
#else // POSIX mmap
    int prot;
    int flag;
#endif
} cumo_narray_filemap_t;


// this will be revised in future.
typedef struct {
    unsigned int element_bits;
    unsigned int element_bytes;
    unsigned int element_stride;
} cumo_narray_type_info_t;

// from ruby/enumerator.c
typedef struct {
    VALUE obj;
    ID    meth;
    VALUE args;
    // use only above in this source
    VALUE fib;
    VALUE dst;
    VALUE lookahead;
    VALUE feedvalue;
    VALUE stop_exc;
    VALUE size;
    // incompatible below depending on ruby version
    //VALUE procs;                      // ruby 2.4
    //rb_enumerator_size_func *size_fn; // ruby 2.1-2.4
    //VALUE (*size_fn)(ANYARGS);        // ruby 2.0
} cumo_enumerator_t;

static inline cumo_narray_t *
cumo_na_get_narray_t(VALUE obj)
{
    cumo_narray_t *na;

    Check_TypedStruct(obj,&cumo_na_data_type);
    na = (cumo_narray_t*)DATA_PTR(obj);
    return na;
}

static inline cumo_narray_t *
_cumo_na_get_narray_t(VALUE obj, unsigned char cumo_na_type)
{
    cumo_narray_t *na;

    Check_TypedStruct(obj,&cumo_na_data_type);
    na = (cumo_narray_t*)DATA_PTR(obj);
    if (na->type != cumo_na_type) {
        rb_bug("unknown type 0x%x (0x%x given)", cumo_na_type, na->type);
    }
    return na;
}

#define cumo_na_get_narray_data_t(obj) (cumo_narray_data_t*)_cumo_na_get_narray_t(obj,CUMO_NARRAY_DATA_T)
#define cumo_na_get_narray_view_t(obj) (cumo_narray_view_t*)_cumo_na_get_narray_t(obj,CUMO_NARRAY_VIEW_T)
#define cumo_na_get_narray_filemap_t(obj) (cumo_narray_filemap_t*)_cumo_na_get_narray_t(obj,CUMO_NARRAY_FILEMAP_T)

#define CumoGetNArray(obj,var)      TypedData_Get_Struct(obj, cumo_narray_t, &cumo_na_data_type, var)
#define CumoGetNArrayView(obj,var)  TypedData_Get_Struct(obj, cumo_narray_view_t, &cumo_na_data_type, var)
#define CumoGetNArrayData(obj,var)  TypedData_Get_Struct(obj, cumo_narray_data_t, &cumo_na_data_type, var)

#define CUMO_SDX_IS_STRIDE(x) ((x).stride&0x1)
#define CUMO_SDX_IS_INDEX(x)  (!CUMO_SDX_IS_STRIDE(x))
#define CUMO_SDX_GET_STRIDE(x) ((x).stride>>1)
#define CUMO_SDX_GET_INDEX(x)  ((x).index)

#define CUMO_SDX_SET_STRIDE(x,s) ((x).stride=((s)<<1)|0x1)
#define CUMO_SDX_SET_INDEX(x,idx) ((x).index=idx)

#define CUMO_RNARRAY(val)            ((cumo_narray_t*)DATA_PTR(val))
#define CUMO_RNARRAY_DATA(val)       ((cumo_narray_data_t*)DATA_PTR(val))
#define CUMO_RNARRAY_VIEW(val)       ((cumo_narray_view_t*)DATA_PTR(val))
#define CUMO_RNARRAY_FILEMAP(val)    ((cumo_narray_filemap_t*)DATA_PTR(val))

#define CUMO_RNARRAY_NDIM(val)       (CUMO_RNARRAY(val)->ndim)
#define CUMO_RNARRAY_TYPE(val)       (CUMO_RNARRAY(val)->type)
#define CUMO_RNARRAY_FLAG(val)       (CUMO_RNARRAY(val)->flag)
#define CUMO_RNARRAY_SIZE(val)       (CUMO_RNARRAY(val)->size)
#define CUMO_RNARRAY_SHAPE(val)      (CUMO_RNARRAY(val)->shape)
#define CUMO_RNARRAY_REDUCE(val)     (CUMO_RNARRAY(val)->reduce)

#define CUMO_RNARRAY_DATA_PTR(val)    (CUMO_RNARRAY_DATA(val)->ptr)
#define CUMO_RNARRAY_VIEW_DATA(val)   (CUMO_RNARRAY_VIEW(val)->data)
#define CUMO_RNARRAY_VIEW_OFFSET(val) (CUMO_RNARRAY_VIEW(val)->offset)
#define CUMO_RNARRAY_VIEW_STRIDX(val) (CUMO_RNARRAY_VIEW(val)->stridx)

#define CUMO_NA_NDIM(na)     (((cumo_narray_t*)na)->ndim)
#define CUMO_NA_TYPE(na)     (((cumo_narray_t*)na)->type)
#define CUMO_NA_SIZE(na)     (((cumo_narray_t*)na)->size)
#define CUMO_NA_SHAPE(na)    (((cumo_narray_t*)na)->shape)
#define CUMO_NA_REDUCE(na)   (((cumo_narray_t*)na)->reduce)

#define CUMO_NA_FLAG(obj)    (cumo_na_get_narray_t(obj)->flag)
#define CUMO_NA_FLAG0(obj)   (CUMO_NA_FLAG(obj)[0])
#define CUMO_NA_FLAG1(obj)   (CUMO_NA_FLAG(obj)[1])

#define CUMO_NA_DATA(na)             ((cumo_narray_data_t*)(na))
#define CUMO_NA_VIEW(na)             ((cumo_narray_view_t*)(na))
#define CUMO_NA_DATA_PTR(na)         (CUMO_NA_DATA(na)->ptr)
#define CUMO_NA_VIEW_DATA(na)        (CUMO_NA_VIEW(na)->data)
#define CUMO_NA_VIEW_OFFSET(na)      (CUMO_NA_VIEW(na)->offset)
#define CUMO_NA_VIEW_STRIDX(na)      (CUMO_NA_VIEW(na)->stridx)

#define CUMO_NA_IS_INDEX_AT(na,i)    (CUMO_SDX_IS_INDEX(CUMO_NA_VIEW_STRIDX(na)[i]))
#define CUMO_NA_IS_STRIDE_AT(na,i)   (CUMO_SDX_IS_STRIDE(CUMO_NA_VIEW_STRIDX(na)[i]))
#define CUMO_NA_INDEX_AT(na,i)       (CUMO_SDX_GET_INDEX(CUMO_NA_VIEW_STRIDX(na)[i]))
#define CUMO_NA_STRIDE_AT(na,i)      (CUMO_SDX_GET_STRIDE(CUMO_NA_VIEW_STRIDX(na)[i]))

#define CUMO_NA_FILEMAP_PTR(na)      (((cumo_narray_filemap_t*)na)->ptr)


#define CUMO_NA_FL0_TEST(x,f) (CUMO_NA_FLAG0(x)&(f))
#define CUMO_NA_FL1_TEST(x,f) (CUMO_NA_FLAG1(x)&(f))

#define CUMO_NA_FL0_SET(x,f) do {CUMO_NA_FLAG0(x) |= (f);} while(0)
#define CUMO_NA_FL1_SET(x,f) do {CUMO_NA_FLAG1(x) |= (f);} while(0)

#define CUMO_NA_FL0_UNSET(x,f) do {CUMO_NA_FLAG0(x) &= ~(f);} while(0)
#define CUMO_NA_FL1_UNSET(x,f) do {CUMO_NA_FLAG1(x) &= ~(f);} while(0)

#define CUMO_NA_FL0_REVERSE(x,f) do {CUMO_NA_FLAG0(x) ^= (f);} while(0)
#define CUMO_NA_FL1_REVERSE(x,f) do {CUMO_NA_FLAG1(x) ^= (f);} while(0)


/* FLAGS
   - row-major / column-major
   - Overwrite or not
   - byteswapp
   - Extensible?
   - matrix or not
*/

#define CUMO_NA_FL0_BIG_ENDIAN     (0x1<<0)
#define CUMO_NA_FL0_COLUMN_MAJOR   (0x1<<1)
#define CUMO_NA_FL1_LOCK           (0x1<<0)
#define CUMO_NA_FL1_INPLACE        (0x1<<1)

#define CUMO_TEST_COLUMN_MAJOR(x)   CUMO_NA_FL0_TEST(x,CUMO_NA_FL0_COLUMN_MAJOR)
#define CUMO_SET_COLUMN_MAJOR(x)    CUMO_NA_FL0_SET(x,CUMO_NA_FL0_COLUMN_MAJOR)
#define CUMO_UNSET_COLUMN_MAJOR(x)  CUMO_NA_FL0_UNSET(x,CUMO_NA_FL0_COLUMN_MAJOR)

#define CUMO_TEST_ROW_MAJOR(x)      (!CUMO_TEST_COLUMN_MAJOR(x))
#define CUMO_SET_ROW_MAJOR(x)       CUMO_UNSET_COLUMN_MAJOR(x)
#define CUMO_UNSET_ROW_MAJOR(x)     CUMO_SET_COLUMN_MAJOR(x)

#define CUMO_TEST_BIG_ENDIAN(x)     CUMO_NA_FL0_TEST(x,CUMO_NA_FL0_BIG_ENDIAN)
#define CUMO_SET_BIG_ENDIAN(x)      CUMO_NA_FL0_SET(x,CUMO_NA_FL0_BIG_ENDIAN)
#define CUMO_UNSET_BIG_ENDIAN(x)    CUMO_NA_FL0_UNSET(x,CUMO_NA_FL0_BIG_ENDIAN)

#define CUMO_TEST_LITTLE_ENDIAN(x)  (!CUMO_TEST_BIG_ENDIAN(x))
#define CUMO_SET_LITTLE_ENDIAN(x)   CUMO_UNSET_BIG_ENDIAN(x)
#define CUMO_UNSET_LITTLE_ENDIAN(x) CUMO_SET_BIG_ENDIAN(x)

#define CUMO_REVERSE_ENDIAN(x)      CUMO_NA_FL0_REVERSE((x),CUMO_NA_FL0_BIG_ENDIAN)

#define CUMO_TEST_LOCK(x)           CUMO_NA_FL1_TEST(x,CUMO_NA_FL1_LOCK)
#define CUMO_SET_LOCK(x)            CUMO_NA_FL1_SET(x,CUMO_NA_FL1_LOCK)
#define CUMO_UNCUMO_SET_LOCK(x)          CUMO_NA_FL1_UNSET(x,CUMO_NA_FL1_LOCK)

#define CUMO_TEST_INPLACE(x)        CUMO_NA_FL1_TEST(x,CUMO_NA_FL1_INPLACE)
#define CUMO_SET_INPLACE(x)         CUMO_NA_FL1_SET(x,CUMO_NA_FL1_INPLACE)
#define CUMO_UNCUMO_SET_INPLACE(x)       CUMO_NA_FL1_UNSET(x,CUMO_NA_FL1_INPLACE)

#ifdef DYNAMIC_ENDIAN
// not supported
#else
#ifdef WORDS_BIGENDIAN
#define CUMO_TEST_HOST_ORDER(x)     CUMO_TEST_BIG_ENDIAN(x)
#define CUMO_SET_HOST_ORDER(x)      CUMO_SET_BIG_ENDIAN(x)
#define CUMO_UNSET_HOST_ORDER(x)    CUMO_UNSET_BIG_ENDIAN(x)
#define CUMO_TEST_BYTE_SWAPPED(x)   CUMO_TEST_LITTLE_ENDIAN(x)
#define CUMO_SET_BYTE_SWAPPED(x)    CUMO_SET_LITTLE_ENDIAN(x)
#define CUMO_UNCUMO_SET_BYTE_SWAPPED(x)  CUMO_UNSET_LITTLE_ENDIAN(x)
#define CUMO_NA_FL0_INIT            CUMO_NA_FL0_BIG_ENDIAN
#else // LITTLE ENDIAN
#define CUMO_TEST_HOST_ORDER(x)     CUMO_TEST_LITTLE_ENDIAN(x)
#define CUMO_SET_HOST_ORDER(x)      CUMO_SET_LITTLE_ENDIAN(x)
#define CUMO_UNSET_HOST_ORDER(x)    CUMO_UNSET_LITTLE_ENDIAN(x)
#define CUMO_TEST_BYTE_SWAPPED(x)   CUMO_TEST_BIG_ENDIAN(x)
#define CUMO_SET_BYTE_SWAPPED(x)    CUMO_SET_BIG_ENDIAN(x)
#define CUMO_UNCUMO_SET_BYTE_SWAPPED(x)  CUMO_UNSET_BIG_ENDIAN(x)
#define CUMO_NA_FL0_INIT            0
#endif
#endif
#define CUMO_NA_FL1_INIT            0


#define CumoIsNArray(obj) (rb_obj_is_kind_of(obj,cNArray)==Qtrue)

#define CUMO_DEBUG_PRINT(v) puts(StringValueCStr(rb_funcall(v,rb_intern("inspect"),0)))

#define CUMO_NA_CumoIsNArray(obj) (rb_obj_is_kind_of(obj,cNArray)==Qtrue)
#define CUMO_NA_IsArray(obj) (TYPE(obj)==T_ARRAY || rb_obj_is_kind_of(obj,cNArray)==Qtrue)

static inline bool
cumo_na_has_idx_p(VALUE obj)
{
    cumo_narray_t *na;
    cumo_narray_view_t *nv;
    int i = 0;
    CumoGetNArray(obj, na);
    if (CUMO_NA_TYPE(na) == CUMO_NARRAY_VIEW_T) {
        CumoGetNArrayView(obj, nv);
        for (; i < nv->base.ndim; ++i) {
            if (nv->stridx[i].index) {
                return true;
            }
        }
    }
    return false;
}

#define CUMO_NUM2REAL(v)  NUM2DBL( rb_funcall((v),cumo_na_id_real,0) )
#define CUMO_NUM2IMAG(v)  NUM2DBL( rb_funcall((v),cumo_na_id_imag,0) )

//#define CUMO_NA_MAX_DIMENSION (int)(sizeof(VALUE)*8-2)
#define CUMO_NA_MAX_DIMENSION 12
#define CUMO_NA_MAX_ELMSZ     65535

typedef unsigned int CUMO_BIT_DIGIT;
#define CUMO_BYTE_BIT_DIGIT sizeof(CUMO_BIT_DIGIT)
#define CUMO_NB     (sizeof(CUMO_BIT_DIGIT)*8)
#define CUMO_BALL   (~(CUMO_BIT_DIGIT)0)
#define CUMO_SLB(n) (((n)==CUMO_NB)?~(CUMO_BIT_DIGIT)0:(~(~(CUMO_BIT_DIGIT)0<<(n))))

#include "cumo/ndloop.h"
#include "cumo/intern.h"

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_NARRAY_H */
