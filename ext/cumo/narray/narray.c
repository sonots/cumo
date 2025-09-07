#define CUMO_NARRAY_C
#include <ruby.h>
#include <assert.h>
#include "cumo.h"
#include "cumo/narray.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"

/* global variables within this module */
VALUE cumo_cNArray;
VALUE rb_mCumo;
VALUE cumo_na_eCastError;
VALUE cumo_na_eShapeError;
VALUE cumo_na_eOperationError;
VALUE cumo_na_eDimensionError;
VALUE cumo_na_eValueError;

static ID cumo_id_contiguous_stride;
static ID cumo_id_allocate;
static ID cumo_id_element_byte_size;
static ID cumo_id_fill;
static ID cumo_id_seq;
static ID cumo_id_logseq;
static ID cumo_id_eye;
static ID cumo_id_UPCAST;
static ID cumo_id_cast;
static ID cumo_id_dup;
static ID cumo_id_to_host;
static ID cumo_id_bracket;
static ID cumo_id_shift_left;
static ID cumo_id_eq;
static ID cumo_id_count_false;
static ID cumo_id_count_false_cpu;
static ID cumo_id_axis;
static ID cumo_id_nan;
static ID cumo_id_keepdims;

VALUE cumo_sym_reduce;
VALUE cumo_sym_option;
VALUE cumo_sym_loop_opt;
VALUE cumo_sym_init;

#ifndef HAVE_RB_CCOMPLEX
VALUE rb_cComplex;
#endif
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
VALUE rb_cArithSeq;
#endif

int cumo_na_inspect_rows_=20;
int cumo_na_inspect_cols_=80;

const rb_data_type_t cumo_na_data_type = {
    "Cumo::NArray",
    {0, 0, 0,}, 0, 0, 0,
};

static void
cumo_na_debug_info_nadata(VALUE self)
{
    cumo_narray_data_t *na;
    CumoGetNArrayData(self,na);

    printf("  ptr    = 0x%"SZF"x\n", (size_t)(na->ptr));
}


static VALUE
cumo_na_debug_info_naview(VALUE self)
{
    int i;
    cumo_narray_view_t *na;
    size_t *idx;
    size_t j;
    CumoGetNArrayView(self,na);

    printf("  data   = 0x%"SZF"x\n", (size_t)na->data);
    printf("  offset = %"SZF"d\n", (size_t)na->offset);
    printf("  stridx = 0x%"SZF"x\n", (size_t)na->stridx);

    if (na->stridx) {
        printf("  stridx = [");
        for (i=0; i<na->base.ndim; i++) {
            if (CUMO_SDX_IS_INDEX(na->stridx[i])) {

                idx = CUMO_SDX_GET_INDEX(na->stridx[i]);
                printf("  index[%d]=[", i);
                for (j=0; j<na->base.shape[i]; j++) {
                    printf(" %"SZF"d", idx[j]);
                }
                printf(" ] ");

            } else {
                printf(" %"SZF"d", CUMO_SDX_GET_STRIDE(na->stridx[i]));
            }
        }
        printf(" ]\n");
    }
    return Qnil;
}


VALUE
cumo_na_debug_info(VALUE self)
{
    int i;
    cumo_narray_t *na;
    CumoGetNArray(self,na);

    printf("%s:\n",rb_class2name(rb_obj_class(self)));
    printf("  id     = 0x%"PRI_VALUE_PREFIX"x\n", self);
    printf("  type   = %d\n", na->type);
    printf("  flag   = [%d,%d]\n", na->flag[0], na->flag[1]);
    printf("  size   = %"SZF"d\n", na->size);
    printf("  ndim   = %d\n", na->ndim);
    printf("  shape  = 0x%"SZF"x\n", (size_t)na->shape);
    if (na->shape) {
        printf("  shape  = [");
        for (i=0;i<na->ndim;i++)
            printf(" %"SZF"d", na->shape[i]);
        printf(" ]\n");
    }

    switch(na->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        cumo_na_debug_info_nadata(self);
        break;
    case CUMO_NARRAY_VIEW_T:
        cumo_na_debug_info_naview(self);
        break;
    }
    return Qnil;
}


static size_t
cumo_na_view_memsize(const void* ptr)
{
    int i;
    size_t size = sizeof(cumo_narray_view_t);
    const cumo_narray_view_t *na = ptr;

    assert(na->base.type == CUMO_NARRAY_VIEW_T);

    if (na->stridx != NULL) {
        for (i=0; i<na->base.ndim; i++) {
            if (CUMO_SDX_IS_INDEX(na->stridx[i])) {
                size += sizeof(size_t) * na->base.shape[i];
            }
        }
        size += sizeof(cumo_stridx_t) * na->base.ndim;
    }
    if (na->base.size > 0) {
        if (na->base.shape != NULL && na->base.shape != &(na->base.size)) {
            size += sizeof(size_t) * na->base.ndim;
        }
    }
    return size;
}

static void
cumo_na_view_free(void* ptr)
{
    int i;
    cumo_narray_view_t *na = (cumo_narray_view_t*)ptr;

    assert(na->base.type == CUMO_NARRAY_VIEW_T);

    if (na->stridx != NULL) {
        for (i=0; i<na->base.ndim; i++) {
            if (CUMO_SDX_IS_INDEX(na->stridx[i])) {
                void *idx = CUMO_SDX_GET_INDEX(na->stridx[i]);
                cumo_cuda_runtime_free(idx);
            }
        }
        xfree(na->stridx);
        na->stridx = NULL;
    }
    if (na->base.size > 0) {
        if (na->base.shape != NULL && na->base.shape != &(na->base.size)) {
            xfree(na->base.shape);
            na->base.shape = NULL;
        }
    }
    xfree(na);
}

static void
cumo_na_view_gc_mark(void* na)
{
    if (((cumo_narray_t*)na)->type == CUMO_NARRAY_VIEW_T) {
        rb_gc_mark(((cumo_narray_view_t*)na)->data);
    }
}

const rb_data_type_t cumo_na_data_type_view = {
    "Cumo::NArrayView",
    {cumo_na_view_gc_mark, cumo_na_view_free, cumo_na_view_memsize,},
    &cumo_na_data_type, 0, 0,
};

VALUE
cumo_na_s_allocate_view(VALUE klass)
{
    cumo_narray_view_t *na = ALLOC(cumo_narray_view_t);

    na->base.ndim = 0;
    na->base.type = CUMO_NARRAY_VIEW_T;
    na->base.flag[0] = CUMO_NA_FL0_INIT;
    na->base.flag[1] = CUMO_NA_FL1_INIT;
    na->base.size = 0;
    na->base.shape = NULL;
    na->base.reduce = INT2FIX(0);
    na->data = Qnil;
    na->offset = 0;
    na->stridx = NULL;
    return TypedData_Wrap_Struct(klass, &cumo_na_data_type_view, (void*)na);
}


//static const size_t zero=0;

void
cumo_na_array_to_internal_shape(VALUE self, VALUE ary, size_t *shape)
{
    size_t    i, n, c, s;
    ssize_t   x;
    VALUE     v;
    int       flag = 0;

    n = RARRAY_LEN(ary);

    if (RTEST(self)) {
        flag = CUMO_TEST_COLUMN_MAJOR(self);
    }
    if (flag) {
        c = n-1;
        s = -1;
    } else {
        c = 0;
        s = 1;
    }
    for (i=0; i<n; i++) {
        v = RARRAY_AREF(ary,i);
        x = NUM2SSIZET(v);
        if (x < 0) {
            rb_raise(rb_eArgError,"size must be non-negative");
        }
        shape[c] = x;
        c += s;
    }
}



void
cumo_na_alloc_shape(cumo_narray_t *na, int ndim)
{
    na->ndim = ndim;
    na->size = 0;
    switch(ndim) {
    case 0:
    case 1:
        na->shape = &(na->size);
        break;
    default:
        if (ndim < 0) {
            rb_raise(cumo_na_eDimensionError,"ndim=%d is negative", ndim);
        }
        if (ndim > CUMO_NA_MAX_DIMENSION) {
            rb_raise(cumo_na_eDimensionError,"ndim=%d is too many", ndim);
        }
        na->shape = ALLOC_N(size_t, ndim);
    }
}

void
cumo_na_setup_shape(cumo_narray_t *na, int ndim, size_t *shape)
{
    int i;
    size_t size;

    cumo_na_alloc_shape(na, ndim);

    if (ndim==0) {
        na->size = 1;
    }
    else if (ndim==1) {
        na->size = shape[0];
    }
    else {
        for (i=0, size=1; i<ndim; i++) {
            na->shape[i] = shape[i];
            size *= shape[i];
        }
        na->size = size;
    }
}

static void
cumo_na_setup(VALUE self, int ndim, size_t *shape)
{
    cumo_narray_t *na;
    CumoGetNArray(self,na);
    cumo_na_setup_shape(na, ndim, shape);
}


/*
  @overload initialize(shape)
  @overload initialize(size0, size1, ...)
  @param [Array] shape (array of sizes along each dimension)
  @param [Integer] sizeN (size along Nth-dimension)
  @return [Cumo::NArray] unallocated narray.

  Constructs an instance of NArray class using the given
  and <i>shape</i> or <i>sizes</i>.
  Note that NArray itself is an abstract super class and
  not suitable to create instances.
  Use Typed Subclasses of NArray (DFloat, Int32, etc) to create instances.
  This method does not allocate memory for array data.
  Memory is allocated on write method such as #fill, #store, #seq, etc.

  @example
    i = Cumo::Int64.new([2,4,3])
    #=> Cumo::Int64#shape=[2,4,3](empty)

    f = Cumo::DFloat.new(3,4)
    #=> Cumo::DFloat#shape=[3,4](empty)

    f.fill(2)
    #=> Cumo::DFloat#shape=[3,4]
    # [[2, 2, 2, 2],
    #  [2, 2, 2, 2],
    #  [2, 2, 2, 2]]

    x = Cumo::NArray.new(5)
    #=> in `new': allocator undefined for Cumo::NArray (TypeError)
    #   	from t.rb:9:in `<main>'

*/
static VALUE
cumo_na_initialize(VALUE self, VALUE args)
{
    VALUE v;
    size_t *shape=NULL;
    int ndim;

    if (RARRAY_LEN(args) == 1) {
        v = RARRAY_AREF(args,0);
        if (TYPE(v) != T_ARRAY) {
            v = args;
        }
    } else {
        v = args;
    }
    ndim = RARRAY_LEN(v);
    if (ndim > CUMO_NA_MAX_DIMENSION) {
        rb_raise(rb_eArgError,"ndim=%d exceeds maximum dimension",ndim);
    }
    shape = ALLOCA_N(size_t, ndim);
    // setup size_t shape[] from VALUE shape argument
    cumo_na_array_to_internal_shape(self, v, shape);
    cumo_na_setup(self, ndim, shape);

    return self;
}


VALUE
cumo_na_new(VALUE klass, int ndim, size_t *shape)
{
    volatile VALUE obj;

    obj = rb_funcall(klass, cumo_id_allocate, 0);
    cumo_na_setup(obj, ndim, shape);
    return obj;
}


VALUE
cumo_na_view_new(VALUE klass, int ndim, size_t *shape)
{
    volatile VALUE obj;

    obj = cumo_na_s_allocate_view(klass);
    cumo_na_setup(obj, ndim, shape);
    return obj;
}


/*
  Replaces the contents of self with the contents of other narray.
  Used in dup and clone method.
  @overload initialize_copy(other)
  @param [Cumo::NArray] other
  @return [Cumo::NArray] self
 */
static VALUE
cumo_na_initialize_copy(VALUE self, VALUE orig)
{
    cumo_narray_t *na;
    CumoGetNArray(orig,na);

    cumo_na_setup(self,CUMO_NA_NDIM(na),CUMO_NA_SHAPE(na));
    cumo_na_store(self,orig);
    cumo_na_copy_flags(orig,self);
    return self;
}


/*
 *  call-seq:
 *     zeros(shape)  => narray
 *     zeros(size1,size2,...)  => narray
 *
 *  Returns a zero-filled narray with <i>shape</i>.
 *  This singleton method is valid not for NArray class itself
 *  but for typed NArray subclasses, e.g., DFloat, Int64.
 *  @example
 *    a = Cumo::DFloat.zeros(3,5)
 *    => Cumo::DFloat#shape=[3,5]
 *    [[0, 0, 0, 0, 0],
 *     [0, 0, 0, 0, 0],
 *     [0, 0, 0, 0, 0]]
 */
static VALUE
cumo_na_s_zeros(int argc, VALUE *argv, VALUE klass)
{
    VALUE obj;
    obj = rb_class_new_instance(argc, argv, klass);
    return rb_funcall(obj, cumo_id_fill, 1, INT2FIX(0));
}


/*
 *  call-seq:
 *     ones(shape)  => narray
 *     ones(size1,size2,...)  => narray
 *
 *  Returns a one-filled narray with <i>shape</i>.
 *  This singleton method is valid not for NArray class itself
 *  but for typed NArray subclasses, e.g., DFloat, Int64.
 *  @example
 *    a = Cumo::DFloat.ones(3,5)
 *    => Cumo::DFloat#shape=[3,5]
 *    [[1, 1, 1, 1, 1],
 *     [1, 1, 1, 1, 1],
 *     [1, 1, 1, 1, 1]]
 */
static VALUE
cumo_na_s_ones(int argc, VALUE *argv, VALUE klass)
{
    VALUE obj;
    obj = rb_class_new_instance(argc, argv, klass);
    return rb_funcall(obj, cumo_id_fill, 1, INT2FIX(1));
}


/*
  Returns an array of N linearly spaced points between x1 and x2.
  This singleton method is valid not for NArray class itself
  but for typed NArray subclasses, e.g., DFloat, Int64.

  @overload linspace(x1, x2, [n])
  @param [Numeric] x1   The start value
  @param [Numeric] x2   The end value
  @param [Integer] n    The number of elements. (default is 100).
  @return [Cumo::NArray]  result array.

  @example
    a = Cumo::DFloat.linspace(-5,5,7)
    => Cumo::DFloat#shape=[7]
    [-5, -3.33333, -1.66667, 0, 1.66667, 3.33333, 5]
 */
static VALUE
cumo_na_s_linspace(int argc, VALUE *argv, VALUE klass)
{
    VALUE obj, vx1, vx2, vstep, vsize;
    double n;
    int narg;

    narg = rb_scan_args(argc,argv,"21",&vx1,&vx2,&vsize);
    if (narg==3) {
        n = NUM2DBL(vsize);
    } else {
        n = 100;
        vsize = INT2FIX(100);
    }

    obj = rb_funcall(vx2, '-', 1, vx1);
    vstep = rb_funcall(obj, '/', 1, DBL2NUM(n-1));

    obj = rb_class_new_instance(1, &vsize, klass);
    return rb_funcall(obj, cumo_id_seq, 2, vx1, vstep);
}

/*
  Returns an array of N logarithmically spaced points between 10^a and 10^b.
  This singleton method is valid not for NArray having +logseq+ method,
  i.e., DFloat, SFloat, DComplex, and SComplex.

  @overload logspace(a, b, [n, base])
  @param [Numeric] a  The start value
  @param [Numeric] b  The end value
  @param [Integer] n  The number of elements. (default is 50)
  @param [Numeric] base  The base of log space. (default is 10)
  @return [Cumo::NArray]  result array.

  @example
    Cumo::DFloat.logspace(4,0,5,2)
    => Cumo::DFloat#shape=[5]
       [16, 8, 4, 2, 1]
    Cumo::DComplex.logspace(0,1i*Math::PI,5,Math::E)
    => Cumo::DComplex#shape=[5]
       [1+4.44659e-323i, 0.707107+0.707107i, 6.12323e-17+1i, -0.707107+0.707107i, ...]
 */
static VALUE
cumo_na_s_logspace(int argc, VALUE *argv, VALUE klass)
{
    VALUE obj, vx1, vx2, vstep, vsize, vbase;
    double n;

    rb_scan_args(argc,argv,"22",&vx1,&vx2,&vsize,&vbase);
    if (vsize == Qnil) {
        vsize = INT2FIX(50);
        n = 50;
    } else {
        n = NUM2DBL(vsize);
    }
    if (vbase == Qnil) {
        vbase = DBL2NUM(10);
    }

    obj = rb_funcall(vx2, '-', 1, vx1);
    vstep = rb_funcall(obj, '/', 1, DBL2NUM(n-1));

    obj = rb_class_new_instance(1, &vsize, klass);
    return rb_funcall(obj, cumo_id_logseq, 3, vx1, vstep, vbase);
}


/*
  Returns a NArray with shape=(n,n) whose diagonal elements are 1, otherwise 0.
  @overload  eye(n)
  @param [Integer] n  Size of NArray. Creates 2-D NArray with shape=(n,n)
  @return [Cumo::NArray]  created NArray.
  @example
    a = Cumo::DFloat.eye(3)
    => Cumo::DFloat#shape=[3,3]
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
*/
static VALUE
cumo_na_s_eye(int argc, VALUE *argv, VALUE klass)
{
    VALUE obj;
    VALUE tmp[2];

    if (argc==0) {
        rb_raise(rb_eArgError,"No argument");
    }
    else if (argc==1) {
        tmp[0] = tmp[1] = argv[0];
        argv = tmp;
        argc = 2;
    }
    obj = rb_class_new_instance(argc, argv, klass);
    return rb_funcall(obj, cumo_id_eye, 0);
}



#define READ 1
#define WRITE 2

static char *
cumo_na_get_pointer_for_rw(VALUE self, int flag)
{
    char *ptr;
    VALUE obj;
    cumo_narray_t *na;

    if ((flag & WRITE) && OBJ_FROZEN(self)) {
        rb_raise(rb_eRuntimeError, "cannot write to frozen NArray.");
    }

    CumoGetNArray(self,na);

    switch(CUMO_NA_TYPE(na)) {
    case CUMO_NARRAY_DATA_T:
        ptr = CUMO_NA_DATA_PTR(na);
        if (CUMO_NA_SIZE(na) > 0 && ptr == NULL) {
            if (flag & READ) {
                rb_raise(rb_eRuntimeError,"cannot read unallocated NArray");
            }
            if (flag & WRITE) {
                rb_funcall(self, cumo_id_allocate, 0);
                ptr = CUMO_NA_DATA_PTR(na);
            }
        }
        return ptr;
    case CUMO_NARRAY_VIEW_T:
        obj = CUMO_NA_VIEW_DATA(na);
        if ((flag & WRITE) && OBJ_FROZEN(obj)) {
            rb_raise(rb_eRuntimeError, "cannot write to frozen NArray.");
        }
        CumoGetNArray(obj,na);
        switch(CUMO_NA_TYPE(na)) {
        case CUMO_NARRAY_DATA_T:
            ptr = CUMO_NA_DATA_PTR(na);
            if (flag & (READ|WRITE)) {
                if (CUMO_NA_SIZE(na) > 0 && ptr == NULL) {
                    rb_raise(rb_eRuntimeError,"cannot read/write unallocated NArray");
                }
            }
            return ptr;
        default:
            rb_raise(rb_eRuntimeError,"invalid CUMO_NA_TYPE of view: %d",CUMO_NA_TYPE(na));
        }
    default:
        rb_raise(rb_eRuntimeError,"invalid CUMO_NA_TYPE: %d",CUMO_NA_TYPE(na));
    }

    return NULL;
}

char *
cumo_na_get_pointer_for_read(VALUE self)
{
    return cumo_na_get_pointer_for_rw(self, READ);
}

char *
cumo_na_get_pointer_for_write(VALUE self)
{
    return cumo_na_get_pointer_for_rw(self, WRITE);
}

char *
cumo_na_get_pointer_for_read_write(VALUE self)
{
    return cumo_na_get_pointer_for_rw(self, READ|WRITE);
}

char *
cumo_na_get_pointer(VALUE self)
{
    return cumo_na_get_pointer_for_rw(self, 0);
}


void
cumo_na_release_lock(VALUE self)
{
    cumo_narray_t *na;

    CUMO_UNCUMO_SET_LOCK(self);
    CumoGetNArray(self,na);

    switch(CUMO_NA_TYPE(na)) {
    case CUMO_NARRAY_VIEW_T:
        cumo_na_release_lock(CUMO_NA_VIEW_DATA(na));
        break;
    }
}


/* method: size() -- returns the total number of typeents */
static VALUE
cumo_na_size(VALUE self)
{
    cumo_narray_t *na;
    CumoGetNArray(self,na);
    return SIZET2NUM(na->size);
}


/* method: size() -- returns the total number of typeents */
static VALUE
cumo_na_ndim(VALUE self)
{
    cumo_narray_t *na;
    CumoGetNArray(self,na);
    return INT2NUM(na->ndim);
}


/*
  Returns true if self.size == 0.
  @overload empty?
*/
static VALUE
cumo_na_empty_p(VALUE self)
{
    cumo_narray_t *na;
    CumoGetNArray(self,na);
    if (CUMO_NA_SIZE(na)==0) {
        return Qtrue;
    }
    return Qfalse;
}


/* method: shape() -- returns shape, array of the size of dimensions */
static VALUE
 cumo_na_shape(VALUE self)
{
    volatile VALUE v;
    cumo_narray_t *na;
    size_t i, n, c, s;

    CumoGetNArray(self,na);
    n = CUMO_NA_NDIM(na);
    if (CUMO_TEST_COLUMN_MAJOR(self)) {
        c = n-1;
        s = -1;
    } else {
        c = 0;
        s = 1;
    }
    v = rb_ary_new2(n);
    for (i=0; i<n; i++) {
        rb_ary_push(v, SIZET2NUM(na->shape[c]));
        c += s;
    }
    return v;
}


unsigned int
cumo_na_element_stride(VALUE v)
{
    cumo_narray_type_info_t *info;
    cumo_narray_t *na;

    CumoGetNArray(v,na);
    if (na->type == CUMO_NARRAY_VIEW_T) {
        v = CUMO_NA_VIEW_DATA(na);
        CumoGetNArray(v,na);
    }
    assert(na->type == CUMO_NARRAY_DATA_T);

    info = (cumo_narray_type_info_t *)(RTYPEDDATA_TYPE(v)->data);
    return info->element_stride;
}

size_t
cumo_na_dtype_element_stride(VALUE klass)
{
    return NUM2SIZET(rb_const_get(klass, cumo_id_contiguous_stride));
}

size_t
cumo_na_get_offset(VALUE self)
{
    cumo_narray_t *na;
    CumoGetNArray(self,na);

    switch(na->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        return 0;
    case CUMO_NARRAY_VIEW_T:
        return CUMO_NA_VIEW_OFFSET(na);
    }
    return 0;
}

char*
cumo_na_get_offset_pointer(VALUE a)
{
    return cumo_na_get_pointer(a) + cumo_na_get_offset(a);
}

char*
cumo_na_get_offset_pointer_for_write(VALUE a)
{
    return cumo_na_get_pointer_for_write(a) + cumo_na_get_offset(a);
}

char*
cumo_na_get_offset_pointer_for_read(VALUE a)
{
    return cumo_na_get_pointer_for_read(a) + cumo_na_get_offset(a);
}

char*
cumo_na_get_offset_pointer_for_read_write(VALUE a)
{
    return cumo_na_get_pointer_for_read_write(a) + cumo_na_get_offset(a);
}

void
cumo_na_index_arg_to_internal_order(int argc, VALUE *argv, VALUE self)
{
    int i,j;
    VALUE tmp;

    if (CUMO_TEST_COLUMN_MAJOR(self)) {
        for (i=0,j=argc-1; i<argc/2; i++,j--) {
            tmp = argv[i];
            argv[i] = argv[j];
            argv[j] = tmp;
        }
    }
}

void
cumo_na_copy_flags(VALUE src, VALUE dst)
{
    cumo_narray_t *na1, *na2;

    CumoGetNArray(src,na1);
    CumoGetNArray(dst,na2);

    na2->flag[0] = na1->flag[0];
    //na2->flag[1] = CUMO_NA_FL1_INIT;

    RBASIC(dst)->flags |= (RBASIC(src)->flags) &
        (FL_USER1|FL_USER2|FL_USER3|FL_USER4|FL_USER5|FL_USER6|FL_USER7);
}


// fix name, ex, allow_stride_for_flatten_view
VALUE
cumo_na_check_ladder(VALUE self, int start_dim)
{
    int i;
    ssize_t st0, st1;
    cumo_narray_t *na;
    CumoGetNArray(self,na);

    if (start_dim < -na->ndim || start_dim >= na->ndim) {
        rb_bug("start_dim (%d) out of range",start_dim);
    }

    switch(na->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        return Qtrue;
    case CUMO_NARRAY_VIEW_T:
        // negative dim -> position from last dim
        if (start_dim < 0) {
            start_dim += CUMO_NA_NDIM(na);
        }
        // not ladder if it has index
        for (i=start_dim; i<CUMO_NA_NDIM(na); i++) {
            if (CUMO_NA_IS_INDEX_AT(na,i))
                return Qfalse;
        }
        // check stride
        st0 = CUMO_NA_STRIDE_AT(na,start_dim);
        for (i=start_dim+1; i<CUMO_NA_NDIM(na); i++) {
            st1 = CUMO_NA_STRIDE_AT(na,i);
            if (st0 != (ssize_t)(st1 * CUMO_NA_SHAPE(na)[i])) {
                return Qfalse;
            }
            st0 = st1;
        }
    }
    return Qtrue;
}

VALUE
cumo_na_check_contiguous(VALUE self)
{
    ssize_t elmsz;
    cumo_narray_t *na;
    CumoGetNArray(self,na);

    switch(na->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        return Qtrue;
    case CUMO_NARRAY_VIEW_T:
        if (CUMO_NA_VIEW_STRIDX(na)==0) {
            return Qtrue;
        }
        if (cumo_na_check_ladder(self,0)==Qtrue) {
            elmsz = cumo_na_element_stride(self);
            if (elmsz == CUMO_NA_STRIDE_AT(na,CUMO_NA_NDIM(na)-1)) {
                return Qtrue;
            }
        }
    }
    return Qfalse;
}

VALUE
cumo_na_check_fortran_contiguous(VALUE self)
{
    int i;
    ssize_t st0;
    cumo_narray_t *na;

    switch(CUMO_RNARRAY_TYPE(self)) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        return Qfalse;
    case CUMO_NARRAY_VIEW_T:
        CumoGetNArray(self,na);

        // not contiguous if it has index
        for (i=0; i < CUMO_NA_NDIM(na); i++) {
            if (CUMO_NA_IS_INDEX_AT(na,i))
                return Qfalse;
        }

        // check f-contiguous
        st0 = cumo_na_element_stride(self); // elmsz
        for (i=0; i < CUMO_NA_NDIM(na); i++) {
            if (CUMO_NA_SHAPE(na)[i] == 1)
                continue;
            if (CUMO_NA_STRIDE_AT(na, i) != st0)
                return Qfalse;
            st0 *= CUMO_NA_SHAPE(na)[i];
        }
    }
    return Qtrue;
}

VALUE
cumo_na_as_contiguous_array(VALUE a)
{
    return cumo_na_check_contiguous(a) == Qtrue ? a : rb_funcall(a, rb_intern("dup"), 0);
}

//----------------------------------------------------------------------

/*
 *  call-seq:
 *     narray.view => narray
 *
 *  Return view of NArray
 */
VALUE
cumo_na_make_view(VALUE self)
{
    int i, nd;
    size_t *idx1, *idx2;
    ssize_t stride;
    cumo_narray_t *na;
    cumo_narray_view_t *na1, *na2;
    volatile VALUE view;

    CumoGetNArray(self,na);
    nd = na->ndim;

    view = cumo_na_s_allocate_view(rb_obj_class(self));

    cumo_na_copy_flags(self, view);
    CumoGetNArrayView(view, na2);

    cumo_na_setup_shape((cumo_narray_t*)na2, nd, na->shape);
    na2->stridx = ALLOC_N(cumo_stridx_t,nd);

    switch(na->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        stride = cumo_na_element_stride(self);
        for (i=nd; i--;) {
            CUMO_SDX_SET_STRIDE(na2->stridx[i],stride);
            stride *= na->shape[i];
        }
        na2->offset = 0;
        na2->data = self;
        break;
    case CUMO_NARRAY_VIEW_T:
        CumoGetNArrayView(self, na1);
        for (i=0; i<nd; i++) {
            if (CUMO_SDX_IS_INDEX(na1->stridx[i])) {
                idx1 = CUMO_SDX_GET_INDEX(na1->stridx[i]);
                // idx2 = ALLOC_N(size_t,na1->base.shape[i]);
                // for (j=0; j<na1->base.shape[i]; j++) {
                //     idx2[j] = idx1[j];
                // }
                idx2 = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*na1->base.shape[i]);
                cumo_cuda_runtime_check_status(cudaMemcpyAsync(idx2,idx1,sizeof(size_t)*na1->base.shape[i],cudaMemcpyDeviceToDevice,0));
                CUMO_SDX_SET_INDEX(na2->stridx[i],idx2);
            } else {
                na2->stridx[i] = na1->stridx[i];
            }
        }
        na2->offset = na1->offset;
        na2->data = na1->data;
        break;
    }

    return view;
}


//----------------------------------------------------------------------

/*
 *  call-seq:
 *     narray.expand_dims(dim) => narray view
 *
 *  Expand the shape of an array. Insert a new axis with size=1
 *  at a given dimension.
 *  @param [Integer] dim  dimension at which new axis is inserted.
 *  @return [Cumo::NArray]  result narray view.
 */
static VALUE
cumo_na_expand_dims(VALUE self, VALUE vdim)
{
    int  i, j, nd, dim;
    size_t *shape, *na2_shape;
    cumo_stridx_t *stridx, *na2_stridx;
    cumo_narray_t *na;
    cumo_narray_view_t *na2;
    VALUE view;

    CumoGetNArray(self,na);
    nd = na->ndim;

    dim = NUM2INT(vdim);
    if (dim < -nd-1 || dim > nd) {
        rb_raise(cumo_na_eDimensionError,"invalid axis (%d for %dD NArray)",
                 dim,nd);
    }
    if (dim < 0) {
        dim += nd+1;
    }

    view = cumo_na_make_view(self);
    CumoGetNArrayView(view, na2);

    shape = ALLOC_N(size_t,nd+1);
    stridx = ALLOC_N(cumo_stridx_t,nd+1);
    na2_shape = na2->base.shape;
    na2_stridx = na2->stridx;

    for (i=j=0; i<=nd; i++) {
        if (i==dim) {
            shape[i] = 1;
            CUMO_SDX_SET_STRIDE(stridx[i],0);
        } else {
            shape[i] = na2_shape[j];
            stridx[i] = na2_stridx[j];
            j++;
        }
    }

    na2->stridx = stridx;
    xfree(na2_stridx);
    na2->base.shape = shape;
    if (na2_shape != &(na2->base.size)) {
        xfree(na2_shape);
    }
    na2->base.ndim++;
    return view;
}

//----------------------------------------------------------------------

/*
 *  call-seq:
 *     narray.reverse([dim0,dim1,..]) => narray
 *
 *  Return reversed view along specified dimeinsion
 */
static VALUE
cumo_na_reverse(int argc, VALUE *argv, VALUE self)
{
    int i, nd;
    size_t  j, n;
    size_t  offset;
    size_t *idx1, *idx2;
    ssize_t stride;
    ssize_t sign;
    cumo_narray_t *na;
    cumo_narray_view_t *na1, *na2;
    VALUE view;
    VALUE reduce;

    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, 0, 0);

    CumoGetNArray(self,na);
    nd = na->ndim;

    view = cumo_na_s_allocate_view(rb_obj_class(self));

    cumo_na_copy_flags(self, view);
    CumoGetNArrayView(view, na2);

    cumo_na_setup_shape((cumo_narray_t*)na2, nd, na->shape);
    na2->stridx = ALLOC_N(cumo_stridx_t,nd);

    switch(na->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        stride = cumo_na_element_stride(self);
        offset = 0;
        for (i=nd; i--;) {
            if (cumo_na_test_reduce(reduce,i)) {
                offset += (na->shape[i]-1)*stride;
                sign = -1;
            } else {
                sign = 1;
            }
            CUMO_SDX_SET_STRIDE(na2->stridx[i],stride*sign);
            stride *= na->shape[i];
        }
        na2->offset = offset;
        na2->data = self;
        break;
    case CUMO_NARRAY_VIEW_T:
        CumoGetNArrayView(self, na1);
        offset = na1->offset;
        for (i=0; i<nd; i++) {
            n = na1->base.shape[i];
            if (CUMO_SDX_IS_INDEX(na1->stridx[i])) {
                idx1 = CUMO_SDX_GET_INDEX(na1->stridx[i]);
                // idx2 = ALLOC_N(size_t,n);
                // if (cumo_na_test_reduce(reduce,i)) {
                //     for (j=0; j<n; j++) {
                //         idx2[n-1-j] = idx1[j];
                //     }
                // } else {
                //     for (j=0; j<n; j++) {
                //         idx2[j] = idx1[j];
                //     }
                // }
                idx2 = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*n);
                if (cumo_na_test_reduce(reduce,i)) {
                    CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("cumo_na_reverse", "any");
                    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
                    for (j=0; j<n; j++) {
                        idx2[n-1-j] = idx1[j];
                    }
                } else {
                    cumo_cuda_runtime_check_status(cudaMemcpyAsync(idx2,idx1,sizeof(size_t)*n,cudaMemcpyDeviceToDevice,0));
                }
                CUMO_SDX_SET_INDEX(na2->stridx[i],idx2);
            } else {
                stride = CUMO_SDX_GET_STRIDE(na1->stridx[i]);
                if (cumo_na_test_reduce(reduce,i)) {
                    offset += (n-1)*stride;
                    CUMO_SDX_SET_STRIDE(na2->stridx[i],-stride);
                } else {
                    na2->stridx[i] = na1->stridx[i];
                }
            }
        }
        na2->offset = offset;
        na2->data = na1->data;
        break;
    }

    return view;
}

//----------------------------------------------------------------------

VALUE
cumo_na_upcast(VALUE type1, VALUE type2)
{
    VALUE upcast_hash;
    VALUE result_type;

    if (type1==type2) {
        return type1;
    }
    upcast_hash = rb_const_get(type1, cumo_id_UPCAST);
    result_type = rb_hash_aref(upcast_hash, type2);
    if (NIL_P(result_type)) {
        if (TYPE(type2)==T_CLASS) {
            if (RTEST(rb_class_inherited_p(type2,cNArray))) {
                upcast_hash = rb_const_get(type2, cumo_id_UPCAST);
                result_type = rb_hash_aref(upcast_hash, type1);
            }
        }
    }
    return result_type;
}

/*
  Returns an array containing other and self,
  both are converted to upcasted type of NArray.
  Note that NArray has distinct UPCAST mechanism.
  Coerce is used for operation between non-NArray and NArray.
  @overload coerce(other)
  @param [Object] other  numeric object.
  @return [Array]  NArray-casted [other,self]
*/
static VALUE
cumo_na_coerce(VALUE x, VALUE y)
{
    VALUE type;

    type = cumo_na_upcast(rb_obj_class(x), rb_obj_class(y));
    y = rb_funcall(type,cumo_id_cast,1,y);
    return rb_assoc_new(y , x);
}


/*
  Returns total byte size of NArray.
  @return [Integer] byte size.
 */
static VALUE
cumo_na_byte_size(VALUE self)
{
    VALUE velmsz;
    cumo_narray_t *na;

    CumoGetNArray(self,na);
    velmsz = rb_const_get(rb_obj_class(self), cumo_id_element_byte_size);
    if (FIXNUM_P(velmsz)) {
        return SIZET2NUM(NUM2SIZET(velmsz) * na->size);
    }
    return SIZET2NUM(ceil(NUM2DBL(velmsz) * na->size));
}

/*
  Returns byte size of one element of NArray.
  @return [Numeric] byte size.
 */
static VALUE
cumo_na_s_byte_size(VALUE type)
{
    return rb_const_get(type, cumo_id_element_byte_size);
}


/*
  Returns a new 1-D array initialized from binary raw data in a string.
  @overload from_binary(string,[shape])
  @param [String] string  Binary raw data.
  @param [Array] shape  array of integers representing array shape.
  @return [Cumo::NArray] NArray containing binary data.
 */
static VALUE
cumo_na_s_from_binary(int argc, VALUE *argv, VALUE type)
{
    size_t len, str_len, byte_size;
    size_t *shape;
    char *ptr;
    int   i, nd, narg;
    VALUE vstr, vshape, vna;
    VALUE velmsz;

    narg = rb_scan_args(argc,argv,"11",&vstr,&vshape);
    Check_Type(vstr,T_STRING);
    str_len = RSTRING_LEN(vstr);
    velmsz = rb_const_get(type, cumo_id_element_byte_size);
    if (narg==2) {
        switch(TYPE(vshape)) {
        case T_FIXNUM:
            nd = 1;
            len = NUM2SIZET(vshape);
            shape = &len;
            break;
        case T_ARRAY:
            nd = RARRAY_LEN(vshape);
            if (nd == 0 || nd > CUMO_NA_MAX_DIMENSION) {
                rb_raise(cumo_na_eDimensionError,"too long or empty shape (%d)", nd);
            }
            shape = ALLOCA_N(size_t,nd);
            len = 1;
            for (i=0; i<nd; ++i) {
                len *= shape[i] = NUM2SIZET(RARRAY_AREF(vshape,i));
            }
            break;
        default:
            rb_raise(rb_eArgError,"second argument must be size or shape");
        }
        if (FIXNUM_P(velmsz)) {
            byte_size = len * NUM2SIZET(velmsz);
        } else {
            byte_size = ceil(len * NUM2DBL(velmsz));
        }
        if (byte_size > str_len) {
            rb_raise(rb_eArgError, "specified size is too large");
        }
    } else {
        nd = 1;
        if (FIXNUM_P(velmsz)) {
            len = str_len / NUM2SIZET(velmsz);
            byte_size = len * NUM2SIZET(velmsz);
        } else {
            len = floor(str_len / NUM2DBL(velmsz));
            byte_size = str_len;
        }
        if (len == 0) {
            rb_raise(rb_eArgError, "string is empty or too short");
        }
        shape = ALLOCA_N(size_t,nd);
        shape[0] = len;
    }

    vna = cumo_na_new(type, nd, shape);
    ptr = cumo_na_get_pointer_for_write(vna);

    memcpy(ptr, RSTRING_PTR(vstr), byte_size);

    return vna;
}

/*
  Returns a new 1-D array initialized from binary raw data in a string.
  @overload store_binary(string,[offset])
  @param [String] string  Binary raw data.
  @param [Integer] (optional) offset  Byte offset in string.
  @return [Integer] stored length.
 */
static VALUE
cumo_na_store_binary(int argc, VALUE *argv, VALUE self)
{
    size_t size, str_len, byte_size, offset;
    char *ptr;
    int   narg;
    VALUE vstr, voffset;
    VALUE velmsz;
    cumo_narray_t *na;

    narg = rb_scan_args(argc,argv,"11",&vstr,&voffset);
    str_len = RSTRING_LEN(vstr);
    if (narg==2) {
        offset = NUM2SIZET(voffset);
        if (str_len < offset) {
            rb_raise(rb_eArgError, "offset is larger than string length");
        }
        str_len -= offset;
    } else {
        offset = 0;
    }

    CumoGetNArray(self,na);
    size = CUMO_NA_SIZE(na);
    velmsz = rb_const_get(rb_obj_class(self), cumo_id_element_byte_size);
    if (FIXNUM_P(velmsz)) {
        byte_size = size * NUM2SIZET(velmsz);
    } else {
        byte_size = ceil(size * NUM2DBL(velmsz));
    }
    if (byte_size > str_len) {
        rb_raise(rb_eArgError, "string is too short to store");
    }

    ptr = cumo_na_get_pointer_for_write(self);
    memcpy(ptr, RSTRING_PTR(vstr)+offset, byte_size);

    return SIZET2NUM(byte_size);
}

/*
  Returns string containing the raw data bytes in NArray.
  @overload to_binary()
  @return [String] String object containing binary raw data.
 */
static VALUE
cumo_na_to_binary(VALUE self)
{
    size_t len, offset=0;
    char *ptr;
    VALUE str;
    cumo_narray_t *na;

    CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("cumo_na_to_binary", "any");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    CumoGetNArray(self,na);
    if (na->type == CUMO_NARRAY_VIEW_T) {
        if (cumo_na_check_contiguous(self)==Qtrue) {
            offset = CUMO_NA_VIEW_OFFSET(na);
        } else {
            self = rb_funcall(self,cumo_id_dup,0);
        }
    }
    len = NUM2SIZET(cumo_na_byte_size(self));
    ptr = cumo_na_get_pointer_for_read(self);
    str = rb_usascii_str_new(ptr+offset,len);
    RB_GC_GUARD(self);
    return str;
}

/*
  Dump marshal data.
  @overload marshal_dump()
  @return [Array] Array containing marshal data.
 */
static VALUE
cumo_na_marshal_dump(VALUE self)
{
    VALUE a;

    CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("cumo_na_marshal_dump", "any");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    a = rb_ary_new();
    rb_ary_push(a, INT2FIX(1));     // version
    rb_ary_push(a, cumo_na_shape(self));
    rb_ary_push(a, INT2FIX(CUMO_NA_FLAG0(self)));
    if (rb_obj_class(self) == cumo_cRObject) {
        cumo_narray_t *na;
        VALUE *ptr;
        size_t offset=0;
        CumoGetNArray(self,na);
        if (na->type == CUMO_NARRAY_VIEW_T) {
            if (cumo_na_check_contiguous(self)==Qtrue) {
                offset = CUMO_NA_VIEW_OFFSET(na);
            } else {
                self = rb_funcall(self,cumo_id_dup,0);
            }
        }
        ptr = (VALUE*)cumo_na_get_pointer_for_read(self);
        rb_ary_push(a, rb_ary_new4(CUMO_NA_SIZE(na), ptr+offset));
    } else {
        rb_ary_push(a, cumo_na_to_binary(self));
    }
    RB_GC_GUARD(self);
    return a;
}

static VALUE cumo_na_inplace( VALUE self );
/*
  Load marshal data.
  @overload marshal_load(data)
  @param [Array] Array containing marshal data.
  @return [nil]
 */
static VALUE
cumo_na_marshal_load(VALUE self, VALUE a)
{
    VALUE v;

    if (TYPE(a) != T_ARRAY) {
        rb_raise(rb_eArgError,"marshal argument should be array");
    }
    if (RARRAY_LEN(a) != 4) {
        rb_raise(rb_eArgError,"marshal array size should be 4");
    }
    if (RARRAY_AREF(a,0) != INT2FIX(1)) {
        rb_raise(rb_eArgError,"NArray marshal version %d is not supported "
                 "(only version 1)", NUM2INT(RARRAY_AREF(a,0)));
    }
    cumo_na_initialize(self,RARRAY_AREF(a,1));
    CUMO_NA_FL0_SET(self,FIX2INT(RARRAY_AREF(a,2)));
    v = RARRAY_AREF(a,3);
    if (rb_obj_class(self) == cumo_cRObject) {
        cumo_narray_t *na;
        char *ptr;
        if (TYPE(v) != T_ARRAY) {
            rb_raise(rb_eArgError,"RObject content should be array");
        }
        CumoGetNArray(self,na);
        if (RARRAY_LEN(v) != (long)CUMO_NA_SIZE(na)) {
            rb_raise(rb_eArgError,"RObject content size mismatch");
        }
        ptr = cumo_na_get_pointer_for_write(self);
        memcpy(ptr, RARRAY_PTR(v), CUMO_NA_SIZE(na)*sizeof(VALUE));
    } else {
        cumo_na_store_binary(1,&v,self);
        if (CUMO_TEST_BYTE_SWAPPED(self)) {
            rb_funcall(cumo_na_inplace(self),cumo_id_to_host,0);
            CUMO_REVERSE_ENDIAN(self); // correct behavior??
        }
    }
    RB_GC_GUARD(a);
    return self;
}


/*
  Cast self to another NArray datatype.
  @overload cast_to(datatype)
  @param [Class] datatype NArray datatype.
  @return [Cumo::NArray]
 */
static VALUE
cumo_na_cast_to(VALUE obj, VALUE type)
{
    return rb_funcall(type, cumo_id_cast, 1, obj);
}



// reduce is dimension indicies to reduce in reduction kernel (in bits), e.g., for an array of shape:
// [2,3,4], 111b for sum(), 010b for sum(axis: 1), 110b for sum(axis: [1,2])
bool
cumo_na_test_reduce(VALUE reduce, int dim)
{
    size_t m;

    if (!RTEST(reduce))
        return 0;
    if (FIXNUM_P(reduce)) {
        m = FIX2LONG(reduce);
        if (m==0) return 1;
        return (m & (1u<<dim)) ? 1 : 0;
    } else {
        return (rb_funcall(reduce,cumo_id_bracket,1,INT2FIX(dim))==INT2FIX(1)) ?
            1 : 0 ;
    }
}


static VALUE
cumo_na_get_reduce_flag_from_narray(int naryc, VALUE *naryv, int *max_arg)
{
    int ndim, ndim0;
    int rowmaj;
    int i;
    size_t j;
    cumo_narray_t *na;
    VALUE reduce;

    if (naryc<1) {
        rb_raise(rb_eRuntimeError,"must be positive: naryc=%d", naryc);
    }
    CumoGetNArray(naryv[0],na);
    if (na->size==0) {
        rb_raise(cumo_na_eShapeError,"cannot reduce empty NArray");
    }
    reduce = na->reduce;
    ndim = ndim0 = na->ndim;
    if (max_arg) *max_arg = 0;
    rowmaj = CUMO_TEST_COLUMN_MAJOR(naryv[0]);
    for (i=0; i<naryc; i++) {
        CumoGetNArray(naryv[i],na);
        if (na->size==0) {
            rb_raise(cumo_na_eShapeError,"cannot reduce empty NArray");
        }
        if (CUMO_TEST_COLUMN_MAJOR(naryv[i]) != rowmaj) {
            rb_raise(cumo_na_eDimensionError,"dimension order is different");
        }
        if (na->ndim > ndim) { // maximum dimension
            ndim = na->ndim;
            if (max_arg) *max_arg = i;
        }
    }
    if (ndim != ndim0) {
        j = NUM2SIZET(reduce) << (ndim-ndim0);
        reduce = SIZET2NUM(j);
    }
    return reduce;
}


static VALUE
cumo_na_get_reduce_flag_from_axes(VALUE cumo_na_obj, VALUE axes)
{
    int i, r;
    int ndim, rowmaj;
    long narg;
    size_t j;
    size_t len;
    ssize_t beg, step;
    VALUE v;
    size_t m;
    VALUE reduce;
    cumo_narray_t *na;

    CumoGetNArray(cumo_na_obj,na);
    ndim = na->ndim;
    rowmaj = CUMO_TEST_COLUMN_MAJOR(cumo_na_obj);

    m = 0;
    reduce = Qnil;
    narg = RARRAY_LEN(axes);
    for (i=0; i<narg; i++) {
        v = RARRAY_AREF(axes,i);
        //printf("argv[%d]=",i);rb_p(v);
        if (TYPE(v)==T_FIXNUM) {
            beg = FIX2INT(v);
            if (beg<0) beg+=ndim;
            if (beg>=ndim || beg<0) {
                rb_raise(cumo_na_eDimensionError,"dimension is out of range");
            }
            len = 1;
            step = 0;
            //printf("beg=%d step=%d len=%d\n",beg,step,len);
        } else if (rb_obj_is_kind_of(v,rb_cRange) ||
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
                   rb_obj_is_kind_of(v,rb_cArithSeq)) {
#else
                   rb_obj_is_kind_of(v,rb_cEnumerator)) {
#endif
            cumo_na_step_array_index( v, ndim, &len, &beg, &step );
        } else {
            rb_raise(cumo_na_eDimensionError, "invalid dimension argument %s",
                     rb_obj_classname(v));
        }
        for (j=0; j<len; j++) {
            r = beg + step*j;
            if (rowmaj) {
                r = ndim-1-r;
            }
            if (reduce==Qnil) {
              if ( r < (ssize_t)sizeof(size_t) ) {
                    m |= ((size_t)1) << r;
                    continue;
                } else {
                    reduce = SIZET2NUM(m);
                }
            }
            v = rb_funcall( INT2FIX(1), cumo_id_shift_left, 1, INT2FIX(r) );
            reduce = rb_funcall( reduce, '|', 1, v );
        }
    }
    if (NIL_P(reduce)) reduce = SIZET2NUM(m);
    return reduce;
}


VALUE
cumo_na_reduce_options(VALUE axes, VALUE *opts, int naryc, VALUE *naryv,
                    cumo_ndfunc_t *ndf)
{
    int  max_arg;
    VALUE reduce;

    // option: axis
    if (opts[0] != Qundef && RTEST(opts[0])) {
        if (!NIL_P(axes)) {
            rb_raise(rb_eArgError,
              "cannot specify axis-arguments and axis-keyword simultaneously");
        }
        if (TYPE(opts[0]) == T_ARRAY) {
            axes = opts[0];
        } else {
            axes = rb_ary_new3(1,opts[0]);
        }
    }
    if (ndf) {
        // option: keepdims
        if (opts[1] != Qundef) {
            if (RTEST(opts[1]))
                ndf->flag |= CUMO_NDF_KEEP_DIM;
        }
    }

    reduce = cumo_na_get_reduce_flag_from_narray(naryc, naryv, &max_arg);

    if (NIL_P(axes)) return reduce;

    return cumo_na_get_reduce_flag_from_axes(naryv[max_arg], axes);
}


VALUE
cumo_na_reduce_dimension(int argc, VALUE *argv, int naryc, VALUE *naryv,
                      cumo_ndfunc_t *ndf, cumo_na_iter_func_t iter_nan)
{
    long narg;
    VALUE axes;
    VALUE kw_hash = Qnil;
    ID kw_table[3] = {cumo_id_axis,cumo_id_keepdims,cumo_id_nan};
    VALUE opts[3] = {Qundef,Qundef,Qundef};

    narg = rb_scan_args(argc, argv, "*:", &axes, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 3, opts);

    if (ndf) {
        // option: nan
        if (iter_nan && opts[2] != Qundef) {
            if (RTEST(opts[2]))
                ndf->func = iter_nan; // replace to nan-aware iterator function
        }
    }

    return cumo_na_reduce_options((narg)?axes:Qnil, opts, naryc, naryv, ndf);
}

/*
  Return true if column major.
*/
static VALUE cumo_na_column_major_p( VALUE self )
{
    if (CUMO_TEST_COLUMN_MAJOR(self))
	return Qtrue;
    else
	return Qfalse;
}

/*
  Return true if row major.
*/
static VALUE cumo_na_row_major_p( VALUE self )
{
    if (CUMO_TEST_ROW_MAJOR(self))
	return Qtrue;
    else
	return Qfalse;
}


/*
  Return true if byte swapped.
*/
static VALUE cumo_na_byte_swapped_p( VALUE self )
{
    if (CUMO_TEST_BYTE_SWAPPED(self))
      return Qtrue;
    return Qfalse;
}

/*
  Return true if not byte swapped.
*/
static VALUE cumo_na_host_order_p( VALUE self )
{
    if (CUMO_TEST_BYTE_SWAPPED(self))
      return Qfalse;
    return Qtrue;
}


/*
  Returns view of narray with inplace flagged.
  @return [Cumo::NArray] view of narray with inplace flag.
*/
static VALUE cumo_na_inplace( VALUE self )
{
    VALUE view = self;
    view = cumo_na_make_view(self);
    CUMO_SET_INPLACE(view);
    return view;
}

/*
  Set inplace flag to self.
  @return [Cumo::NArray] self
*/
static VALUE cumo_na_inplace_bang( VALUE self )
{
    CUMO_SET_INPLACE(self);
    return self;
}

/*
  Return true if inplace flagged.
*/
static VALUE cumo_na_inplace_p( VALUE self )
{
    if (CUMO_TEST_INPLACE(self))
        return Qtrue;
    else
        return Qfalse;
}

/*
  Unset inplace flag to self.
  @return [Cumo::NArray] self
*/
static VALUE cumo_na_out_of_place_bang( VALUE self )
{
    CUMO_UNCUMO_SET_INPLACE(self);
    return self;
}

int cumo_na_debug_flag=0;

static VALUE cumo_na_debug_set(VALUE mod, VALUE flag)
{
    cumo_na_debug_flag = RTEST(flag);
    return Qnil;
}

static double cumo_na_profile_value=0;

static VALUE cumo_na_profile(VALUE mod)
{
    return rb_float_new(cumo_na_profile_value);
}

static VALUE cumo_na_profile_set(VALUE mod, VALUE val)
{
    cumo_na_profile_value = NUM2DBL(val);
    return val;
}


/*
  Returns the number of rows used for NArray#inspect
  @overload inspect_rows
  @return [Integer or nil]  the number of rows.
*/
static VALUE cumo_na_inspect_rows(VALUE mod)
{
    if (cumo_na_inspect_rows_ > 0) {
        return INT2NUM(cumo_na_inspect_rows_);
    } else {
        return Qnil;
    }
}

/*
  Set the number of rows used for NArray#inspect
  @overload inspect_rows=(rows)
  @param [Integer or nil] rows  the number of rows
  @return [nil]
*/
static VALUE cumo_na_inspect_rows_set(VALUE mod, VALUE num)
{
    if (RTEST(num)) {
        cumo_na_inspect_rows_ = NUM2INT(num);
    } else {
        cumo_na_inspect_rows_ = 0;
    }
    return Qnil;
}

/*
  Returns the number of cols used for NArray#inspect
  @overload inspect_cols
  @return [Integer or nil]  the number of cols.
*/
static VALUE cumo_na_inspect_cols(VALUE mod)
{
    if (cumo_na_inspect_cols_ > 0) {
        return INT2NUM(cumo_na_inspect_cols_);
    } else {
        return Qnil;
    }
}

/*
  Set the number of cols used for NArray#inspect
  @overload inspect_cols=(cols)
  @param [Integer or nil] cols  the number of cols
  @return [nil]
*/
static VALUE cumo_na_inspect_cols_set(VALUE mod, VALUE num)
{
    if (RTEST(num)) {
        cumo_na_inspect_cols_ = NUM2INT(num);
    } else {
        cumo_na_inspect_cols_ = 0;
    }
    return Qnil;
}


/*
  Equality of self and other in view of numerical array.
  i.e., both arrays have same shape and corresponding elements are equal.
  @overload == other
  @param [Object] other
  @return [Boolean] true if self and other is equal.
*/
static VALUE
cumo_na_equal(VALUE self, volatile VALUE other)
{
    volatile VALUE vbool;
    cumo_narray_t *na1, *na2;
    int i;

    CumoGetNArray(self,na1);

    if (!rb_obj_is_kind_of(other,cNArray)) {
        other = rb_funcall(rb_obj_class(self), cumo_id_cast, 1, other);
    }

    CumoGetNArray(other,na2);
    if (na1->ndim != na2->ndim) {
        return Qfalse;
    }
    for (i=0; i<na1->ndim; i++) {
        if (na1->shape[i] != na2->shape[i]) {
            return Qfalse;
        }
    }
    vbool = rb_funcall(self, cumo_id_eq, 1, other);
    return (rb_funcall(vbool, cumo_id_count_false_cpu, 0)==INT2FIX(0)) ? Qtrue : Qfalse;
}

/*
  Free data memory explicitly without waiting GC.

  @return [Boolean] true if free
*/
VALUE
cumo_na_free_data(VALUE self)
{
    cumo_narray_t *na;
    CumoGetNArray(self, na);

    if (na->type == CUMO_NARRAY_DATA_T) {
        void *ptr = CUMO_NA_DATA_PTR(na);
        if (ptr != NULL) {
            if (cumo_cuda_runtime_is_device_memory(ptr)) {
                cumo_cuda_runtime_free(ptr);
            } else {
                xfree(ptr);
            }
            CUMO_NA_DATA_PTR(na) = NULL;
            return Qtrue;
        }
    }

    return Qfalse;
}

/* initialization of NArray Class */
void
Init_cumo_narray()
{
    mCumo = rb_define_module("Cumo");

    /*
      Document-class: Cumo::NArray

      Cumo::NArray is the abstract super class for
      Numerical N-dimensional Array in the Ruby/Cumo module.
      Use Typed Subclasses of NArray (Cumo::DFloat, Int32, etc)
      to create data array instances.
    */
    cNArray = rb_define_class_under(mCumo, "NArray", rb_cObject);

#ifndef HAVE_RB_CCOMPLEX
    rb_require("complex");
    rb_cComplex = rb_const_get(rb_cObject, rb_intern("Complex"));
#endif
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
    rb_cArithSeq = rb_path2class("Enumerator::ArithmeticSequence");
#endif

    rb_define_const(cNArray, "VERSION", rb_str_new2(CUMO_VERSION));

    cumo_na_eCastError = rb_define_class_under(cNArray, "CastError", rb_eStandardError);
    cumo_na_eShapeError = rb_define_class_under(cNArray, "ShapeError", rb_eStandardError);
    cumo_na_eOperationError = rb_define_class_under(cNArray, "OperationError", rb_eStandardError);
    cumo_na_eDimensionError = rb_define_class_under(cNArray, "DimensionError", rb_eStandardError);
    cumo_na_eValueError = rb_define_class_under(cNArray, "ValueError", rb_eStandardError);

    rb_define_singleton_method(cNArray, "debug=", cumo_na_debug_set, 1);
    rb_define_singleton_method(cNArray, "profile", cumo_na_profile, 0);
    rb_define_singleton_method(cNArray, "profile=", cumo_na_profile_set, 1);

    rb_define_singleton_method(cNArray, "inspect_rows", cumo_na_inspect_rows, 0);
    rb_define_singleton_method(cNArray, "inspect_rows=", cumo_na_inspect_rows_set, 1);
    rb_define_singleton_method(cNArray, "inspect_cols", cumo_na_inspect_cols, 0);
    rb_define_singleton_method(cNArray, "inspect_cols=", cumo_na_inspect_cols_set, 1);

    /* Ruby allocation framework  */
    rb_undef_alloc_func(cNArray);
    rb_define_method(cNArray, "initialize", cumo_na_initialize, -2);
    rb_define_method(cNArray, "initialize_copy", cumo_na_initialize_copy, 1);

    rb_define_method(cNArray, "free", cumo_na_free_data, 0);

    rb_define_singleton_method(cNArray, "zeros", cumo_na_s_zeros, -1);
    rb_define_singleton_method(cNArray, "ones", cumo_na_s_ones, -1);
    rb_define_singleton_method(cNArray, "linspace", cumo_na_s_linspace, -1);
    rb_define_singleton_method(cNArray, "logspace", cumo_na_s_logspace, -1);
    rb_define_singleton_method(cNArray, "eye", cumo_na_s_eye, -1);

    rb_define_method(cNArray, "size", cumo_na_size, 0);
    rb_define_alias (cNArray, "length","size");
    rb_define_alias (cNArray, "total","size");
    rb_define_method(cNArray, "shape", cumo_na_shape, 0);
    rb_define_method(cNArray, "ndim", cumo_na_ndim,0);
    rb_define_alias (cNArray, "rank","ndim");
    rb_define_method(cNArray, "empty?", cumo_na_empty_p, 0);

    rb_define_method(cNArray, "debug_info", cumo_na_debug_info, 0);

    rb_define_method(cNArray, "contiguous?", cumo_na_check_contiguous, 0);
    rb_define_method(cNArray, "fortran_contiguous?", cumo_na_check_fortran_contiguous, 0);

    rb_define_method(cNArray, "view", cumo_na_make_view, 0);
    rb_define_method(cNArray, "expand_dims", cumo_na_expand_dims, 1);
    rb_define_method(cNArray, "reverse", cumo_na_reverse, -1);

    rb_define_singleton_method(cNArray, "upcast", cumo_na_upcast, 1);
    rb_define_singleton_method(cNArray, "byte_size", cumo_na_s_byte_size, 0);

    rb_define_singleton_method(cNArray, "from_binary", cumo_na_s_from_binary, -1);
    rb_define_alias (rb_singleton_class(cNArray), "from_string", "from_binary");
    rb_define_method(cNArray, "store_binary",  cumo_na_store_binary, -1);
    rb_define_method(cNArray, "to_binary",  cumo_na_to_binary, 0);
    rb_define_alias (cNArray, "to_string", "to_binary");
    rb_define_method(cNArray, "marshal_dump",  cumo_na_marshal_dump, 0);
    rb_define_method(cNArray, "marshal_load",  cumo_na_marshal_load, 1);

    rb_define_method(cNArray, "byte_size",  cumo_na_byte_size, 0);

    rb_define_method(cNArray, "cast_to", cumo_na_cast_to, 1);

    rb_define_method(cNArray, "coerce", cumo_na_coerce, 1);

    rb_define_method(cNArray, "column_major?", cumo_na_column_major_p, 0);
    rb_define_method(cNArray, "row_major?", cumo_na_row_major_p, 0);
    rb_define_method(cNArray, "byte_swapped?", cumo_na_byte_swapped_p, 0);
    rb_define_method(cNArray, "host_order?", cumo_na_host_order_p, 0);

    rb_define_method(cNArray, "inplace", cumo_na_inplace, 0);
    rb_define_method(cNArray, "inplace?", cumo_na_inplace_p, 0);
    rb_define_method(cNArray, "inplace!", cumo_na_inplace_bang, 0);
    rb_define_method(cNArray, "out_of_place!", cumo_na_out_of_place_bang, 0);
    rb_define_alias (cNArray, "not_inplace!", "out_of_place!");

    rb_define_method(cNArray, "==", cumo_na_equal, 1);

    cumo_id_allocate = rb_intern("allocate");
    cumo_id_contiguous_stride = rb_intern("CONTIGUOUS_STRIDE");
    //cumo_id_element_bit_size = rb_intern("ELEMENT_BIT_SIZE");
    cumo_id_element_byte_size = rb_intern("ELEMENT_BYTE_SIZE");

    cumo_id_fill            = rb_intern("fill");
    cumo_id_seq             = rb_intern("seq");
    cumo_id_logseq          = rb_intern("logseq");
    cumo_id_eye             = rb_intern("eye");
    cumo_id_UPCAST          = rb_intern("UPCAST");
    cumo_id_cast            = rb_intern("cast");
    cumo_id_dup             = rb_intern("dup");
    cumo_id_to_host         = rb_intern("to_host");
    cumo_id_bracket         = rb_intern("[]");
    cumo_id_shift_left      = rb_intern("<<");
    cumo_id_eq              = rb_intern("eq");
    cumo_id_count_false     = rb_intern("count_false");
    cumo_id_count_false_cpu = rb_intern("count_false_cpu");
    cumo_id_axis            = rb_intern("axis");
    cumo_id_nan             = rb_intern("nan");
    cumo_id_keepdims        = rb_intern("keepdims");

    cumo_sym_reduce   = ID2SYM(rb_intern("reduce"));
    cumo_sym_option   = ID2SYM(rb_intern("option"));
    cumo_sym_loop_opt = ID2SYM(rb_intern("loop_opt"));
    cumo_sym_init     = ID2SYM(rb_intern("init"));
}
