#include <ruby.h>
#include "cumo.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"
#include "cumo/narray.h"
#include "cumo/template.h"

static VALUE cumo_sym_mulsum;
static ID cumo_id_mulsum;
static ID cumo_id_respond_to_p;
static ID cumo_id_store;
static ID cumo_id_swap_byte;

// ---------------------------------------------------------------------

#define LOOP_UNARY_PTR(lp,proc)                    \
{                                                  \
    size_t  i;                                     \
    ssize_t s1, s2;                                \
    char   *p1, *p2;                               \
    size_t *idx1, *idx2;                           \
    CUMO_INIT_COUNTER(lp, i);                           \
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);             \
    CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);             \
    if (idx1) {                                    \
        if (idx2) {                                \
            for (; i--;) {                         \
                proc((p1+*idx1), (p2+*idx2));      \
                idx1++;                            \
                idx2++;                            \
            }                                      \
        } else {                                   \
            for (; i--;) {                         \
                proc((p1+*idx1), p2);              \
                idx1++;                            \
                p2 += s2;                          \
            }                                      \
        }                                          \
    } else {                                       \
        if (idx2) {                                \
            for (; i--;) {                         \
                proc(p1, (p1+*idx2));              \
                p1 += s1;                          \
                idx2++;                            \
            }                                      \
        } else {                                   \
            for (; i--;) {                         \
                proc(p1, p2);                      \
                p1 += s1;                          \
                p2 += s2;                          \
            }                                      \
        }                                          \
    }                                              \
}

void cumo_iter_copy_bytes_kernel_launch(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t *idx1, size_t *idx2, size_t n, int elmsz);
// #define m_memcpy(src,dst) memcpy(dst,src,e)

static void
iter_copy_bytes(cumo_na_loop_t *const lp)
{
    size_t  n;
    ssize_t s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    cumo_iter_copy_bytes_kernel_launch(p1, p2, s1, s2, idx1, idx2, n, lp->args[0].elmsz);
    // size_t e;
    // e = lp->args[0].elmsz;
    // LOOP_UNARY_PTR(lp,m_memcpy);
}

VALUE
cumo_na_copy(VALUE self)
{
    VALUE v;
    cumo_ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{INT2FIX(0),0}};
    cumo_ndfunc_t ndf = { iter_copy_bytes, CUMO_FULL_LOOP, 1, 1, ain, aout };

    v = cumo_na_ndloop(&ndf, 1, self);
    return v;
}

VALUE
cumo_na_store(VALUE self, VALUE src)
{
    return rb_funcall(self,cumo_id_store,1,src);
}

// ---------------------------------------------------------------------

#define m_swap_byte(q1,q2)       \
    {                            \
        size_t j;                \
        memcpy(b1,q1,e);         \
        for (j=0; j<e; j++) {    \
            b2[e-1-j] = b1[j];   \
        }                        \
        memcpy(q2,b2,e);         \
    }

static void
iter_swap_byte(cumo_na_loop_t *const lp)
{
    char   *b1, *b2;
    size_t  e;

    e = lp->args[0].elmsz;
    b1 = ALLOCA_N(char, e);
    b2 = ALLOCA_N(char, e);
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("iter_swap_bytes", "any");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    LOOP_UNARY_PTR(lp,m_swap_byte);
}

static VALUE
cumo_na_swap_byte(VALUE self)
{
    VALUE v;
    cumo_ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{INT2FIX(0),0}};
    cumo_ndfunc_t ndf = { iter_swap_byte, CUMO_FULL_LOOP|CUMO_NDF_ACCEPT_BYTESWAP,
                     1, 1, ain, aout };

    v = cumo_na_ndloop(&ndf, 1, self);
    if (self!=v) {
        cumo_na_copy_flags(self, v);
    }
    CUMO_REVERSE_ENDIAN(v);
    return v;
}


static VALUE
cumo_na_to_network(VALUE self)
{
    if (CUMO_TEST_BIG_ENDIAN(self)) {
        return self;
    }
    return rb_funcall(self, cumo_id_swap_byte, 0);
}

static VALUE
cumo_na_to_vacs(VALUE self)
{
    if (CUMO_TEST_LITTLE_ENDIAN(self)) {
        return self;
    }
    return rb_funcall(self, cumo_id_swap_byte, 0);
}

static VALUE
cumo_na_to_host(VALUE self)
{
    if (CUMO_TEST_HOST_ORDER(self)) {
        return self;
    }
    return rb_funcall(self, cumo_id_swap_byte, 0);
}

static VALUE
cumo_na_to_swapped(VALUE self)
{
    if (CUMO_TEST_BYTE_SWAPPED(self)) {
        return self;
    }
    return rb_funcall(self, cumo_id_swap_byte, 0);
}


//----------------------------------------------------------------------

static inline int
check_axis(int axis, int ndim)
{
    if (axis < -ndim || axis >= ndim) {
        rb_raise(cumo_na_eDimensionError,"invalid axis (%d for %d-dimension)",
                 axis, ndim);
    }
    if (axis < 0) {
        axis += ndim;
    }
    return axis;
}

/*
  Interchange two axes.
  @overload  swapaxes(axis1,axis2)
  @param [Integer] axis1
  @param [Integer] axis2
  @return [Cumo::NArray]  view of NArray.
  @example
    x = Cumo::Int32[[1,2,3]]

    p x.swapaxes(0,1)
    # Cumo::Int32(view)#shape=[3,1]
    # [[1],
    #  [2],
    #  [3]]

    p x = Cumo::Int32[[[0,1],[2,3]],[[4,5],[6,7]]]
    # Cumo::Int32#shape=[2,2,2]
    # [[[0, 1],
    #   [2, 3]],
    #  [[4, 5],
    #   [6, 7]]]

    p x.swapaxes(0,2)
    # Cumo::Int32(view)#shape=[2,2,2]
    # [[[0, 4],
    #   [2, 6]],
    #  [[1, 5],
    #   [3, 7]]]
*/
static VALUE
cumo_na_swapaxes(VALUE self, VALUE a1, VALUE a2)
{
    int  i, j, ndim;
    size_t tmp_shape;
    cumo_stridx_t tmp_stridx;
    cumo_narray_view_t *na;
    volatile VALUE view;

    view = cumo_na_make_view(self);
    CumoGetNArrayView(view,na);

    ndim = na->base.ndim;
    i = check_axis(NUM2INT(a1), ndim);
    j = check_axis(NUM2INT(a2), ndim);

    tmp_shape = na->base.shape[i];
    tmp_stridx = na->stridx[i];
    na->base.shape[i] = na->base.shape[j];
    na->stridx[i] = na->stridx[j];
    na->base.shape[j] = tmp_shape;
    na->stridx[j] = tmp_stridx;

    return view;
}

static VALUE
cumo_na_transpose_map(VALUE self, int *map)
{
    int  i, ndim;
    size_t *shape;
    cumo_stridx_t *stridx;
    cumo_narray_view_t *na;
    volatile VALUE view;

    view = cumo_na_make_view(self);
    CumoGetNArrayView(view,na);

    ndim = na->base.ndim;
    shape = ALLOCA_N(size_t,ndim);
    stridx = ALLOCA_N(cumo_stridx_t,ndim);

    for (i=0; i<ndim; i++) {
	shape[i] = na->base.shape[i];
	stridx[i] = na->stridx[i];
    }
    for (i=0; i<ndim; i++) {
	na->base.shape[i] = shape[map[i]];
	na->stridx[i] = stridx[map[i]];
    }
    return view;
}


#define SWAP(a,b,tmp) {tmp=a;a=b;b=tmp;}

static VALUE
cumo_na_transpose(int argc, VALUE *argv, VALUE self)
{
    int ndim, *map, *permute;
    int i, d;
    bool is_positive, is_negative;
    cumo_narray_t *na1;

    CumoGetNArray(self,na1);
    ndim = na1->ndim;
    if (ndim < 2) {
        if (argc > 0) {
            rb_raise(rb_eArgError, "unnecessary argument for 1-d array");
        }
        return cumo_na_make_view(self);
    }
    map = ALLOCA_N(int,ndim);
    if (argc == 0) {
        for (i=0; i < ndim; i++) {
            map[i] = ndim-1-i;
        }
        return cumo_na_transpose_map(self,map);
    }
    // with argument
    if (argc > ndim) {
        rb_raise(rb_eArgError, "more arguments than ndim");
    }
    for (i=0; i < ndim; i++) {
        map[i] = i;
    }
    permute = ALLOCA_N(int,argc);
    for (i=0; i < argc; i++) {
        permute[i] = 0;
    }
    is_positive = is_negative = 0;
    for (i=0; i < argc; i++) {
	if (TYPE(argv[i]) != T_FIXNUM) {
            rb_raise(rb_eArgError, "invalid argument");
        }
        d = FIX2INT(argv[i]);
        if (d >= 0) {
            if (d >= argc) {
                rb_raise(rb_eArgError, "out of dimension range");
            }
            if (is_negative) {
                rb_raise(rb_eArgError, "dimension must be non-negative only or negative only");
            }
            if (permute[d]) {
                rb_raise(rb_eArgError, "not permutation");
            }
            map[i] = d;
            permute[d] = 1;
            is_positive = 1;
        } else {
            if (d < -argc) {
                rb_raise(rb_eArgError, "out of dimension range");
            }
            if (is_positive) {
                rb_raise(rb_eArgError, "dimension must be non-negative only or negative only");
            }
            if (permute[argc+d]) {
                rb_raise(rb_eArgError, "not permutation");
            }
            map[ndim-argc+i] = ndim+d;
            permute[argc+d] = 1;
            is_negative = 1;
        }
    }
    return cumo_na_transpose_map(self,map);
}

//----------------------------------------------------------------------

static void
cumo_na_check_reshape(int argc, VALUE *argv, VALUE self, size_t *shape)
{
    int    i, unfixed=-1;
    size_t total=1;
    cumo_narray_t *na;

    if (argc == 0) {
        rb_raise(rb_eArgError, "No argrument");
    }
    CumoGetNArray(self,na);
    if (CUMO_NA_SIZE(na) == 0) {
        rb_raise(rb_eRuntimeError, "cannot reshape empty array");
    }

    /* get shape from argument */
    for (i=0; i<argc; ++i) {
        switch(TYPE(argv[i])) {
        case T_FIXNUM:
            total *= shape[i] = NUM2INT(argv[i]);
            break;
        case T_NIL:
        case T_TRUE:
            if (unfixed >= 0) {
                rb_raise(rb_eArgError,"multiple unfixed dimension");
            }
            unfixed = i;
            break;
        default:
            rb_raise(rb_eArgError,"illegal type");
        }
    }

    if (unfixed>=0) {
        if (CUMO_NA_SIZE(na) % total != 0) {
            rb_raise(rb_eArgError, "Total size size must be divisor");
        }
        shape[unfixed] = CUMO_NA_SIZE(na) / total;
    }
    else if (total !=  CUMO_NA_SIZE(na)) {
        rb_raise(rb_eArgError, "Total size must be same");
    }
}

/*
  Change the shape of self NArray without coping.
  Raise exception if self is non-contiguous.

  @overload  reshape!(size0,size1,...)
  @param sizeN [Integer] new shape
  @return [Cumo::NArray] return self.
  @example
*/
static VALUE
cumo_na_reshape_bang(int argc, VALUE *argv, VALUE self)
{
    size_t *shape;
    cumo_narray_t *na;
    cumo_narray_view_t *na2;
    ssize_t stride;
    cumo_stridx_t *stridx;
    int i;

    if (cumo_na_check_contiguous(self)==Qfalse) {
        rb_raise(rb_eStandardError, "cannot change shape of non-contiguous NArray");
    }
    shape = ALLOCA_N(size_t, argc);
    cumo_na_check_reshape(argc, argv, self, shape);

    CumoGetNArray(self, na);
    if (na->type == CUMO_NARRAY_VIEW_T) {
        CumoGetNArrayView(self, na2);
        if (na->ndim < argc) {
            stridx = ALLOC_N(cumo_stridx_t,argc);
        } else {
            stridx = na2->stridx;
        }
        stride = CUMO_SDX_GET_STRIDE(na2->stridx[na->ndim-1]);
        for (i=argc; i--;) {
            CUMO_SDX_SET_STRIDE(stridx[i],stride);
            stride *= shape[i];
        }
        if (stridx != na2->stridx) {
            xfree(na2->stridx);
            na2->stridx = stridx;
        }
    }
    cumo_na_setup_shape(na, argc, shape);
    return self;
}

/*
  Copy and change the shape of NArray.
  Returns a copied NArray.

  @overload  reshape(size0,size1,...)
  @param sizeN [Integer] new shape
  @return [Cumo::NArray] return self.
  @example
*/
static VALUE
cumo_na_reshape(int argc, VALUE *argv, VALUE self)
{
    size_t *shape;
    cumo_narray_t *na;
    VALUE    copy;

    shape = ALLOCA_N(size_t, argc);
    cumo_na_check_reshape(argc, argv, self, shape);

    copy = rb_funcall(self, rb_intern("dup"), 0);
    CumoGetNArray(copy, na);
    cumo_na_setup_shape(na, argc, shape);
    return copy;
}

//----------------------------------------------------------------------

VALUE
cumo_na_flatten_dim(VALUE self, int sd)
{
    int i, nd, fd;
    size_t j;
    size_t *c, *pos, *idx1, *idx2;
    size_t stride;
    size_t  *shape, size;
    cumo_stridx_t sdx;
    cumo_narray_t *na;
    cumo_narray_view_t *na1, *na2;
    volatile VALUE view;

    CumoGetNArray(self,na);
    nd = na->ndim;

     if (nd==0 || na->size==0) {
        return cumo_na_make_view(self);
    }
    if (sd<0 || sd>=nd) {
        rb_bug("cumo_na_flaten_dim: start_dim (%d) out of range",sd);
    }

    // new shape
    shape = ALLOCA_N(size_t,sd+1);
    for (i=0; i<sd; i++) {
        shape[i] = na->shape[i];
    }
    size = 1;
    for (i=sd; i<nd; i++) {
        size *= na->shape[i];
    }
    shape[sd] = size;

    // new object
    view = cumo_na_s_allocate_view(rb_obj_class(self));
    cumo_na_copy_flags(self, view);
    CumoGetNArrayView(view, na2);

    // new stride
    cumo_na_setup_shape((cumo_narray_t*)na2, sd+1, shape);
    na2->stridx = ALLOC_N(cumo_stridx_t,sd+1);

    switch(na->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        stride = cumo_na_element_stride(self);
        for (i=sd+1; i--; ) {
            CUMO_SDX_SET_STRIDE(na2->stridx[i],stride);
            stride *= shape[i];
        }
        na2->offset = 0;
        na2->data = self;
        break;
    case CUMO_NARRAY_VIEW_T:
        CumoGetNArrayView(self, na1);
        na2->data = na1->data;
        na2->offset = na1->offset;
        for (i=0; i<sd; i++) {
            if (CUMO_SDX_IS_INDEX(na1->stridx[i])) {
                idx1 = CUMO_SDX_GET_INDEX(na1->stridx[i]);
                // idx2 = ALLOC_N(size_t, shape[i]);
                // for (j=0; j<shape[i]; j++) {
                //     idx2[j] = idx1[j];
                // }
                idx2 = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*shape[i]);
                cumo_cuda_runtime_check_status(cudaMemcpyAsync(idx2,idx1,sizeof(size_t)*shape[i],cudaMemcpyDeviceToDevice,0));
                CUMO_SDX_SET_INDEX(na2->stridx[i],idx2);
            } else {
                na2->stridx[i] = na1->stridx[i];
            }
        }
        // flat dimenion == last dimension
        if (RTEST(cumo_na_check_ladder(self,sd))) {
            na2->stridx[sd] = na1->stridx[nd-1];
        } else {
            // set index
            // idx2 = ALLOC_N(size_t, shape[sd]);
            idx2 = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*shape[sd]);
            CUMO_SDX_SET_INDEX(na2->stridx[sd],idx2);
            // init for md-loop
            fd = nd-sd;
            c = ALLOC_N(size_t, fd);
            for (i=0; i<fd; i++) c[i]=0;
            pos = ALLOC_N(size_t, fd+1);
            pos[0] = 0;
            // md-loop
            CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("na_flatten_dim", "any");
            cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
            for (i=j=0;;) {
                for (; i<fd; i++) {
                    sdx = na1->stridx[i+sd];
                    if (CUMO_SDX_IS_INDEX(sdx)) {
                        pos[i+1] = pos[i] + CUMO_SDX_GET_INDEX(sdx)[c[i]];
                    } else {
                        pos[i+1] = pos[i] + CUMO_SDX_GET_STRIDE(sdx)*c[i];
                    }
                }
                idx2[j++] = pos[i];
                for (;;) {
                    if (i==0) goto loop_end;
                    i--;
                    c[i]++;
                    if (c[i] < na1->base.shape[i+sd]) break;
                    c[i] = 0;
                }
            }
        loop_end:
            xfree(pos);
            xfree(c);
        }
        break;
    }
    return view;
}

VALUE
cumo_na_flatten(VALUE self)
{
    return cumo_na_flatten_dim(self,0);
}

//----------------------------------------------------------------------

#define MIN(a,b) (((a)<(b))?(a):(b))

void cumo_na_diagonal_index_index_kernel_launch(size_t *idx, size_t *idx0, size_t *idx1, size_t k0, size_t k1, uint64_t n);
void cumo_na_diagonal_index_stride_kernel_launch(size_t *idx, size_t *idx0, ssize_t s1, size_t k0, size_t k1, uint64_t n);
void cumo_na_diagonal_stride_index_kernel_launch(size_t *idx, ssize_t s0, size_t *idx1, size_t k0, size_t k1, uint64_t n);

/*
  Returns a diagonal view of NArray
  @overload  diagonal([offset,axes])
  @param [Integer] offset  Diagonal offset from the main diagonal.
    The default is 0. k>0 for diagonals above the main diagonal,
    and k<0 for diagonals below the main diagonal.
  @param [Array] axes  Array of axes to be used as the 2-d sub-arrays
    from which the diagonals should be taken. Defaults to last-two
    axes ([-2,-1]).
  @return [Cumo::NArray]  diagonal view of NArray.
  @example
    a = Cumo::DFloat.new(4,5).seq
    => Cumo::DFloat#shape=[4,5]
    [[0, 1, 2, 3, 4],
     [5, 6, 7, 8, 9],
     [10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19]]
    b = a.diagonal(1)
    => Cumo::DFloat(view)#shape=[4]
    [1, 7, 13, 19]
    b.store(0)
    a
    => Cumo::DFloat#shape=[4,5]
    [[0, 0, 2, 3, 4],
     [5, 6, 0, 8, 9],
     [10, 11, 12, 0, 14],
     [15, 16, 17, 18, 0]]
    b.store([1,2,3,4])
    a
    => Cumo::DFloat#shape=[4,5]
    [[0, 1, 2, 3, 4],
     [5, 6, 2, 8, 9],
     [10, 11, 12, 3, 14],
     [15, 16, 17, 18, 4]]
 */
static VALUE
cumo_na_diagonal(int argc, VALUE *argv, VALUE self)
{
    int  i, k, nd;
    size_t *idx0, *idx1, *diag_idx;
    size_t *shape;
    size_t  diag_size;
    ssize_t stride, stride0, stride1;
    cumo_narray_t *na;
    cumo_narray_view_t *na1, *na2;
    VALUE view;
    VALUE vofs=0, vaxes=0;
    ssize_t kofs;
    size_t k0, k1;
    int ax[2];

    // check arguments
    if (argc>2) {
        rb_raise(rb_eArgError,"too many arguments (%d for 0..2)",argc);
    }

    for (i=0; i<argc; i++) {
        switch(TYPE(argv[i])) {
        case T_FIXNUM:
            if (vofs) {
                rb_raise(rb_eArgError,"offset is given twice");
            }
            vofs = argv[i];
            break;
        case T_ARRAY:
            if (vaxes) {
                rb_raise(rb_eArgError,"axes-array is given twice");
            }
            vaxes = argv[i];
            break;
        }
    }

    if (vofs) {
        kofs = NUM2SSIZET(vofs);
    } else {
        kofs = 0;
    }

    CumoGetNArray(self,na);
    nd = na->ndim;
    if (nd < 2) {
        rb_raise(cumo_na_eDimensionError,"less than 2-d array");
    }

    if (vaxes) {
        if (RARRAY_LEN(vaxes) != 2) {
            rb_raise(rb_eArgError,"axes must be 2-element array");
        }
        ax[0] = NUM2INT(RARRAY_AREF(vaxes,0));
        ax[1] = NUM2INT(RARRAY_AREF(vaxes,1));
        if (ax[0]<-nd || ax[0]>=nd || ax[1]<-nd || ax[1]>=nd) {
            rb_raise(rb_eArgError,"axis out of range:[%d,%d]",ax[0],ax[1]);
        }
        if (ax[0]<0) {ax[0] += nd;}
        if (ax[1]<0) {ax[1] += nd;}
        if (ax[0]==ax[1]) {
            rb_raise(rb_eArgError,"same axes:[%d,%d]",ax[0],ax[1]);
        }
    } else {
        ax[0] = nd-2;
        ax[1] = nd-1;
    }

    // Diagonal offset from the main diagonal.
    if (kofs >= 0) {
        k0 = 0;
        k1 = kofs;
        if (k1 >= na->shape[ax[1]]) {
            rb_raise(rb_eArgError,"invalid diagonal offset(%"SZF"d) for "
                     "last dimension size(%"SZF"d)",kofs,na->shape[ax[1]]);
        }
    } else {
        k0 = -kofs;
        k1 = 0;
        if (k0 >= na->shape[ax[0]]) {
            rb_raise(rb_eArgError,"invalid diagonal offset(=%"SZF"d) for "
                     "last-1 dimension size(%"SZF"d)",kofs,na->shape[ax[0]]);
        }
    }

    diag_size = MIN(na->shape[ax[0]]-k0,na->shape[ax[1]]-k1);

    // new shape
    shape = ALLOCA_N(size_t,nd-1);
    for (i=k=0; i<nd; i++) {
        if (i != ax[0] && i != ax[1]) {
            shape[k++] = na->shape[i];
        }
    }
    shape[k] = diag_size;

    // new object
    view = cumo_na_s_allocate_view(rb_obj_class(self));
    cumo_na_copy_flags(self, view);
    CumoGetNArrayView(view, na2);

    // new stride
    cumo_na_setup_shape((cumo_narray_t*)na2, nd-1, shape);
    na2->stridx = ALLOC_N(cumo_stridx_t, nd-1);

    switch(na->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        na2->offset = 0;
        na2->data = self;
        stride = stride0 = stride1 = cumo_na_element_stride(self);
        for (i=nd,k=nd-2; i--; ) {
            if (i==ax[1]) {
                stride1 = stride;
                if (kofs > 0) {
                    na2->offset = kofs*stride;
                }
            } else if (i==ax[0]) {
                stride0 = stride;
                if (kofs < 0) {
                    na2->offset = (-kofs)*stride;
                }
            } else {
                CUMO_SDX_SET_STRIDE(na2->stridx[--k],stride);
            }
            stride *= na->shape[i];
        }
        CUMO_SDX_SET_STRIDE(na2->stridx[nd-2],stride0+stride1);
        break;

    case CUMO_NARRAY_VIEW_T:
        CumoGetNArrayView(self, na1);
        na2->data = na1->data;
        na2->offset = na1->offset;
        for (i=k=0; i<nd; i++) {
            if (i != ax[0] && i != ax[1]) {
                if (CUMO_SDX_IS_INDEX(na1->stridx[i])) {
                    idx0 = CUMO_SDX_GET_INDEX(na1->stridx[i]);
                    // idx1 = ALLOC_N(size_t, na->shape[i]);
                    // for (j=0; j<na->shape[i]; j++) {
                    //     idx1[j] = idx0[j];
                    // }
                    idx1 = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*na->shape[i]);
                    cumo_cuda_runtime_check_status(cudaMemcpyAsync(idx1,idx0,sizeof(size_t)*na->shape[i],cudaMemcpyDeviceToDevice,0));
                    CUMO_SDX_SET_INDEX(na2->stridx[k],idx1);
                } else {
                    na2->stridx[k] = na1->stridx[i];
                }
                k++;
            }
        }
        if (CUMO_SDX_IS_INDEX(na1->stridx[ax[0]])) {
            idx0 = CUMO_SDX_GET_INDEX(na1->stridx[ax[0]]);
            // diag_idx = ALLOC_N(size_t, diag_size);
            diag_idx = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*diag_size);
            if (CUMO_SDX_IS_INDEX(na1->stridx[ax[1]])) {
                idx1 = CUMO_SDX_GET_INDEX(na1->stridx[ax[1]]);
                cumo_na_diagonal_index_index_kernel_launch(diag_idx, idx0, idx1, k0, k1, diag_size);
            } else {
                stride1 = CUMO_SDX_GET_STRIDE(na1->stridx[ax[1]]);
                cumo_na_diagonal_index_stride_kernel_launch(diag_idx, idx0, stride1, k0, k1, diag_size);
            }
            CUMO_SDX_SET_INDEX(na2->stridx[nd-2],diag_idx);
        } else {
            stride0 = CUMO_SDX_GET_STRIDE(na1->stridx[ax[0]]);
            if (CUMO_SDX_IS_INDEX(na1->stridx[ax[1]])) {
                idx1 = CUMO_SDX_GET_INDEX(na1->stridx[ax[1]]);
                // diag_idx = ALLOC_N(size_t, diag_size);
                diag_idx = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*diag_size);
                cumo_na_diagonal_stride_index_kernel_launch(diag_idx, stride0, idx1, k0, k1, diag_size);
                CUMO_SDX_SET_INDEX(na2->stridx[nd-2],diag_idx);
            } else {
                stride1 = CUMO_SDX_GET_STRIDE(na1->stridx[ax[1]]);
                na2->offset += stride0*k0 + stride1*k1;
                CUMO_SDX_SET_STRIDE(na2->stridx[nd-2],stride0+stride1);
            }
        }
        break;
    }
    return view;
}

//----------------------------------------------------------------------


#if 0
#ifdef SWAP
#undef SWAP
#endif
#define SWAP(a,b,t) {t=a;a=b;b=t;}

static VALUE
cumo_na_new_dimension_for_dot(VALUE self, int pos, int len, bool transpose)
{
    int i, k, l, nd;
    size_t  j;
    size_t *idx1, *idx2;
    size_t *shape;
    ssize_t stride;
    cumo_narray_t *na;
    cumo_narray_view_t *na1, *na2;
    size_t shape_n;
    cumo_stridx_t stridx_n;
    volatile VALUE view;

    CumoGetNArray(self,na);
    nd = na->ndim;

    view = cumo_na_s_allocate_view(rb_obj_class(self));

    cumo_na_copy_flags(self, view);
    CumoGetNArrayView(view, na2);

    // new dimension
    if (pos < 0) pos += nd;
    if (pos > nd || pos < 0) {
        rb_raise(rb_eRangeError,"new dimension is out of range");
    }
    nd += len;
    shape = ALLOCA_N(size_t,nd);
    na2->stridx = ALLOC_N(cumo_stridx_t,nd);

    switch(na->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        i = k = 0;
        while (i < nd) {
            if (i == pos && len > 0) {
                for (l=0; l<len; l++) {
                    shape[i++] = 1;
                }
            } else {
                shape[i++] = na->shape[k++];
            }
        }
        cumo_na_setup_shape((cumo_narray_t*)na2, nd, shape);
        stride = cumo_na_element_stride(self);
        for (i=nd; i--;) {
            CUMO_SDX_SET_STRIDE(na2->stridx[i], stride);
            stride *= shape[i];
        }
        na2->offset = 0;
        na2->data = self;
        break;
    case CUMO_NARRAY_VIEW_T:
        CumoGetNArrayView(self, na1);
        i = k = 0;
        while (i < nd) {
            if (i == pos && len > 0) {
                if (CUMO_SDX_IS_INDEX(na1->stridx[k])) {
                    stride = CUMO_SDX_GET_INDEX(na1->stridx[k])[0];
                } else {
                    stride = CUMO_SDX_GET_STRIDE(na1->stridx[k]);
                }
                for (l=0; l<len; l++) {
                    shape[i] = 1;
                    CUMO_SDX_SET_STRIDE(na2->stridx[i], stride);
                    i++;
                }
            } else {
                shape[i] = na1->base.shape[k];
                if (CUMO_SDX_IS_INDEX(na1->stridx[k])) {
                    idx1 = CUMO_SDX_GET_INDEX(na1->stridx[k]);
                    idx2 = ALLOC_N(size_t,na1->base.shape[k]);
                    for (j=0; j<na1->base.shape[k]; j++) {
                        idx2[j] = idx1[j];
                    }
                    CUMO_SDX_SET_INDEX(na2->stridx[i], idx2);
                } else {
                    na2->stridx[i] = na1->stridx[k];
                }
                i++; k++;
            }
        }
        cumo_na_setup_shape((cumo_narray_t*)na2, nd, shape);
        na2->offset = na1->offset;
        na2->data = na1->data;
        break;
    }

    if (transpose) {
	SWAP(na2->base.shape[nd-1], na2->base.shape[nd-2], shape_n);
	SWAP(na2->stridx[nd-1], na2->stridx[nd-2], stridx_n);
    }

    return view;
}


//----------------------------------------------------------------------

/*
 *  call-seq:
 *     narray.dot(other) => narray
 *
 *  Returns dot product.
 *
 */

static VALUE
cumo_na_dot(VALUE self, VALUE other)
{
    VALUE test;
    volatile VALUE a1=self, a2=other;
    cumo_narray_t *na1, *na2;

    test = rb_funcall(a1, cumo_id_respond_to_p, 1, cumo_sym_mulsum);
    if (!RTEST(test)) {
        rb_raise(rb_eNoMethodError,"requires mulsum method for dot method");
    }
    CumoGetNArray(a1,na1);
    CumoGetNArray(a2,na2);
    if (na1->ndim==0 || na2->ndim==0) {
        rb_raise(cumo_na_eDimensionError,"zero dimensional narray");
    }
    if (na2->ndim > 1) {
        if (na1->shape[na1->ndim-1] != na2->shape[na2->ndim-2]) {
            rb_raise(cumo_na_eShapeError,"shape mismatch: self.shape[-1](=%"SZF"d) != other.shape[-2](=%"SZF"d)",
                     na1->shape[na1->ndim-1], na2->shape[na2->ndim-2]);
        }
        // insert new axis [ ..., last-1-dim, newaxis*other.ndim, last-dim ]
        a1 = cumo_na_new_dimension_for_dot(a1, na1->ndim-1, na2->ndim-1, 0);
        // insert & transpose [ newaxis*self.ndim, ..., last-dim, last-1-dim ]
        a2 = cumo_na_new_dimension_for_dot(a2, 0, na1->ndim-1, 1);
    }
    return rb_funcall(a1,cumo_id_mulsum,2,a2,INT2FIX(-1));
}
#endif

void
Init_cumo_na_data(void)
{
    rb_define_method(cNArray, "copy", cumo_na_copy, 0); // deprecated

    rb_define_method(cNArray, "flatten", cumo_na_flatten, 0);
    rb_define_method(cNArray, "swapaxes", cumo_na_swapaxes, 2);
    rb_define_method(cNArray, "transpose", cumo_na_transpose, -1);

    rb_define_method(cNArray, "reshape", cumo_na_reshape,-1);
    rb_define_method(cNArray, "reshape!", cumo_na_reshape_bang,-1);
    /*
    rb_define_alias(cNArray,  "shape=","reshape!");
    */
    rb_define_method(cNArray, "diagonal", cumo_na_diagonal,-1);

    rb_define_method(cNArray, "swap_byte", cumo_na_swap_byte, 0);
#ifdef DYNAMIC_ENDIAN
#else
#ifdef WORDS_BIGENDIAN
#else // LITTLE_ENDIAN
    rb_define_alias(cNArray, "hton", "swap_byte");
    rb_define_alias(cNArray, "network_order?", "byte_swapped?");
    rb_define_alias(cNArray, "little_endian?", "host_order?");
    rb_define_alias(cNArray, "vacs_order?", "host_order?");
#endif
#endif
    rb_define_method(cNArray, "to_network", cumo_na_to_network, 0);
    rb_define_method(cNArray, "to_vacs", cumo_na_to_vacs, 0);
    rb_define_method(cNArray, "to_host", cumo_na_to_host, 0);
    rb_define_method(cNArray, "to_swapped", cumo_na_to_swapped, 0);

    //rb_define_method(cNArray, "dot", cumo_na_dot, 1);

    cumo_id_mulsum       = rb_intern("mulsum");
    cumo_sym_mulsum      = ID2SYM(cumo_id_mulsum);
    cumo_id_respond_to_p = rb_intern("respond_to?");
    cumo_id_store        = rb_intern("store");
    cumo_id_swap_byte    = rb_intern("swap_byte");
}
