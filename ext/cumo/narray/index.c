#include <string.h>
#include <ruby.h>
#include "cumo.h"
#include "cumo/narray.h"
#include "cumo/cuda/runtime.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/template.h"

#if   SIZEOF_VOIDP == 8
#define cIndex cumo_cInt64
#elif SIZEOF_VOIDP == 4
#define cIndex cumo_cInt32
#endif


// note: the memory refed by this pointer is not freed and causes memroy leak.
//
// @example
//     a[1..3,1] generates two cumo_na_index_arg_t(s). First is for 1..3, and second is for 1.
typedef struct {
    size_t  n; // the number of elements of the dimesnion
    size_t  beg; // the starting point in the dimension
    ssize_t step; // the step size of the dimension
    size_t *idx; // list of indices
    int     reduce; // true if the dimension is reduced by addition
    int     orig_dim; // the dimension of original array
} cumo_na_index_arg_t;


static void
print_index_arg(cumo_na_index_arg_t *q, int n)
{
    int i;
    printf("cumo_na_index_arg_t = 0x%"SZF"x {\n",(size_t)q);
    for (i=0; i<n; i++) {
        printf("  q[%d].n=%"SZF"d\n",i,q[i].n);
        printf("  q[%d].beg=%"SZF"d\n",i,q[i].beg);
        printf("  q[%d].step=%"SZF"d\n",i,q[i].step);
        printf("  q[%d].idx=0x%"SZF"x (cuda:%d)\n",i,(size_t)q[i].idx, cumo_cuda_runtime_is_device_memory(q[i].idx));
        // printf("  q[%d].idx=0x%"SZF"x\n",i,(size_t)q[i].idx);
        printf("  q[%d].reduce=0x%x\n",i,q[i].reduce);
        printf("  q[%d].orig_dim=%d\n",i,q[i].orig_dim);
    }
    printf("}\n");
}

static VALUE cumo_sym_ast;
static VALUE cumo_sym_all;
//static VALUE cumo_sym_reduce;
static VALUE cumo_sym_minus;
static VALUE cumo_sym_new;
static VALUE cumo_sym_reverse;
static VALUE cumo_sym_plus;
static VALUE cumo_sym_sum;
static VALUE cumo_sym_tilde;
static VALUE cumo_sym_rest;
static ID cumo_id_beg;
static ID cumo_id_end;
static ID cumo_id_exclude_end;
static ID cumo_id_each;
static ID cumo_id_step;
static ID cumo_id_dup;
static ID cumo_id_bracket;
static ID cumo_id_shift_left;
static ID cumo_id_mask;


static void
cumo_na_index_set_step(cumo_na_index_arg_t *q, int i, size_t n, size_t beg, ssize_t step)
{
    q->n    = n;
    q->beg  = beg;
    q->step = step;
    q->idx  = NULL;
    q->reduce = 0;
    q->orig_dim = i;
}

static void
cumo_na_index_set_scalar(cumo_na_index_arg_t *q, int i, ssize_t size, ssize_t x)
{
    if (x < -size || x >= size)
        rb_raise(rb_eRangeError,
                  "array index (%"SZF"d) is out of array size (%"SZF"d)",
                  x, size);
    if (x < 0)
        x += size;
    q->n    = 1;
    q->beg  = x;
    q->step = 0;
    q->idx  = NULL;
    q->reduce = 0;
    q->orig_dim = i;
}

static inline ssize_t
cumo_na_range_check(ssize_t pos, ssize_t size, int dim)
{
    ssize_t idx=pos;

    if (idx < 0) idx += size;
    if (idx < 0 || idx >= size) {
        rb_raise(rb_eIndexError, "index=%"SZF"d out of shape[%d]=%"SZF"d",
                 pos, dim, size);
    }
    return idx;
}

static void CUDART_CB
cumo_na_parse_array_callback(cudaStream_t stream, cudaError_t status, void *data)
{
    cudaFreeHost(data);
}

// copy ruby array to idx
static void
cumo_na_parse_array(VALUE ary, int orig_dim, ssize_t size, cumo_na_index_arg_t *q)
{
    int k;
    size_t* idx;
    cudaError_t status;
    int n = RARRAY_LEN(ary);
    //q->idx = ALLOC_N(size_t, n);
    //for (k=0; k<n; k++) {
    //    q->idx[k] = na_range_check(NUM2SSIZET(RARRAY_AREF(ary,k)), size, orig_dim);
    //}
    // make a contiguous pinned memory on host => copy to device => release pinned memory after copy finished on callback
    q->idx = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*n);
    cudaHostAlloc((void**)&idx, sizeof(size_t)*n, cudaHostAllocDefault);
    for (k=0; k<n; k++) {
        idx[k] = cumo_na_range_check(NUM2SSIZET(RARRAY_AREF(ary,k)), size, orig_dim);
    }
    status = cudaMemcpyAsync(q->idx,idx,sizeof(size_t)*n,cudaMemcpyHostToDevice,0);
    if (status == 0) {
        cumo_cuda_runtime_check_status(cudaStreamAddCallback(0,cumo_na_parse_array_callback,idx,0));
    } else {
        cudaFreeHost(idx);
    }
    cumo_cuda_runtime_check_status(status);

    q->n    = n;
    q->beg  = 0;
    q->step = 1;
    q->reduce = 0;
    q->orig_dim = orig_dim;
}

// copy narray to idx
static void
cumo_na_parse_narray_index(VALUE a, int orig_dim, ssize_t size, cumo_na_index_arg_t *q)
{
    VALUE idx;
    cumo_narray_t *na;
    cumo_narray_data_t *nidx;
    size_t n;
    ssize_t *nidxp;

    CumoGetNArray(a,na);
    if (CUMO_NA_NDIM(na) != 1) {
        rb_raise(rb_eIndexError, "should be 1-d NArray");
    }
    n = CUMO_NA_SIZE(na);
    idx = cumo_na_new(cIndex,1,&n);
    cumo_na_store(idx,a);

    CumoGetNArrayData(idx,nidx);
    nidxp   = (ssize_t*)nidx->ptr; // Cumo::NArray data resides on GPU
    //q->idx  = ALLOC_N(size_t, n);
    //for (k=0; k<n; k++) {
    //    q->idx[k] = na_range_check(nidxp[k], size, orig_dim);
    //}
    q->idx = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*n);
    cumo_cuda_runtime_check_status(cudaMemcpyAsync(q->idx,nidxp,sizeof(size_t)*n,cudaMemcpyDeviceToDevice,0));

    q->n    = n;
    q->beg  = 0;
    q->step = 1;
    q->reduce = 0;
    q->orig_dim = orig_dim;
}

static void
cumo_na_parse_range(VALUE range, ssize_t step, int orig_dim, ssize_t size, cumo_na_index_arg_t *q)
{
    int n;
    VALUE excl_end;
    ssize_t beg, end, beg_orig, end_orig;
    const char *dot = "..", *edot = "...";

#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
    rb_arithmetic_sequence_components_t x;
    rb_arithmetic_sequence_extract(range, &x);
    step = NUM2SSIZET(x.step);

    beg = beg_orig = NUM2SSIZET(x.begin);
    if (beg < 0) {
        beg += size;
    }
    if (T_NIL == TYPE(x.end)) { // endless range
        end = size -1;
        if (RTEST(x.exclude_end)) {
            dot = edot;
        }
    } else {
        end = end_orig = NUM2SSIZET(x.end);
        if (end < 0) {
            end += size;
        }
        if (RTEST(x.exclude_end)) {
            end--;
            dot = edot;
        }
    }
    if (beg < 0 || beg >= size || end < 0 || end >= size) {
        if (T_NIL == TYPE(x.end)) { // endless range
            rb_raise(rb_eRangeError,
                     "%"SZF"d%s is out of range for size=%"SZF"d",
                     beg_orig, dot, size);
        } else {
            rb_raise(rb_eRangeError,
                     "%"SZF"d%s%"SZF"d is out of range for size=%"SZF"d",
                     beg_orig, dot, end_orig, size);
        }
    }
#else
    beg = beg_orig = NUM2SSIZET(rb_funcall(range,cumo_id_beg,0));
    if (beg < 0) {
        beg += size;
    }
    end = end_orig = NUM2SSIZET(rb_funcall(range,cumo_id_end,0));
    if (end < 0) {
        end += size;
    }
    excl_end = rb_funcall(range,cumo_id_exclude_end,0);
    if (RTEST(excl_end)) {
        end--;
        dot = edot;
    }
    if (beg < 0 || beg >= size || end < 0 || end >= size) {
        rb_raise(rb_eRangeError,
                 "%"SZF"d%s%"SZF"d is out of range for size=%"SZF"d",
                 beg_orig, dot, end_orig, size);
    }
#endif
    n = (end-beg)/step+1;
    if (n<0) n=0;
    cumo_na_index_set_step(q,orig_dim,n,beg,step);

}

void
cumo_na_parse_enumerator_step(VALUE enum_obj, VALUE *pstep)
{
    int len;
    VALUE step;
    cumo_enumerator_t *e;

    if (!RB_TYPE_P(enum_obj, T_DATA)) {
        rb_raise(rb_eTypeError,"wrong argument type (not T_DATA)");
    }
    e = CUMO_RENUMERATOR_PTR(enum_obj);

    if (!rb_obj_is_kind_of(e->obj, rb_cRange)) {
        rb_raise(rb_eTypeError,"not Range object");
    }

    if (e->meth == cumo_id_each) {
        step = INT2NUM(1);
    }
    else if (e->meth == cumo_id_step) {
        if (TYPE(e->args) != T_ARRAY) {
            rb_raise(rb_eArgError,"no argument for step");
        }
        len = RARRAY_LEN(e->args);
        if (len != 1) {
            rb_raise(rb_eArgError,"invalid number of step argument (1 for %d)",len);
        }
        step = RARRAY_AREF(e->args,0);
    } else {
        rb_raise(rb_eTypeError,"unknown Range method: %s",rb_id2name(e->meth));
    }
    if (pstep) *pstep = step;
}

static void
cumo_na_parse_enumerator(VALUE enum_obj, int orig_dim, ssize_t size, cumo_na_index_arg_t *q)
{
    VALUE step;
    cumo_enumerator_t *e;

    if (!RB_TYPE_P(enum_obj, T_DATA)) {
        rb_raise(rb_eTypeError,"wrong argument type (not T_DATA)");
    }
    cumo_na_parse_enumerator_step(enum_obj, &step);
    e = CUMO_RENUMERATOR_PTR(enum_obj);
    cumo_na_parse_range(e->obj, NUM2SSIZET(step), orig_dim, size, q); // e->obj : Range Object
}

// Analyze *a* which is *i*-th index object and store the information to q
//
// a: a ruby object of i-th index
// size: size of i-th dimension of original NArray
// i: parse i-th index
// q: parsed information is stored to *q
static void
cumo_na_index_parse_each(volatile VALUE a, ssize_t size, int i, cumo_na_index_arg_t *q)
{
    switch(TYPE(a)) {

    case T_FIXNUM:
        cumo_na_index_set_scalar(q,i,size,FIX2LONG(a));
        break;

    case T_BIGNUM:
        cumo_na_index_set_scalar(q,i,size,NUM2SSIZET(a));
        break;

    case T_FLOAT:
        cumo_na_index_set_scalar(q,i,size,NUM2SSIZET(a));
        break;

    case T_NIL:
    case T_TRUE:
        cumo_na_index_set_step(q,i,size,0,1);
        break;

    case T_SYMBOL:
        if (a==cumo_sym_all || a==cumo_sym_ast) {
            cumo_na_index_set_step(q,i,size,0,1);
        }
        else if (a==cumo_sym_reverse) {
            cumo_na_index_set_step(q,i,size,size-1,-1);
        }
        else if (a==cumo_sym_new) {
            cumo_na_index_set_step(q,i,1,0,1);
        }
        else if (a==cumo_sym_reduce || a==cumo_sym_sum || a==cumo_sym_plus) {
            cumo_na_index_set_step(q,i,size,0,1);
            q->reduce = 1;
        } else {
            rb_raise(rb_eIndexError, "invalid symbol for index");
        }
        break;

    case T_ARRAY:
        cumo_na_parse_array(a, i, size, q);
        break;

    default:
        if (rb_obj_is_kind_of(a, rb_cRange)) {
            cumo_na_parse_range(a, 1, i, size, q);
        }
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
        else if (rb_obj_is_kind_of(a, rb_cArithSeq)) {
            cumo_na_parse_range(a, 1, i, size, q);
        }
#endif
        else if (rb_obj_is_kind_of(a, rb_cEnumerator)) {
            cumo_na_parse_enumerator(a, i, size, q);
        }
        // NArray index
        else if (CUMO_NA_CumoIsNArray(a)) {
            cumo_na_parse_narray_index(a, i, size, q);
        }
        else {
            rb_raise(rb_eIndexError, "not allowed type");
        }
    }
}


static size_t
cumo_na_index_parse_args(VALUE args, cumo_narray_t *na, cumo_na_index_arg_t *q, int ndim)
{
    int i, j, k, l, nidx;
    size_t total=1;
    VALUE v;

    if (ndim == 0) {
        return /*total*/1;
    }

    nidx = RARRAY_LEN(args);

    for (i=j=k=0; i<nidx; i++) {
        v = RARRAY_AREF(args,i);
        // rest (ellipsis) dimension
        if (v==Qfalse) {
            for (l = ndim - (nidx-1); l>0; l--) {
                //printf("i=%d j=%d k=%d l=%d ndim=%d nidx=%d\n",i,j,k,l,ndim,nidx);
                cumo_na_index_parse_each(Qtrue, na->shape[k], k, &q[j]);
                if (q[j].n > 1) {
                    total *= q[j].n;
                }
                j++;
                k++;
            }
        }
        // new dimension
        else if (v==cumo_sym_new) {
            cumo_na_index_parse_each(v, 1, k, &q[j]);
            j++;
        }
        // other dimention
        else {
            cumo_na_index_parse_each(v, na->shape[k], k, &q[j]);
            if (q[j].n > 1) {
                total *= q[j].n;
            }
            j++;
            k++;
        }
    }
    return total;
}


static void
cumo_na_get_strides_nadata(const cumo_narray_data_t *na, ssize_t *strides, ssize_t elmsz)
{
    int i = na->base.ndim - 1;
    strides[i] = elmsz;
    for (; i>0; i--) {
        strides[i-1] = strides[i] * na->base.shape[i];
    }
}

void cumo_na_index_aref_nadata_index_stride_kernel_launch(size_t *idx, ssize_t s1, uint64_t n);

static void
cumo_na_index_aref_nadata(cumo_narray_data_t *na1, cumo_narray_view_t *na2,
                     cumo_na_index_arg_t *q, ssize_t elmsz, int ndim, int keep_dim)
{
    int i, j;
    ssize_t size, total=1;
    ssize_t stride1;
    ssize_t *strides_na1;
    size_t  *index;
    ssize_t beg, step;
    VALUE m;

    strides_na1 = ALLOCA_N(ssize_t, na1->base.ndim);
    cumo_na_get_strides_nadata(na1, strides_na1, elmsz);

    for (i=j=0; i<ndim; i++) {
        stride1 = strides_na1[q[i].orig_dim];

        // numeric index -- trim dimension
        if (!keep_dim && q[i].n==1 && q[i].step==0) {
            beg  = q[i].beg;
            na2->offset += stride1 * beg;
            continue;
        }

        na2->base.shape[j] = size = q[i].n;

        if (q[i].reduce != 0) {
            m = rb_funcall(INT2FIX(1),cumo_id_shift_left,1,INT2FIX(j));
            na2->base.reduce = rb_funcall(m,'|',1,na2->base.reduce);
        }

        // array index
        if (q[i].idx != NULL) {
            index = q[i].idx;
            CUMO_SDX_SET_INDEX(na2->stridx[j],index);
            q[i].idx = NULL;
            cumo_na_index_aref_nadata_index_stride_kernel_launch(index, stride1, size);
        } else {
            beg  = q[i].beg;
            step = q[i].step;
            na2->offset += stride1*beg;
            CUMO_SDX_SET_STRIDE(na2->stridx[j], stride1*step);
        }
        j++;
        total *= size;
    }
    na2->base.size = total;
}


void cumo_na_index_aref_naview_index_index_kernel_launch(size_t *idx, size_t *idx1, uint64_t n);
void cumo_na_index_aref_naview_index_stride_last_kernel_launch(size_t *idx, ssize_t s1, size_t last, uint64_t n);
void cumo_na_index_aref_naview_index_stride_kernel_launch(size_t *idx, ssize_t s1, uint64_t n);
void cumo_na_index_aref_naview_index_index_beg_step_kernel_launch(size_t *idx, size_t *idx1, size_t beg, ssize_t step, uint64_t n);

static void
cumo_na_index_aref_naview(cumo_narray_view_t *na1, cumo_narray_view_t *na2,
                     cumo_na_index_arg_t *q, ssize_t elmsz, int ndim, int keep_dim)
{
    int i, j;
    ssize_t total=1;

    for (i=j=0; i<ndim; i++) {
        cumo_stridx_t sdx1 = na1->stridx[q[i].orig_dim];
        ssize_t size;

        // numeric index -- trim dimension
        if (!keep_dim && q[i].n==1 && q[i].step==0) {
            if (CUMO_SDX_IS_INDEX(sdx1)) {
                na2->offset += CUMO_SDX_GET_INDEX(sdx1)[q[i].beg];
            } else {
                na2->offset += CUMO_SDX_GET_STRIDE(sdx1)*q[i].beg;
            }
            continue;
        }

        na2->base.shape[j] = size = q[i].n;

        if (q[i].reduce != 0) {
            VALUE m = rb_funcall(INT2FIX(1),cumo_id_shift_left,1,INT2FIX(j));
            na2->base.reduce = rb_funcall(m,'|',1,na2->base.reduce);
        }

        if (q[i].orig_dim >= na1->base.ndim) {
            // new dimension
            CUMO_SDX_SET_STRIDE(na2->stridx[j], elmsz);
        }
        else if (q[i].idx != NULL && CUMO_SDX_IS_INDEX(sdx1)) {
            // index <- index
            size_t *index = q[i].idx;
            size_t *index1 = CUMO_SDX_GET_INDEX(sdx1);
            CUMO_SDX_SET_INDEX(na2->stridx[j], index);
            q[i].idx = NULL;
            cumo_na_index_aref_naview_index_index_kernel_launch(index, index1, size);
        }
        else if (q[i].idx != NULL && CUMO_SDX_IS_STRIDE(sdx1)) {
            // index <- step
            ssize_t stride1 = CUMO_SDX_GET_STRIDE(sdx1);
            size_t *index = q[i].idx;
            CUMO_SDX_SET_INDEX(na2->stridx[j],index);
            q[i].idx = NULL;

            if (stride1<0) {
                size_t  last;
                stride1 = -stride1;
                last = na1->base.shape[q[i].orig_dim] - 1;
                if (na2->offset < last * stride1) {
                    rb_raise(rb_eStandardError,"bug: negative offset");
                }
                na2->offset -= last * stride1;
                cumo_na_index_aref_naview_index_stride_last_kernel_launch(index, stride1, last, size);
            } else {
                cumo_na_index_aref_naview_index_stride_kernel_launch(index, stride1, size);
            }
        }
        else if (q[i].idx == NULL && CUMO_SDX_IS_INDEX(sdx1)) {
            // step <- index
            size_t beg  = q[i].beg;
            ssize_t step = q[i].step;
            // size_t *index = ALLOC_N(size_t, size);
            size_t *index = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*size);
            size_t *index1 = CUMO_SDX_GET_INDEX(sdx1);
            CUMO_SDX_SET_INDEX(na2->stridx[j],index);
            cumo_na_index_aref_naview_index_index_beg_step_kernel_launch(index, index1, beg, step, size);
        }
        else if (q[i].idx == NULL && CUMO_SDX_IS_STRIDE(sdx1)) {
            // step <- step
            size_t beg  = q[i].beg;
            ssize_t step = q[i].step;
            ssize_t stride1 = CUMO_SDX_GET_STRIDE(sdx1);
            na2->offset += stride1*beg;
            CUMO_SDX_SET_STRIDE(na2->stridx[j], stride1*step);
        }

        j++;
        total *= size;
    }
    na2->base.size = total;
}

void cumo_na_index_at_nadata_index_stride_add_kernel_launch(size_t *idx, size_t *idx1, ssize_t s1, uint64_t n);
void cumo_na_index_at_nadata_index_beg_step_stride_kernel_launch(size_t *idx, size_t beg, ssize_t step, ssize_t s1, uint64_t n);
void cumo_na_index_at_nadata_index_beg_step_stride_add_kernel_launch(size_t *idx, size_t beg, ssize_t step, ssize_t s1, uint64_t n);

static void
cumo_na_index_at_nadata(cumo_narray_data_t *na1, cumo_narray_view_t *na2,
                     cumo_na_index_arg_t *q, ssize_t elmsz, int ndim, int keep_dim)
{
    int i;
    ssize_t size = q[ndim-1].n;
    ssize_t stride1;
    ssize_t *strides_na1;
    size_t  *index;
    ssize_t beg, step;
    int use_cumo_cuda_runtime_malloc = 0;

    strides_na1 = ALLOCA_N(ssize_t, na1->base.ndim);
    cumo_na_get_strides_nadata(na1, strides_na1, elmsz);

    if (q[ndim-1].idx != NULL) {
        index = q[ndim-1].idx;
    } else {
        //index = ALLOC_N(size_t, size);
        index = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*size);
        use_cumo_cuda_runtime_malloc = 1;
    }
    CUMO_SDX_SET_INDEX(na2->stridx[0], index);

    for (i=ndim-1; i>=0; i--) {
        stride1 = strides_na1[q[i].orig_dim];
        if (i==ndim-1) {
            if (size == 0) {
                rb_raise(cumo_na_eShapeError, "cannot get element of empty array");
            }
        } else {
            if (size != q[i].n) {
                rb_raise(cumo_na_eShapeError, "index array sizes mismatch");
            }
        }

        if (q[i].idx != NULL) {
            if (i==ndim-1) {
                cumo_na_index_aref_nadata_index_stride_kernel_launch(index, stride1, size);
            } else {
                cumo_na_index_at_nadata_index_stride_add_kernel_launch(index, q[i].idx, stride1, size);
            }
            q[i].idx = NULL;
        } else {
            beg  = q[i].beg;
            step = q[i].step;
            if (i==ndim-1) {
                cumo_na_index_at_nadata_index_beg_step_stride_kernel_launch(index, beg, step, stride1, size);
            } else {
                cumo_na_index_at_nadata_index_beg_step_stride_add_kernel_launch(index, beg, step, stride1, size);
            }
        }

    }
    na2->base.size = size;
    na2->base.shape[0] = size;
    if (use_cumo_cuda_runtime_malloc) {
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("index", "cumo_na_index_at_nadata");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    }
}

void cumo_na_index_at_naview_index_index_index_add_kernel_launch(size_t *idx, size_t *idx1, size_t *idx2, uint64_t n);
void cumo_na_index_at_naview_index_index_beg_step_add_kernel_launch(size_t *idx, size_t *idx1, size_t beg, ssize_t step, uint64_t n);
void cumo_na_index_at_naview_index_stride_last_add_kernel_launch(size_t *idx, ssize_t s1, size_t last, uint64_t n);

static void
cumo_na_index_at_naview(cumo_narray_view_t *na1, cumo_narray_view_t *na2,
                     cumo_na_index_arg_t *q, ssize_t elmsz, int ndim, int keep_dim)
{
    int i;
    size_t  *index;
    ssize_t size = q[ndim-1].n;
    int use_cumo_cuda_runtime_malloc = 0;

    if (q[ndim-1].idx != NULL) {
        index = q[ndim-1].idx;
    } else {
        //index = ALLOC_N(size_t, size);
        index = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*size);
        use_cumo_cuda_runtime_malloc = 1;
    }
    CUMO_SDX_SET_INDEX(na2->stridx[0], index);

    for (i=ndim-1; i>=0; i--) {
        cumo_stridx_t sdx1 = na1->stridx[q[i].orig_dim];
        if (i==ndim-1) {
            if (size == 0) {
                rb_raise(cumo_na_eShapeError, "cannot get element of empty array");
            }
        } else {
            if (size != q[i].n) {
                rb_raise(cumo_na_eShapeError, "index array sizes mismatch");
            }
        }

        if (q[i].idx != NULL && CUMO_SDX_IS_INDEX(sdx1)) {
            // index <- index
            size_t *index1 = CUMO_SDX_GET_INDEX(sdx1);
            if (i==ndim-1) {
                cumo_na_index_aref_naview_index_index_kernel_launch(index, index1, size);
            } else {
                cumo_na_index_at_naview_index_index_index_add_kernel_launch(index, index1, q[i].idx, size);
            }
            q[i].idx = NULL;
        }
        else if (q[i].idx == NULL && CUMO_SDX_IS_INDEX(sdx1)) {
            // step <- index
            size_t beg  = q[i].beg;
            ssize_t step = q[i].step;
            size_t *index1 = CUMO_SDX_GET_INDEX(sdx1);
            if (i==ndim-1) {
                cumo_na_index_aref_naview_index_index_beg_step_kernel_launch(index, index1, beg, step, size);
            } else {
                cumo_na_index_at_naview_index_index_beg_step_add_kernel_launch(index, index1, beg, step, size);
            }
        }
        else if (q[i].idx != NULL && CUMO_SDX_IS_STRIDE(sdx1)) {
            // index <- step
            ssize_t stride1 = CUMO_SDX_GET_STRIDE(sdx1);
            if (stride1<0) {
                size_t  last;
                stride1 = -stride1;
                last = na1->base.shape[q[i].orig_dim] - 1;
                if (na2->offset < last * stride1) {
                    rb_raise(rb_eStandardError,"bug: negative offset");
                }
                na2->offset -= last * stride1;
                if (i==ndim-1) {
                    cumo_na_index_aref_naview_index_stride_last_kernel_launch(index, stride1, last, size);
                } else {
                    cumo_na_index_at_naview_index_stride_last_add_kernel_launch(index, stride1, last, size);
                }
            } else {
                if (i==ndim-1) {
                    cumo_na_index_aref_nadata_index_stride_kernel_launch(index, stride1, size);
                } else {
                    cumo_na_index_at_nadata_index_stride_add_kernel_launch(index, q[i].idx, stride1, size);
                }
            }
            q[i].idx = NULL;
        }
        else if (q[i].idx == NULL && CUMO_SDX_IS_STRIDE(sdx1)) {
            // step <- step
            size_t beg  = q[i].beg;
            ssize_t step = q[i].step;
            ssize_t stride1 = CUMO_SDX_GET_STRIDE(sdx1);
            if (stride1<0) {
                size_t  last;
                stride1 = -stride1;
                last = na1->base.shape[q[i].orig_dim] - 1;
                if (na2->offset < last * stride1) {
                    rb_raise(rb_eStandardError,"bug: negative offset");
                }
                na2->offset -= last * stride1;
                if (i==ndim-1) {
                    cumo_na_index_at_nadata_index_beg_step_stride_kernel_launch(index, last - beg, -step, stride1, size);
                } else {
                    cumo_na_index_at_nadata_index_beg_step_stride_add_kernel_launch(index, last - beg, -step, stride1, size);
                }
            } else {
                if (i==ndim-1) {
                    cumo_na_index_at_nadata_index_beg_step_stride_kernel_launch(index, beg, step, stride1, size);
                } else {
                    cumo_na_index_at_nadata_index_beg_step_stride_add_kernel_launch(index, beg, step, stride1, size);
                }
            }
        }
    }
    na2->base.size = size;
    na2->base.shape[0] = size;
    if (use_cumo_cuda_runtime_malloc) {
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("index", "cumo_na_index_at_naview");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    }
}

static int
cumo_na_ndim_new_narray(int ndim, const cumo_na_index_arg_t *q)
{
    int i, ndim_new=0;
    for (i=0; i<ndim; i++) {
        if (q[i].n>1 || q[i].step!=0) {
            ndim_new++;
        }
    }
    return ndim_new;
}

typedef struct {
    VALUE args, self, store;
    int ndim;
    cumo_na_index_arg_t *q; // multi-dimensional index args
    cumo_narray_t *na1;
    int keep_dim;
    size_t pos; // offset position for 0-dimensional narray. 0-dimensional array does not use q.
    int at_mode; // 0: aref, 1: at
} cumo_na_aref_md_data_t;

static cumo_na_index_arg_t*
cumo_na_allocate_index_args(int ndim)
{
    cumo_na_index_arg_t *q;
    int i;
    if (ndim == 0) return NULL;

    q = ALLOC_N(cumo_na_index_arg_t, ndim);
    for (i=0; i<ndim; i++) {
        q[i].idx = NULL;
    }
    return q;
}

static
VALUE cumo_na_aref_md_protected(VALUE data_value)
{
    cumo_na_aref_md_data_t *data = (cumo_na_aref_md_data_t*)(data_value);
    VALUE self = data->self;
    VALUE args = data->args;
    VALUE store = data->store;
    int ndim = data->ndim;
    cumo_na_index_arg_t *q = data->q;
    cumo_narray_t *na1 = data->na1;
    int keep_dim = data->keep_dim;
    int at_mode = data->at_mode;

    int ndim_new;
    VALUE view;
    cumo_narray_view_t *na2;
    ssize_t elmsz;

    cumo_na_index_parse_args(args, na1, q, ndim);

    if (cumo_na_debug_flag) print_index_arg(q,ndim);

    if (at_mode) {
        ndim_new = 1;
    } else {
        if (keep_dim) {
            ndim_new = ndim;
        } else {
            ndim_new = cumo_na_ndim_new_narray(ndim, q);
        }
    }
    view = cumo_na_s_allocate_view(rb_obj_class(self));

    cumo_na_copy_flags(self, view);
    CumoGetNArrayView(view,na2);

    cumo_na_alloc_shape((cumo_narray_t*)na2, ndim_new);

    na2->stridx = ALLOC_N(cumo_stridx_t,ndim_new);

    elmsz = cumo_na_element_stride(self);

    switch(na1->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        if (ndim == 0) {
            na2->offset = data->pos;
            na2->base.size = 1;
        } else {
            if (at_mode) {
                cumo_na_index_at_nadata((cumo_narray_data_t *)na1,na2,q,elmsz,ndim,keep_dim);
            } else {
                cumo_na_index_aref_nadata((cumo_narray_data_t *)na1,na2,q,elmsz,ndim,keep_dim);
            }
        }
        na2->data = self;
        break;
    case CUMO_NARRAY_VIEW_T:
        if (ndim == 0) {
            na2->offset = ((cumo_narray_view_t *)na1)->offset + data->pos;
            na2->data = ((cumo_narray_view_t *)na1)->data;
            na2->base.size = 1;
        } else {
            na2->offset = ((cumo_narray_view_t *)na1)->offset;
            na2->data = ((cumo_narray_view_t *)na1)->data;
            if (at_mode) {
               cumo_na_index_at_naview((cumo_narray_view_t *)na1,na2,q,elmsz,ndim,keep_dim);
            } else {
               cumo_na_index_aref_naview((cumo_narray_view_t *)na1,na2,q,elmsz,ndim,keep_dim);
            }
        }
        break;
    }
    if (store) {
        cumo_na_get_pointer_for_write(store); // allocate memory
        cumo_na_store(cumo_na_flatten_dim(store,0),view);
        return store;
    }
    return view;
}

static VALUE
cumo_na_aref_md_ensure(VALUE data_value)
{
    cumo_na_aref_md_data_t *data = (cumo_na_aref_md_data_t*)(data_value);
    int i;
    for (i=0; i<data->ndim; i++) {
        cumo_cuda_runtime_free((char*)(data->q[i].idx));
    }
    if (data->q) xfree(data->q);
    return Qnil;
}

static VALUE
cumo_na_aref_md(int argc, VALUE *argv, VALUE self, int keep_dim, int result_nd, size_t pos, int at_mode)
{
    VALUE args; // should be GC protected
    cumo_narray_t *na1;
    cumo_na_aref_md_data_t data;
    VALUE store = 0;
    VALUE idx;
    cumo_narray_t *nidx;

    CumoGetNArray(self,na1);

    args = rb_ary_new4(argc,argv);
    if (at_mode && na1->ndim == 0) {
        rb_raise(cumo_na_eDimensionError,"argument length does not match dimension size");
    }

    if (argc == 1 && result_nd == 1) {
        idx = argv[0];
        if (rb_obj_is_kind_of(idx, rb_cArray)) {
            idx = rb_apply(cumo_cNArray,cumo_id_bracket,idx);
        }
        if (rb_obj_is_kind_of(idx, cumo_cNArray)) {
            CumoGetNArray(idx,nidx);
            if (CUMO_NA_NDIM(nidx)>1) {
                store = cumo_na_new(rb_obj_class(self),CUMO_NA_NDIM(nidx),CUMO_NA_SHAPE(nidx));
                idx = cumo_na_flatten(idx);
                RARRAY_ASET(args,0,idx);
            }
        }
        // flatten should be done only for narray-view with non-uniform stride.
        if (na1->ndim > 1) {
            self = cumo_na_flatten(self);
            CumoGetNArray(self,na1);
        }
    }

    data.args = args;
    data.self = self;
    data.store = store;
    data.ndim = result_nd;
    data.q = cumo_na_allocate_index_args(result_nd);
    data.na1 = na1;
    data.keep_dim = keep_dim;
    data.at_mode = at_mode;

    switch(na1->type) {
    case CUMO_NARRAY_DATA_T:
        data.pos = pos;
        break;
    case CUMO_NARRAY_FILEMAP_T:
        data.pos = pos; // correct? I have never used..
        break;
    case CUMO_NARRAY_VIEW_T:
        {
            cumo_narray_view_t *nv;
            CumoGetNArrayView(self,nv);
            // pos obtained by cumo_na_get_result_dimension adds view->offset.
            data.pos = pos - nv->offset;
        }
        break;
    }

    return rb_ensure(cumo_na_aref_md_protected, (VALUE)&data, cumo_na_aref_md_ensure, (VALUE)&data);
}


/* method: [](idx1,idx2,...,idxN) */
VALUE
cumo_na_aref_main(int nidx, VALUE *idx, VALUE self, int keep_dim, int result_nd, size_t pos)
{
    cumo_na_index_arg_to_internal_order(nidx, idx, self);

    if (nidx==0) {
        return rb_funcall(self,cumo_id_dup,0);
    }
    if (nidx==1) {
        if (rb_obj_class(*idx)==cumo_cBit) {
            return rb_funcall(*idx,cumo_id_mask,1,self);
        }
    }
    return cumo_na_aref_md(nidx, idx, self, keep_dim, result_nd, pos, 0);
}

/* method: at([idx1,idx2,...,idxN], [idx1,idx2,...,idxN]) */
VALUE
cumo_na_at_main(int nidx, VALUE *idx, VALUE self, int keep_dim, int result_nd, size_t pos)
{
    cumo_na_index_arg_to_internal_order(nidx, idx, self);
    return cumo_na_aref_md(nidx, idx, self, keep_dim, result_nd, pos, 1);
}


/* method: slice(idx1,idx2,...,idxN) */
static VALUE cumo_na_slice(int argc, VALUE *argv, VALUE self)
{
    int result_nd;
    size_t pos;

    result_nd = cumo_na_get_result_dimension(self, argc, argv, 0, &pos);
    return cumo_na_aref_main(argc, argv, self, 1, result_nd, pos);
}


static int
check_index_count(int argc, int cumo_na_ndim, int count_new, int count_rest)
{
    int result_nd = cumo_na_ndim + count_new;

    switch(count_rest) {
    case 0:
        if (argc == 1 && count_new == 0) return 1;
        if (argc == result_nd) return result_nd;
        rb_raise(rb_eIndexError,"# of index(=%i) should be "
                 "equal to ndim(=%i) or 1", argc,cumo_na_ndim);
        break;
    case 1:
        if (argc-1 <= result_nd) return result_nd;
        rb_raise(rb_eIndexError,"# of index(=%i) > ndim(=%i) with :rest",
                 argc,cumo_na_ndim);
        break;
    default:
        rb_raise(rb_eIndexError,"multiple rest-dimension is not allowd");
    }
    return -1;
}

int
cumo_na_get_result_dimension(VALUE self, int argc, VALUE *argv, ssize_t stride, size_t *pos_idx)
{
    int i, j;
    int count_new=0;
    int count_rest=0;
    ssize_t x, s, m, pos, *idx;
    cumo_narray_t *na;
    cumo_narray_view_t *nv;
    cumo_stridx_t sdx;
    VALUE a;

    CumoGetNArray(self,na);
    if (na->size == 0) {
        rb_raise(cumo_na_eShapeError, "cannot get element of empty array");
    }
    idx = ALLOCA_N(ssize_t, argc);
    for (i=j=0; i<argc; i++) {
        a = argv[i];
        switch(TYPE(a)) {
        case T_FIXNUM:
            idx[j++] = FIX2LONG(a);
            break;
        case T_BIGNUM:
        case T_FLOAT:
            idx[j++] = NUM2SSIZET(a);
            break;
        case T_FALSE:
        case T_SYMBOL:
            if (a==cumo_sym_rest || a==cumo_sym_tilde || a==Qfalse) {
                argv[i] = Qfalse;
                count_rest++;
                break;
            } else if (a==cumo_sym_new || a==cumo_sym_minus) {
                argv[i] = cumo_sym_new;
                count_new++;
            }
        }
    }

    if (j != argc) {
        return check_index_count(argc, na->ndim, count_new, count_rest);
    }

    switch(na->type) {
    case CUMO_NARRAY_VIEW_T:
        CumoGetNArrayView(self,nv);
        pos = nv->offset;
        if (j == na->ndim) {
            for (i=j-1; i>=0; i--) {
                x = cumo_na_range_check(idx[i], na->shape[i], i);
                sdx = nv->stridx[i];
                if (CUMO_SDX_IS_INDEX(sdx)) {
                    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("index", "cumo_na_get_result_dimension");
                    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
                    pos += CUMO_SDX_GET_INDEX(sdx)[x];
                } else {
                    pos += CUMO_SDX_GET_STRIDE(sdx)*x;
                }
            }
            *pos_idx = pos;
            return 0;
        }
        if (j == 1) {
            x = cumo_na_range_check(idx[0], na->size, 0);
            for (i=na->ndim-1; i>=0; i--) {
                s = na->shape[i];
                m = x % s;
                x = x / s;
                sdx = nv->stridx[i];
                if (CUMO_SDX_IS_INDEX(sdx)) {
                    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("index", "cumo_na_get_result_dimension");
                    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
                    pos += CUMO_SDX_GET_INDEX(sdx)[m];
                } else {
                    pos += CUMO_SDX_GET_STRIDE(sdx)*m;
                }
            }
            *pos_idx = pos;
            return 0;
        }
        break;
    default:
        if (!stride) {
            stride = cumo_na_element_stride(self);
        }
        if (j == 1) {
            x = cumo_na_range_check(idx[0], na->size, 0);
            *pos_idx = stride * x;
            return 0;
        }
        if (j == na->ndim) {
            pos = 0;
            for (i=j-1; i>=0; i--) {
                x = cumo_na_range_check(idx[i], na->shape[i], i);
                pos += stride * x;
                stride *= na->shape[i];
            }
            *pos_idx = pos;
            return 0;
        }
    }
    rb_raise(rb_eIndexError,"# of index(=%i) should be "
             "equal to ndim(=%i) or 1", argc,na->ndim);
    return -1;
}


void
Init_cumo_na_index()
{
    rb_define_method(cNArray, "slice", cumo_na_slice, -1);

    cumo_sym_ast        = ID2SYM(rb_intern("*"));
    cumo_sym_all        = ID2SYM(rb_intern("all"));
    cumo_sym_minus      = ID2SYM(rb_intern("-"));
    cumo_sym_new        = ID2SYM(rb_intern("new"));
    cumo_sym_reverse    = ID2SYM(rb_intern("reverse"));
    cumo_sym_plus       = ID2SYM(rb_intern("+"));
    //cumo_sym_reduce   = ID2SYM(rb_intern("reduce"));
    cumo_sym_sum        = ID2SYM(rb_intern("sum"));
    cumo_sym_tilde      = ID2SYM(rb_intern("~"));
    cumo_sym_rest       = ID2SYM(rb_intern("rest"));
    cumo_id_beg         = rb_intern("begin");
    cumo_id_end         = rb_intern("end");
    cumo_id_exclude_end = rb_intern("exclude_end?");
    cumo_id_each        = rb_intern("each");
    cumo_id_step        = rb_intern("step");
    cumo_id_dup         = rb_intern("dup");
    cumo_id_bracket     = rb_intern("[]");
    cumo_id_shift_left  = rb_intern("<<");
    cumo_id_mask        = rb_intern("mask");
}
