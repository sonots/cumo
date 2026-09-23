#include <ruby.h>
#include "cumo.h"
#include "cumo/indexer.h"
#include "cumo/narray.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"

#if 0
#define DBG(x) x
#else
#define DBG(x)
#endif

#ifdef HAVE_STDARG_PROTOTYPES
#include <stdarg.h>
#define va_init_list(a,b) va_start(a,b)
#else
#include <varargs.h>
#define va_init_list(a,b) va_start(a)
#endif

typedef struct CUMO_NA_BUFFER_COPY {
    int ndim;
    size_t elmsz;
    size_t *n;
    char *src_ptr;
    char *buf_ptr;
    bool buf_on_device;
    cumo_na_loop_iter_t *src_iter;
    cumo_na_loop_iter_t *buf_iter;
} cumo_na_buffer_copy_t;

typedef struct CUMO_NA_LOOP_XARGS {
    cumo_na_loop_iter_t *iter;     // moved from cumo_na_loop_t
    cumo_na_buffer_copy_t *bufcp;  // copy data to buffer
    int flag;                 // CUMO_NDL_READ CUMO_NDL_WRITE
    bool free_user_iter;   // alloc LARG(lp,j).iter=lp->xargs[j].iter
} cumo_na_loop_xargs_t;

typedef struct CUMO_NA_MD_LOOP {
    int  narg;
    int  nin;
    int  ndim;                // n of total dimension looped at loop_narray. NOTE: lp->ndim + lp-.user.ndim is the total dimension.
    unsigned int copy_flag;   // set i-th bit if i-th arg is cast
    void    *ptr;             // memory for n
    cumo_na_loop_iter_t *iter_ptr; // memory for iter
    size_t  *n;               // n of elements for each dim (shape)
    cumo_na_loop_t  user;          // loop in user function
    cumo_na_loop_xargs_t *xargs;   // extra data for each arg
    int    writeback;         // write back result to i-th arg
    int    init_aidx;         // index of initializer argument
    int    reduce_dim;        // number of dimensions to reduce in reduction kernel, e.g., for an array of shape: [2,3,4],
                              // 3 for sum(), 1 for sum(axis: 1), 2 for sum(axis: [1,2])
    int   *trans_map;
    size_t sub_narray_len;    // elements the sub-narray of the current row reaches with, for loop_store_subnarray
    VALUE  vargs;
    VALUE  reduce;            // dimension indicies to reduce in reduction kernel (in bits), e.g., for an array of shape:
                              // [2,3,4], 111b for sum(), 010b for sum(axis: 1), 110b for sum(axis: [1,2])
    VALUE  loop_opt;
    cumo_ndfunc_t  *ndfunc;
    void (*loop_func)(cumo_ndfunc_t *, struct CUMO_NA_MD_LOOP *);
    struct CUMO_NA_HOST_STAGE *hs; // inputs staged for a host loop, or NULL
} cumo_na_md_loop_t;

// What a host loop reads through: the pinned copy of the byte range each
// input walks, and host copies of its index arrays. The originals are kept
// to put back at release.
typedef struct CUMO_NA_HOST_STAGE {
    int      narg;
    int      ndim;       // dimensions walked, the outer loop's and the user function's
    int      rows_only;  // the whole range did not fit: rows are copied as reached
    int      direct;     // inputs read where they are, which the device is waited for
    char   **ptr;        // per arg: the device pointer LARG(lp,j).ptr held, NULL if not staged
    size_t  *min;        // per arg: first byte of the range copied
    size_t  *len;        // per arg: bytes in the range, or in a row
    size_t  *off;        // per arg: where its copy starts in the buffer
    size_t **idx;        // per (dim, arg): the device index array LITER held
    size_t **host_idx;   // per (dim, arg): its host copy
    cumo_cuda_stage_t stage;
    uint64_t epoch;      // the launch count when the whole range was copied
    uint64_t row_epoch;  // the launch count when the current row was copied
} cumo_na_host_stage_t;

#define LARG(lp,iarg) ((lp)->user.args[iarg])
#define LITER(lp,idim,iarg) ((lp)->xargs[iarg].iter[idim])
#define LITER_SRC(lp,idim) ((lp)->src_iter[idim])
#define LBUFCP(lp,j) ((lp)->xargs[j].bufcp)

#define CASTABLE(t) (RTEST(t) && (t)!=CUMO_OVERWRITE)

#define CUMO_NDL_READ 1
#define CUMO_NDL_WRITE 2
#define CUMO_NDL_READ_WRITE (CUMO_NDL_READ|CUMO_NDL_WRITE)

static ID cumo_id_cast;
static ID cumo_id_extract;

static inline VALUE
cumo_na_type_s_cast(VALUE type, VALUE obj)
{
    return rb_funcall(type,cumo_id_cast,1,obj);
}

static void
print_ndfunc(cumo_ndfunc_t *nf) {
    volatile VALUE t;
    int i, k;
    printf("cumo_ndfunc_t = 0x%"SZF"x {\n",(size_t)nf);
    printf("  func  = 0x%"SZF"x\n", (size_t)nf->func);
    printf("  flag  = 0x%"SZF"x\n", (size_t)nf->flag);
    printf("  nin   = %d\n", nf->nin);
    printf("  nout  = %d\n", nf->nout);
    printf("  ain   = 0x%"SZF"x\n", (size_t)nf->ain);
    for (i=0; i<nf->nin; i++) {
        t = rb_inspect(nf->ain[i].type);
        printf("  ain[%d].type = %s\n", i, StringValuePtr(t));
        printf("  ain[%d].dim = %d\n", i, nf->ain[i].dim);
    }
    printf("  aout  = 0x%"SZF"x\n", (size_t)nf->aout);
    for (i=0; i<nf->nout; i++) {
        t = rb_inspect(nf->aout[i].type);
        printf("  aout[%d].type = %s\n", i, StringValuePtr(t));
        printf("  aout[%d].dim = %d\n", i, nf->aout[i].dim);
        for (k=0; k<nf->aout[i].dim; k++) {
            printf("  aout[%d].shape[%d] = %"SZF"u\n", i, k, nf->aout[i].shape[k]);
        }
    }
    printf("}\n");
}


static void
print_ndloop(cumo_na_md_loop_t *lp) {
    int i,j,nd;
    printf("cumo_na_md_loop_t = 0x%"SZF"x {\n",(size_t)lp);
    printf("  narg = %d\n", lp->narg);
    printf("  nin  = %d\n", lp->nin);
    printf("  ndim = %d\n", lp->ndim);
    printf("  copy_flag = %x\n", lp->copy_flag);
    printf("  writeback = %d\n", lp->writeback);
    printf("  init_aidx = %d\n", lp->init_aidx);
    printf("  reduce_dim = %d\n", lp->reduce_dim);
    printf("  trans_map = 0x%"SZF"x\n", (size_t)lp->trans_map);
    nd = lp->ndim + lp->user.ndim;
    for (i=0; i<nd; i++) {
        printf("  trans_map[%d] = %d\n", i, lp->trans_map[i]);
    }
    printf("  n = 0x%"SZF"x\n", (size_t)lp->n);
    nd = lp->ndim + lp->user.ndim;
    for (i=0; i<=lp->ndim; i++) {
        printf("  n[%d] = %"SZF"u\n", i, lp->n[i]);
    }
    printf("  user.n = 0x%"SZF"x\n", (size_t)lp->user.n);
    if (lp->user.n) {
        for (i=0; i<=lp->user.ndim; i++) {
            printf("  user.n[%d] = %"SZF"u\n", i, lp->user.n[i]);
        }
    }
    printf("  xargs = 0x%"SZF"x\n", (size_t)lp->xargs);
    printf("  iter_ptr = 0x%"SZF"x\n", (size_t)lp->iter_ptr);
    printf("  user.narg = %d\n", lp->user.narg);
    printf("  user.ndim = %d\n", lp->user.ndim);
    printf("  user.args = 0x%"SZF"x\n", (size_t)lp->user.args);
    for (j=0; j<lp->narg; j++) {
    }
    printf("  user.opt_ptr = 0x%"SZF"x\n", (size_t)lp->user.opt_ptr);
    if (lp->reduce==Qnil) {
        printf("  reduce  = nil\n");
    } else {
        printf("  reduce  = 0x%x\n", NUM2INT(lp->reduce));
    }
    for (j=0; j<lp->narg; j++) {
        printf("--user.args[%d]--\n", j);
        printf("  user.args[%d].ptr = 0x%"SZF"x\n", j, (size_t)LARG(lp,j).ptr);
        printf("  user.args[%d].elmsz = %"SZF"d\n", j, LARG(lp,j).elmsz);
        printf("  user.args[%d].value = 0x%"PRI_VALUE_PREFIX"x\n", j, LARG(lp,j).value);
        printf("  user.args[%d].ndim = %d\n", j, LARG(lp,j).ndim);
        printf("  user.args[%d].shape = 0x%"SZF"x\n", j, (size_t)LARG(lp,j).shape);
        if (LARG(lp,j).shape) {
            for (i=0; i<LARG(lp,j).ndim; i++) {
                printf("  user.args[%d].shape[%d] = %"SZF"d\n", j, i, LARG(lp,j).shape[i]);
            }
        }
        printf("  user.args[%d].iter = 0x%"SZF"x\n", j,(size_t)lp->user.args[j].iter);
        if (lp->user.args[j].iter) {
            for (i=0; i<lp->user.ndim; i++) {
                printf(" &user.args[%d].iter[%d] = 0x%"SZF"x\n", j,i, (size_t)&lp->user.args[j].iter[i]);
                printf("  user.args[%d].iter[%d].pos = %"SZF"u\n", j,i, lp->user.args[j].iter[i].pos);
                printf("  user.args[%d].iter[%d].step = %"SZF"d\n", j,i, lp->user.args[j].iter[i].step);
                printf("  user.args[%d].iter[%d].idx = 0x%"SZF"x (cuda:%d)\n", j,i, (size_t)lp->user.args[j].iter[i].idx, cumo_cuda_runtime_is_device_memory(lp->user.args[j].iter[i].idx));
                // printf("  user.args[%d].iter[%d].idx = 0x%"SZF"x\n", j,i, (size_t)lp->user.args[j].iter[i].idx);
            }
        }
        //
        printf("  xargs[%d].flag = %d\n", j, lp->xargs[j].flag);
        printf("  xargs[%d].free_user_iter = %d\n", j, lp->xargs[j].free_user_iter);
        for (i=0; i<=nd; i++) {
            printf(" &xargs[%d].iter[%d] = 0x%"SZF"x\n", j,i, (size_t)&LITER(lp,i,j));
            printf("  xargs[%d].iter[%d].pos = %"SZF"u\n", j,i, LITER(lp,i,j).pos);
            printf("  xargs[%d].iter[%d].step = %"SZF"d\n", j,i, LITER(lp,i,j).step);
            printf("  xargs[%d].iter[%d].idx = 0x%"SZF"x (cuda:%d)\n", j,i, (size_t)LITER(lp,i,j).idx, cumo_cuda_runtime_is_device_memory(LITER(lp,i,j).idx));
            // printf("  xargs[%d].iter[%d].idx = 0x%"SZF"x\n", j,i, (size_t)LITER(lp,i,j).idx);
        }
        printf("  xargs[%d].bufcp = 0x%"SZF"x\n", j, (size_t)lp->xargs[j].bufcp);
        if (lp->xargs[j].bufcp) {
            printf("  xargs[%d].bufcp->ndim = %d\n", j, lp->xargs[j].bufcp->ndim);
            printf("  xargs[%d].bufcp->elmsz = %"SZF"d\n", j, lp->xargs[j].bufcp->elmsz);
            printf("  xargs[%d].bufcp->n = 0x%"SZF"x\n", j, (size_t)lp->xargs[j].bufcp->n);
            for (i=0; i<=lp->xargs[j].bufcp->ndim; i++) {
                printf("  xargs[%d].bufcp->n[%d] = %"SZF"u\n", j, i, lp->xargs[j].bufcp->n[i]);
            }
            printf("  xargs[%d].bufcp->src_ptr = 0x%"SZF"x\n", j, (size_t)lp->xargs[j].bufcp->src_ptr);
            printf("  xargs[%d].bufcp->buf_ptr = 0x%"SZF"x\n", j, (size_t)lp->xargs[j].bufcp->buf_ptr);
            printf("  xargs[%d].bufcp->src_iter = 0x%"SZF"x\n", j, (size_t)lp->xargs[j].bufcp->src_iter);
            printf("  xargs[%d].bufcp->buf_iter = 0x%"SZF"x\n", j, (size_t)lp->xargs[j].bufcp->buf_iter);
        }
    }
    printf("}\n");
}


// returns 0x01 if CUMO_NDF_HAS_LOOP, but not supporting CUMO_NDF_STRIDE_LOOP
// returns 0x02 if CUMO_NDF_HAS_LOOP, but not supporting CUMO_NDF_INDEX_LOOP
static unsigned int
ndloop_func_loop_spec(cumo_ndfunc_t *nf, int user_ndim)
{
    unsigned int f=0;
    // If user function supports LOOP
    if (user_ndim > 0 || CUMO_NDF_TEST(nf,CUMO_NDF_HAS_LOOP)) {
        if (!CUMO_NDF_TEST(nf,CUMO_NDF_STRIDE_LOOP)) {
            f |= 1;
        }
        if (!CUMO_NDF_TEST(nf,CUMO_NDF_INDEX_LOOP)) {
            f |= 2;
        }
    }
    return f;
}




static int
ndloop_cast_required(VALUE type, VALUE value)
{
    return CASTABLE(type) && type != rb_obj_class(value);
}

static int
ndloop_castable_type(VALUE type)
{
    return rb_obj_is_kind_of(type, rb_cClass) && RTEST(rb_class_inherited_p(type, cNArray));
}

static void
ndloop_cast_error(VALUE type, VALUE value)
{
    VALUE x = rb_inspect(type);
    char* s = StringValueCStr(x);
    rb_bug("fail cast from %s to %s", rb_obj_classname(value),s);
    rb_raise(rb_eTypeError,"fail cast from %s to %s",
             rb_obj_classname(value), s);
}

// convert input argeuments given by RARRAY_PTR(args)[j]
//              to type specified by nf->args[j].type
// returns copy_flag where nth-bit is set if nth argument is converted.
static unsigned int
ndloop_cast_args(cumo_ndfunc_t *nf, VALUE args)
{
    int j;
    unsigned int copy_flag=0;
    VALUE type, value;

    for (j=0; j<nf->nin; j++) {

        type = nf->ain[j].type;
        if (TYPE(type)==T_SYMBOL)
            continue;
        value = RARRAY_AREF(args,j);
        if (!ndloop_cast_required(type, value))
            continue;

        if (ndloop_castable_type(type)) {
            RARRAY_ASET(args,j,cumo_na_type_s_cast(type, value));
            copy_flag |= 1<<j;
        } else {
            ndloop_cast_error(type, value);
        }
    }

    RB_GC_GUARD(type); RB_GC_GUARD(value);
    return copy_flag;
}


static void
ndloop_handle_symbol_in_ain(VALUE type, VALUE value, int at, cumo_na_md_loop_t *lp)
{
    if (type==cumo_sym_reduce) {
        lp->reduce = value;
    }
    else if (type==cumo_sym_option) {
        lp->user.option = value;
    }
    else if (type==cumo_sym_loop_opt) {
        lp->loop_opt = value;
    }
    else if (type==cumo_sym_init) {
        lp->init_aidx = at;
    }
    else {
        rb_bug("ndloop parse_options: unknown type");
    }
}

static inline int
max2(int x, int y)
{
    return x > y ? x : y;
}

static void
ndloop_find_max_dimension(cumo_na_md_loop_t *lp, cumo_ndfunc_t *nf, VALUE args)
{
    int j;
    int nin=0; // number of input objects (except for symbols)
    int user_nd=0; // max dimension of user function
    int loop_nd=0; // max dimension of md-loop

    for (j=0; j<RARRAY_LEN(args); j++) {
        VALUE t = nf->ain[j].type;
        VALUE v = RARRAY_AREF(args,j);
        if (TYPE(t)==T_SYMBOL) {
            ndloop_handle_symbol_in_ain(t, v, j, lp);
        } else {
            nin++;
            user_nd = max2(user_nd, nf->ain[j].dim);
            if (CumoIsNArray(v))
                loop_nd = max2(loop_nd, CUMO_RNARRAY_NDIM(v) - nf->ain[j].dim);
        }
    }

    lp->narg = lp->user.narg = nin + nf->nout;
    lp->nin = nin;
    lp->ndim = loop_nd;
    lp->user.ndim = user_nd;
}

/*
  user-dimension:
    user_nd = MAX( nf->args[j].dim )

  user-support dimension:

  loop dimension:
    loop_nd
*/

static void
ndloop_alloc(cumo_na_md_loop_t *lp, cumo_ndfunc_t *nf, VALUE args,
             void *opt_ptr, unsigned int copy_flag,
             void (*loop_func)(cumo_ndfunc_t*, cumo_na_md_loop_t*))
{
    int i,j;
    int narg;
    int max_nd;

    char *buf;
    size_t n1, n2, n3, n4, n5;

    long args_len;

    cumo_na_loop_iter_t *iter;

    int trans_dim;
    unsigned int f;

    args_len = RARRAY_LEN(args);

    if (args_len != nf->nin) {
        rb_bug("wrong number of arguments for ndfunc (%lu for %d)",
               args_len, nf->nin);
    }

    lp->vargs = args;
    lp->ndfunc = nf;
    lp->loop_func = loop_func;
    lp->copy_flag = copy_flag;

    lp->reduce = Qnil;
    lp->user.option = Qnil;
    lp->user.opt_ptr = opt_ptr;
    lp->user.err_type = Qfalse;
    lp->loop_opt = Qnil;
    lp->writeback = -1;
    lp->init_aidx = -1;

    lp->ptr = NULL;
    lp->user.n = NULL;
    lp->hs = NULL;

    ndloop_find_max_dimension(lp, nf, args);
    narg = lp->nin + nf->nout;
    max_nd = lp->ndim + lp->user.ndim;

    n1 = sizeof(size_t)*(max_nd+1);
    n2 = sizeof(cumo_na_loop_xargs_t)*narg;
    n2 = ((n2-1)/8+1)*8;
    n3 = sizeof(cumo_na_loop_args_t)*narg;
    n3 = ((n3-1)/8+1)*8;
    n4 = sizeof(cumo_na_loop_iter_t)*narg*(max_nd+1);
    n4 = ((n4-1)/8+1)*8;
    n5 = sizeof(int)*(max_nd+1);

    lp->ptr = buf = (char*)xmalloc(n1+n2+n3+n4+n5);
    lp->n = (size_t*)buf; buf+=n1;
    lp->xargs = (cumo_na_loop_xargs_t*)buf; buf+=n2;
    lp->user.args = (cumo_na_loop_args_t*)buf; buf+=n3;
    lp->iter_ptr = iter = (cumo_na_loop_iter_t*)buf; buf+=n4;
    lp->trans_map = (int*)buf;

    for (j=0; j<narg; j++) {
        LARG(lp,j).value = Qnil;
        LARG(lp,j).iter = NULL;
        LARG(lp,j).shape = NULL;
        LARG(lp,j).ndim = 0;
        lp->xargs[j].iter = &(iter[(max_nd+1)*j]);
        lp->xargs[j].bufcp = NULL;
        lp->xargs[j].flag = (j<lp->nin) ? CUMO_NDL_READ : CUMO_NDL_WRITE;
        lp->xargs[j].free_user_iter = 0;
    }

    for (i=0; i<=max_nd; i++) {
        lp->n[i] = 1;
        for (j=0; j<narg; j++) {
            LITER(lp,i,j).pos = 0;
            LITER(lp,i,j).step = 0;
            LITER(lp,i,j).idx = NULL;
        }
    }

    // transpose reduce-dimensions to last dimensions
    //              array          loop
    //           [*,+,*,+,*] => [*,*,*,+,+]
    // trans_map=[0,3,1,4,2] <= [0,1,2,3,4]
    if (CUMO_NDF_TEST(nf,CUMO_NDF_FLAT_REDUCE) && RTEST(lp->reduce)) {
        trans_dim = 0;
        for (i=0; i<max_nd; i++) {
            if (cumo_na_test_reduce(lp->reduce, i)) {
                lp->trans_map[i] = -1;
            } else {
                lp->trans_map[i] = trans_dim++;
            }
        }
        j = trans_dim;
        for (i=0; i<max_nd; i++) {
            if (lp->trans_map[i] == -1) {
                lp->trans_map[i] = j++;
            }
        }
        lp->reduce_dim = max_nd - trans_dim;
        f = 0;
        for (i=trans_dim; i<max_nd; i++) {
            f |= 1<<i;
        }
        lp->reduce = INT2FIX(f);
    } else {
        for (i=0; i<max_nd; i++) {
            lp->trans_map[i] = i;
        }
        lp->reduce_dim = 0;
    }
}


static void ndloop_unstage_host_reads(cumo_na_md_loop_t *lp);

static VALUE
ndloop_release(VALUE vlp)
{
    int j;
    VALUE v;
    cumo_na_md_loop_t *lp = (cumo_na_md_loop_t*)(vlp);

    ndloop_unstage_host_reads(lp);
    for (j=0; j < lp->narg; j++) {
        v = LARG(lp,j).value;
        if (CumoIsNArray(v)) {
            cumo_na_release_lock(v);
        }
    }
    for (j=0; j<lp->narg; j++) {
        //printf("lp->xargs[%d].bufcp=%lx\n",j,(size_t)(lp->xargs[j].bufcp));
        if (lp->xargs[j].bufcp) {
            xfree(lp->xargs[j].bufcp->buf_iter);
            if (lp->xargs[j].bufcp->buf_on_device) {
                // Raising here would replace the exception and skip the frees
                // below. The wait stays out, as it has been.
                cumo_cuda_runtime_return_scratch(lp->xargs[j].bufcp->buf_ptr, 0, NULL);
            }
            else {
                xfree(lp->xargs[j].bufcp->buf_ptr);
            }
            xfree(lp->xargs[j].bufcp->n);
            xfree(lp->xargs[j].bufcp);
            if (lp->xargs[j].free_user_iter) {
                xfree(LARG(lp,j).iter);
            }
        }
    }
    xfree(lp->ptr);
    return Qnil;
}


/*
  set lp->n[i] (shape of n-d iteration) here
*/
static void
ndloop_check_shape(cumo_na_md_loop_t *lp, int nf_dim, cumo_narray_t *na)
{
    int i, k;
    size_t n;
    int dim_beg;

    dim_beg = lp->ndim + nf_dim - na->ndim;

    for (k = na->ndim - nf_dim - 1; k>=0; k--) {
        i = lp->trans_map[k + dim_beg];
        n = na->shape[k];
        // if n==1 then repeat this dimension
        if (n != 1) {
            if (lp->n[i] == 1) {
                lp->n[i] = n;
            } else if (lp->n[i] != n) {
                // inconsistent array shape
                rb_raise(cumo_na_eShapeError,"shape1[%d](=%"SZF"u) != shape2[%d](=%"SZF"u)",
                         i, lp->n[i], k, n);
            }
        }
    }
}


/*
na->shape[i] == lp->n[ dim_map[i] ]
 */
static void
ndloop_set_stepidx(cumo_na_md_loop_t *lp, int j, VALUE vna, int *dim_map, int rwflag)
{
    size_t n, s;
    int i, k, nd;
    cumo_stridx_t sdx;
    cumo_narray_t *na;

    LARG(lp,j).value = vna;
    LARG(lp,j).elmsz = cumo_na_element_stride(vna);
    if (rwflag == CUMO_NDL_READ) {
        LARG(lp,j).ptr = cumo_na_get_pointer_for_read(vna);
    } else
    if (rwflag == CUMO_NDL_WRITE) {
        LARG(lp,j).ptr = cumo_na_get_pointer_for_write(vna);
    } else
    if (rwflag == CUMO_NDL_READ_WRITE) {
        LARG(lp,j).ptr = cumo_na_get_pointer_for_read_write(vna);
    } else {
        rb_bug("invalid value for read-write flag");
    }
    CumoGetNArray(vna,na);
    nd = LARG(lp,j).ndim;

    switch(CUMO_NA_TYPE(na)) {
    case CUMO_NARRAY_DATA_T:
        if (CUMO_NA_DATA_PTR(na)==NULL && CUMO_NA_SIZE(na)>0) {
            rb_bug("cannot read no-data NArray");
            rb_raise(rb_eRuntimeError,"cannot read no-data NArray");
        }
        // through
    case CUMO_NARRAY_FILEMAP_T:
        s = LARG(lp,j).elmsz;
        for (k=na->ndim; k--;) {
            n = na->shape[k];
            if (n > 1 || nd > 0) {
                i = dim_map[k];
                //printf("n=%d k=%d i=%d\n",n,k,i);
                LITER(lp,i,j).step = s;
                //LITER(lp,i,j).idx = NULL;
            }
            s *= n;
            nd--;
        }
        LITER(lp,0,j).pos = 0;
        break;
    case CUMO_NARRAY_VIEW_T:
        LITER(lp,0,j).pos = CUMO_NA_VIEW_OFFSET(na);
        for (k=0; k<na->ndim; k++) {
            n = na->shape[k];
            sdx = CUMO_NA_VIEW_STRIDX(na)[k];
            if (n > 1 || nd > 0) {
                i = dim_map[k];
                if (CUMO_SDX_IS_INDEX(sdx)) {
                    LITER(lp,i,j).step = 0;
                    LITER(lp,i,j).idx = CUMO_SDX_GET_INDEX(sdx);
                } else {
                    LITER(lp,i,j).step = CUMO_SDX_GET_STRIDE(sdx);
                    //LITER(lp,i,j).idx = NULL;
                }
            } else if (n==1) {
                if (CUMO_SDX_IS_INDEX(sdx)) {
                    cumo_na_index_wait_fill((cumo_narray_view_t *)na);
                    LITER(lp,0,j).pos += CUMO_SDX_GET_INDEX(sdx)[0];
                }
            }
            nd--;
        }
        break;
    default:
        rb_bug("invalid narray internal type");
    }
}



static void
ndloop_init_args(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp, VALUE args)
{
    int i, j;
    VALUE v;
    cumo_narray_t *na;
    int nf_dim;
    int dim_beg;
    int *dim_map;
    int max_nd = lp->ndim + lp->user.ndim;
    int flag;

/*
na->shape[i] == lp->n[ dim_map[i] ]
 */
    dim_map = ALLOCA_N(int, max_nd);

    // input arguments
    for (j=0; j<nf->nin; j++) {
        if (TYPE(nf->ain[j].type)==T_SYMBOL) {
            continue;
        }
        v = RARRAY_AREF(args,j);
        if (CumoIsNArray(v)) {
            // set LARG(lp,j) with v
            CumoGetNArray(v,na);
            nf_dim = nf->ain[j].dim;
            if (nf_dim > na->ndim) {
                rb_raise(cumo_na_eDimensionError,"requires >= %d-dimensioal array "
                         "while %d-dimensional array is given",nf_dim,na->ndim);
            }
            ndloop_check_shape(lp, nf_dim, na);
            dim_beg = lp->ndim + nf->ain[j].dim - na->ndim;
            for (i=0; i<na->ndim; i++) {
                dim_map[i] = lp->trans_map[i+dim_beg];
                //printf("dim_map[%d]=%d na->shape[%d]=%d\n",i,dim_map[i],i,na->shape[i]);
            }
            if (nf->ain[j].type==CUMO_OVERWRITE) {
                lp->xargs[j].flag = flag = CUMO_NDL_WRITE;
            } else {
                lp->xargs[j].flag = flag = CUMO_NDL_READ;
            }
            LARG(lp,j).ndim = nf_dim;
            ndloop_set_stepidx(lp, j, v, dim_map, flag);
            if (nf_dim > 0) {
                LARG(lp,j).shape = na->shape + (na->ndim - nf_dim);
            }
        } else if (TYPE(v)==T_ARRAY) {
            LARG(lp,j).value = v;
            LARG(lp,j).elmsz = sizeof(VALUE);
            LARG(lp,j).ptr   = NULL;
            for (i=0; i<=max_nd; i++) {
                LITER(lp,i,j).step = 1;
            }
        }
    }
}


static int
ndloop_check_inplace(VALUE type, int cumo_na_ndim, size_t *cumo_na_shape, VALUE v)
{
    int i;
    cumo_narray_t *na;

    // type check
    if (type != rb_obj_class(v)) {
        return 0;
    }
    CumoGetNArray(v,na);
    // shape check
    if (na->ndim != cumo_na_ndim) {
        return 0;
    }
    for (i=0; i<cumo_na_ndim; i++) {
        if (cumo_na_shape[i] != na->shape[i]) {
            return 0;
        }
    }
    // v is selected as output
    return 1;
}

static VALUE
ndloop_find_inplace(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp, VALUE type,
                    int cumo_na_ndim, size_t *cumo_na_shape, VALUE args)
{
    int j;
    VALUE v;

    // find inplace
    for (j=0; j<nf->nin; j++) {
        v = RARRAY_AREF(args,j);
        if (CumoIsNArray(v)) {
            if (CUMO_TEST_INPLACE(v)) {
                if (ndloop_check_inplace(type,cumo_na_ndim,cumo_na_shape,v)) {
                    // if already copied, create outary and write-back
                    if (lp->copy_flag & (1<<j)) {
                        lp->writeback = j;
                    }
                    return v;
                }
            }
        }
    }
    // find casted or copied input array
    for (j=0; j<nf->nin; j++) {
        if (lp->copy_flag & (1<<j)) {
            v = RARRAY_AREF(args,j);
            if (ndloop_check_inplace(type,cumo_na_ndim,cumo_na_shape,v)) {
                return v;
            }
        }
    }
    return Qnil;
}



static VALUE
ndloop_get_arg_type(cumo_ndfunc_t *nf, VALUE args, VALUE t)
{
    int i;

    // if type is FIXNUM, get the type of i-th argument
    if (FIXNUM_P(t)) {
        i = FIX2INT(t);
        if (i<0 || i>=nf->nin) {
            rb_bug("invalid type: index (%d) out of # of args",i);
        }
        t = nf->ain[i].type;
        // if i-th type is Qnil, get the type of i-th input value
        if (!CASTABLE(t)) {
            t = rb_obj_class(RARRAY_AREF(args,i));
        }
    }
    return t;
}


static VALUE
ndloop_set_output_narray(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp, int k,
                         VALUE type, VALUE args)
{
    int i, j;
    int cumo_na_ndim;
    int lp_dim;
    volatile VALUE v=Qnil;
    size_t *cumo_na_shape;
    int *dim_map;
    int flag = CUMO_NDL_READ_WRITE;
    int nd;
    int max_nd = lp->ndim + nf->aout[k].dim;

    cumo_na_shape = ALLOCA_N(size_t, max_nd);
    dim_map = ALLOCA_N(int, max_nd);

    //printf("max_nd=%d lp->ndim=%d\n",max_nd,lp->ndim);

    // md-loop shape
    cumo_na_ndim = 0;
    for (i=0; i<lp->ndim; i++) {
        // cumo_na_shape[i] == lp->n[lp->trans_map[i]]
        lp_dim = lp->trans_map[i];
        //printf("i=%d lp_dim=%d\n",i,lp_dim);
        if (CUMO_NDF_TEST(nf,CUMO_NDF_CUM)) {   // cumulate with shape kept
            cumo_na_shape[cumo_na_ndim] = lp->n[lp_dim];
        } else
        if (cumo_na_test_reduce(lp->reduce,lp_dim)) {   // accumulate dimension
            if (CUMO_NDF_TEST(nf,CUMO_NDF_KEEP_DIM)) {
                cumo_na_shape[cumo_na_ndim] = 1;         // leave it
            } else {
                continue;  // delete dimension
            }
        } else {
            cumo_na_shape[cumo_na_ndim] = lp->n[lp_dim];
        }
        //printf("i=%d lp_dim=%d cumo_na_shape[%d]=%ld\n",i,lp_dim,i,cumo_na_shape[i]);
        dim_map[cumo_na_ndim++] = lp_dim;
        //dim_map[lp_dim] = cumo_na_ndim++;
    }

    // user-specified shape
    for (i=0; i<nf->aout[k].dim; i++) {
        cumo_na_shape[cumo_na_ndim] = nf->aout[k].shape[i];
        dim_map[cumo_na_ndim++] = i + lp->ndim;
    }

    // find inplace from input arrays
    if (k==0 && CUMO_NDF_TEST(nf,CUMO_NDF_INPLACE)) {
        v = ndloop_find_inplace(nf,lp,type,cumo_na_ndim,cumo_na_shape,args);
    }
    if (!RTEST(v)) {
        // new object
        v = cumo_na_new(type, cumo_na_ndim, cumo_na_shape);
        flag = CUMO_NDL_WRITE;
    }

    j = lp->nin + k;
    LARG(lp,j).ndim = nd = nf->aout[k].dim;
    ndloop_set_stepidx(lp, j, v, dim_map, flag);
    if (nd > 0) {
        LARG(lp,j).shape = nf->aout[k].shape;
    }

    return v;
}

static VALUE
ndloop_set_output(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp, VALUE args)
{
    int i, j, k, idx;
    volatile VALUE v, t, results;
    VALUE init;

    int max_nd = lp->ndim + lp->user.ndim;

    // output results
    results = rb_ary_new2(nf->nout);

    for (k=0; k<nf->nout; k++) {
        t = nf->aout[k].type;
        t = ndloop_get_arg_type(nf,args,t);

        if (rb_obj_is_kind_of(t, rb_cClass)) {
            if (RTEST(rb_class_inherited_p(t, cNArray))) {
                // NArray
                v = ndloop_set_output_narray(nf,lp,k,t,args);
                rb_ary_push(results, v);
            }
            else if (RTEST(rb_class_inherited_p(t, rb_cArray))) {
                // Ruby Array
                j = lp->nin + k;
                for (i=0; i<=max_nd; i++) {
                    LITER(lp,i,j).step = sizeof(VALUE);
                }
                LARG(lp,j).value = t;
                LARG(lp,j).elmsz = sizeof(VALUE);
            } else {
                rb_raise(rb_eRuntimeError,"ndloop_set_output: invalid for type");
            }
        }
    }

    // initialilzer
    k = lp->init_aidx;
    if (k > -1) {
        idx = nf->ain[k].dim;
        v = RARRAY_AREF(results,idx);
        init = RARRAY_AREF(args,k);
        cumo_na_store(v,init);
    }

    return results;
}


// Compressing dimesions.
//
// For example, compressing [2,3] shape into [6] so that we can process
// all elements with one user loop.
static void
cumo_ndfunc_contract_loop(cumo_na_md_loop_t *lp)
{
    int i,j,k,success,cnt=0;
    int red0, redi;

    redi = cumo_na_test_reduce(lp->reduce,0);

    //for (i=0; i<lp->ndim; i++) {
    //    printf("lp->n[%d]=%lu\n",i,lp->n[i]);
    //}

    for (i=1; i<lp->ndim; i++) {
        red0 = redi;
        redi = cumo_na_test_reduce(lp->reduce,i);
        //printf("contract i=%d reduce_cond=%d %d\n",i,red0,redi);
        if (red0 != redi) {
            continue;
        }
        success = 1;
        for (j=0; j<lp->narg; j++) {
            if (!(LITER(lp,i,j).idx == NULL &&
                  LITER(lp,i-1,j).idx == NULL &&
                  (lp->n[i] == 1 || lp->n[i-1] == 1 ||
                   LITER(lp,i-1,j).step == LITER(lp,i,j).step*(ssize_t)(lp->n[i])))) {
                success = 0;
                break;
            }
        }
        if (success) {
            //printf("contract i=%d-th and %d-th, lp->n[%d]=%"SZF"d, lp->n[%d]=%"SZF"d\n",
            //       i-1,i, i,lp->n[i], i-1,lp->n[i-1]);
            // A dimension of length 1 never moves the pointer, so its step
            // is left unset; the merged one walks with the other's.
            if (lp->n[i] == 1) {
                for (j=0; j<lp->narg; j++) {
                    LITER(lp,i,j).step = LITER(lp,i-1,j).step;
                }
            }
            // contract (i-1)-th and i-th dimension
            lp->n[i] *= lp->n[i-1];
            // shift dimensions
            for (k=i-1; k>cnt; k--) {
                lp->n[k] = lp->n[k-1];
            }
            //printf("k=%d\n",k);
            for (; k>=0; k--) {
                lp->n[k] = 1;
            }
            for (j=0; j<lp->narg; j++) {
                for (k=i-1; k>cnt; k--) {
                    LITER(lp,k,j) = LITER(lp,k-1,j);
                }
            }
            if (redi) {
                lp->reduce_dim--;
            }
            cnt++;
        }
    }
    //printf("contract cnt=%d\n",cnt);
    if (cnt>0) {
        for (j=0; j<lp->narg; j++) {
            LITER(lp,cnt,j).pos = LITER(lp,0,j).pos;
            lp->xargs[j].iter = &LITER(lp,cnt,j);
        }
        lp->n = &(lp->n[cnt]);
        lp->ndim -= cnt;
        //for (i=0; i<lp->ndim; i++) {printf("lp->n[%d]=%lu\n",i,lp->n[i]);}
    }
}


// Order the axes so that the one the written operand walks with the smallest
// step is innermost, which is the one consecutive threads of an indexer kernel
// take. A view that transposes a contiguous array otherwise hands the kernel
// an inner axis whose step is a whole row, and every warp touches a line per
// element: a + b on two transposed [4096,1024] views ran at 74 GB/s where the
// contiguous add ran at 1500. Only for functions that declared the order of
// their elements does not matter, and never through an index array.
static void
cumo_ndfunc_reorder_loop(cumo_na_md_loop_t *lp)
{
    int i, j, k, nd = lp->ndim;
    int order[CUMO_NA_MAX_DIMENSION];
    ssize_t key[CUMO_NA_MAX_DIMENSION];
    size_t n[CUMO_NA_MAX_DIMENSION];
    cumo_na_loop_iter_t iter[CUMO_NA_MAX_DIMENSION];
    int written = (lp->narg > lp->nin) ? lp->nin : 0;

    if (nd < 2) return;
    for (i = 0; i < nd; i++) {
        for (j = 0; j < lp->narg; j++) {
            if (LITER(lp,i,j).idx) return;
        }
        key[i] = LITER(lp,i,written).step;
        if (key[i] < 0) key[i] = -key[i];
        order[i] = i;
    }
    // A stable insertion sort on the step magnitude, largest first, so that
    // axes the written operand does not tell apart keep their order.
    for (i = 1; i < nd; i++) {
        int d = order[i];
        for (k = i; k > 0 && key[order[k-1]] < key[d]; k--) {
            order[k] = order[k-1];
        }
        order[k] = d;
    }
    for (i = 0; i < nd; i++) {
        if (order[i] != i) break;
    }
    if (i == nd) return;

    for (i = 0; i < nd; i++) {
        n[i] = lp->n[order[i]];
    }
    for (i = 0; i < nd; i++) {
        lp->n[i] = n[i];
    }
    for (j = 0; j < lp->narg; j++) {
        // Only iter[0].pos carries the offset, so it stays where it is.
        ssize_t pos = LITER(lp,0,j).pos;
        for (i = 0; i < nd; i++) {
            iter[i] = LITER(lp,order[i],j);
        }
        for (i = 0; i < nd; i++) {
            LITER(lp,i,j) = iter[i];
            LITER(lp,i,j).pos = 0;
        }
        LITER(lp,0,j).pos = pos;
    }
}

// Ndloop does loop at two places, loop_narray and user loop.
// loop_narray is an outer loop, and the user loop is an internal loop.
//
// lp->ndim: ndim to be looped at loop_narray
// lp->user.ndim: ndim to be looped at user function
//
// For example, for element-wise function, lp->user.ndim is 1, and lp->ndim -= 1.
static void
cumo_ndfunc_set_user_loop(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp)
{
    int j, ud=0;

    if (lp->reduce_dim > 0) {
        // Increase user.ndim by number of dimensions to reduce for reduction function.
        ud = lp->reduce_dim;
    }
    else if (lp->ndim > 0 && CUMO_NDF_TEST(nf,CUMO_NDF_HAS_LOOP)) {
        // Set user.ndim to 1 (default is 0) for element-wise function.
        ud = 1;
    }
    else {
        goto skip_ud;
    }
    if (ud > lp->ndim) {
        rb_bug("Reduce-dimension is larger than loop-dimension");
    }
    // Increase user loop dimension. NOTE: lp->ndim + lp->user.ndim is the total dimension.
    lp->user.ndim += ud;
    lp->ndim -= ud;
    for (j=0; j<lp->narg; j++) {
        if (LARG(lp,j).shape) {
            rb_bug("HAS_LOOP or reduce-dimension=%d conflicts with user-dimension",lp->reduce_dim);
        }
        LARG(lp,j).ndim += ud;
        LARG(lp,j).shape = &(lp->n[lp->ndim]);
        //printf("LARG(lp,j).ndim=%d,LARG(lp,j).shape=%lx\n",LARG(lp,j).ndim,(size_t)LARG(lp,j).shape);
    }
    //printf("lp->reduce_dim=%d lp->user.ndim=%d lp->ndim=%d\n",lp->reduce_dim,lp->user.ndim,lp->ndim);

 skip_ud:
    // user function shape is the latter part of cumo_na_md_loop shape.
    lp->user.n = &(lp->n[lp->ndim]);
    for (j=0; j<lp->narg; j++) {
        LARG(lp,j).iter = &LITER(lp,lp->ndim,j);
        //printf("in cumo_ndfunc_set_user_loop: lp->user.args[%d].iter=%lx\n",j,(size_t)(LARG(lp,j).iter));
    }
}


// Initialize lp->user for indexer loop.
static void
cumo_ndfunc_set_user_indexer_loop(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp)
{
    int j;

    lp->user.ndim = lp->ndim;
    lp->ndim = 0;

    if (CUMO_NDF_TEST(nf,CUMO_NDF_FLAT_REDUCE)) {
        // in
        LARG(lp,0).ndim = lp->user.ndim;
        LARG(lp,0).shape = &(lp->n[lp->ndim]);
        // out is constructed at cumo_na_make_reduction_arg from in and lp->reduce,
        // except for a cumulative function, whose output keeps the shape it was
        // given and so is laid out like the input rather than like a reduction.
        if (CUMO_NDF_TEST(nf,CUMO_NDF_CUM)) {
            for (j=1; j<lp->narg; j++) {
                LARG(lp,j).ndim = lp->user.ndim;
                LARG(lp,j).shape = &(lp->n[lp->ndim]);
            }
        }

        lp->user.n = &(lp->n[lp->ndim]);
        for (j=0; j<lp->narg; j++) {
            LARG(lp,j).iter = &LITER(lp,lp->ndim,j);
        }

        lp->user.reduce_dim = lp->reduce_dim;
        lp->user.reduce = lp->reduce;
    } else { // element-wise
        for (j=0; j<lp->narg; j++) {
            LARG(lp,j).ndim = lp->user.ndim;
            LARG(lp,j).shape = &(lp->n[lp->ndim]);
        }

        lp->user.n = &(lp->n[lp->ndim]);
        for (j=0; j<lp->narg; j++) {
            LARG(lp,j).iter = &LITER(lp,lp->ndim,j);
        }

        lp->user.reduce_dim = 0;
        lp->user.reduce = 0;
    }
}


// Judge whether a (contiguous) buffer copy is required or not, and malloc if it is required.
//
// CASES TO REQUIRE A BUFFER COPY:
// 1) ndloop has `idx` but does not support CUMO_NDF_INDEX_LOOP.
// 2) ndloop has non-contiguous arrays but does not support CUMO_NDF_STRIDE_LOOP.
static void
cumo_ndfunc_set_bufcp(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp)
{
    unsigned int f;
    int i, j;
    int nd, ndim;
    bool zero_step;
    ssize_t n, sz, elmsz, stride, n_total; //, last_step;
    size_t *buf_shape;
    cumo_na_loop_iter_t *buf_iter=NULL, *src_iter;

    unsigned int loop_spec = ndloop_func_loop_spec(nf, lp->user.ndim);
    //if (loop_spec==0) return;

    n_total = lp->user.n[0];
    for (i=1; i<lp->user.ndim; i++) {
        n_total *= lp->user.n[i];
    }

    //for (j=0; j<lp->nin; j++) {
    for (j=0; j<lp->narg; j++) {
        //ndim = nd = lp->user.ndim;
        ndim = nd = LARG(lp,j).ndim;
        sz = elmsz = LARG(lp,j).elmsz;
        src_iter = LARG(lp,j).iter;
        //last_step = src_iter[ndim-1].step;
        f = 0;
        zero_step = 1;
        for (i=ndim; i>0; ) {
            i--;
            if (LARG(lp,j).shape) {
                n = LARG(lp,j).shape[i];
            } else {
                //printf("shape is NULL\n");
                n = lp->user.n[i];
            }
            stride = sz * n;
            //printf("{j=%d,i=%d,ndim=%d,nd=%d,idx=%lx,step=%ld,n=%ld,sz=%ld,stride=%ld}\n",j,i,ndim,nd,(size_t)src_iter[i].idx,src_iter[i].step,n,sz,stride);
            if (src_iter[i].idx) {
                f |= 2;  // INDEX LOOP
                zero_step = 0;
            } else {
                if (src_iter[i].step != sz) {
                    f |= 1;  // NON_CONTIGUOUS LOOP
                } else {
                    // CONTIGUOUS LOOP
                    if (i==ndim-1) {  // contract if last dimension
                        ndim = i;
                        elmsz = stride;
                    }
                }
                if (src_iter[i].step != 0) {
                    zero_step = 0;
                }
            }
            sz = stride;
        }
        //printf("[j=%d f=%d loop_spec=%d zero_step=%d]\n",j,f,loop_spec,zero_step);

        if (zero_step) {
            // no buffer needed
            continue;
        }

        // should check flatten-able loop to avoid buffering


        // over loop_spec or reduce_loop is not contiguous
        if (f & loop_spec || (lp->reduce_dim > 1 && ndim > 0)) {
            //printf("(buf,nd=%d)",nd);
            buf_iter = ALLOC_N(cumo_na_loop_iter_t,nd+3);
            buf_shape = ALLOC_N(size_t,nd);
            buf_iter[nd].pos = 0;
            buf_iter[nd].step = 0;
            buf_iter[nd].idx = NULL;
            sz = LARG(lp,j).elmsz;
            //last_step = sz;
            for (i=nd; i>0; ) {
                i--;
                buf_iter[i].pos = 0;
                buf_iter[i].step = sz;
                buf_iter[i].idx = NULL;
                //n = lp->user.n[i];
                n = LARG(lp,j).shape[i];
                buf_shape[i] = n;
                sz *= n;
            }
            LBUFCP(lp,j) = ALLOC(cumo_na_buffer_copy_t);
            LBUFCP(lp,j)->buf_on_device = cumo_cuda_runtime_is_device_memory(LARG(lp,j).ptr);
            if (LBUFCP(lp,j)->buf_on_device) {
                // The copy below runs one thread per element, so a contracted
                // element leaves a whole row to a single thread.
                ndim = nd;
                elmsz = LARG(lp,j).elmsz;
            }
            LBUFCP(lp,j)->ndim = ndim;
            LBUFCP(lp,j)->elmsz = elmsz;
            LBUFCP(lp,j)->n = buf_shape;
            LBUFCP(lp,j)->src_iter = src_iter;
            LBUFCP(lp,j)->buf_iter = buf_iter;
            LARG(lp,j).iter = buf_iter;
            //printf("in cumo_ndfunc_set_bufcp(1): lp->user.args[%d].iter=%lx\n",j,(size_t)(LARG(lp,j).iter));
            LBUFCP(lp,j)->src_ptr = LARG(lp,j).ptr;
            if (LBUFCP(lp,j)->buf_on_device) {
                LARG(lp,j).ptr = LBUFCP(lp,j)->buf_ptr = cumo_cuda_runtime_malloc(sz);
            }
            else {
                LARG(lp,j).ptr = LBUFCP(lp,j)->buf_ptr = xmalloc(sz);
            }
            //printf("(LBUFCP(lp,%d)->buf_ptr=%lx)\n",j,(size_t)(LBUFCP(lp,j)->buf_ptr));
        }
    }

#if 0
    for (j=0; j<lp->narg; j++) {
        ndim = lp->user.ndim;
        src_iter = LARG(lp,j).iter;
        last_step = src_iter[ndim-1].step;
        if (lp->reduce_dim>1) {
            //printf("(reduce_dim=%d,ndim=%d,nd=%d,n=%ld,lst=%ld)\n",lp->reduce_dim,ndim,nd,n_total,last_step);
            buf_iter = ALLOC_N(cumo_na_loop_iter_t,2);
            buf_iter[0].pos = LARG(lp,j).iter[0].pos;
            buf_iter[0].step = last_step;
            buf_iter[0].idx = NULL;
            buf_iter[1].pos = 0;
            buf_iter[1].step = 0;
            buf_iter[1].idx = NULL;
            LARG(lp,j).iter = buf_iter;
            //printf("in cumo_ndfunc_set_bufcp(2): lp->user.args[%d].iter=%lx\n",j,(size_t)(LARG(lp,j).iter));
            lp->xargs[j].free_user_iter = 1;
        }
    }
#endif

    // flatten reduce dimensions
    if (lp->reduce_dim > 1) {
#if 1
        for (j=0; j<lp->narg; j++) {
            ndim = lp->user.ndim;
            LARG(lp,j).iter[0].step = LARG(lp,j).iter[ndim-1].step;
            LARG(lp,j).iter[0].idx = NULL;
        }
#endif
        lp->user.n[0] = n_total;
        lp->user.ndim = 1;
    }
}

static cumo_na_iarray_stridx_t
cumo_na_make_iarray_buffer_copy(cumo_na_buffer_copy_t* lp)
{
    cumo_na_iarray_stridx_t iarray;
    int i;
    int ndim = lp->ndim;
    iarray.ptr = lp->src_ptr + lp->src_iter[0].pos;
    for (i = 0; i < ndim; ++i) {
        if (LITER_SRC(lp,i).idx) {
            CUMO_SDX_SET_INDEX(iarray.stridx[i], LITER_SRC(lp,i).idx);
        } else {
            CUMO_SDX_SET_STRIDE(iarray.stridx[i], LITER_SRC(lp,i).step);
        }
    }
    return iarray;
}

static cumo_na_indexer_t
cumo_na_make_indexer_buffer_copy(cumo_na_buffer_copy_t* lp)
{
    cumo_na_indexer_t indexer;
    int i;
    indexer.ndim = lp->ndim;
    indexer.total_size = 1;
    for (i = 0; i< lp->ndim; ++i) {
        indexer.shape[i] = lp->n[i];
        indexer.total_size *= lp->n[i];
    }
    return indexer;
}

// The index arrays live in device memory and are filled by kernels, so a host
// loop must not read them until those kernels are done.
static void
ndloop_sync_src_index(cumo_na_buffer_copy_t *lp)
{
    int i;
    for (i=0; i<lp->ndim; i++) {
        if (LITER_SRC(lp,i).idx) {
            CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("ndloop buffer copy", "any");
            cumo_cuda_runtime_device_synchronize();
            return;
        }
    }
}

// Same for the index arrays the user function itself walks, which it reads on
// the host whenever the data it indexes is host memory.
static void
ndloop_sync_user_index(cumo_na_md_loop_t *lp)
{
    int i, j;
    for (j=0; j<lp->narg; j++) {
        if (LARG(lp,j).iter == NULL) continue;
        for (i=0; i<LARG(lp,j).ndim; i++) {
            if (LARG(lp,j).iter[i].idx == NULL) continue;
            if (cumo_cuda_runtime_is_device_memory(LARG(lp,j).ptr)) break;
            CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("ndloop", "any");
            cumo_cuda_runtime_device_synchronize();
            return;
        }
    }
}

void cumo_ndloop_copy_to_buffer_kernel_launch(cumo_na_iarray_stridx_t *a, cumo_na_indexer_t* indexer, char *buf, size_t elmsz);

// Make contiguous memory for ops not supporting index or stride (step) loop
static void
ndloop_copy_to_buffer(cumo_na_buffer_copy_t *lp)
{
    size_t *c;
    char *src, *buf;
    int  i;
    int  nd = lp->ndim;
    size_t elmsz = lp->elmsz;
    size_t buf_pos = 0;
    DBG(size_t j);

    if (lp->buf_on_device) {
        cumo_na_iarray_stridx_t a = cumo_na_make_iarray_buffer_copy(lp);
        cumo_na_indexer_t indexer = cumo_na_make_indexer_buffer_copy(lp);
        cumo_ndloop_copy_to_buffer_kernel_launch(&a, &indexer, lp->buf_ptr, lp->elmsz);
        return;
    }

    // A host buffer is read by a host loop, which an async kernel is not
    // ordered against, so copy it on the host as well.
    ndloop_sync_src_index(lp);

    //printf("\nto_buf nd=%d elmsz=%ld\n",nd,elmsz);
    DBG(printf("<to buf> ["));
    // zero-dimension
    if (nd==0) {
        src = lp->src_ptr + LITER_SRC(lp,0).pos;
        buf = lp->buf_ptr;
        memcpy(buf,src,elmsz);
        DBG(for (j=0; j<elmsz/8; j++) {printf("%g,",((double*)(buf))[j]);});
        goto loop_end;
    }
    // initialize loop counter
    c = ALLOCA_N(size_t, nd+1);
    for (i=0; i<=nd; i++) c[i]=0;
    // loop body
    for (i=0;;) {
        // i-th dimension
        for (; i<nd; i++) {
            if (LITER_SRC(lp,i).idx) {
                LITER_SRC(lp,i+1).pos = LITER_SRC(lp,i).pos + LITER_SRC(lp,i).idx[c[i]];
            } else {
                LITER_SRC(lp,i+1).pos = LITER_SRC(lp,i).pos + LITER_SRC(lp,i).step*c[i];
            }
        }
        src = lp->src_ptr + LITER_SRC(lp,nd).pos;
        buf = lp->buf_ptr + buf_pos;
        memcpy(buf,src,elmsz);
        DBG(for (j=0; j<elmsz/8; j++) {printf("%g,",((double*)(buf))[j]);});
        buf_pos += elmsz;
        // count up
        for (;;) {
            if (i<=0) goto loop_end;
            i--;
            if (++c[i] < lp->n[i]) break;
            c[i] = 0;
        }
    }
 loop_end:
    ;
    DBG(printf("]\n"));
}

void cumo_ndloop_copy_from_buffer_kernel_launch(cumo_na_iarray_stridx_t *a, cumo_na_indexer_t* indexer, char *buf, size_t elmsz);

static void
ndloop_copy_from_buffer(cumo_na_buffer_copy_t *lp)
{
    size_t *c;
    char *src, *buf;
    int  i;
    int  nd = lp->ndim;
    size_t elmsz = lp->elmsz;
    size_t buf_pos = 0;
    DBG(size_t j);

    if (lp->buf_on_device) {
        cumo_na_iarray_stridx_t a = cumo_na_make_iarray_buffer_copy(lp);
        cumo_na_indexer_t indexer = cumo_na_make_indexer_buffer_copy(lp);
        cumo_ndloop_copy_from_buffer_kernel_launch(&a, &indexer, lp->buf_ptr, lp->elmsz);
        return;
    }

    ndloop_sync_src_index(lp);

    //printf("\nfrom_buf nd=%d elmsz=%ld\n",nd,elmsz);
    DBG(printf("<from buf> ["));
    // zero-dimension
    if (nd==0) {
        src = lp->src_ptr + LITER_SRC(lp,0).pos;
        buf = lp->buf_ptr;
        memcpy(src,buf,elmsz);
        DBG(for (j=0; j<elmsz/8; j++) {printf("%g,",((double*)(src))[j]);});
        goto loop_end;
    }
    // initialize loop counter. c[i] indicates an element index of i-th dim.
    c = ALLOCA_N(size_t, nd+1);
    for (i=0; i<=nd; i++) c[i]=0;
    // loop body
    for (i=0;;) {
        // i-th dimension
        for (; i<nd; i++) {
            if (LITER_SRC(lp,i).idx) {
                LITER_SRC(lp,i+1).pos = LITER_SRC(lp,i).pos + LITER_SRC(lp,i).idx[c[i]];
            } else {
                LITER_SRC(lp,i+1).pos = LITER_SRC(lp,i).pos + LITER_SRC(lp,i).step*c[i];
            }
        }
        src = lp->src_ptr + LITER_SRC(lp,nd).pos;
        buf = lp->buf_ptr + buf_pos;
        memcpy(src,buf,elmsz);
        DBG(for (j=0; j<elmsz/8; j++) {printf("%g,",((double*)(src))[j]);});
        buf_pos += elmsz;
        // count up
        // ++c[i] and goes to the next element of i-th dim if c[i] < lp->n[i]
        // If c[i] == lp->n[i], a loop for i-th dim ends. Goes to (i-1)-th dim.
        for (;;) {
            if (i<=0) goto loop_end;
            i--;
            ++c[i];
            if (c[i] < lp->n[i]) break;
            c[i] = 0;
        }
    }
 loop_end:
    DBG(printf("]\n"));
}


static void
cumo_ndfunc_write_back(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp, VALUE orig_args, VALUE results)
{
    VALUE src, dst;

    if (lp->writeback >= 0) {
        dst = RARRAY_AREF(orig_args,lp->writeback);
        src = RARRAY_AREF(results,0);
        cumo_na_store(dst,src);
        RARRAY_ASET(results,0,dst);
    }
}


static VALUE
ndloop_extract(VALUE results, cumo_ndfunc_t *nf)
{
    // long n, i;
    // VALUE x, y;
    // cumo_narray_t *na;

    // extract result objects
    switch(nf->nout) {
    case 0:
        return Qnil;
    case 1:
        return RARRAY_AREF(results,0);
        // x = RARRAY_AREF(results,0);
        // if (CUMO_NDF_TEST(nf,CUMO_NDF_EXTRACT)) {
        //     if (CumoIsNArray(x)){
        //         CumoGetNArray(x,na);
        //         if (CUMO_NA_NDIM(na)==0) {
        //             x = rb_funcall(x, cumo_id_extract, 0);
        //         }
        //     }
        // }
        // return x;
    }
    // if (CUMO_NDF_TEST(nf,CUMO_NDF_EXTRACT)) {
    //     n = RARRAY_LEN(results);
    //     for (i=0; i<n; i++) {
    //         x = RARRAY_AREF(results,i);
    //         if (CumoIsNArray(x)){
    //             CumoGetNArray(x,na);
    //             if (CUMO_NA_NDIM(na)==0) {
    //                 y = rb_funcall(x, cumo_id_extract, 0);
    //                 RARRAY_ASET(results,i,y);
    //             }
    //         }
    //     }
    // }
    return results;
}

// The md-loop dimensions only. An index on one of the argument's own user
// dimensions sits past lp->ndim and is not reported here.
static bool
loop_arg_is_using_idx(cumo_na_md_loop_t *lp, int j)
{
    int  i;

    if (lp->ndim<0) {
        rb_bug("bug? lp->ndim = %d\n", lp->ndim);
    }

    // i-th dimension
    for (i=0; i<lp->ndim; i++) {
        if (LITER(lp,i,j).idx) {
            return true;
        }
    }
    return false;
}

static bool
loop_is_using_idx(cumo_na_md_loop_t *lp)
{
    int  j;

    // j-th argument
    for (j=0; j<lp->narg; j++) {
        if (loop_arg_is_using_idx(lp, j)) {
            return true;
        }
    }
    return false;
}

// The md-loops below find each position by reading an index array on the host,
// whatever memory the data it indexes lives in, and a kernel filled that array.
// ndloop_sync_user_index covers the dimensions the user function walks; these
// are the ones ndloop walks itself.
static void
ndloop_sync_device(void)
{
    CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("ndloop", "any");
    cumo_cuda_runtime_device_synchronize();
}

// An argument walked through an index array is a view whose fills were counted,
// so ask that view rather than stopping the device outright: one that settled
// since costs nothing, and one that has not settles every other argument with
// it. The index an iterator walks belongs to the view this argument was built
// from, which ndloop_set_stepidx put in LARG(lp,j).value.
static void
ndloop_wait_arg_index(cumo_na_md_loop_t *lp, int j)
{
    cumo_narray_t *na;
    VALUE v;

    if (!loop_arg_is_using_idx(lp, j)) {
        return;
    }
    v = LARG(lp,j).value;
    if (!CumoIsNArray(v)) {
        ndloop_sync_device();
        return;
    }
    CumoGetNArray(v, na);
    if (na->type != CUMO_NARRAY_VIEW_T) {
        ndloop_sync_device();
        return;
    }
    cumo_na_index_wait_fill((cumo_narray_view_t *)na);
}

static void
ndloop_sync_md_index(cumo_na_md_loop_t *lp)
{
    int j;

    for (j=0; j<lp->narg; j++) {
        ndloop_wait_arg_index(lp, j);
    }
}

static void
loop_narray(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp);

// The data object a view reads through, or the array itself. A view made from
// another one already points at the original, so one hop is enough.
static VALUE
ndloop_alias_base(VALUE v)
{
    cumo_narray_t *na;

    CumoGetNArray(v,na);
    if (CUMO_NA_TYPE(na) == CUMO_NARRAY_VIEW_T) {
        return CUMO_NA_VIEW(na)->data;
    }
    return v;
}

// The offset and per-dimension strides an argument walks with. A whole array
// is the contiguous case of a view. Answers 0 for an index array, which walks
// wherever the index says.
static int
ndloop_addressing(VALUE v, int nd, size_t *offset, ssize_t *stride)
{
    cumo_narray_t *na;
    cumo_narray_view_t *nv;
    ssize_t s;
    int i;

    CumoGetNArray(v,na);
    if (CUMO_NA_TYPE(na) != CUMO_NARRAY_VIEW_T) {
        *offset = 0;
        s = (ssize_t)cumo_na_element_stride(v);
        for (i=nd; i--;) {
            stride[i] = s;
            s *= (ssize_t)na->shape[i];
        }
        return 1;
    }
    nv = CUMO_NA_VIEW(na);
    for (i=0; i<nd; i++) {
        if (CUMO_SDX_IS_INDEX(nv->stridx[i])) {
            return 0;
        }
        stride[i] = CUMO_SDX_GET_STRIDE(nv->stridx[i]);
    }
    *offset = nv->offset;
    return 1;
}

// Whether the two put their i-th element at the same address for every i.
static int
ndloop_walks_alike(VALUE v1, VALUE v2)
{
    cumo_narray_t *na1, *na2;
    size_t o1, o2;
    ssize_t *s1, *s2;
    int i, nd;

    CumoGetNArray(v1,na1);
    CumoGetNArray(v2,na2);
    nd = na1->ndim;
    if (nd != na2->ndim) {
        return 0;
    }
    for (i=0; i<nd; i++) {
        if (na1->shape[i] != na2->shape[i]) {
            return 0;
        }
    }
    s1 = ALLOCA_N(ssize_t, nd+1);
    s2 = ALLOCA_N(ssize_t, nd+1);
    if (!ndloop_addressing(v1,nd,&o1,s1) || !ndloop_addressing(v2,nd,&o2,s2)) {
        return 0;
    }
    if (o1 != o2) {
        return 0;
    }
    for (i=0; i<nd; i++) {
        if (na1->shape[i] > 1 && s1[i] != s2[i]) {
            return 0;
        }
    }
    return 1;
}

// An elementwise op writes each output element from the operands at the same
// position, so an operand sharing the output's memory is only safe where it
// sits at the same addresses. A transposed or broadcast one does not: the
// element a thread reads is one another thread is writing, and on a device
// nothing orders the two. Take a copy, so the answer is the one the same
// expression gives out of place.
static void
ndloop_copy_aliased_args(cumo_ndfunc_t *nf, VALUE args)
{
    VALUE out = Qnil, type, base, v;
    int j;

    // With one input there is nothing to look at but the one being written.
    if (nf->nin < 2 || !CUMO_NDF_TEST(nf,CUMO_NDF_INPLACE)) {
        return;
    }
    for (j=0; j<nf->nin; j++) {
        v = RARRAY_AREF(args,j);
        if (CumoIsNArray(v) && CUMO_TEST_INPLACE(v)) {
            out = v;
            break;
        }
    }
    if (NIL_P(out)) {
        return;
    }
    // A named output class the flagged one is not, such as the Bit a comparison
    // answers, means nothing here can be written in place.
    type = nf->nout > 0 ? nf->aout[0].type : Qnil;
    if (rb_obj_is_kind_of(type, rb_cClass) && type != rb_obj_class(out)) {
        return;
    }
    base = ndloop_alias_base(out);
    for (j=0; j<nf->nin; j++) {
        if (nf->ain[j].type == CUMO_OVERWRITE) {
            continue;
        }
        v = RARRAY_AREF(args,j);
        if (!CumoIsNArray(v) || ndloop_alias_base(v) != base) {
            continue;
        }
        // ndloop_find_inplace weighs shape and class as well, so a later
        // flagged argument can be the one written. Copying it would send the
        // answer to the copy and leave the caller's array alone.
        if (CUMO_TEST_INPLACE(v) || ndloop_walks_alike(out,v)) {
            continue;
        }
        rb_ary_store(args, j, cumo_na_copy(v));
    }
}

static VALUE
ndloop_run(VALUE vlp)
{
    volatile VALUE args, orig_args, results;
    cumo_na_md_loop_t *lp = (cumo_na_md_loop_t*)(vlp);
    cumo_ndfunc_t *nf;

    orig_args = lp->vargs;
    nf = lp->ndfunc;

    // Every caller of ndloop_alloc builds this with rb_ary_new3/4 or
    // rb_assoc_new, so the general path through Ruby's dispatch buys nothing.
    args = rb_ary_dup(orig_args);
    ndloop_copy_aliased_args(nf, args);

    // setup ndloop iterator with arguments
    ndloop_init_args(nf, lp, args);
    results = ndloop_set_output(nf, lp, args);
    //if (cumo_na_debug_flag) {
    //    printf("-- ndloop_set_output --\n");
    //    print_ndloop(lp);
    //}

    // contract loop (compact dimessions)
    if (CUMO_NDF_TEST(nf,CUMO_NDF_INDEXER_LOOP) && CUMO_NDF_TEST(nf,CUMO_NDF_FLAT_REDUCE)) {
        // do nothing
    } else {
        if (lp->loop_func == loop_narray) {
            if (CUMO_NDF_TEST(nf,CUMO_NDF_INDEXER_LOOP) && CUMO_NDF_TEST(nf,CUMO_NDF_ANY_ORDER) && lp->reduce_dim == 0) {
                cumo_ndfunc_reorder_loop(lp);
                if (cumo_na_debug_flag) {
                    printf("-- cumo_ndfunc_reorder_loop --\n");
                    print_ndloop(lp);
                }
            }
            cumo_ndfunc_contract_loop(lp);
            if (cumo_na_debug_flag) {
                printf("-- cumo_ndfunc_contract_loop --\n");
                print_ndloop(lp);
            }
        }
    }

    // setup lp->user
    if (CUMO_NDF_TEST(nf,CUMO_NDF_INDEXER_LOOP)) {
        cumo_ndfunc_set_user_indexer_loop(nf, lp);
        if (cumo_na_debug_flag) {
            printf("-- cumo_ndfunc_set_user_indexer_loop --\n");
            print_ndloop(lp);
        }
    } else {
        cumo_ndfunc_set_user_loop(nf, lp);
        if (cumo_na_debug_flag) {
            printf("-- cumo_ndfunc_set_user_loop --\n");
            print_ndloop(lp);
        }
    }

    // setup buffering during loop
    if (CUMO_NDF_TEST(nf,CUMO_NDF_INDEXER_LOOP) && CUMO_NDF_TEST(nf,CUMO_NDF_FLAT_REDUCE) && !loop_is_using_idx(lp)) {
        // do nothing
    } else {
        if (lp->loop_func == loop_narray) {
            cumo_ndfunc_set_bufcp(nf, lp);
        }
        if (cumo_na_debug_flag) {
            printf("-- cumo_ndfunc_set_bufcp --\n");
            print_ndloop(lp);
        }
    }

    // loop
    (*(lp->loop_func))(nf, lp);

    if (RTEST(lp->user.err_type)) {
        rb_raise(lp->user.err_type, "error in NArray operation");
    }

    // write-back will be placed here
    cumo_ndfunc_write_back(nf, lp, orig_args, results);

    // extract result objects
    return ndloop_extract(results, nf);
}


// ---------------------------------------------------------------------------

// A zero in any dimension, not only the first, leaves nothing to visit.
static int
ndloop_is_empty(cumo_na_md_loop_t *lp)
{
    int i, n = lp->ndim + lp->user.ndim;

    for (i=0; i<n; i++) {
        if (lp->n[i] == 0) return 1;
    }
    return 0;
}


// The lowest and highest position an argument's dimensions from..to reach,
// relative to base, from their steps and, for an index dimension, its indices.
static void
ndloop_stage_extent(cumo_na_md_loop_t *lp, int j, int from, int to, ssize_t base, ssize_t *lo, ssize_t *hi)
{
    int i;

    *lo = *hi = base;
    for (i=from; i<to; i++) {
        size_t n = lp->n[i];
        size_t *h = LITER(lp,i,j).idx;
        if (h) {
            // An index is an offset from the position, and one that walks
            // backwards is negative, stored as it wraps.
            ssize_t mn = (ssize_t)h[0], mx = (ssize_t)h[0];
            size_t k;
            for (k=1; k<n; k++) {
                if ((ssize_t)h[k] < mn) mn = (ssize_t)h[k];
                if ((ssize_t)h[k] > mx) mx = (ssize_t)h[k];
            }
            *lo += mn;
            *hi += mx;
        } else {
            ssize_t d = LITER(lp,i,j).step * (ssize_t)(n - 1);
            if (d < 0) *lo += d; else *hi += d;
        }
    }
}

// The bytes positions lo..hi of argument j cover. Bit positions are bits,
// and the loop reads whole words.
static void
ndloop_stage_bytes(cumo_na_md_loop_t *lp, int j, ssize_t lo, ssize_t hi, size_t *min, size_t *len)
{
    if (rb_obj_is_kind_of(LARG(lp,j).value, cumo_cBit)) {
        *min = (size_t)(lo / CUMO_NB) * sizeof(CUMO_BIT_DIGIT);
        *len = (size_t)(hi / CUMO_NB + 1) * sizeof(CUMO_BIT_DIGIT) - *min;
    } else {
        *min = (size_t)lo;
        *len = (size_t)(hi - lo) + LARG(lp,j).elmsz;
    }
}

static void
ndloop_stage_dma(cumo_na_md_loop_t *lp, int j, size_t min, size_t len)
{
    cumo_na_host_stage_t *hs = lp->hs;
    char *dst;

    if (hs->rows_only) {
        if (len > hs->len[j]) {
            rb_bug("a staged row of %"SZF"u bytes outgrew its %"SZF"u", len, hs->len[j]);
        }
        dst = hs->stage.ptr + hs->off[j];
        LARG(lp,j).ptr = dst - min;
    } else {
        dst = LARG(lp,j).ptr + min;
    }
    cumo_cuda_runtime_check_status(cumo_cuda_runtime_memcpy_to_pinned(dst, hs->ptr[j] + min, len));
}

// Copies the row the loop is about to hand the user function, for every
// staged input. The row is what the user dimensions walk from the position
// the outer loop has reached.
static void
ndloop_stage_row(cumo_na_md_loop_t *lp)
{
    cumo_na_host_stage_t *hs = lp->hs;
    int j;

    for (j=0; j<hs->narg; j++) {
        ssize_t lo, hi;
        size_t min, len;
        if (hs->ptr[j] == NULL) continue;
        ndloop_stage_extent(lp, j, lp->ndim, hs->ndim, LITER(lp,lp->ndim,j).pos, &lo, &hi);
        ndloop_stage_bytes(lp, j, lo, hi, &min, &len);
        ndloop_stage_dma(lp, j, min, len);
    }
    hs->row_epoch = cumo_cuda_launch_epoch;
}

// Reads the inputs of a host loop into pinned memory and points the
// iterators at the copy, so that the loop's pointer arithmetic lands there
// unchanged. Index arrays come over first, since the range an input walks
// is read off them. The whole range is copied once when it fits the cache;
// otherwise each row is copied as the loop reaches it, and an input whose
// rows alone do not fit is read where it is.
static void
ndloop_stage_host_reads(cumo_na_md_loop_t *lp)
{
    cumo_na_host_stage_t *hs;
    int nd, narg, i, j;
    size_t total = 0, rows = 0;

    if (!CUMO_NDF_TEST(lp->ndfunc, CUMO_NDF_HOST_READ) || lp->hs) return;
    if (ndloop_is_empty(lp)) return;
    nd = lp->ndim + lp->user.ndim;
    narg = lp->narg;

    hs = ZALLOC(cumo_na_host_stage_t);
    lp->hs = hs;
    hs->narg = narg;
    hs->ndim = nd;
    hs->ptr = ZALLOC_N(char*, narg);
    hs->min = ZALLOC_N(size_t, narg);
    hs->len = ZALLOC_N(size_t, narg);
    hs->off = ZALLOC_N(size_t, narg);
    hs->idx = ZALLOC_N(size_t*, (size_t)nd * narg);
    hs->host_idx = ZALLOC_N(size_t*, (size_t)nd * narg);

    for (j=0; j<narg; j++) {
        VALUE v = LARG(lp,j).value;
        ssize_t lo, hi;

        if (lp->xargs[j].flag != CUMO_NDL_READ || lp->xargs[j].bufcp) continue;
        if (!CumoIsNArray(v) || LARG(lp,j).ptr == NULL) continue;

        for (i=0; i<nd; i++) {
            size_t *idx = LITER(lp,i,j).idx;
            size_t n = lp->n[i];
            size_t *h;
            if (idx == NULL) continue;
            h = ALLOC_N(size_t, n);
            hs->idx[i*narg+j] = idx;
            hs->host_idx[i*narg+j] = h;
            LITER(lp,i,j).idx = h;
            cumo_cuda_runtime_check_status(
                cumo_cuda_runtime_memcpy_to_host(h, idx, n * sizeof(size_t)));
        }
        // Its data is host memory already.
        if (rb_obj_is_kind_of(v, cumo_cRObject)) { hs->direct++; continue; }

        ndloop_stage_extent(lp, j, 0, nd, LITER(lp,0,j).pos, &lo, &hi);
        ndloop_stage_bytes(lp, j, lo, hi, &hs->min[j], &hs->len[j]);
        hs->ptr[j] = LARG(lp,j).ptr;
        total += (hs->len[j] + 7) & ~(size_t)7;
    }

    if (total > CUMO_CUDA_STAGE_CACHE_MAX) {
        hs->rows_only = 1;
        total = 0;
        for (j=0; j<narg; j++) {
            ssize_t lo, hi;
            size_t len;
            if (hs->ptr[j] == NULL) continue;
            ndloop_stage_extent(lp, j, lp->ndim, nd, 0, &lo, &hi);
            if (rb_obj_is_kind_of(LARG(lp,j).value, cumo_cBit)) {
                // Which words a row covers depends on where it starts.
                len = ((size_t)(hi - lo) / CUMO_NB + 2) * sizeof(CUMO_BIT_DIGIT);
            } else {
                len = (size_t)(hi - lo) + LARG(lp,j).elmsz;
            }
            if (len > CUMO_CUDA_STAGE_CACHE_MAX) {
                hs->ptr[j] = NULL;
                hs->direct++;
                continue;
            }
            hs->off[j] = total;
            hs->len[j] = len;
            total += (len + 7) & ~(size_t)7;
            rows++;
        }
        if (hs->direct) cumo_cuda_runtime_device_synchronize();
        if (rows == 0) return;
        cumo_cuda_runtime_stage_alloc(&hs->stage, total);
        return;
    }
    if (hs->direct) cumo_cuda_runtime_device_synchronize();
    if (total == 0) return;
    cumo_cuda_runtime_stage_alloc(&hs->stage, total);
    total = 0;
    for (j=0; j<narg; j++) {
        if (hs->ptr[j] == NULL) continue;
        hs->off[j] = total;
        LARG(lp,j).ptr = hs->stage.ptr + total - hs->min[j];
        ndloop_stage_dma(lp, j, hs->min[j], hs->len[j]);
        total += (hs->len[j] + 7) & ~(size_t)7;
    }
    hs->epoch = hs->row_epoch = cumo_cuda_launch_epoch;
}

// Before each row: a row that is only copied as reached, or one that
// device memory was written since the copy.
static inline void
ndloop_stage_before_row(cumo_na_md_loop_t *lp)
{
    cumo_na_host_stage_t *hs = lp->hs;
    if (hs == NULL) return;
    if (hs->direct) cumo_cuda_runtime_sync_if_busy();
    if (hs->stage.ptr && (hs->rows_only || hs->epoch != cumo_cuda_launch_epoch)) {
        ndloop_stage_row(lp);
    }
}

// After a yield: the block may have written the element the user function
// reads next, which is copied again on its own so that a block writing at
// every element costs an element, not a row, each time.
void
cumo_na_ndloop_refresh_next(cumo_na_loop_t *user, const void *next, size_t bytes)
{
    cumo_na_md_loop_t *lp = (cumo_na_md_loop_t*)((char*)user - offsetof(cumo_na_md_loop_t, user));
    cumo_na_host_stage_t *hs = lp->hs;
    int j;

    if (hs == NULL) return;
    if (hs->direct) cumo_cuda_runtime_sync_if_busy();
    if (hs->stage.ptr == NULL || hs->row_epoch == cumo_cuda_launch_epoch) return;
    for (j=0; j<hs->narg; j++) {
        char *base = hs->stage.ptr + hs->off[j];
        if (hs->ptr[j] == NULL) continue;
        if ((const char*)next >= base && (const char*)next + bytes <= base + hs->len[j]) {
            cumo_cuda_runtime_check_status(cumo_cuda_runtime_memcpy_to_pinned(
                (void*)next, hs->ptr[j] + ((const char*)next - LARG(lp,j).ptr), bytes));
            return;
        }
    }
}

static void
ndloop_unstage_host_reads(cumo_na_md_loop_t *lp)
{
    cumo_na_host_stage_t *hs = lp->hs;
    int i, j;

    if (hs == NULL) return;
    lp->hs = NULL;
    for (j=0; j<hs->narg; j++) {
        if (hs->ptr[j]) LARG(lp,j).ptr = hs->ptr[j];
        for (i=0; i<hs->ndim; i++) {
            if (hs->idx[i*hs->narg+j]) LITER(lp,i,j).idx = hs->idx[i*hs->narg+j];
            if (hs->host_idx[i*hs->narg+j]) xfree(hs->host_idx[i*hs->narg+j]);
        }
    }
    cumo_cuda_runtime_stage_free(&hs->stage);
    xfree(hs->ptr);
    xfree(hs->min);
    xfree(hs->len);
    xfree(hs->off);
    xfree(hs->idx);
    xfree(hs->host_idx);
    xfree(hs);
}

static void
loop_narray(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp)
{
    size_t *c;
    int  i, j;
    int  nd = lp->ndim;

    if (nd<0) {
        rb_bug("bug? lp->ndim = %d\n", lp->ndim);
    }
    if (ndloop_is_empty(lp)) {
        return;
    }

    // One wait settles every index array the loop below reads, so the loop
    // reads them without one of its own.
    ndloop_sync_md_index(lp);
    ndloop_sync_user_index(lp);
    ndloop_stage_host_reads(lp);

    if (nd==0 || CUMO_NDF_TEST(nf,CUMO_NDF_INDEXER_LOOP)) {
        for (j=0; j<lp->nin; j++) {
            if (lp->xargs[j].bufcp) {
                //printf("copy_to_buffer j=%d\n",j);
                ndloop_copy_to_buffer(lp->xargs[j].bufcp);
            }
        }
        ndloop_stage_before_row(lp);
        (*(nf->func))(&(lp->user));
        for (j=0; j<lp->narg; j++) {
            if (lp->xargs[j].bufcp && (lp->xargs[j].flag & CUMO_NDL_WRITE)) {
                //printf("copy_from_buffer j=%d\n",j);
                // copy data to work buffer
                ndloop_copy_from_buffer(lp->xargs[j].bufcp);
            }
        }
        return;
    }

    // initialize loop counter
    c = ALLOCA_N(size_t, nd+1);
    for (i=0; i<=nd; i++) c[i]=0;

    // loop body
    for (i=0;;) {
        // i-th dimension
        for (; i<nd; i++) {
            // j-th argument
            for (j=0; j<lp->narg; j++) {
                if (LITER(lp,i,j).idx) {
                    LITER(lp,i+1,j).pos = LITER(lp,i,j).pos + LITER(lp,i,j).idx[c[i]];
                } else {
                    LITER(lp,i+1,j).pos = LITER(lp,i,j).pos + LITER(lp,i,j).step*c[i];
                }
                //printf("j=%d c[i=%d]=%lu pos=%lu\n",j,i,c[i],LITER(lp,i+1,j).pos);
            }
        }
        for (j=0; j<lp->nin; j++) {
            if (lp->xargs[j].bufcp) {
                // copy data to work buffer
                // cp lp->iter[j][nd..*] to lp->user.args[j].iter[0..*]
                //printf("copy_to_buffer j=%d\n",j);
                ndloop_copy_to_buffer(lp->xargs[j].bufcp);
            }
        }
        ndloop_stage_before_row(lp);
        (*(nf->func))(&(lp->user));
        for (j=0; j<lp->narg; j++) {
            if (lp->xargs[j].bufcp && (lp->xargs[j].flag & CUMO_NDL_WRITE)) {
                // copy data to work buffer
                //printf("copy_from_buffer j=%d\n",j);
                ndloop_copy_from_buffer(lp->xargs[j].bufcp);
            }
        }
        if (RTEST(lp->user.err_type)) {return;}

        for (;;) {
            if (i<=0) goto loop_end;
            i--;
            if (++c[i] < lp->n[i]) break;
            c[i] = 0;
        }
    }
 loop_end:
    ;
}


static VALUE
cumo_na_ndloop_main(cumo_ndfunc_t *nf, VALUE args, void *opt_ptr)
{
    unsigned int copy_flag;
    cumo_na_md_loop_t lp;

    if (cumo_na_debug_flag) print_ndfunc(nf);

    // cast arguments to NArray
    copy_flag = ndloop_cast_args(nf, args);

    // allocate ndloop struct
    ndloop_alloc(&lp, nf, args, opt_ptr, copy_flag, loop_narray);

    return rb_ensure(ndloop_run, (VALUE)&lp, ndloop_release, (VALUE)&lp);
}


VALUE
#ifdef HAVE_STDARG_PROTOTYPES
cumo_na_ndloop(cumo_ndfunc_t *nf, int argc, ...)
#else
cumo_na_ndloop(nf, argc, va_alist)
  cumo_ndfunc_t *nf;
  int argc;
  va_dcl
#endif
{
    va_list ar;

    int i;
    VALUE *argv;
    volatile VALUE args;

    argv = ALLOCA_N(VALUE,argc);

    va_init_list(ar, argc);
    for (i=0; i<argc; i++) {
        argv[i] = va_arg(ar, VALUE);
    }
    va_end(ar);

    args = rb_ary_new4(argc, argv);

    return cumo_na_ndloop_main(nf, args, NULL);
}


VALUE
cumo_na_ndloop2(cumo_ndfunc_t *nf, VALUE args)
{
    return cumo_na_ndloop_main(nf, args, NULL);
}

VALUE
#ifdef HAVE_STDARG_PROTOTYPES
cumo_na_ndloop3(cumo_ndfunc_t *nf, void *ptr, int argc, ...)
#else
cumo_na_ndloop3(nf, ptr, argc, va_alist)
  cumo_ndfunc_t *nf;
  void *ptr;
  int argc;
  va_dcl
#endif
{
    va_list ar;

    int i;
    VALUE *argv;
    volatile VALUE args;

    argv = ALLOCA_N(VALUE,argc);

    va_init_list(ar, argc);
    for (i=0; i<argc; i++) {
        argv[i] = va_arg(ar, VALUE);
    }
    va_end(ar);

    args = rb_ary_new4(argc, argv);

    return cumo_na_ndloop_main(nf, args, ptr);
}

VALUE
cumo_na_ndloop4(cumo_ndfunc_t *nf, void *ptr, VALUE args)
{
    return cumo_na_ndloop_main(nf, args, ptr);
}

//----------------------------------------------------------------------

VALUE
cumo_na_info_str(VALUE ary)
{
    int nd, i;
    char tmp[32];
    VALUE buf;
    cumo_narray_t *na;

    CumoGetNArray(ary,na);
    nd = na->ndim;

    buf = rb_str_new2(rb_class2name(rb_obj_class(ary)));
    if (CUMO_NA_TYPE(na) == CUMO_NARRAY_VIEW_T) {
        rb_str_cat(buf,"(view)",6);
    }
    rb_str_cat(buf,"#shape=[",8);
    if (nd>0) {
        for (i=0;;) {
            sprintf(tmp,"%"SZF"u",na->shape[i]);
            rb_str_cat2(buf,tmp);
            if (++i==nd) break;
            rb_str_cat(buf,",",1);
        }
    }
    rb_str_cat(buf,"]",1);
    return buf;
}


//----------------------------------------------------------------------

extern int cumo_na_inspect_cols_;
extern int cumo_na_inspect_rows_;
#define ncol cumo_na_inspect_cols_
#define nrow cumo_na_inspect_rows_

static void
loop_inspect(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp)
{
    int nd, i, ii;
    size_t *c;
    int col=0, row=0;
    long len;
    VALUE str;
    cumo_na_text_func_t func = (cumo_na_text_func_t)(nf->func);
    VALUE buf, opt;

    ndloop_sync_md_index(lp);
    ndloop_stage_host_reads(lp);

    nd = lp->ndim;
    buf = lp->loop_opt;
    //opt = *(VALUE*)(lp->user.opt_ptr);
    opt = lp->user.option;

    if (ndloop_is_empty(lp)) {
        rb_str_cat(buf,"[]",2);
        return;
    }

    rb_str_cat(buf,"\n",1);

    c = ALLOCA_N(size_t, nd+1);
    for (i=0; i<=nd; i++) c[i]=0;

    if (nd>0) {
        rb_str_cat(buf,"[",1);
    } else {
        rb_str_cat(buf,"",0);
    }

    col = nd*2;
    for (i=0;;) {
        if (i<nd-1) {
            for (ii=0; ii<i; ii++) rb_str_cat(buf," ",1);
            for (; ii<nd-1; ii++) rb_str_cat(buf,"[",1);
        }
        for (; i<nd; i++) {
            if (LITER(lp,i,0).idx) {
                LITER(lp,i+1,0).pos = LITER(lp,i,0).pos + LITER(lp,i,0).idx[c[i]];
            } else {
                LITER(lp,i+1,0).pos = LITER(lp,i,0).pos + LITER(lp,i,0).step*c[i];
            }
        }
        ndloop_stage_before_row(lp);
        str = (*func)(LARG(lp,0).ptr, LITER(lp,i,0).pos, opt);

        len = RSTRING_LEN(str) + 2;
        if (ncol>0 && col+len > ncol-3) {
            rb_str_cat(buf,"...",3);
            c[i-1] = lp->n[i-1];
        } else {
            rb_str_append(buf, str);
            col += len;
        }
        for (;;) {
            if (i==0) goto loop_end;
            i--;
            if (++c[i] < lp->n[i]) break;
            rb_str_cat(buf,"]",1);
            c[i] = 0;
        }
        //line_break:
        rb_str_cat(buf,", ",2);
        if (i<nd-1) {
            rb_str_cat(buf,"\n ",2);
            col = nd*2;
            row++;
            if (row==nrow) {
                rb_str_cat(buf,"...",3);
                goto loop_end;
            }
        }
    }
 loop_end:
    ;
}


VALUE
cumo_na_ndloop_inspect(VALUE nary, cumo_na_text_func_t func, VALUE opt)
{
    volatile VALUE args;
    cumo_na_md_loop_t lp;
    VALUE buf;
    cumo_ndfunc_arg_in_t ain[3] = {{Qnil,0},{cumo_sym_loop_opt},{cumo_sym_option}};
    cumo_ndfunc_t nf = { (cumo_na_iter_func_t)func, CUMO_NO_LOOP|CUMO_NDF_HOST_READ, 3, 0, ain, 0 };
    //nf = cumo_ndfunc_alloc(NULL, CUMO_NO_LOOP, 1, 0, Qnil);

    buf = cumo_na_info_str(nary);

    if (cumo_na_get_pointer(nary)==NULL) {
        return rb_str_cat(buf,"(empty)",7);
    }

    //rb_p(args);
    //if (cumo_na_debug_flag) print_ndfunc(&nf);

    args = rb_ary_new3(3,nary,buf,opt);

    // cast arguments to NArray
    //ndloop_cast_args(nf, args);

    // allocate ndloop struct
    ndloop_alloc(&lp, &nf, args, NULL, 0, loop_inspect);

    rb_ensure(ndloop_run, (VALUE)&lp, ndloop_release, (VALUE)&lp);

    return buf;
}


//----------------------------------------------------------------------

static void
loop_store_subnarray(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp, int i0, size_t *c, VALUE a)
{
    int nd = lp->ndim;
    int i, j;
    bool *reach;
    cumo_narray_t *na;
    int *dim_map;
    size_t *saved_shape = LARG(lp,1).shape;
    ssize_t saved_elmsz = LARG(lp,1).elmsz;
    VALUE a_type;

    a_type = rb_obj_class(LARG(lp,0).value);
    if (rb_obj_class(a) != a_type) {
        a = rb_funcall(a_type, cumo_id_cast, 1, a);
    }
    CumoGetNArray(a,na);
    if (na->ndim != nd-i0+1) {
        rb_raise(cumo_na_eShapeError, "mismatched dimension of sub-narray: "
                 "nd_src=%d, nd_dst=%d", na->ndim, nd-i0+1);
    }
    dim_map = ALLOCA_N(int, na->ndim);
    for (i=0; i<na->ndim; i++) {
        dim_map[i] = lp->trans_map[i+i0];
        //printf("dim_map[i=%d] = %d, i0=%d\n", i, dim_map[i], i0);
    }
    for (i=0; i<=nd+lp->user.ndim; i++) {
        LITER(lp,i,1).pos = 0;
        LITER(lp,i,1).step = 0;
        LITER(lp,i,1).idx = NULL;
    }
    ndloop_set_stepidx(lp, 1, a, dim_map, CUMO_NDL_READ);
    LITER(lp,i0,1).pos = LITER(lp,0,1).pos;
    LARG(lp,1).shape = &lp->sub_narray_len;

    // The sub-narray binds its own index array here, after the entry point
    // looked, so the wait for the kernel that filled it belongs here too. The
    // destination was asked about at the entry point, which waits only if it
    // has not settled, so a row that binds no index the loop below walks has
    // nothing left to ask. An index on the row's last dimension goes to the
    // per-dtype store iterator, which waits for itself where it reads one on
    // the host.
    ndloop_wait_arg_index(lp, 1);

    // loop body
    reach = ALLOCA_N(bool, nd+1);
    reach[i0] = true;
    for (i=i0;;) {
        for (; i<nd; i++) {
            reach[i+1] = reach[i] && (c[i] < na->shape[i-i0]);
            for (j=0; j<lp->narg; j++) {
                if (j==1 && !reach[i+1]) {
                    LITER(lp,i+1,j).pos = LITER(lp,i,j).pos;
                } else if (LITER(lp,i,j).idx) {
                    LITER(lp,i+1,j).pos = LITER(lp,i,j).pos + LITER(lp,i,j).idx[c[i]];
                } else {
                    LITER(lp,i+1,j).pos = LITER(lp,i,j).pos + LITER(lp,i,j).step*c[i];
                }
            }
        }
        lp->sub_narray_len = reach[nd] ? na->shape[na->ndim-1] : 0;

        (*(nf->func))(&(lp->user));

        for (;;) {
            if (i<=i0) goto loop_end;
            i--; c[i]++;
            if (c[i] < lp->n[i]) break;
            c[i] = 0;
        }
    }
 loop_end:
    LARG(lp,1).ptr = NULL;
    LARG(lp,1).shape = saved_shape;
    LARG(lp,1).elmsz = saved_elmsz;
}


static void
loop_store_rarray(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp)
{
    size_t *c;
    int     i;
    VALUE  *a;
    int nd = lp->ndim;

    if (ndloop_is_empty(lp)) {
        return;
    }

    ndloop_sync_md_index(lp);

    // counter
    c = ALLOCA_N(size_t, nd+1);
    for (i=0; i<=nd; i++) c[i]=0;

    // array at each dimension
    a = ALLOCA_N(VALUE, nd+1);
    a[0] = LARG(lp,1).value;

    //print_ndloop(lp);

    // loop body
    for (i=0;;) {
        for (; i<nd; i++) {
            if (LITER(lp,i,0).idx) {
                LITER(lp,i+1,0).pos = LITER(lp,i,0).pos + LITER(lp,i,0).idx[c[i]];
            } else {
                LITER(lp,i+1,0).pos = LITER(lp,i,0).pos + LITER(lp,i,0).step*c[i];
            }
            if (TYPE(a[i])==T_ARRAY) {
                if (c[i] < (size_t)RARRAY_LEN(a[i])) {
                    a[i+1] = RARRAY_AREF(a[i],c[i]);
                } else {
                    a[i+1] = Qnil;
                }
            } else if (CumoIsNArray(a[i])) {
                //printf("a[i=%d]=0x%lx\n",i,a[i]);
                loop_store_subnarray(nf,lp,i,c,a[i]);
                goto loop_next;
            } else {
                if (c[i]==0) {
                    a[i+1] = a[i];
                } else {
                    a[i+1] = Qnil;
                }
            }
            //printf("c[%d]=%lu\n",i,c[i]);
        }

        //printf("a[i=%d]=0x%lx\n",i,a[i]);
        if (CumoIsNArray(a[i])) {
            loop_store_subnarray(nf,lp,i,c,a[i]);
        } else {
            LARG(lp,1).value = a[i];
            (*(nf->func))(&(lp->user));
        }

    loop_next:
        for (;;) {
            if (i<=0) goto loop_end;
            i--; c[i]++;
            if (c[i] < lp->n[i]) break;
            c[i] = 0;
        }
    }
 loop_end:
    ;
}

struct cumo_na_stage_pool {
    char       *ptr;
    size_t      bytes;
    int         launched;
    cudaError_t st;
};

static VALUE ndloop_store_rarray_walk(VALUE arg);
static VALUE ndloop_stage_pool_release(VALUE pool);

typedef struct {
    cumo_ndfunc_t        *ndf;
    VALUE                 nary;
    VALUE                 rary;
    cumo_na_stage_pool_t *pool;
} ndloop_store_rarray_arg_t;

static VALUE
ndloop_store_rarray_opt(cumo_ndfunc_t *nf, VALUE nary, VALUE rary, void *opt_ptr)
{
    cumo_na_md_loop_t lp;
    VALUE args;

    //rb_p(args);
    if (cumo_na_debug_flag) print_ndfunc(nf);

    args = rb_assoc_new(nary,rary);

    // cast arguments to NArray
    //ndloop_cast_args(nf, args);

    // allocate ndloop struct
    ndloop_alloc(&lp, nf, args, opt_ptr, 0, loop_store_rarray);

    return rb_ensure(ndloop_run, (VALUE)&lp, ndloop_release, (VALUE)&lp);
}

static VALUE
ndloop_store_rarray_walk(VALUE arg)
{
    ndloop_store_rarray_arg_t *r = (ndloop_store_rarray_arg_t*)arg;

    return ndloop_store_rarray_opt(r->ndf, r->nary, r->rary, r->pool);
}

// Only a row wanting more takes the wait, and a walk's rows are all the length
// of the destination's last axis, so that is the first row and no other.
char *
cumo_na_stage_pool_get(cumo_na_stage_pool_t *pool, size_t bytes)
{
    if (pool->bytes < bytes) {
        ndloop_stage_pool_release((VALUE)pool);
        pool->ptr = cumo_cuda_runtime_malloc(bytes);
        pool->bytes = bytes;
    }
    pool->launched = 1;
    return pool->ptr;
}

// Takes the pool as its argument so it can be handed to rb_ensure as it is.
// The wait belongs to whoever reports, so it is folded in rather than dropped.
static VALUE
ndloop_stage_pool_release(VALUE pool)
{
    cumo_na_stage_pool_t *p = (cumo_na_stage_pool_t*)pool;

    cumo_cuda_runtime_return_scratch(p->ptr, p->launched, &p->st);
    p->ptr = NULL;
    p->bytes = 0;
    p->launched = 0;
    return Qnil;
}

// The pool is made here so that no caller can forget it: the iterators read it
// through the loop's opt_ptr and have no other way to reach one.
VALUE
cumo_na_ndloop_store_rarray(cumo_ndfunc_t *nf, VALUE nary, VALUE rary)
{
    cumo_na_stage_pool_t pool = {NULL, 0, 0, cudaSuccess};
    ndloop_store_rarray_arg_t r = {nf, nary, rary, &pool};
    VALUE v;

    v = rb_ensure(ndloop_store_rarray_walk, (VALUE)&r, ndloop_stage_pool_release, (VALUE)&pool);
    cumo_cuda_runtime_check_status(pool.st);
    return v;
}


VALUE
cumo_na_ndloop_store_rarray2(cumo_ndfunc_t *nf, VALUE nary, VALUE rary, VALUE opt)
{
    cumo_na_md_loop_t lp;
    VALUE args;

    //rb_p(args);
    if (cumo_na_debug_flag) print_ndfunc(nf);

    //args = rb_assoc_new(rary,nary);
    args = rb_ary_new3(3,nary,rary,opt);

    // cast arguments to NArray
    //ndloop_cast_args(nf, args);

    // allocate ndloop struct
    ndloop_alloc(&lp, nf, args, NULL, 0, loop_store_rarray);

    return rb_ensure(ndloop_run, (VALUE)&lp, ndloop_release, (VALUE)&lp);
}


//----------------------------------------------------------------------

static void
loop_narray_to_rarray(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp)
{
    size_t *c;
    int i, zd;
    //int nargs = nf->narg + nf->nres;
    int nd = lp->ndim;
    VALUE *a;
    volatile VALUE a0;

    ndloop_sync_md_index(lp);
    ndloop_stage_host_reads(lp);

    // alloc counter
    c = ALLOCA_N(size_t, nd+1);
    for (i=0; i<=nd; i++) c[i]=0;
    //c[i]=1; // for zero-dim
    //fprintf(stderr,"in loop_narray_to_rarray, nd=%d\n",nd);

    a = ALLOCA_N(VALUE, nd+1);
    a[0] = a0 = lp->loop_opt;

    // The first loop dimension of zero, or nd. Nothing lies inside it, so no
    // array is nested past it and no row is read; the iterator reads a row of
    // its own dimension's length, which is zero only when that is the zero.
    for (zd=0; zd<nd && lp->n[zd]!=0; zd++) ;

    // loop body
    for (i=0;;) {
        for (; i<nd && i<=zd; i++) {
            if (LITER(lp,i,0).idx) {
                LITER(lp,i+1,0).pos = LITER(lp,i,0).pos + LITER(lp,i,0).idx[c[i]];
            } else {
                LITER(lp,i+1,0).pos = LITER(lp,i,0).pos + LITER(lp,i,0).step*c[i];
            }
            if (c[i]==0) {
                a[i+1] = rb_ary_new2(lp->n[i]);
                rb_ary_push(a[i],a[i+1]);
            }
        }

        if (zd == nd) {
            //lp->user.info = a[i];
            LARG(lp,1).value = a[i];
            ndloop_stage_before_row(lp);
            (*(nf->func))(&(lp->user));
        }

        for (;;) {
            if (i<=0) goto loop_end;
            i--;
            if (++c[i] < lp->n[i]) break;
            c[i] = 0;
        }
    }
 loop_end:
    ;
}

VALUE
cumo_na_ndloop_cast_narray_to_rarray(cumo_ndfunc_t *nf, VALUE nary, VALUE fmt)
{
    cumo_na_md_loop_t lp;
    VALUE args, a0;

    //rb_p(args);
    if (cumo_na_debug_flag) print_ndfunc(nf);

    a0 = rb_ary_new();
    args = rb_ary_new3(3,nary,a0,fmt);

    // cast arguments to NArray
    //ndloop_cast_args(nf, args);

    // allocate ndloop struct
    ndloop_alloc(&lp, nf, args, NULL, 0, loop_narray_to_rarray);

    rb_ensure(ndloop_run, (VALUE)&lp, ndloop_release, (VALUE)&lp);
    return RARRAY_AREF(a0,0);
}


//----------------------------------------------------------------------

static void
loop_narray_with_index(cumo_ndfunc_t *nf, cumo_na_md_loop_t *lp)
{
    size_t *c;
    int i,j;
    int nd = lp->ndim;

    if (nd < 0) {
        rb_bug("bug? lp->ndim = %d\n", lp->ndim);
    }
    if (ndloop_is_empty(lp)) {
        return;
    }

    ndloop_sync_md_index(lp);
    ndloop_stage_host_reads(lp);

    // pass total ndim to iterator
    lp->user.ndim += nd;

    // alloc counter
    lp->user.opt_ptr = c = ALLOCA_N(size_t, nd+1);
    for (i=0; i<=nd; i++) c[i]=0;

    // loop body
    for (i=0;;) {
        for (; i<nd; i++) {
            // j-th argument
            for (j=0; j<lp->narg; j++) {
                if (LITER(lp,i,j).idx) {
                    LITER(lp,i+1,j).pos = LITER(lp,i,j).pos + LITER(lp,i,j).idx[c[i]];
                } else {
                    LITER(lp,i+1,j).pos = LITER(lp,i,j).pos + LITER(lp,i,j).step*c[i];
                }
                //printf("j=%d c[i=%d]=%lu pos=%lu\n",j,i,c[i],LITER(lp,i+1,j).pos);
            }
        }

        ndloop_stage_before_row(lp);
        (*(nf->func))(&(lp->user));

        for (;;) {
            if (i<=0) goto loop_end;
            i--;
            if (++c[i] < lp->n[i]) break;
            c[i] = 0;
        }
    }
 loop_end:
    ;
}


VALUE
#ifdef HAVE_STDARG_PROTOTYPES
cumo_na_ndloop_with_index(cumo_ndfunc_t *nf, int argc, ...)
#else
cumo_na_ndloop_with_index(nf, argc, va_alist)
  cumo_ndfunc_t *nf;
  int argc;
  va_dcl
#endif
{
    va_list ar;

    int i;
    VALUE *argv;
    volatile VALUE args;
    cumo_na_md_loop_t lp;

    argv = ALLOCA_N(VALUE,argc);

    va_init_list(ar, argc);
    for (i=0; i<argc; i++) {
        argv[i] = va_arg(ar, VALUE);
    }
    va_end(ar);

    args = rb_ary_new4(argc, argv);

    //return cumo_na_ndloop_main(nf, args, NULL);
    if (cumo_na_debug_flag) print_ndfunc(nf);

    // cast arguments to NArray
    //copy_flag = ndloop_cast_args(nf, args);

    // allocate ndloop struct
    ndloop_alloc(&lp, nf, args, 0, 0, loop_narray_with_index);

    return rb_ensure(ndloop_run, (VALUE)&lp, ndloop_release, (VALUE)&lp);
}


void
Init_cumo_na_ndloop()
{
    cumo_id_cast    = rb_intern("cast");
    cumo_id_extract = rb_intern("extract");
}
