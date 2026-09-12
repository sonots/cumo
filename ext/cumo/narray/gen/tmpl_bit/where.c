typedef struct {
    size_t count;
    char  *idx0;
    char  *idx1;
    size_t elmsz;
    char  *scratch;
    size_t cap1;
    size_t cap0;
    size_t wrote1;
    size_t wrote0;
    int    used_kernel;
} where_opt_t;

#define STORE_INT(ptr, esz, x) memcpy(ptr,&(x),esz)

char *cumo_bit_where_scratch_new(void);
void cumo_bit_where_kernel_launch(CUMO_BIT_DIGIT *a, size_t p, ssize_t s, size_t *idx, uint64_t n, int invert, char *out, size_t elmsz, uint64_t cap, uint64_t count, char *scratch);

// Taking a pointer runs allocate, which a Ruby class can redefine, so both are
// read once every allocation has run.
static void
bit_where_take(VALUE idx, char **ptr, size_t *cap)
{
    *ptr = cumo_na_get_pointer_for_write(idx);
    *cap = CUMO_RNARRAY_SIZE(idx);
}

// The scatter keeps counting what it wanted to write after the room ran out, so
// the cursors say whether the walk still matched the counts that sized it.
static cudaError_t
bit_where_cursors(char *scratch, uint64_t *v, int n)
{
    return cudaMemcpy(v, scratch, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
}

static void
bit_where_check(size_t a, size_t b)
{
    if (a != b) {
        rb_raise(rb_eRuntimeError, "the bit array and its index array no longer agree");
    }
}

// The number of ones, read back with a copy of the count rather than through
// the managed pointer, which would fault the whole page. The one sync it costs
// sizes the output; the compaction itself never reads anything back.
static size_t
bit_where_count_true(VALUE self)
{
    VALUE v = <%=find_tmpl("count_true").c_func%>(0, NULL, self);
    uint64_t count;
    char *ptr;

    if (RB_INTEGER_TYPE_P(v)) {
        return NUM2SIZET(v);
    }
    ptr = cumo_na_get_pointer_for_read(v) + cumo_na_get_offset(v);
    cumo_cuda_runtime_check_status(cudaMemcpy(&count, ptr, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    cumo_na_release_lock(v);
    return (size_t)count;
}

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  i;
    CUMO_BIT_DIGIT *a;
    size_t  p;
    ssize_t s;
    size_t *idx;
    CUMO_BIT_DIGIT x=0;
    char   *idx1;
    size_t  count;
    size_t  e, cap, wrote;
    where_opt_t *g;

    g = (where_opt_t*)(lp->opt_ptr);
    count = g->count;
    idx1  = g->idx1;
    e     = g->elmsz;
    cap   = g->cap1;
    wrote = g->wrote1;
    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a, p, s, idx);
    if (i >= CUMO_BIT_WHERE_MIN_KERNEL_SIZE && g->scratch) {
        // idx1 stays put: the device-side cursor in the scratch orders the
        // writes of successive calls instead.
        cumo_bit_where_kernel_launch(a,p,s,idx,i,0,idx1,e,cap,count,g->scratch);
        g->count = count + i;
        g->used_kernel = 1;
        return;
    }

    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    if (idx) {
        for (; i--;) {
            CUMO_LOAD_BIT(a, p+*idx, x);
            idx++;
            if (x!=0) {
                if (wrote < cap) {
                    STORE_INT(idx1,e,count);
                    idx1 += e;
                }
                wrote++;
            }
            count++;
        }
    } else {
        for (; i--;) {
            CUMO_LOAD_BIT(a, p, x);
            p+=s;
            if (x!=0) {
                if (wrote < cap) {
                    STORE_INT(idx1,e,count);
                    idx1 += e;
                }
                wrote++;
            }
            count++;
        }
    }
    g->count  = count;
    g->idx1   = idx1;
    g->wrote1 = wrote;
}

/*
  Returns the array of index where the bit is one (true).
  @overload <%=op_map%>
  @return [Cumo::Int32,Cumo::Int64]
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    volatile VALUE idx_1;
    size_t size, n_1;
    uint64_t cur[1];
    cudaError_t st = cudaSuccess;
    where_opt_t *g;

    cumo_ndfunc_arg_in_t ain[1] = {{cT,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 1, 0, ain, 0 };

    n_1 = bit_where_count_true(self);
    size = CUMO_RNARRAY_SIZE(self);
    g = ALLOCA_N(where_opt_t,1);
    g->count = 0;
    if (size>4294967295ul) {
        idx_1 = cumo_na_new(cumo_cInt64, 1, &n_1);
        g->elmsz = 8;
    } else {
        idx_1 = cumo_na_new(cumo_cInt32, 1, &n_1);
        g->elmsz = 4;
    }
    bit_where_take(idx_1, &g->idx1, &g->cap1);
    bit_where_check(CUMO_RNARRAY_SIZE(self), size);
    g->wrote1 = 0;
    g->idx0 = NULL;
    g->cap0 = 0;
    g->wrote0 = 0;
    g->scratch = NULL;
    g->used_kernel = 0;
    if (size >= CUMO_BIT_WHERE_MIN_KERNEL_SIZE) {
        g->scratch = cumo_bit_where_scratch_new();
    }
    cumo_na_ndloop3(&ndf, g, 1, self);
    if (g->used_kernel) {
        st = bit_where_cursors(g->scratch, cur, 1);
        if (st == cudaSuccess) { g->wrote1 += (size_t)cur[0]; }
    }
    if (g->scratch) {
        cumo_cuda_runtime_free(g->scratch);
    }
    cumo_cuda_runtime_check_status(st);
    cumo_na_release_lock(idx_1);
    bit_where_check(g->wrote1, g->cap1);
    return idx_1;
}
