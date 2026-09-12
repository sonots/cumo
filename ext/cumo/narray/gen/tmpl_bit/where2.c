static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  i;
    CUMO_BIT_DIGIT *a;
    size_t  p;
    ssize_t s;
    size_t *idx;
    CUMO_BIT_DIGIT x=0;
    char   *idx0, *idx1;
    size_t  count;
    size_t  e, cap1, cap0, wrote1, wrote0;
    where_opt_t *g;

    g = (where_opt_t*)(lp->opt_ptr);
    count = g->count;
    idx0  = g->idx0;
    idx1  = g->idx1;
    e     = g->elmsz;
    cap1  = g->cap1;
    cap0  = g->cap0;
    wrote1 = g->wrote1;
    wrote0 = g->wrote0;
    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a, p, s, idx);
    if (i >= CUMO_BIT_WHERE_MIN_KERNEL_SIZE && g->scratch) {
        cumo_bit_where_kernel_launch(a,p,s,idx,i,0,idx1,e,cap1,count,g->scratch);
        cumo_bit_where_kernel_launch(a,p,s,idx,i,1,idx0,e,cap0,count,g->scratch);
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
            if (x==0) {
                if (wrote0 < cap0) {
                    STORE_INT(idx0,e,count);
                    idx0 += e;
                }
                wrote0++;
            } else {
                if (wrote1 < cap1) {
                    STORE_INT(idx1,e,count);
                    idx1 += e;
                }
                wrote1++;
            }
            count++;
        }
    } else {
        for (; i--;) {
            CUMO_LOAD_BIT(a, p, x);
            p+=s;
            if (x==0) {
                if (wrote0 < cap0) {
                    STORE_INT(idx0,e,count);
                    idx0 += e;
                }
                wrote0++;
            } else {
                if (wrote1 < cap1) {
                    STORE_INT(idx1,e,count);
                    idx1 += e;
                }
                wrote1++;
            }
            count++;
        }
    }
    g->count  = count;
    g->idx0   = idx0;
    g->idx1   = idx1;
    g->wrote0 = wrote0;
    g->wrote1 = wrote1;
}

/*
  Returns two index arrays.
  The first array contains index where the bit is one (true).
  The second array contains index where the bit is zero (false).
  @overload <%=op_map%>
  @return [Cumo::Int32,Cumo::Int64]*2
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    volatile VALUE idx_1, idx_0;
    size_t size, n_1, n_0;
    uint64_t cur[2];
    cudaError_t st = cudaSuccess;
    where_opt_t *g;

    cumo_ndfunc_arg_in_t ain[1] = {{cT,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 1, 0, ain, 0 };

    n_1 = bit_where_count_true(self);
    size = CUMO_RNARRAY_SIZE(self);
    n_0 = size - n_1;
    g = ALLOCA_N(where_opt_t,1);
    g->count = 0;
    if (size>4294967295ul) {
        idx_1 = cumo_na_new(cumo_cInt64, 1, &n_1);
        idx_0 = cumo_na_new(cumo_cInt64, 1, &n_0);
        g->elmsz = 8;
    } else {
        idx_1 = cumo_na_new(cumo_cInt32, 1, &n_1);
        idx_0 = cumo_na_new(cumo_cInt32, 1, &n_0);
        g->elmsz = 4;
    }
    cumo_na_get_pointer_for_write(idx_1);
    cumo_na_get_pointer_for_write(idx_0);
    bit_where_take(idx_1, &g->idx1, &g->cap1);
    bit_where_take(idx_0, &g->idx0, &g->cap0);
    bit_where_check(CUMO_RNARRAY_SIZE(self), size);
    g->wrote1 = 0;
    g->wrote0 = 0;
    g->scratch = NULL;
    g->used_kernel = 0;
    if (size >= CUMO_BIT_WHERE_MIN_KERNEL_SIZE) {
        g->scratch = cumo_bit_where_scratch_new();
    }
    cumo_na_ndloop3(&ndf, g, 1, self);
    if (g->used_kernel) {
        st = bit_where_cursors(g->scratch, cur, 2);
        if (st == cudaSuccess) {
            g->wrote1 += (size_t)cur[0];
            g->wrote0 += (size_t)cur[1];
        }
    }
    if (g->scratch) {
        cumo_cuda_runtime_free(g->scratch);
    }
    cumo_cuda_runtime_check_status(st);
    cumo_na_release_lock(idx_0);
    cumo_na_release_lock(idx_1);
    bit_where_check(g->wrote1, g->cap1);
    bit_where_check(g->wrote0, g->cap0);
    return rb_assoc_new(idx_1,idx_0);
}
