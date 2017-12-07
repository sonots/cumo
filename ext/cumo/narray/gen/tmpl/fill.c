__global__ void <%="#{c_iter}_kernel"%>(dtype *ptr, ssize_t step, dtype val, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x * step; i < N; i += blockDim.x * gridDim.x) {
        ptr[i] = val;
    }
}

static void
<%=c_iter%>(na_loop_t *const lp)
{
    size_t   i;
    char    *p1;
    ssize_t  s1;
    size_t  *idx1;
    VALUE    x = lp->option;
    dtype    y;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    y = m_num_to_data(x);
    if (idx1) {
        for (; i--;) {
            SET_DATA_INDEX(p1,idx1,dtype,y);
        }
    } else {
        //for (; i--;) {
        //    SET_DATA_STRIDE(p1,s1,dtype,y);
        //}
        size_t maxBlockDim = 128;
        size_t gridDim = (i / maxBlockDim) + 1;
        size_t blockDim = (i > maxBlockDim) ? maxBlockDim : i;
        // ref. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
        if (gridDim > 2147483647) gridDim = 2147483647;
        <%="#{c_iter}_kernel"%><<<gridDim, blockDim>>>(p1,s1,y,i);
    }
}

/*
  Fill elements with other.
  @overload <%=name%> other
  @param [Numeric] other
  @return [Cumo::<%=class_name%>] self.
*/
static VALUE
<%=c_func(1)%>(VALUE self, VALUE val)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{sym_option}};
    ndfunc_t ndf = { <%=c_iter%>, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, val);
    return self;
}
