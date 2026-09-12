static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t i, s1;
    char *p1;
    size_t *idx1;
    dtype x;
    VALUE *a;
    size_t *c;
    int nd, md;

    c = (size_t*)(lp->opt_ptr);
    nd = lp->ndim;
    if (nd > 0) {nd--;}
    md = nd + 2;
    a = ALLOCA_N(VALUE,md);

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    c[nd] = 0;


    if (idx1) {
        for (; i--;) {
            if (cumo_cuda_runtime_sync_if_busy()) { CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>"); }
            CUMO_GET_DATA_INDEX(p1,idx1,dtype,x);
            cumo_na_yield_with_index(m_data_to_num(x),c,a,nd,md);
            c[nd]++;
        }
    } else {
        for (; i--;) {
            if (cumo_cuda_runtime_sync_if_busy()) { CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>"); }
            CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
            cumo_na_yield_with_index(m_data_to_num(x),c,a,nd,md);
            c[nd]++;
        }
    }
}

/*
  Invokes the given block once for each element of self,
  passing that element and indices along each axis as parameters.
  @overload <%=name%>
  For a block `{|x,i,j,...| ... }`,
  @yieldparam [Numeric] x  an element
  @yieldparam [Integer] i,j,...  multitimensional indices
  @return [Cumo::NArray] self
  @see #each
  @see #map_with_index
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    cumo_ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP_NIP, 1,0, ain,0};

    cumo_na_ndloop_with_index(&ndf, 1, self);
    return self;
}
