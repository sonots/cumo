static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t   i;
    CUMO_BIT_DIGIT *a1, x=0;
    size_t   p1;
    ssize_t  s1;
    size_t  *idx1;

    VALUE *a;
    size_t *c;
    int nd, md;

    c = (size_t*)(lp->opt_ptr);
    cumo_na_with_index_dims(lp->ndim, &nd, &md);
    a = ALLOCA_N(VALUE,md);

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a1, p1, s1, idx1);
    c[nd] = 0;


    if (idx1) {
        for (; i--;) {
            if (cumo_cuda_runtime_sync_if_busy()) { CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>"); }
            CUMO_LOAD_BIT(a1, p1+*idx1, x); idx1++;
            cumo_na_yield_with_index(m_data_to_num(x),c,a,nd,md);
            c[nd]++;
        }
    } else {
        for (; i--;) {
            if (cumo_cuda_runtime_sync_if_busy()) { CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>"); }
            CUMO_LOAD_BIT(a1, p1, x); p1+=s1;
            cumo_na_yield_with_index(m_data_to_num(x),c,a,nd,md);
            c[nd]++;
        }
    }
}

/*
  Invokes the given block once for each element of self,
  passing that element and indices along each axis as parameters.
  @overload <%=name%>
  @return [Cumo::NArray] self
  For a block {|x,i,j,...| ... }
  @yield [x,i,j,...]  x is an element, i,j,... are multidimensional indices.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    cumo_ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP_NIP, 1,0, ain,0};

    cumo_na_ndloop_with_index(&ndf, 1, self);
    return self;
}
