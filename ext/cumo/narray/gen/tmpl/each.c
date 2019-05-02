static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t i, s1;
    char *p1;
    size_t *idx1;
    dtype x;
    VALUE y;

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);

    CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");

    if (idx1) {
        for (; i--;) {
            cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
            CUMO_GET_DATA_INDEX(p1,idx1,dtype,x);
            y = m_data_to_num(x);
            rb_yield(y);
        }
    } else {
        for (; i--;) {
            cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
            CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
            y = m_data_to_num(x);
            rb_yield(y);
        }
    }
}

/*
  Calls the given block once for each element in self,
  passing that element as a parameter.
  @overload <%=name%>
  @return [Cumo::NArray] self
  For a block {|x| ... }
  @yield [x]  x is element of NArray.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    cumo_ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP_NIP, 1,0, ain,0};

    cumo_na_ndloop(&ndf, 1, self);
    return self;
}
