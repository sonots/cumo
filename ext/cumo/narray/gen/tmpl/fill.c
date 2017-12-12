void
<%="#{c_iter}_kernel_launch"%>(dtype *ptr, ssize_t step, dtype val, size_t N);

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
    dtype *ptr = (dtype*)p1;
    ssize_t step = s1 / sizeof(dtype);
    y = m_num_to_data(x);
    if (idx1) {
        for (; i--;) {
            SET_DATA_INDEX(p1,idx1,dtype,y);
        }
    } else {
        //for (; i--;) {
        //    SET_DATA_STRIDE(p1,s1,dtype,y);
        //}
        <%="#{c_iter}_kernel_launch"%>(ptr,step,y,i);
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
