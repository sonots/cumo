static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  i;
    char   *p1;
    ssize_t s1;
    size_t *idx1;
    dtype *x;
    VALUE y;
    volatile VALUE a;
    VALUE fmt = lp->option;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    a = rb_ary_new2(i);
    rb_ary_push(lp->args[1].value, a);
    //SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    if (idx1) {
        for (; i--;) {
            x = (dtype*)(p1 + *idx1);  idx1++;
            y = format_<%=type_name%>(fmt, x);
            rb_ary_push(a,y);
        }
    } else {
        for (; i--;) {
            x = (dtype*)p1;  p1+=s1;
            y = format_<%=type_name%>(fmt, x);
            rb_ary_push(a,y);
        }
    }
}

/*
  Format elements into strings.
  @overload <%=name%> format
  @param [String] format
  @return [Array] array of formated strings.
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE fmt=Qnil;
    cumo_ndfunc_arg_in_t ain[3] = {{Qnil,0},{cumo_sym_loop_opt},{cumo_sym_option}};
    cumo_ndfunc_arg_out_t aout[1] = {{rb_cArray,0}}; // dummy?
    cumo_ndfunc_t ndf = { <%=c_iter%>, FULL_LOOP_NIP, 3, 1, ain, aout };

    rb_scan_args(argc, argv, "01", &fmt);
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    return cumo_na_ndloop_cast_narray_to_rarray(&ndf, self, fmt);
}
