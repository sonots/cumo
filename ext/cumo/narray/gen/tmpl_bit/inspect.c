static VALUE
<%=c_iter%>(char *ptr, size_t pos, VALUE fmt)
{
    dtype x;
    LOAD_BIT(ptr,pos,x);
    return format_<%=type_name%>(fmt, x);
}

/*
  Returns a string containing a human-readable representation of NArray.
  @overload inspect
  @return [String]
*/
static VALUE
<%=c_func(0)%>(VALUE ary)
{
    SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    return na_ndloop_inspect(ary, <%=c_iter%>, Qnil);
}
