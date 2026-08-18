static VALUE
<%=c_iter%>(char *ptr, size_t pos, VALUE fmt)
{
<% if is_object %>
    return rb_inspect(*(VALUE*)(ptr+pos));
<% else %>
    return format_<%=type_name%>(fmt, (dtype*)(ptr+pos));
<% end %>
}

/*
  Returns a string containing a human-readable representation of NArray.
  @overload inspect
  @return [String]
*/
static VALUE
<%=c_func(0)%>(VALUE ary)
{
    CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    return cumo_na_ndloop_inspect(ary, <%=c_iter%>, Qnil);
}
