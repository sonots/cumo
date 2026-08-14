<% unless is_object %>
void <%="cumo_#{c_func(:nodef)}_kernel_launch"%>(dtype *ptr, dtype x);
<% end %>

static VALUE
<%=c_func(:nodef)%>(dtype x)
{
    VALUE v;
    dtype *ptr;

    v = cumo_na_new(cT, 0, NULL);
    ptr = (dtype*)cumo_na_get_pointer_for_write(v);
<% if is_object %>
    // RObject data is host memory. A kernel would write it asynchronously with
    // nothing ordering that against the host loop that reads it back.
    *ptr = x;
<% else %>
    <%="cumo_#{c_func(:nodef)}_kernel_launch"%>(ptr, x);
<% end %>

    cumo_na_release_lock(v);
    return v;
}
