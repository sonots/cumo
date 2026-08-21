<% unless is_object %>
void <%="cumo_#{c_func(:nodef)}_kernel_launch"%>(dtype *ptr, dtype x);
<% end %>

<% if !is_object && !is_bit %>
// The value stays a Ruby numeric until something reads the array, because
// filling one element on the device costs a whole kernel launch. An operand
// cast this way is usually consumed by the very next operator, which takes the
// value from here instead and never makes it reach the device at all.
static VALUE
<%=c_func(:nodef)%>_lazy(VALUE num)
{
    VALUE v;

    m_num_to_data(num); // raises for a value the element cannot hold, as filling it did
    v = cumo_na_new(cT, 0, NULL);
    rb_ivar_set(v, cumo_id_pending_scalar, num);
    return v;
}
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
