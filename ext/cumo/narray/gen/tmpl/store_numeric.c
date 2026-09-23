static VALUE <%=type_name%>_fill(VALUE self, VALUE val);

// The value goes to the kernel as an argument. Making it a 0-dimensional
// array first cost a kernel to fill that array and an allocation per store.
static VALUE
<%=c_func(:nodef)%>(VALUE self, VALUE obj)
{
    <% if is_bit %>
    // A store reads any number as a bit, and fill takes only 0 and 1.
    obj = INT2FIX(m_num_to_data(obj));
    <% end %>
    return <%=type_name%>_fill(self, obj);
}
