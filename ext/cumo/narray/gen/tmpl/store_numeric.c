static VALUE <%=type_name%>_fill(VALUE self, VALUE val);

static VALUE
<%=c_func(:nodef)%>(VALUE self, VALUE obj)
{
    dtype x = m_num_to_data(obj);
    <% if is_bit %>
    obj = INT2FIX(x);
    <% else %>
    (void)x;
    <% end %>
    return <%=type_name%>_fill(self, obj);
}
