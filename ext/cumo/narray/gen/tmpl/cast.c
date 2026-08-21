<% children.each do |c|%>
<%= c.result %>

<% end %>
/*
  Cast object to Cumo::<%=class_name%>.
  @overload [](elements)
  @overload <%=name%>(array)
  @param [Numeric,Array] elements
  @param [Array] array
  @return [Cumo::<%=class_name%>]
*/
static VALUE
<%=c_func(1)%>(VALUE type, VALUE obj)
{
    VALUE v;
    cumo_narray_t *na;
    //<% if is_object || is_bit %>
    dtype x;
    //<% end %>

    if (rb_obj_class(obj)==cT) {
        return obj;
    }
    if (RTEST(rb_obj_is_kind_of(obj,rb_cNumeric))) {
        //<% if is_object || is_bit %>
        x = m_num_to_data(obj);
        return <%=type_name%>_new_dim0(x);
        //<% else %>
        return <%=type_name%>_new_dim0_lazy(obj);
        //<% end %>
    }
    if (RTEST(rb_obj_is_kind_of(obj,rb_cArray))) {
        return <%=find_tmpl("cast_array").c_func%>(obj);
    }
    if (CumoIsNArray(obj)) {
        CumoGetNArray(obj,na);
        v = cumo_na_new(cT, CUMO_NA_NDIM(na), CUMO_NA_SHAPE(na));
        if (CUMO_NA_SIZE(na) > 0) {
            <%=find_tmpl("store").c_func%>(v,obj);
        }
        return v;
    }
    if (rb_respond_to(obj,cumo_id_to_a)) {
        obj = rb_funcall(obj,cumo_id_to_a,0);
        if (TYPE(obj)!=T_ARRAY) {
            rb_raise(rb_eTypeError, "`to_a' did not return Array");
        }
        return <%=find_tmpl("cast_array").c_func%>(obj);
    }
    <% if is_object %>
    return robject_new_dim0(obj);
    <% else %>
    rb_raise(cumo_na_eCastError,"cannot cast to %s",rb_class2name(type));
    return Qnil;
    <% end %>
}
