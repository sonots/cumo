<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, char *p2, CUMO_BIT_DIGIT *a3, size_t p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    CUMO_BIT_DIGIT *a3;
    size_t  p3;
    ssize_t s1, s2, s3;
    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR(lp, 0, p1, s1);
    CUMO_INIT_PTR(lp, 1, p2, s2);
    CUMO_INIT_PTR_BIT(lp, 2, a3, p3, s3);
    <% if type_name == 'robject' %>
    {
        dtype x, y;
        CUMO_BIT_DIGIT b;
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        for (; i--;) {
            CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
            CUMO_GET_DATA_STRIDE(p2,s2,dtype,y);
            b = (m_<%=name%>(x,y)) ? 1:0;
            CUMO_STORE_BIT(a3,p3,b);
            p3+=s3;
        }
    }
    <% else %>
    {
        <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,p2,a3,p3,s1,s2,s3,i);
    }
    <% end %>
}

static VALUE
<%=c_func%>_self(VALUE self, VALUE other)
{
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cumo_cBit,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP, 2, 1, ain, aout };

    return cumo_na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison <%=name%> other.
  @overload <%=op_map%> other
  @param [Cumo::NArray,Numeric] other
  @return [Cumo::Bit] result of self <%=name%> other.
*/
static VALUE
<%=c_func(1)%>(VALUE self, VALUE other)
{
    <% if is_object %>
    return <%=c_func%>_self(self, other);
    <% else %>
    VALUE klass, v;
    klass = cumo_na_upcast(rb_obj_class(self),rb_obj_class(other));
    if (klass==cT) {
        return <%=c_func%>_self(self, other);
    } else {
        v = rb_funcall(klass, cumo_id_cast, 1, self);
        return rb_funcall(v, <%=cumo_id_op%>, 1, other);
    }
    <% end %>
}
