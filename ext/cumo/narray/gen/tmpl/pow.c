<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n);
void <%="cumo_#{c_iter}_int32_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  i;
    char    *p1, *p2, *p3;
    ssize_t s1, s2, s3;
    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR(lp, 0, p1, s1);
    CUMO_INIT_PTR(lp, 1, p2, s2);
    CUMO_INIT_PTR(lp, 2, p3, s3);
    <% if type_name == 'robject' %>
    {
        dtype x, y;
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        for (; i--;) {
            CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
            CUMO_GET_DATA_STRIDE(p2,s2,dtype,y);
            x = m_pow(x,y);
            CUMO_SET_DATA_STRIDE(p3,s3,dtype,x);
        }
    }
    <% else %>
    <%="cumo_#{c_iter}_kernel_launch"%>(p1,p2,p3,s1,s2,s3,i);
    <% end %>
}

static void
<%=c_iter%>_int32(cumo_na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2, *p3;
    ssize_t s1, s2, s3;
    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR(lp, 0, p1, s1);
    CUMO_INIT_PTR(lp, 1, p2, s2);
    CUMO_INIT_PTR(lp, 2, p3, s3);
    <% if type_name == 'robject' %>
    {
        dtype   x;
        int32_t y;
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>_int32", "<%=type_name%>");
        for (; i--;) {
            CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
            CUMO_GET_DATA_STRIDE(p2,s2,int32_t,y);
            x = m_pow_int(x,y);
            CUMO_SET_DATA_STRIDE(p3,s3,dtype,x);
        }
    }
    <% else %>
    <%="cumo_#{c_iter}_int32_kernel_launch"%>(p1,p2,p3,s1,s2,s3,i);
    <% end %>
}

static VALUE
<%=c_func%>_self(VALUE self, VALUE other)
{
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    cumo_ndfunc_arg_in_t ain_i[2] = {{cT,0},{cumo_cInt32,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP, 2, 1, ain, aout };
    cumo_ndfunc_t ndf_i = { <%=c_iter%>_int32, CUMO_STRIDE_LOOP, 2, 1, ain_i, aout };

    // fixme : use na.integer?
    if (FIXNUM_P(other) || rb_obj_is_kind_of(other,cumo_cInt32)) {
        return cumo_na_ndloop(&ndf_i, 2, self, other);
    } else {
        return cumo_na_ndloop(&ndf, 2, self, other);
    }
}

/*
  Binary power.
  @overload <%=op_map%> other
  @param [Cumo::NArray,Numeric] other
  @return [Cumo::NArray] self to the other-th power.
*/
static VALUE
<%=c_func(1)%>(VALUE self, VALUE other)
{
    <% if is_object %>
    return <%=c_func%>_self(self,other);
    <% else %>
    VALUE klass, v;
    klass = cumo_na_upcast(rb_obj_class(self),rb_obj_class(other));
    if (klass==cT) {
        return <%=c_func%>_self(self,other);
    } else {
        v = rb_funcall(klass, cumo_id_cast, 1, self);
        return rb_funcall(v, cumo_id_pow, 1, other);
    }
    <% end %>
}
