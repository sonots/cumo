/*
  Element-wise <%=name%> of two arrays.

<% if is_float %>
  @overload <%=name%>(a1, a2, nan:false)
  @param [Cumo::NArray,Numeric] a1  The array to be compared.
  @param [Cumo::NArray,Numeric] a2  The array to be compared.
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (return NaN if exist).
<% else %>
  @overload <%=name%>(a1, a2)
  @param [Cumo::NArray,Numeric] a1,a2  The arrays holding the elements to be compared.
<% end %>
  @return [Cumo::<%=class_name%>]
*/

<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

<% unless type_name == 'robject' %>
void cumo_<%=type_name%>_<%=name%><%=nan%>_kernel_launch(char *p1, char* p2, char* p3, ssize_t s1, ssize_t s2, ssize_t s3, size_t n);
<% end %>

static void
<%=c_iter%><%=nan%>(cumo_na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    <% if type_name == 'robject' %>
    {
        size_t i;
        SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%><%=nan%>", "<%=type_name%>");
        for (i=0; i<n; i++) {
            dtype x, y, z;
            GET_DATA_STRIDE(p1,s1,dtype,x);
            GET_DATA_STRIDE(p2,s2,dtype,y);
            GET_DATA(p3,dtype,z);
            z = f_<%=name%><%=nan%>(x,y);
            SET_DATA_STRIDE(p3,s3,dtype,z);
        }
    }
    <% else %>
    {
        cumo_<%=type_name%>_<%=name%><%=nan%>_kernel_launch(p1,p2,p3,s1,s2,s3,n);
    }
    <% end %>
}
<% end %>

static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE mod)
{
    VALUE a1 = Qnil;
    VALUE a2 = Qnil;
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, STRIDE_LOOP_NIP, 2, 1, ain, aout };

    <% if is_float %>
    VALUE kw_hash = Qnil;
    ID kw_table[1] = {cumo_id_nan};
    VALUE opts[1] = {Qundef};

    rb_scan_args(argc, argv, "20:", &a1, &a2, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 1, opts);
    if (opts[0] != Qundef) {
        ndf.func = <%=c_iter%>_nan;
    }
    <% else %>
    rb_scan_args(argc, argv, "20", &a1, &a2);
    <% end %>

    return cumo_na_ndloop(&ndf, 2, a1, a2);
}
