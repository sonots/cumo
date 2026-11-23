<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

<% unless type_name == 'robject' %>
void cumo_<%=type_name%>_<%=name%><%=nan%>_kernel_launch(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer);
<% end %>

static void
<%=c_iter%><%=nan%>(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    {
        size_t   i, n;
        char    *p1, *p2, *p3;
        ssize_t  s1, s2, s3;

        CUMO_INIT_COUNTER(lp, n);
        CUMO_INIT_PTR(lp, 0, p1, s1);
        CUMO_INIT_PTR(lp, 1, p2, s2);
        CUMO_INIT_PTR(lp, 2, p3, s3);

        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%><%=nan%>", "<%=type_name%>");
        for (i=0; i<n; i++) {
            dtype x, y, z;
            CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
            CUMO_GET_DATA_STRIDE(p2,s2,dtype,y);
            CUMO_GET_DATA(p3,dtype,z);
            z = f_<%=name%><%=nan%>(x,y);
            CUMO_SET_DATA_STRIDE(p3,s3,dtype,z);
        }
    }
    <% else %>
    {
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);
        cumo_na_iarray_t a3 = cumo_na_make_iarray(&lp->args[2]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        cumo_<%=type_name%>_<%=name%><%=nan%>_kernel_launch(&a1,&a2,&a3,&indexer);
    }
    <% end %>
}
<% end %>

/*
  Element-wise <%=name%> of two arrays, cast to <%=class_name%>.
<% if is_float && !is_object %>
  Where one of the two is NaN, it returns the other, unless nan: is given.

  @overload <%=name%>(a1, a2)
    @param [Cumo::NArray,Numeric,Array] a1  The array to be compared.
    @param [Cumo::NArray,Numeric,Array] a2  The array to be compared.
    @return [Cumo::<%=class_name%>]
  @overload <%=name%>(a1, a2, nan:)
    @param [Cumo::NArray,Numeric,Array] a1  The array to be compared.
    @param [Cumo::NArray,Numeric,Array] a2  The array to be compared.
    @param [Boolean] nan  Given at all, true or false, it returns NaN where either of the two is NaN.
    @note nan: false does the same as nan: true, unlike the nan: of max.
<% elsif is_object %>

  @overload <%=name%>(a1, a2, nan:false)
    @param [Cumo::NArray,Numeric,Array] a1  The array to be compared.
    @param [Cumo::NArray,Numeric,Array] a2  The array to be compared.
    @param [Boolean] nan  Accepted, and does not change the result.
<% else %>

  @overload <%=name%>(a1, a2)
    @param [Cumo::NArray,Numeric,Array] a1  The array to be compared.
    @param [Cumo::NArray,Numeric,Array] a2  The array to be compared.
<% end %>
    @return [Cumo::<%=class_name%>]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE mod)
{
    VALUE a1 = Qnil;
    VALUE a2 = Qnil;
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    <% if type_name == 'robject' %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP_NIP, 2, 1, ain, aout };
    <% else %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 2, 1, ain, aout };
    <% end %>

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
