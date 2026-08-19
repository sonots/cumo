<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_indexer_t* indexer, dtype val);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    VALUE x = lp->option;
    dtype y = m_num_to_data(x);
    <% if type_name == 'robject' %>
    {
        size_t   i;
        char    *p1;
        ssize_t  s1;
        size_t  *idx1;

        CUMO_INIT_COUNTER(lp, i);
        CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        if (idx1) {
            for (; i--;) {
                CUMO_SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                CUMO_SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
    <% else %>
    {
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&indexer,y);
    }
    <% end %>
}

/*
  Fill elements with other.
  @overload <%=name%> other
  @param [Numeric] other
  @return [Cumo::<%=class_name%>] self.
*/
static VALUE
<%=c_func(1)%>(VALUE self, VALUE val)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{cumo_sym_option}};
    <% if type_name == 'robject' %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 2, 0, ain, 0 };
    <% else %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP, 2, 0, ain, 0 };
    <% end %>

    cumo_na_ndloop(&ndf, 2, self, val);
    return self;
}
