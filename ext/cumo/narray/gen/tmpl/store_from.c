//<% unless c_iter.include? 'robject' %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer);
//<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    //<% if c_iter.include? 'robject' %>
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    {
        <%=dtype%> x;
        dtype y;
        if (idx2) {
            if (idx1) {
                for (; i--;) {
                    CUMO_GET_DATA_INDEX(p2,idx2,<%=dtype%>,x);
                    y = <%=macro%>(x);
                    CUMO_SET_DATA_INDEX(p1,idx1,dtype,y);
                }
            } else {
                for (; i--;) {
                    CUMO_GET_DATA_INDEX(p2,idx2,<%=dtype%>,x);
                    y = <%=macro%>(x);
                    CUMO_SET_DATA_STRIDE(p1,s1,dtype,y);
                }
            }
        } else {
            if (idx1) {
                for (; i--;) {
                    CUMO_GET_DATA_STRIDE(p2,s2,<%=dtype%>,x);
                    y = <%=macro%>(x);
                    CUMO_SET_DATA_INDEX(p1,idx1,dtype,y);
                }
            } else {
                for (; i--;) {
                    CUMO_GET_DATA_STRIDE(p2,s2,<%=dtype%>,x);
                    y = <%=macro%>(x);
                    CUMO_SET_DATA_STRIDE(p1,s1,dtype,y);
                }
            }
        }
    }
    //<% else %>
    {
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&indexer);
    }
    //<% end %>
}


static VALUE
<%=c_func(:nodef)%>(VALUE self, VALUE obj)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{Qnil,0}};
    //<% if c_iter.include? 'robject' %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 2, 0, ain, 0 };
    <% else %>
    // Without INDEXER_LOOP ndloop walks the outer dimensions itself and the
    // kernel runs once per row, which for a column slice costs one launch per
    // row and nothing else. An index on either side is buffered into
    // contiguous memory first, in one launch of its own.
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP, 2, 0, ain, 0 };
    <% end %>

    cumo_na_ndloop(&ndf, 2, self, obj);
    return self;
}
