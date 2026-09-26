<% unless type_name == 'robject' %>
<% (is_float ? ["","_nan"] : [""]).each do |j| %>
cudaError_t <%="cumo_#{type_name}_#{name}#{j}_batched_kernel_launch"%>(
        cumo_na_iarray_stridx_t* a_in, cumo_na_iarray_stridx_t* a_out,
        cumo_na_indexer_t* indexer, uint64_t row_len, ssize_t step_in, ssize_t step_out);
<% end %>
<% end %>

<% unless type_name == 'robject' %>
// How far the scan moves for one step along the last axis, in elements, or 0
// where one number cannot say it. A reversed view answers a negative step and
// a flat one answers 1; an index or a transpose answers 0 and is gathered.
// An axis of extent one is never indexed, so whatever stride it carries is
// not reached.
static ssize_t
<%=c_iter%>_walk_step(cumo_na_iarray_stridx_t *a, cumo_na_indexer_t *indexer)
{
    ssize_t step, want;
    int k, last = indexer->ndim - 1;

    if (indexer->ndim == 0) { return 1; }
    if (!CUMO_SDX_IS_STRIDE(a->stridx[last])) { return 0; }
    step = CUMO_SDX_GET_STRIDE(a->stridx[last]);
    if (step == 0 || step % (ssize_t)sizeof(dtype) != 0) { return 0; }
    step /= (ssize_t)sizeof(dtype);

    want = step;
    for (k = last; --k >= 0;) {
        want *= (ssize_t)indexer->shape[k + 1];
        if (indexer->shape[k] == 1) { continue; }
        if (!CUMO_SDX_IS_STRIDE(a->stridx[k])) { return 0; }
        if (CUMO_SDX_GET_STRIDE(a->stridx[k]) != want * (ssize_t)sizeof(dtype)) { return 0; }
    }
    return step;
}
<% end %>

<% (is_float ? ["","_nan"] : [""]).each do |j| %>
static void
<%=c_iter%><%=j%>(cumo_na_loop_t *const lp)
{
  <% unless type_name == 'robject' %>
    // Every row goes in one call, so a scan down a short axis costs no more
    // launches than a scan down a long one. The rows the scan must not run
    // across are the ones the reduction names.
    cumo_na_iarray_stridx_t a_in = cumo_na_make_iarray_stridx(&lp->args[0]);
    cumo_na_iarray_stridx_t a_out = cumo_na_make_iarray_stridx(&lp->args[1]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);
    uint64_t row_len = 1;
    int k;
    ssize_t step_in, step_out;

    for (k = indexer.ndim - lp->reduce_dim; k < indexer.ndim; ++k) {
        row_len *= (uint64_t)indexer.shape[k];
    }
    // The scan reads base[i * step], so an operand that walks one stride needs
    // no buffer whichever way it walks. Anything else is copied into one that
    // does.
    step_in = <%=c_iter%>_walk_step(&a_in, &indexer);
    step_out = <%=c_iter%>_walk_step(&a_out, &indexer);

    cumo_cuda_runtime_check_status(<%="cumo_#{type_name}_#{name}#{j}_batched_kernel_launch"%>(
            &a_in, &a_out, &indexer, row_len, step_in, step_out));
  <% else %>
    size_t   i;
    char    *p1, *p2;
    ssize_t  s1, s2;
    dtype    x, y;

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR(lp, 0, p1, s1);
    CUMO_INIT_PTR(lp, 1, p2, s2);

    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%><%=j%>", "<%=type_name%>");
    cumo_cuda_runtime_device_synchronize();

    CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
    CUMO_SET_DATA_STRIDE(p2,s2,dtype,x);
    for (i--; i--;) {
        CUMO_GET_DATA_STRIDE(p1,s1,dtype,y);
        m_<%=name%><%=j%>(x,y);
        CUMO_SET_DATA_STRIDE(p2,s2,dtype,x);
    }
  <% end %>
}
<% end %>

/*
  <%=name%> of self.
  @overload <%=name%>(axis:nil, nan:false)
  @param [Numeric,Array,Range] axis  Performs <%=name%> along the axis.
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN if exists).
  @return [Cumo::<%=class_name%>] <%=name%> of self.
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cumo_sym_reduce,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_CUM,
                     2, 1, ain, aout };
  <% unless type_name == 'robject' %>
    // The whole array reaches the iterator at once, which is what lets the scan
    // take every row in one call.
    ndf.flag = CUMO_NDF_HAS_LOOP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_CUM;
  <% end %>

  <% if is_float %>
    reduce = cumo_na_parse_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_nan, 1, 3);
  <% else %>
    reduce = cumo_na_parse_reduce_dimension(argc, argv, 1, &self, &ndf, 0, 1, 3);
  <% end %>
  <% unless type_name == 'robject' %>
    // or rather than assign: cumo_na_reduce_dimension may have set
    // CUMO_NDF_KEEP_DIM by then, and assigning would drop it
    ndf.flag |= CUMO_NDF_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP;
  <% end %>
    return cumo_na_ndloop(&ndf, 2, self, reduce);
}
