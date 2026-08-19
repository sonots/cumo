void cumo_<%=type_name%>_sort_kernel_launch(cumo_na_iarray_stridx_t* a, cumo_na_indexer_t* indexer, int64_t n_rows, int64_t row_len, int flat);

// One call sorts every row, so a loop over many short rows costs no more
// launches than one long row. nan:true keeps the host path: its comparator
// leaves NaN unordered, and what numo answers there is not a sorted array.
static void
<%=c_iter%>_kernel(cumo_na_loop_t *const lp)
{
    cumo_na_iarray_stridx_t a = cumo_na_make_iarray_stridx(&lp->args[0]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);
    int64_t row_len = 1;
    int64_t n_rows;
    ssize_t expect = sizeof(dtype);
    int i, flat = 1;

    for (i = indexer.ndim - lp->reduce_dim; i < indexer.ndim; ++i) {
        row_len *= (int64_t)indexer.shape[i];
    }
    n_rows = row_len > 0 ? (int64_t)indexer.total_size / row_len : 0;

    // The rows have to be laid out end to end for a segmented sort to address
    // them; anything else is gathered into a buffer of its own first.
    for (i = indexer.ndim; --i >= 0;) {
        if (!CUMO_SDX_IS_STRIDE(a.stridx[i]) || CUMO_SDX_GET_STRIDE(a.stridx[i]) != expect) {
            flat = 0;
            break;
        }
        expect *= (ssize_t)indexer.shape[i];
    }

    cumo_<%=type_name%>_sort_kernel_launch(&a, &indexer, n_rows, row_len, flat);
}

<% (is_float ? ["_ignan","_prnan"] : [""]).each do |j| %>
static void
<%=c_iter%><%=j%>(cumo_na_loop_t *const lp)
{
    size_t n;
    char *ptr;
    ssize_t step;

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR(lp, 0, ptr, step);
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    <%=type_name%>_qsort<%=j%>(ptr, n, step);
}
<% end %>

/*
  <%=name%> of self.
<% if is_float %>
  @overload <%=name%>(axis:nil, nan:false)
  @param [TrueClass] nan  If true, propagete NaN. If false, ignore NaN.
<% else %>
  @overload <%=name%>(axis:nil)
<% end %>
  @param [Numeric,Array,Range] axis  Performs <%=name%> along the axis.
  @return [Cumo::<%=class_name%>] returns result of <%=name%>.
  @example
      Cumo::DFloat[3,4,1,2].sort # => Cumo::DFloat[1,2,3,4]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{cumo_sym_reduce,0}};
    cumo_ndfunc_t ndf = {0, CUMO_NDF_HAS_LOOP|CUMO_NDF_FLAT_REDUCE, 2,0, ain,0};

    if (!CUMO_TEST_INPLACE(self)) {
        self = cumo_na_copy(self);
    }
  <% if is_float %>
    ndf.func = <%=c_iter%>_ignan;
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_prnan);
    if (ndf.func == <%=c_iter%>_ignan) {
        ndf.func = <%=c_iter%>_kernel;
        // or rather than assign: cumo_na_reduce_dimension may have set
        // CUMO_NDF_KEEP_DIM by then, and assigning would drop it
        ndf.flag |= CUMO_NDF_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP;
    }
  <% else %>
    ndf.func = <%=c_iter%>;
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
    ndf.func = <%=c_iter%>_kernel;
    ndf.flag |= CUMO_NDF_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP;
  <% end %>
    cumo_na_ndloop(&ndf, 2, self, reduce);
    return self;
}
