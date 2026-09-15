void cumo_<%=type_name%>_sort_kernel_launch(cumo_na_iarray_stridx_t* a, cumo_na_indexer_t* indexer, int64_t n_rows, int64_t row_len, int flat);

// One call sorts every row, so a loop over many short rows costs no more
// launches than one long row.
<% if is_float %>
// The keys put every NaN last, so nan:true asks for the order this already
// produces.
<% end %>
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

/*
  <%=name%> of self.
<% if is_float %>
  @overload <%=name%>(axis:nil, nan:false)
  @param [TrueClass] nan  A NaN sorts after every number whether this is true or false.
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
    ndf.func = <%=c_iter%>_kernel;
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
    // or rather than assign: cumo_na_reduce_dimension may have set
    // CUMO_NDF_KEEP_DIM by then, and assigning would drop it
    ndf.flag |= CUMO_NDF_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP;
    cumo_na_ndloop(&ndf, 2, self, reduce);
    return self;
}
