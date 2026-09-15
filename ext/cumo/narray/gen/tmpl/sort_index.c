void cumo_<%=type_name%>_sort_index_kernel_launch(cumo_na_iarray_t* a, cumo_na_indexer_t* indexer,
        cumo_na_iarray_t* idx, cumo_na_iarray_t* out, int64_t n_rows, int64_t row_len, int flat, int idx_bytes);

// args[1] holds the answer for each position, so the sort only has to say
// which position each rank came from.
<% if is_float %>
// The keys put every NaN last, so nan:true asks for the order this already
// produces.
<% end %>
static void
<%=c_iter%>_kernel(cumo_na_loop_t *const lp)
{
    cumo_na_iarray_t a = cumo_na_make_iarray(&lp->args[0]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);
    cumo_na_iarray_t idx = cumo_na_make_iarray_given_ndim(&lp->args[1], indexer.ndim);
    cumo_na_iarray_t out = cumo_na_make_iarray_given_ndim(&lp->args[2], indexer.ndim);
    int64_t row_len = 1;
    int64_t n_rows;
    ssize_t expect = sizeof(dtype);
    int i, flat = 1;

    for (i = indexer.ndim - lp->reduce_dim; i < indexer.ndim; ++i) {
        row_len *= (int64_t)indexer.shape[i];
    }
    n_rows = row_len > 0 ? (int64_t)indexer.total_size / row_len : 0;

    for (i = indexer.ndim; --i >= 0;) {
        if (a.step[i] != expect) {
            flat = 0;
            break;
        }
        expect *= (ssize_t)indexer.shape[i];
    }

    cumo_<%=type_name%>_sort_index_kernel_launch(&a, &indexer, &idx, &out, n_rows, row_len, flat,
                                                 (int)lp->args[2].elmsz);
}

/*
  <%=name%>. Returns an index array of sort result.
<% if is_float %>
  @overload <%=name%>(axis:nil, nan:false)
  @param [TrueClass] nan  A NaN sorts after every number whether this is true or false.
<% else %>
  @overload <%=name%>(axis:nil)
<% end %>
  @param [Numeric,Array,Range] axis  Performs <%=name%> along the axis.
  @return [Integer,Cumo::Int] returns result index of <%=name%>.
  @example
      Cumo::NArray[3,4,1,2].sort_index # => Cumo::Int32[2,3,0,1]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    cumo_narray_t *na;
    VALUE idx, reduce;
    cumo_ndfunc_arg_in_t ain[3] = {{cT,0},{0,0},{cumo_sym_reduce,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{0,0,0}};
    cumo_ndfunc_t ndf = {0, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_CUM, 3,1, ain,aout};

    CumoGetNArray(self,na);
    if (na->ndim==0) {
        return INT2FIX(0);
    }
    if (na->size > (~(u_int32_t)0)) {
        ain[1].type =
        aout[0].type = cumo_cInt64;
        idx = cumo_na_new(cumo_cInt64, na->ndim, na->shape);
    } else {
        ain[1].type =
        aout[0].type = cumo_cInt32;
        idx = cumo_na_new(cumo_cInt32, na->ndim, na->shape);
    }
    ndf.func = <%=c_iter%>_kernel;
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
    rb_funcall(idx, rb_intern("seq"), 0);

    ndf.flag |= CUMO_NDF_INDEXER_LOOP;
    // The index the kernel writes counts along the operand's memory, so it
    // only means the same thing as the logical index when the operand is
    // contiguous.
    if (cumo_na_check_contiguous(self) != Qtrue) {
        self = cumo_na_copy(self);
    }
    return cumo_na_ndloop3(&ndf, 0, 3, self, idx, reduce);
}
