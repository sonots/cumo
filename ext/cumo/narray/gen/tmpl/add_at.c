void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_stridx_t* a1, cumo_na_iarray_stridx_t* a2, cumo_na_indexer_t* indexer);

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);
    cumo_na_iarray_stridx_t a1 = cumo_na_make_iarray_stridx(&lp->args[0]);
    cumo_na_iarray_stridx_t a2 = cumo_na_make_iarray_stridx(&lp->args[1]);

    <%="cumo_#{c_iter}_kernel_launch"%>(&a1, &a2, &indexer);
}

static VALUE
<%=c_func%>_root(VALUE a)
{
    cumo_narray_t *na;

    CumoGetNArray(a, na);
    return (CUMO_NA_TYPE(na) == CUMO_NARRAY_VIEW_T) ? CUMO_NA_VIEW_DATA(na) : a;
}

// numpy refuses values that would have to grow the selection to fit, where
// ndloop would broadcast the selection instead and add each row of them.
static void
<%=c_func%>_check_shape(VALUE selected, VALUE values)
{
    cumo_narray_t *nw, *nv;
    int i;

    CumoGetNArray(selected, nw);
    CumoGetNArray(values, nv);
    for (i = 1; i <= nv->ndim; ++i) {
        if (i > nw->ndim ||
            (nv->shape[nv->ndim - i] != 1 && nv->shape[nv->ndim - i] != nw->shape[nw->ndim - i])) {
            rb_raise(cumo_na_eShapeError, "values do not broadcast to the shape of the places selected");
        }
    }
}

/*
  Adds values at the places the subscripts select, in place, as numpy's
  add.at does: a place selected more than once gets every value added to it,
  where a[index] += values would add only one of them.
  The subscripts are those of []=, so a single subscript of a
  multi-dimensional array counts its elements in order, and a row is
  a.add_at(i, true, values). An index array of more than one dimension
  selects a place for each of its elements, and the values broadcast to its
  shape.
  @overload add_at(*index, values)
    @param [Numeric,Array,Range,Cumo::NArray] index  subscripts, as for []=.
    @param [Numeric,Array,Cumo::NArray] values  what to add, cast to
      Cumo::<%=class_name%> and broadcast to the places selected.
    @return [Cumo::<%=class_name%>] self.
  @example
    a = Cumo::<%=class_name%>.zeros(5)
    a.add_at(Cumo::Int32[1, 3, 1], 1)
    # => Cumo::<%=class_name%>#shape=[5]
    #    [0, 2, 0, 1, 0]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{cT,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 2, 0, ain, 0 };
    VALUE view = self, values, sub, idx;
    int nd;
    size_t pos;

    rb_check_arity(argc, 1, UNLIMITED_ARGUMENTS);
    argc--;
    values = argv[argc];
    // As in []=, the view below has the root for its data object, so the frozen
    // check of the pointer it takes never reaches self.
    if (OBJ_FROZEN(self)) {
        rb_raise(rb_eRuntimeError, "cannot write to frozen NArray.");
    }
    if (!rb_obj_is_kind_of(values, cT)) {
        values = rb_funcall(cT, cumo_id_cast, 1, values);
    } else if (<%=c_func%>_root(values) == <%=c_func%>_root(self)) {
        // The kernel reads the values while it adds into self, so values held
        // in self's own memory are read from a copy.
        values = cumo_na_copy(values);
    }
    if (argc == 1) {
        sub = argv[0];
        idx = RB_TYPE_P(sub, T_ARRAY) ? rb_apply(cumo_cNArray, rb_intern("[]"), sub) : sub;
        // An index array of more than one dimension selects into a new array
        // rather than a view, so it goes in flattened, and the values with it.
        if (CumoIsNArray(idx) && !rb_obj_is_kind_of(idx, cumo_cBit) && CUMO_RNARRAY_NDIM(idx) > 1) {
            <%=c_func%>_check_shape(idx, values);
            if (CUMO_RNARRAY_NDIM(values) > 0) {
                VALUE spread = cumo_na_new(cT, CUMO_RNARRAY_NDIM(idx), CUMO_RNARRAY_SHAPE(idx));
                <%=c_func.sub(/_add_at/, "_store")%>(spread, values);
                values = cumo_na_flatten(spread);
            }
            sub = cumo_na_flatten(idx);
        }
        nd = cumo_na_get_result_dimension(self, 1, &sub, sizeof(dtype), &pos);
        view = cumo_na_aref_main(1, &sub, self, 0, nd, pos);
    } else if (argc > 1) {
        nd = cumo_na_get_result_dimension(self, argc, argv, sizeof(dtype), &pos);
        view = cumo_na_aref_main(argc, argv, self, 0, nd, pos);
    }
    <%=c_func%>_check_shape(view, values);
    cumo_na_ndloop(&ndf, 2, view, values);
    return self;
}
