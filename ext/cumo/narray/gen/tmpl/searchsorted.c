void cumo_<%=type_name%>_searchsorted_kernel_launch(const dtype *a, uint64_t n, const int64_t *sorter, const dtype *v, int64_t *out, uint64_t m, int right, int *oor);

/*
  The places of v in self for Cumo::NArray#searchsorted, which has cast self
  and v to this type and sorter to Int64 by the time it calls here.
*/
static VALUE
<%=c_func(3)%>(VALUE self, VALUE v, VALUE right, VALUE sorter)
{
    VALUE a_cont, v_cont, s_cont = Qnil, out;
    cumo_narray_t *na, *nv, *ns, *nout;
    char *a_ptr, *v_ptr, *s_ptr = NULL, *out_ptr;
    int *oor = NULL;
    int i;

    CUMO_CHECK_NARRAY_TYPE(self, cT, "self");
    CUMO_CHECK_NARRAY_TYPE(v, cT, "v");
    if (!NIL_P(sorter)) {
        CUMO_CHECK_NARRAY_TYPE(sorter, cumo_cInt64, "sorter");
    }

    a_cont = cumo_na_as_contiguous_array(self);
    v_cont = cumo_na_as_contiguous_array(v);
    if (!NIL_P(sorter)) {
        s_cont = cumo_na_as_contiguous_array(sorter);
    }
    CumoGetNArray(v_cont, nv);
    out = cumo_na_new(cumo_cInt64, nv->ndim, nv->shape);

    // Taking a pointer runs allocate, which is Ruby, so every one is taken
    // before anything is measured, and checked afterwards to be still the same.
    a_ptr = cumo_na_get_offset_pointer_for_read(a_cont);
    v_ptr = cumo_na_get_offset_pointer_for_read(v_cont);
    if (!NIL_P(s_cont)) {
        s_ptr = cumo_na_get_offset_pointer_for_read(s_cont);
    }
    out_ptr = cumo_na_get_offset_pointer_for_write(out);

    // cumo_na_as_contiguous_array answers what dup gave it, which a class is
    // free to define.
    CUMO_CHECK_NARRAY_TYPE(a_cont, cT, "self");
    CUMO_CHECK_NARRAY_TYPE(v_cont, cT, "v");
    if (!NIL_P(s_cont)) {
        CUMO_CHECK_NARRAY_TYPE(s_cont, cumo_cInt64, "sorter");
    }
    CumoGetNArray(a_cont, na);
    CumoGetNArray(v_cont, nv);
    CumoGetNArray(out, nout);
    if (na->ndim != 1) {
        rb_raise(cumo_na_eShapeError, "searchsorted needs a 1-dimensional array");
    }
    if (nout->ndim != nv->ndim) {
        rb_raise(cumo_na_eShapeError, "v or the result was reshaped while it was measured");
    }
    for (i = 0; i < nv->ndim; ++i) {
        if (nout->shape[i] != nv->shape[i]) {
            rb_raise(cumo_na_eShapeError, "v or the result was reshaped while it was measured");
        }
    }
    if (!NIL_P(s_cont)) {
        CumoGetNArray(s_cont, ns);
        if (ns->ndim != 1 || ns->size != na->size) {
            rb_raise(cumo_na_eShapeError, "sorter must be 1-dimensional and %"SZF"u long", na->size);
        }
        CUMO_ROW_CHECK_READ_BUFFER(s_cont, s_ptr, "sorter");
    }
    CUMO_ROW_CHECK_READ_BUFFER(a_cont, a_ptr, "self");
    CUMO_ROW_CHECK_READ_BUFFER(v_cont, v_ptr, "v");
    CUMO_ROW_CHECK_WRITE_BUFFER(out, out_ptr, "the result");

    if (nv->size > 0) {
        if (s_ptr != NULL) {
            oor = cumo_cuda_runtime_error_flag_new();
        }
        cumo_<%=type_name%>_searchsorted_kernel_launch(
            (const dtype*)a_ptr, na->size, (const int64_t*)s_ptr, (const dtype*)v_ptr,
            (int64_t*)out_ptr, nv->size, RTEST(right), oor);
        if (oor != NULL) {
            CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("searchsorted", "<%=type_name%>");
            if (cumo_cuda_runtime_error_flag_get(oor)) {
                rb_raise(rb_eIndexError, "sorter has an index out of range");
            }
        }
    }
    RB_GC_GUARD(a_cont);
    RB_GC_GUARD(v_cont);
    RB_GC_GUARD(s_cont);
    return out;
}
