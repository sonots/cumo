void <%="cumo_#{c_iter}_kernel_launch"%>(char *px, char *pq, char *ps, uint64_t rows, uint64_t cols);

/*
  Quantizes self to 8-bit integers, one scale per row of the last axis.
  Each row is taken on its own: the scale is the largest magnitude in it over
  127, and the row is that scale's multiples, rounded to nearest with ties away
  from zero. A row of zeros answers a scale of zero and a row of zeros, which
  stands for the same values either way. A row holding an infinity or a NaN
  answers that in its scale and a row of zeros, since neither has a value 8 bits
  could stand for.
  @overload quantize_symmetric
  Answers new arrays: inplace! is not honoured.
  @return [Array] returns [quantized, scale], where quantized is a
    Cumo::Int8 shaped like self and scale is shaped like self without its last
    axis. The scale is the class the reduction accumulates in, which is
    Cumo::SFloat for the 16-bit classes and self's own otherwise.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    VALUE cont, q, s;
    cumo_narray_t *nx, *nq, *ns;
    char *px, *pq, *ps;
    size_t rows = 1, cols;
    int i;

    CUMO_CHECK_NARRAY_TYPE(self, cT);
    cont = cumo_na_as_contiguous_array(self);
    CumoGetNArray(cont, nx);
    if (nx->ndim < 1) {
        rb_raise(cumo_na_eShapeError, "quantize_symmetric needs an axis to run along");
    }

    /* Both allocations run Ruby, so every pointer is taken after the last of
     * them and the shape is measured again below. */
    /* The kernel reduces into the accumulator type, which is wider than the
     * element for the 16-bit classes, so the scale is that class and not cT. */
    q = cumo_na_new(cumo_cInt8, nx->ndim, nx->shape);
    s = nx->ndim == 1 ? cumo_na_new(<%=acc_class%>, 0, nx->shape)
                      : cumo_na_new(<%=acc_class%>, nx->ndim - 1, nx->shape);

    px = cumo_na_get_offset_pointer_for_read(cont);
    pq = cumo_na_get_offset_pointer_for_write(q);
    ps = cumo_na_get_offset_pointer_for_write(s);

    CUMO_CHECK_NARRAY_TYPE(cont, cT);
    CumoGetNArray(cont, nx);
    CumoGetNArray(q, nq);
    CumoGetNArray(s, ns);
    if (nx->ndim < 1) {
        rb_raise(cumo_na_eShapeError, "quantize_symmetric needs an axis to run along");
    }
    CUMO_ROW_CHECK_SAME_SHAPE(nq, nx);
    cols = nx->shape[nx->ndim - 1];
    for (i = 0; i < nx->ndim - 1; ++i) {
        rows *= nx->shape[i];
    }
    if (CUMO_NA_SIZE(ns) != rows) {
        rb_raise(cumo_na_eShapeError, "quantize_symmetric: the scale no longer fits its rows");
    }
    CUMO_ROW_CHECK_READ_BUFFER(cont, px, "self");
    CUMO_ROW_CHECK_WRITE_BUFFER(q, pq, "the quantized array");
    CUMO_ROW_CHECK_WRITE_BUFFER(s, ps, "the scale");

    /* A row with nothing in it is never reduced, so it has no scale of its own
     * to be written, and cumo_na_new does not clear what it hands out. */
    if (cols == 0 && rows > 0) {
        cumo_na_store(s, INT2FIX(0));
    }
    <%="cumo_#{c_iter}_kernel_launch"%>(px, pq, ps, (uint64_t)rows, (uint64_t)cols);
    RB_GC_GUARD(cont);

    return rb_assoc_new(q, s);
}
