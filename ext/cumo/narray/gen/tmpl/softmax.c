void <%="cumo_#{c_iter}_kernel_launch"%>(char *px, char *py, uint64_t rows, uint64_t cols);

/*
  Softmax of self along its last axis. Each row is taken on its own and comes out
  summing to one: y = exp(x - max) / sum(exp(x - max)), where the maximum is of
  that row and is subtracted so that exp cannot overflow.
  @overload softmax
  Answers a new array: inplace! is not honoured.
  @return [Cumo::<%=class_name%>] returns the softmax, shaped like self.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    VALUE x=self, y, x_cont;
    cumo_narray_t *nx, *ny;
    char *x_ptr, *y_ptr;
    size_t cols, rows;

    CUMO_CHECK_NARRAY_TYPE(x, cT);
    x_cont = cumo_na_as_contiguous_array(x);

    CumoGetNArray(x_cont, nx);
    if (nx->ndim < 1) {
        rb_raise(cumo_na_eShapeError, "softmax needs an axis to run along");
    }
    y = cumo_na_new(cT, nx->ndim, nx->shape);

    // Taking a pointer runs allocate, which is Ruby, so both are taken before
    // anything is measured and the checks below run over what is left. See
    // row_method.h for why the pointers themselves have to be looked at again.
    x_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    y_ptr = cumo_na_get_offset_pointer_for_write(y);

    // cumo_na_as_contiguous_array answers what dup gave it, which a class is
    // free to define, so what it carries is not what was checked above.
    CUMO_CHECK_NARRAY_TYPE(x_cont, cT);
    CumoGetNArray(x_cont, nx);
    CumoGetNArray(y, ny);

    if (nx->ndim < 1) {
        rb_raise(cumo_na_eShapeError, "softmax needs an axis to run along");
    }
    CUMO_CHECK_SIZE_EQ(ny->size, nx->size);
    CUMO_ROW_CHECK_READ_BUFFER(x_cont, x_ptr, "self");
    CUMO_ROW_CHECK_WRITE_BUFFER(y, y_ptr, "the result");

    if (nx->size == 0) {
        return y;
    }
    cols = nx->shape[nx->ndim - 1];
    rows = nx->size / cols;

    <%="cumo_#{c_iter}_kernel_launch"%>(x_ptr, y_ptr, (uint64_t)rows, (uint64_t)cols);

    return y;
}
