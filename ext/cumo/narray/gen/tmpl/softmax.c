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
    cumo_row_args_t a;
    VALUE y;

    y = cumo_row_prepare(self, cT, "softmax", &a, 0);
    <%="cumo_#{c_iter}_kernel_launch"%>(a.x_ptr, a.y_ptr, (uint64_t)a.rows, (uint64_t)a.cols);

    return y;
}
