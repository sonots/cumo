/*
  Multi-dimensional array indexing.
  Same as [] for one-dimensional NArray.
  Similar to numpy's tuple indexing, i.e., `a[[1,2,..],[3,4,..]]`
  @overload at(*indices)
  @param [Numeric,Range,etc] *indices  Multi-dimensional Index Arrays.
  @return [Cumo::NArray::<%=class_name%>] one-dimensional NArray view.

  @example
      x = Cumo::DFloat.new(3,3,3).seq
      => Cumo::DFloat#shape=[3,3,3]
       [[[0, 1, 2],
         [3, 4, 5],
         [6, 7, 8]],
        [[9, 10, 11],
         [12, 13, 14],
         [15, 16, 17]],
        [[18, 19, 20],
         [21, 22, 23],
         [24, 25, 26]]]

      x.at([0,1,2],[0,1,2],[-1,-2,-3])
      => Cumo::DFloat(view)#shape=[3]
       [2, 13, 24]
 */
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    int result_nd;
    size_t pos;

    result_nd = cumo_na_get_result_dimension(self, argc, argv, sizeof(dtype), &pos);
    return cumo_na_at_main(argc, argv, self, 0, result_nd, pos);
}
