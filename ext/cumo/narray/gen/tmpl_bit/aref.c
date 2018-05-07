/*
  Array element referenece or slice view.
  @overload [](dim0,...,dimL)
  @param [Numeric,Range,etc] dim0,...,dimL  Multi-dimensional Index.
  @return [Numeric,NArray::<%=class_name%>] Element object or NArray view.

  --- Returns the element at +dim0+, +dim1+, ... are Numeric indices
  for each dimension, or returns a NArray View as a sliced subarray if
  +dim0+, +dim1+, ... includes other than Numeric index, e.g., Range
  or Array or true.

  @example
      a = Cumo::DFloat.new(4,5).seq
      => Cumo::DFloat#shape=[4,5]
      [[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]]

      a[1,1]
      => 6.0

      a[1..3,1]
      => Cumo::DFloat#shape=[3]
      [6, 11, 16]

      a[1,[1,3,4]]
      => Cumo::DFloat#shape=[3]
      [6, 8, 9]

      a[true,2].fill(99)
      a
      => Cumo::DFloat#shape=[4,5]
      [[0, 1, 99, 3, 4],
       [5, 6, 99, 8, 9],
       [10, 11, 99, 13, 14],
       [15, 16, 99, 18, 19]]
 */
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    int nd;
    size_t pos;
    char *ptr;
    dtype x;

    nd = na_get_result_dimension(self, argc, argv, 1, &pos);
    if (nd) {
        return na_aref_main(argc, argv, self, 0, nd, pos);
    } else {
        // TODO(sonots): Return 0-dimensional narray rather than Synchronize()
        ptr = na_get_pointer_for_read(self);
        SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        LOAD_BIT(ptr,pos,x);
        return m_data_to_num(x);
    }
}
