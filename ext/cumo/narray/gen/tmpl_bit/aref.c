static VALUE
<%=c_func(-1)%>_cpu(int argc, VALUE *argv, VALUE self);

/*
  Array indexing.
  @overload [](dim0,...,dimL)
  @param [Numeric,Range,Array,Cumo::Bit,Cumo::Int32,Cumo::Int64] dim0,...,dimL  Multi-dimensional Index.
  @return [Cumo::Bit,Numeric] Element value or NArray view.

  --- Returns an element at `dim0`, `dim1`, ... are Numeric indices for each dimension, or returns a NArray View as a sliced array if `dim0`, `dim1`, ... includes other than Numeric index, e.g., Range or Array or true.

  @see #[]=

  @example
      a = Cumo::Int32.new(3,4).seq
      # => Cumo::Int32#shape=[3,4]
      # [[0, 1, 2, 3],
      #  [4, 5, 6, 7],
      #  [8, 9, 10, 11]]

      b = (a%2).eq(0)
      # => Cumo::Bit#shape=[3,4]
      # [[1, 0, 1, 0],
      #  [1, 0, 1, 0],
      #  [1, 0, 1, 0]]

      b[true,(0..-1)%2]
      # => Cumo::Bit(view)#shape=[3,2]
      # [[1, 1],
      #  [1, 1],
      #  [1, 1]]

      b[1,1]
      # => 0
 */
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    if (cumo_compatible_mode_enabled_p()) {
        return <%=c_func(-1)%>_cpu(argc, argv, self);
    } else {
        int result_nd;
        size_t pos;

        result_nd = cumo_na_get_result_dimension(self, argc, argv, 1, &pos);
        return cumo_na_aref_main(argc, argv, self, 0, result_nd, pos);
    }
}
