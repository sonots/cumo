/*
  Array element reference or slice view.
  @overload aref_cpu(*indices)
    @param [Numeric,Range,Array,true,false,Cumo::NArray] indices  Multi-dimensional Index.
    @return [Integer,Cumo::Bit] Element value or NArray view.

  --- Returns the element at +dim0+, +dim1+, ... are Numeric indices
  for each dimension, or returns a NArray View as a sliced subarray if
  +dim0+, +dim1+, ... includes other than Numeric index, e.g., Range
  or Array or true.

  @example
      a = Cumo::Int32.new(3,4).seq
      b = (a%2).eq(0)
      # => Cumo::Bit#shape=[3,4]
      # [[1, 0, 1, 0],
      #  [1, 0, 1, 0],
      #  [1, 0, 1, 0]]

      b.aref_cpu(1,1)
      # => 0

      b.aref_cpu(true,(0..-1)%2)
      # => Cumo::Bit(view)#shape=[3,2]
      # [[1, 1],
      #  [1, 1],
      #  [1, 1]]
 */
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    int nd;
    size_t pos;
    char *ptr;
    dtype x;

    nd = cumo_na_get_result_dimension(self, argc, argv, 1, &pos);
    if (nd) {
        return cumo_na_aref_main(argc, argv, self, 0, nd, pos);
    } else {
        CUMO_BIT_DIGIT word;
        ptr = cumo_na_get_pointer_for_read(self);
        CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cumo_cuda_runtime_memcpy_to_host(&word, ptr+(pos)/CUMO_NB*sizeof(CUMO_BIT_DIGIT), sizeof(CUMO_BIT_DIGIT)));
        x = (word >> ((pos)%CUMO_NB)) & 1u;
        return m_data_to_num(x);
    }
}
