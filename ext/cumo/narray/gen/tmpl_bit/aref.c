static VALUE
<%=c_func(-1)%>_cpu(int argc, VALUE *argv, VALUE self);

/*
  Array indexing.
  @overload [](*indices)
    @param [Numeric,Range,Array,Cumo::Bit,Cumo::Int32,Cumo::Int64] indices  Multi-dimensional Index.
    @return [Cumo::Bit,Integer] NArray view, or an Integer for an element under compatible mode.

  --- With one Numeric index for each dimension, it returns a 0-dimensional view of the element there. With anything else among the indices, e.g., a Range or an Array or true, it returns a NArray View as a sliced array.

  Note that Cumo::NArray always returns NArray and does not return a Ruby numeric object as Numo::NArray does to avoid synchronization between GPU and CPU for performance. Call Cumo.enable_compatible_mode to make this method behave compatibly with Numo, or you can use `aref_cpu(*idx)` method instead.

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
      # => Cumo::Bit(view)#shape=[]
      # 0
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
        if (result_nd == 0) {
            // Numeric indices read an element, so an unallocated array has to
            // raise here. Returning the view instead only defers it to whatever
            // touches the data, and each_over_axis never does.
            (void)cumo_na_get_pointer_for_read(self);
        }
        return cumo_na_aref_main(argc, argv, self, 0, result_nd, pos);
    }
}
