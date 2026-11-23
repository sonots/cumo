static VALUE
<%=c_func(-1)%>_cpu(int argc, VALUE *argv, VALUE self);

/*
  Array element reference or slice view.
  @overload [](*indices)
    @param [Numeric,Range,Array,true,false,Cumo::NArray] indices  Multi-dimensional Index.
    @return [Cumo::<%=class_name%>, <%= is_object ? 'Object' : 'Numeric' %>] NArray view, or the element as a Ruby object under compatible mode. An integer index array of two or more dimensions gives a copy.

  --- With one Numeric index for each dimension, it returns the element
  there. With anything else among the indices, e.g., a Range or an Array
  or true, it returns a NArray View as a sliced subarray.

  Note that Cumo::NArray always returns NArray and does not return a
  Ruby numeric object as Numo::NArray does to avoid synchronization
  between GPU and CPU for performance.

  Call Cumo.enable_compatible_mode to make this method behave compatibly
  with Numo, or you can use `aref_cpu(*idx)` method instead.

  @example
      a = Cumo::DFloat.new(4,5).seq
      # => Cumo::DFloat#shape=[4,5]
      # [[0, 1, 2, 3, 4],
      #  [5, 6, 7, 8, 9],
      #  [10, 11, 12, 13, 14],
      #  [15, 16, 17, 18, 19]]

      a[7]
      # => Cumo::DFloat(view)#shape=[]
      # 7

      a[1,1]
      # => Cumo::DFloat(view)#shape=[]
      # 6

      a[1..3,1]
      # => Cumo::DFloat(view)#shape=[3]
      # [6, 11, 16]

      a[1,[1,3,4]]
      # => Cumo::DFloat(view)#shape=[3]
      # [6, 8, 9]

      a[true,2].fill(99)
      a
      # => Cumo::DFloat#shape=[4,5]
      # [[0, 1, 99, 3, 4],
      #  [5, 6, 99, 8, 9],
      #  [10, 11, 99, 13, 14],
      #  [15, 16, 99, 18, 19]]
 */
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    if (cumo_compatible_mode_enabled_p()) {
        return <%=c_func(-1)%>_cpu(argc, argv, self);
    } else {
        int result_nd;
        size_t pos;

        result_nd = cumo_na_get_result_dimension(self, argc, argv, sizeof(dtype), &pos);
        if (result_nd == 0) {
            // Numeric indices read an element, so an unallocated array has to
            // raise here. Returning the view instead only defers it to whatever
            // touches the data, and each_over_axis never does.
            (void)cumo_na_get_pointer_for_read(self);
        }
        return cumo_na_aref_main(argc, argv, self, 0, result_nd, pos);
    }
}
