/*
  Array element(s) assignment.
  @overload []=(dim0,..,dimL,val)
  @param [Numeric,Range,Array,Cumo::Bit,Cumo::Int32,Cumo::Int64] dim0,..,dimL  Multi-dimensional Index.
  @param [Numeric,Cumo::NArray,etc] val  Value(s) to be set to self.
  @return [Numeric] returns `val` (last argument).

  Replaces element(s) at `dim0`, `dim1`, ... . Broadcasting mechanism is applied.

  @see #[]

  @example
      a = Cumo::Bit.new(4,5).fill(0)
      # => Cumo::Bit#shape=[4,5]
      # [[0, 0, 0, 0, 0],
      #  [0, 0, 0, 0, 0],
      #  [0, 0, 0, 0, 0],
      #  [0, 0, 0, 0, 0]]

      a[(0..-1)%2,(1..-1)%2] = 1
      a
      # => Cumo::Bit#shape=[4,5]
      # [[0, 1, 0, 1, 0],
      #  [0, 0, 0, 0, 0],
      #  [0, 1, 0, 1, 0],
      #  [0, 0, 0, 0, 0]]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    int nd;
    size_t pos;
    VALUE a;

    argc--;
    if (argc==0) {
        <%=c_func.sub(/_aset/,"_store")%>(self, argv[argc]);
    } else {
        nd = cumo_na_get_result_dimension(self, argc, argv, 1, &pos);
        a = cumo_na_aref_main(argc, argv, self, 0, nd, pos);
        <%=c_func.sub(/_aset/,"_store")%>(a, argv[argc]);
    }
    return argv[argc];
}
