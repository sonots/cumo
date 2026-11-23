/*
  Array element(s) assignment.
  @overload []=(*indices, val)
    @param [Numeric,Range,Array,Cumo::Bit,Cumo::Int32,Cumo::Int64] indices  Multi-dimensional Index.
    @param [Numeric,Cumo::NArray,etc] val  Value(s) to be set to self.
    @return [Numeric,Cumo::NArray,Array] returns `val` (last argument) as given.

  Replaces the element(s) the indices select. Broadcasting mechanism is applied.
  A single index array of more than one dimension selects a place for each of
  its elements, counting them in order, and the value broadcasts to its shape.

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
    VALUE a, idx = Qnil;

    argc--;
    if (argc==0) {
        <%=c_func.sub(/_aset/,"_store")%>(self, argv[argc]);
    } else {
        // The store goes to a view derived by cumo_na_aref_main, whose data
        // object is the root rather than self, so the frozen check inside
        // cumo_na_get_pointer_for_rw never reaches self. Numo has a scalar
        // subscript path that writes through self and raises there; cumo has
        // none, since the element write has to happen on the device.
        if (OBJ_FROZEN(self)) {
            rb_raise(rb_eRuntimeError, "cannot write to frozen NArray.");
        }
        if (argc == 1) {
            idx = RB_TYPE_P(argv[0], T_ARRAY) ? rb_apply(cumo_cNArray, rb_intern("[]"), argv[0]) : argv[0];
            if (!CumoIsNArray(idx) || rb_obj_is_kind_of(idx, cumo_cBit) || CUMO_RNARRAY_NDIM(idx) <= 1) {
                idx = Qnil;
            }
        }
        if (RTEST(idx)) {
            // An index array of more than one dimension selects into a new
            // array rather than a view, so it goes in flattened, with the
            // value spread over its shape, as add_at does. A store through
            // the array it selects into would be lost with it.
            VALUE val = argv[argc], sub;

            if (!rb_obj_is_kind_of(val, cT)) {
                val = rb_funcall(cT, cumo_id_cast, 1, val);
            }
            if (CUMO_RNARRAY_NDIM(val) > 0) {
                VALUE spread = cumo_na_new(cT, CUMO_RNARRAY_NDIM(idx), CUMO_RNARRAY_SHAPE(idx));
                <%=c_func.sub(/_aset/,"_store")%>(spread, val);
                val = cumo_na_flatten(spread);
            }
            sub = cumo_na_flatten(idx);
            nd = cumo_na_get_result_dimension(self, 1, &sub, 1, &pos);
            a = cumo_na_aref_main(1, &sub, self, 0, nd, pos);
            <%=c_func.sub(/_aset/,"_store")%>(a, val);
        } else {
            nd = cumo_na_get_result_dimension(self, argc, argv, 1, &pos);
            a = cumo_na_aref_main(argc, argv, self, 0, nd, pos);
            <%=c_func.sub(/_aset/,"_store")%>(a, argv[argc]);
        }
    }
    return argv[argc];
}
