/*
  Array element(s) set.
  @overload []=(*indices, val)
    @param [Numeric,Range,Array,true,false,Cumo::NArray] indices  Multi-dimensional Index.
    @param [Numeric,Cumo::NArray,etc] val  Value(s) to be set to self.
    @return [Numeric,Cumo::NArray,Array] returns val (last argument) as given.

  --- Replaces the element(s) the indices select, one index, range, array or
  true for each dimension. Broadcasting mechanism is applied. A single index
  array of more than one dimension gives a place in the flattened receiver for
  each of its elements, and the value goes in as #store puts it into an array
  of the shape of that index array.

  @example
      a = Cumo::DFloat.new(3,4).seq
      # => Cumo::DFloat#shape=[3,4]
      # [[0, 1, 2, 3],
      #  [4, 5, 6, 7],
      #  [8, 9, 10, 11]]

      a[1,2]=99
      a
      # => Cumo::DFloat#shape=[3,4]
      # [[0, 1, 2, 3],
      #  [4, 5, 99, 7],
      #  [8, 9, 10, 11]]

      a[1,[0,2]] = [101,102]
      a
      # => Cumo::DFloat#shape=[3,4]
      # [[0, 1, 2, 3],
      #  [101, 5, 102, 7],
      #  [8, 9, 10, 11]]

      a[1,true]=99
      a
      # => Cumo::DFloat#shape=[3,4]
      # [[0, 1, 2, 3],
      #  [99, 99, 99, 99],
      #  [8, 9, 10, 11]]

*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    int nd;
    size_t pos = 0;
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
            idx = argv[0];
            if (RB_TYPE_P(idx, T_ARRAY)) {
                long i;
                // Only an Array or an NArray among the elements can make
                // NArray[] answer more than one dimension, and building it for
                // a plain list of indices would be work for nothing.
                for (i = 0; i < RARRAY_LEN(idx); ++i) {
                    VALUE e = RARRAY_AREF(idx, i);
                    if (RB_TYPE_P(e, T_ARRAY) || CumoIsNArray(e)) {
                        idx = rb_apply(cumo_cNArray, rb_intern("[]"), idx);
                        break;
                    }
                }
            }
            if (!CumoIsNArray(idx) || rb_obj_is_kind_of(idx, cumo_cBit) || CUMO_RNARRAY_NDIM(idx) <= 1) {
                idx = Qnil;
            }
        }
        if (RTEST(idx)) {
            // An index array of more than one dimension selects into a new
            // array rather than a view, so it goes in flattened, and a store
            // through the array it selects into would be lost with it. A value
            // with a shape of its own goes through the same store into the
            // shape of the index, so a shape that does not fit is taken the
            // way every other subscript takes it.
            VALUE val = argv[argc], sub;

            sub = cumo_na_flatten(idx);
            nd = cumo_na_get_result_dimension(self, 1, &sub, sizeof(dtype), &pos);
            // Before the value, so that an index out of range is answered
            // first, as it is for every other subscript.
            a = cumo_na_aref_main(1, &sub, self, 0, nd, pos);
            if (RB_TYPE_P(val, T_ARRAY) || (CumoIsNArray(val) && CUMO_RNARRAY_NDIM(val) > 0)) {
                VALUE spread = cumo_na_new(cT, CUMO_RNARRAY_NDIM(idx), CUMO_RNARRAY_SHAPE(idx));
                <%=c_func.sub(/_aset/,"_store")%>(spread, val);
                // A value with nothing in it leaves the array it is stored
                // into as it was, so nothing goes out to the receiver either.
                // A shape that does not broadcast has raised above.
                if (CumoIsNArray(val) && CUMO_RNARRAY_SIZE(val) == 0) {
                    return argv[argc];
                }
                val = cumo_na_flatten(spread);
            }
            <%=c_func.sub(/_aset/,"_store")%>(a, val);
        } else {
            nd = cumo_na_get_result_dimension(self, argc, argv, sizeof(dtype), &pos);
            a = cumo_na_aref_main(argc, argv, self, 0, nd, pos);
            <%=c_func.sub(/_aset/,"_store")%>(a, argv[argc]);
        }
    }
    return argv[argc];
}
