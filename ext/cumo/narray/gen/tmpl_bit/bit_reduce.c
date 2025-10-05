static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t     i;
    CUMO_BIT_DIGIT *a1, *a2;
    size_t     p1,  p2;
    ssize_t    s1,  s2;
    size_t    *idx1, *idx2;
    CUMO_BIT_DIGIT  x=0, y=0;

    // TODO(sonots): CUDA kernelize
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a1, p1, s1, idx1);
    CUMO_INIT_PTR_BIT_IDX(lp, 1, a2, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                CUMO_LOAD_BIT(a2, p2+*idx2, y);
                if (y == <%=init_bit%>) {
                    CUMO_LOAD_BIT(a1, p1+*idx1, x);
                    if (x != <%=init_bit%>) {
                        CUMO_STORE_BIT(a2, p2+*idx2, x);
                    }
                }
                idx1++;
                idx2++;
            }
        } else {
            for (; i--;) {
                CUMO_LOAD_BIT(a2, p2+*idx2, y);
                if (y == <%=init_bit%>) {
                    CUMO_LOAD_BIT(a1, p1, x);
                    if (x != <%=init_bit%>) {
                        CUMO_STORE_BIT(a2, p2+*idx2, x);
                    }
                }
                p1 += s1;
                idx2++;
            }
        }
    } else if (s2) {
        if (idx1) {
            for (; i--;) {
                CUMO_LOAD_BIT(a2, p2, y);
                if (y == <%=init_bit%>) {
                    CUMO_LOAD_BIT(a1, p1+*idx1, x);
                    if (x != <%=init_bit%>) {
                        CUMO_STORE_BIT(a2, p2, x);
                    }
                }
                idx1++;
                p2 += s2;
            }
        } else {
            for (; i--;) {
                CUMO_LOAD_BIT(a2, p2, y);
                if (y == <%=init_bit%>) {
                    CUMO_LOAD_BIT(a1, p1, x);
                    if (x != <%=init_bit%>) {
                        CUMO_STORE_BIT(a2, p2, x);
                    }
                }
                p1 += s1;
                p2 += s2;
            }
        }
    } else {
        CUMO_LOAD_BIT(a2, p2, x);
        if (x != <%=init_bit%>) {
            return;
        }
        if (idx1) {
            for (; i--;) {
                CUMO_LOAD_BIT(a1, p1+*idx1, y);
                if (y != <%=init_bit%>) {
                    CUMO_STORE_BIT(a2, p2, y);
                    return;
                }
                idx1++;
            }
        } else {
            for (; i--;) {
                CUMO_LOAD_BIT(a1, p1, y);
                if (y != <%=init_bit%>) {
                    CUMO_STORE_BIT(a2, p2, y);
                    return;
                }
                p1 += s1;
            }
        }
    }
}

/*
<% case name
   when /^any/ %>
  Return true if any of bits is one (true).
<% when /^all/ %>
  Return true if all of bits are one (true).
<% end %>
  If argument is supplied, return Bit-array reduced along the axes.
  @overload <%=op_map%>(axis:nil, keepdims:false)
  @param [Integer,Array,Range] axis (keyword) axes to be reduced.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Cumo::Bit] .
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    cumo_narray_t *na;
    cumo_ndfunc_arg_in_t ain[3] = {{cT,0},{cumo_sym_reduce,0},{cumo_sym_init,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cumo_cBit,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP_NIP, 3,1, ain,aout};

    CumoGetNArray(self,na);
    if (CUMO_NA_SIZE(na)==0) {
        return INT2FIX(0);
    }
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
    v = cumo_na_ndloop(&ndf, 3, self, reduce, INT2FIX(<%=init_bit%>));
    if (argc > 0) {
        return v;
    }
    v = <%=find_tmpl("extract").c_func%>(v);
    switch (v) {
    case INT2FIX(0):
        return Qfalse;
    case INT2FIX(1):
        return Qtrue;
    default:
        rb_bug("unexpected result");
        return v;
    }
}
