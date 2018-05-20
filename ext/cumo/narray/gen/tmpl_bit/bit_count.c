#undef int_t
#define int_t uint64_t

void <%="cumo_#{c_iter}_index_kernel_launch"%>(size_t p1, char *p2, BIT_DIGIT *a1, size_t *idx1, uint64_t n);
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(size_t p1, char *p2, BIT_DIGIT *a1, ssize_t s1, uint64_t n);
void <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(size_t p1, char *p2, BIT_DIGIT *a1, size_t *idx1, ssize_t s2, uint64_t n);
void <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(size_t p1, char *p2, BIT_DIGIT *a1, ssize_t s1, ssize_t s2, uint64_t n);

static void
<%=c_iter%>(na_loop_t *const lp)
{
    size_t  i;
    BIT_DIGIT *a1;
    size_t  p1;
    char   *p2;
    ssize_t s1, s2;
    size_t *idx1;

    INIT_COUNTER(lp, i);
    INIT_PTR_BIT_IDX(lp, 0, a1, p1, s1, idx1);
    INIT_PTR(lp, 1, p2, s2);

    if (s2==0) {
        if (idx1) {
            <%="cumo_#{c_iter}_index_kernel_launch"%>(p1,p2,a1,idx1,i);
        } else {
            <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,p2,a1,s1,i);
        }
    } else {
        if (idx1) {
            <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(p1,p2,a1,idx1,s2,i);
        } else {
            <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(p1,p2,a1,s1,s2,i);
        }
    }
}

static VALUE
<%=c_func(-1)%>_cpu(int argc, VALUE *argv, VALUE self);

/*
  Returns the number of bits.
  If argument is supplied, return Int-array counted along the axes.
  @overload <%=op_map%>(axis:nil, keepdims:false)
  @param [Integer,Array,Range] axis (keyword) axes to be counted.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Cumo::UInt64]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    if (cumo_compatible_mode_enabled_p()) {
        return <%=c_func(-1)%>_cpu(argc, argv, self);
    }
    VALUE v, reduce;
    ndfunc_arg_in_t ain[3] = {{cT,0},{sym_reduce,0},{sym_init,0}};
    ndfunc_arg_out_t aout[1] = {{cumo_cUInt64,0}};
    ndfunc_t ndf = { <%=c_iter%>, FULL_LOOP_NIP, 3, 1, ain, aout };

    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
    v = na_ndloop(&ndf, 3, self, reduce, INT2FIX(0));
    return v;
}
