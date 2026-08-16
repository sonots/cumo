<%
if is_int && !is_object
  if /Int64$/ =~ class_name
    rand_bit = 64
  else
    rand_bit = 32
  end
  m_rand = "m_rand(max,shift)"
  shift_def = "int shift;"
  shift_set = "shift = #{rand_bit-1} - msb_pos(max);"
  rand_type = "uint#{rand_bit}_t"
%>

#define HWID (sizeof(dtype)*4)

static int msb_pos(<%=rand_type%> a)
{
    int width = HWID;
    int pos = 0;
    <%=rand_type%> mask = (((dtype)1 << HWID)-1) << HWID;

    if (a==0) {return -1;}

    while (width) {
        if (a & mask) {
            pos += width;
        } else {
            mask >>= width;
        }
        width >>= 1;
        mask &= mask << width;
    }
    return pos;
}

/* generates a random number on [0,max) */
<% if rand_bit == 64 %>
inline static dtype m_rand(uint64_t max, int shift)
{
    uint64_t x;
    do {
        x = gen_rand32();
        x <<= 32;
        x |= gen_rand32();
        x >>= shift;
    } while (x >= max);
    return x;
}
<% else %>
inline static dtype m_rand(uint32_t max, int shift)
{
    uint32_t x;
    do {
        x = gen_rand32();
        x >>= shift;
    } while (x >= max);
    return x;
}
<% end %>
<%
else
  m_rand = "m_rand(max)"
  shift_def = ""
  shift_set = ""
  rand_type = "dtype"
end
%>

typedef struct {
    dtype low;
    <%=rand_type%> max;
    u_int64_t seed;
    u_int64_t offset;
} rand_opt_t;

<% unless is_object %>
void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, size_t *idx1, uint64_t seed, uint64_t offset, dtype low, <%=rand_type%> max, int shift, uint64_t n);
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, ssize_t s1, uint64_t seed, uint64_t offset, dtype low, <%=rand_type%> max, int shift, uint64_t n);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t   i;
    char    *p1;
    ssize_t  s1;
    size_t  *idx1;
    rand_opt_t *g;
    dtype    low;
    <%=rand_type%> max;
    <%=shift_def%>

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    g = (rand_opt_t*)(lp->opt_ptr);
    low = g->low;
    max = g->max;
    <%=shift_set%>

    <% if is_object %>
    {
        dtype x;

        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        if (idx1) {
            for (; i--;) {
                x = m_add(<%=m_rand%>,low);
                CUMO_SET_DATA_INDEX(p1,idx1,dtype,x);
            }
        } else {
            for (; i--;) {
                x = m_add(<%=m_rand%>,low);
                CUMO_SET_DATA_STRIDE(p1,s1,dtype,x);
            }
        }
    }
    <% else %>
    {
        size_t n = i;
        int shift_arg = 0;
        <% if is_int %>
        shift_arg = shift;
        <% end %>
        if (idx1) {
            <%="cumo_#{c_iter}_index_kernel_launch"%>(p1,idx1,g->seed,g->offset,low,max,shift_arg,n);
        } else {
            <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,s1,g->seed,g->offset,low,max,shift_arg,n);
        }
        g->offset += n;
    }
    <% end %>
}


/*
  Generate uniformly distributed random numbers on self narray.
  @overload rand([[low],high])
  @param [Numeric] low  lower inclusive boundary of random numbers. (default=0)
  @param [Numeric] high  upper exclusive boundary of random numbers. (default=1 or 1+1i for complex types)
  @return [Cumo::<%=class_name%>] self.
  @example
    Cumo::DFloat.new(6).rand
    # => Cumo::DFloat#shape=[6]
    #    [0.600954, 0.483321, 0.97507, 0.0599206, 0.0541459, 0.203269]

    Cumo::DComplex.new(6).rand(5+5i)
    # => Cumo::DComplex#shape=[6]
    #    [3.00477+0.597399i, 2.4166+0.30171i, 4.87535+1.43536i, 0.299603+4.66121i, ...]

    Cumo::Int32.new(6).rand(2,5)
    # => Cumo::Int32#shape=[6]
    #    [3, 4, 2, 2, 4, 4]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *args, VALUE self)
{
    rand_opt_t g;
    VALUE v1=Qnil, v2=Qnil;
    dtype high;
    cumo_ndfunc_arg_in_t ain[1] = {{CUMO_OVERWRITE,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 1,0, ain,0};

    <% if is_int && !is_object %>
    rb_scan_args(argc, args, "11", &v1, &v2);
    if (v2==Qnil) {
        g.low = m_zero;
        g.max = high = m_num_to_data(v1);
    <% else %>
    rb_scan_args(argc, args, "02", &v1, &v2);
    if (v2==Qnil) {
        g.low = m_zero;
        if (v1==Qnil) {
            <% if is_complex %>
            g.max = high = c_new(1,1);
            <% else %>
            g.max = high = m_one;
            <% end %>
        } else {
            g.max = high = m_num_to_data(v1);
        }
    <% end %>
    } else {
        g.low = m_num_to_data(v1);
        high = m_num_to_data(v2);
        g.max = m_sub(high,g.low);
    }
    <% if is_int && !is_object %>
    if (high <= g.low) {
        rb_raise(rb_eArgError,"high must be larger than low");
    }
    <% end %>
    g.seed = cumo_cuda_rand_seed();
    g.offset = cumo_cuda_rand_offset();
    cumo_na_ndloop3(&ndf, &g, 1, self);
    cumo_cuda_rand_set_offset(g.offset);
    return self;
}
