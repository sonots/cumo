<% if is_int && !is_object %>
typedef double seq_data_t;
<% else %>
typedef dtype seq_data_t;
<% end %>

<% if is_object %>
typedef size_t seq_count_t;
<% else %>
typedef double seq_count_t;
<% end %>

typedef struct {
    seq_data_t beg;
    seq_data_t step;
    seq_count_t count;
} seq_opt_t;

<% unless is_object %>
void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, size_t* idx1, seq_data_t beg, seq_data_t step, seq_count_t c, uint64_t n);
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, ssize_t s1, seq_data_t beg, seq_data_t step, seq_count_t c, uint64_t n);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  i;
    char   *p1;
    ssize_t s1;
    size_t *idx1;
    seq_data_t beg, step;
    seq_count_t c;
    seq_opt_t *g;

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    g = (seq_opt_t*)(lp->opt_ptr);
    beg  = g->beg;
    step = g->step;
    c    = g->count;
    <% if is_object %>
    {
        dtype x;
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        if (idx1) {
            for (; i--;) {
                x = f_seq(beg,step,c++);
                *(dtype*)(p1+*idx1) = x;
                idx1++;
            }
        } else {
            for (; i--;) {
                x = f_seq(beg,step,c++);
                *(dtype*)(p1) = x;
                p1 += s1;
            }
        }
        g->count = c;
    }
    <% else %>
    {
        size_t n = i;
        if (idx1) {
            <%="cumo_#{c_iter}_index_kernel_launch"%>(p1,idx1,beg,step,c,n);
        } else {
            <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,s1,beg,step,c,n);
        }
        g->count += n;
    }
    <% end %>
}

/*
  Set linear sequence of numbers to self. The sequence is obtained from
     beg+i*step
  where i is 1-dimensional index.
  @overload seq([beg,[step]])
  @param [Numeric] beg  begining of sequence. (default=0)
  @param [Numeric] step  step of sequence. (default=1)
  @return [Cumo::<%=class_name%>] self.
  @example
    Cumo::DFloat.new(6).seq(1,-0.2)
    # => Cumo::DFloat#shape=[6]
    #    [1, 0.8, 0.6, 0.4, 0.2, 0]

    Cumo::DComplex.new(6).seq(1,-0.2+0.2i)
    # => Cumo::DComplex#shape=[6]
    #    [1+0i, 0.8+0.2i, 0.6+0.4i, 0.4+0.6i, 0.2+0.8i, 0+1i]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *args, VALUE self)
{
    seq_opt_t *g;
    VALUE vbeg=Qnil, vstep=Qnil;
    cumo_ndfunc_arg_in_t ain[1] = {{CUMO_OVERWRITE,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 1,0, ain,0};

    g = ALLOCA_N(seq_opt_t,1);
    g->beg = m_zero;
    g->step = m_one;
    g->count = 0;
    rb_scan_args(argc, args, "02", &vbeg, &vstep);
<% if is_int && !is_object %>
    if (vbeg!=Qnil) {g->beg = NUM2DBL(vbeg);}
    if (vstep!=Qnil) {g->step = NUM2DBL(vstep);}
<% else %>
    if (vbeg!=Qnil) {g->beg = m_num_to_data(vbeg);}
    if (vstep!=Qnil) {g->step = m_num_to_data(vstep);}
<% end %>

    cumo_na_ndloop3(&ndf, g, 1, self);
    return self;
}
