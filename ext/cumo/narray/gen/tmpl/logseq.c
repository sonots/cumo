typedef struct {
    seq_data_t beg;
    seq_data_t step;
    seq_data_t base;
    seq_count_t count;
} logseq_opt_t;

<% unless is_object %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_indexer_t* indexer, seq_data_t beg, seq_data_t step, seq_data_t base, seq_count_t c);
void <%="cumo_#{c_iter}_stridx_kernel_launch"%>(cumo_na_iarray_stridx_t* a1, cumo_na_indexer_t* indexer, seq_data_t beg, seq_data_t step, seq_data_t base, seq_count_t c);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    seq_data_t beg, step, base;
    seq_count_t c;
    logseq_opt_t *g;

    g = (logseq_opt_t*)(lp->opt_ptr);
    beg  = g->beg;
    step = g->step;
    base = g->base;
    c    = g->count;
    <% if is_object %>
    {
        size_t  i;
        char   *p1;
        ssize_t s1;
        size_t *idx1;
        dtype x;

        CUMO_INIT_COUNTER(lp, i);
        CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        if (idx1) {
            for (; i--;) {
                x = f_seq(beg,step,c++);
                *(dtype*)(p1+*idx1) = m_pow(base,x);
                idx1++;
            }
        } else {
            for (; i--;) {
                x = f_seq(beg,step,c++);
                *(dtype*)(p1) = m_pow(base,x);
                p1 += s1;
            }
        }
        g->count = c;
    }
    <% else %>
    {
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        if (cumo_na_loop_has_index(lp)) {
            cumo_na_iarray_stridx_t b1 = cumo_na_make_iarray_stridx(&lp->args[0]);
            <%="cumo_#{c_iter}_stridx_kernel_launch"%>(&b1,&indexer,beg,step,base,c);
        } else {
            cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
            <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&indexer,beg,step,base,c);
        }
        g->count += indexer.total_size;
    }
    <% end %>
}

/*
  Set logarithmic sequence of numbers to self. The sequence is obtained from
     `base**(beg+i*step)`
  where i is 1-dimensional index.
  Applicable classes: DFloat, SFloat, DComplex, SCopmplex.

  @overload logseq(beg,step,[base])
  @param [Numeric] beg  The beginning of sequence.
  @param [Numeric] step  The step of sequence.
  @param [Numeric] base  The base of log space. (default=10)
  @return [Cumo::<%=class_name%>] self.

  @example
    Cumo::DFloat.new(5).logseq(4,-1,2)
    # => Cumo::DFloat#shape=[5]
    #   [16, 8, 4, 2, 1]

    Cumo::DComplex.new(5).logseq(0,1i*Math::PI/3,Math::E)
    # => Cumo::DComplex#shape=[5]
    #   [1+7.26156e-310i, 0.5+0.866025i, -0.5+0.866025i, -1+1.22465e-16i, ...]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *args, VALUE self)
{
    logseq_opt_t *g;
    VALUE vbeg, vstep, vbase;
    cumo_ndfunc_arg_in_t ain[1] = {{CUMO_OVERWRITE,0}};
    <% if is_object %>
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 1,0, ain,0};
    <% else %>
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP|CUMO_NDF_INDEXER_LOOP, 1,0, ain,0};
    <% end %>

    g = ALLOCA_N(logseq_opt_t,1);
    g->count = 0;
    rb_scan_args(argc, args, "21", &vbeg, &vstep, &vbase);
    g->beg = m_num_to_data(vbeg);
    g->step = m_num_to_data(vstep);
    if (vbase==Qnil) {
        g->base = m_from_real(10);
    } else {
        g->base = m_num_to_data(vbase);
    }
    cumo_na_ndloop3(&ndf, g, 1, self);
    return self;
}
