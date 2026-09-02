<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_iarray_t* a4, cumo_na_indexer_t* indexer, int* invalid);
void <%="cumo_#{c_iter}_s_kernel_launch"%>(cumo_na_iarray_t* a1, dtype sv_min, dtype sv_max, cumo_na_iarray_t* a4, cumo_na_indexer_t* indexer);
void <%="cumo_#{c_iter}_min_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer);
void <%="cumo_#{c_iter}_min_s_kernel_launch"%>(cumo_na_iarray_t* a1, dtype sv, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer);
void <%="cumo_#{c_iter}_max_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer);
void <%="cumo_#{c_iter}_max_s_kernel_launch"%>(cumo_na_iarray_t* a1, dtype sv, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    {
        size_t  i, n;
        char   *p1, *p2, *p3, *p4;
        ssize_t s1, s2, s3, s4;
        dtype  x, min, max;

        CUMO_INIT_COUNTER(lp, n);
        CUMO_INIT_PTR(lp, 0, p1, s1);
        CUMO_INIT_PTR(lp, 1, p2, s2);
        CUMO_INIT_PTR(lp, 2, p3, s3);
        CUMO_INIT_PTR(lp, 3, p4, s4);

        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        for (i=n; i--;) {
            CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
            CUMO_GET_DATA_STRIDE(p2,s2,dtype,min);
            CUMO_GET_DATA_STRIDE(p3,s3,dtype,max);
            if (m_gt(min,max)) {rb_raise(cumo_na_eOperationError,"min is greater than max");}
            if (m_lt(x,min)) {x=min;}
            if (m_gt(x,max)) {x=max;}
            CUMO_SET_DATA_STRIDE(p4,s4,dtype,x);
        }
    }
    <% else %>
    {
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);
        cumo_na_iarray_t a3 = cumo_na_make_iarray(&lp->args[2]);
        cumo_na_iarray_t a4 = cumo_na_make_iarray(&lp->args[3]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);
        int *invalid = cumo_cuda_runtime_error_flag_new();

        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&a3,&a4,&indexer,invalid);
        CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        if (cumo_cuda_runtime_error_flag_get(invalid)) {
            rb_raise(cumo_na_eOperationError,"min is greater than max");
        }
    }
    <% end %>
}

static void
<%=c_iter%>_min(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    {
        size_t  i, n;
        char   *p1, *p2, *p3;
        ssize_t s1, s2, s3;
        dtype  x, min;

        CUMO_INIT_COUNTER(lp, n);
        CUMO_INIT_PTR(lp, 0, p1, s1);
        CUMO_INIT_PTR(lp, 1, p2, s2);
        CUMO_INIT_PTR(lp, 2, p3, s3);

        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>_min", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        for (i=n; i--;) {
            CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
            CUMO_GET_DATA_STRIDE(p2,s2,dtype,min);
            if (m_lt(x,min)) {x=min;}
            CUMO_SET_DATA_STRIDE(p3,s3,dtype,x);
        }
    }
    <% else %>
    {
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);
        cumo_na_iarray_t a3 = cumo_na_make_iarray(&lp->args[2]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        <%="cumo_#{c_iter}_min_kernel_launch"%>(&a1,&a2,&a3,&indexer);
    }
    <% end %>
}

static void
<%=c_iter%>_max(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    {
        size_t  i, n;
        char   *p1, *p2, *p3;
        ssize_t s1, s2, s3;
        dtype  x, max;

        CUMO_INIT_COUNTER(lp, n);
        CUMO_INIT_PTR(lp, 0, p1, s1);
        CUMO_INIT_PTR(lp, 1, p2, s2);
        CUMO_INIT_PTR(lp, 2, p3, s3);

        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>_max", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        for (i=n; i--;) {
            CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
            CUMO_GET_DATA_STRIDE(p2,s2,dtype,max);
            if (m_gt(x,max)) {x=max;}
            CUMO_SET_DATA_STRIDE(p3,s3,dtype,x);
        }
    }
    <% else %>
    {
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);
        cumo_na_iarray_t a3 = cumo_na_make_iarray(&lp->args[2]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        <%="cumo_#{c_iter}_max_kernel_launch"%>(&a1,&a2,&a3,&indexer);
    }
    <% end %>
}

<% unless type_name == 'robject' %>
// A Ruby numeric bound rides in through opt_ptr instead of being cast to a
// 0-dimensional array, which costs a kernel launch to fill. With both bounds
// numeric the caller rejects min > max itself, so the kernel needs no error
// flag and the loop no longer synchronizes to read one.
static void
<%=c_iter%>_s(cumo_na_loop_t *const lp)
{
    dtype *sv = (dtype*)(lp->opt_ptr);
    cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
    cumo_na_iarray_t a4 = cumo_na_make_iarray(&lp->args[1]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

    <%="cumo_#{c_iter}_s_kernel_launch"%>(&a1,sv[0],sv[1],&a4,&indexer);
}

static void
<%=c_iter%>_min_s(cumo_na_loop_t *const lp)
{
    dtype sv = *(dtype*)(lp->opt_ptr);
    cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
    cumo_na_iarray_t a3 = cumo_na_make_iarray(&lp->args[1]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

    <%="cumo_#{c_iter}_min_s_kernel_launch"%>(&a1,sv,&a3,&indexer);
}

static void
<%=c_iter%>_max_s(cumo_na_loop_t *const lp)
{
    dtype sv = *(dtype*)(lp->opt_ptr);
    cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
    cumo_na_iarray_t a3 = cumo_na_make_iarray(&lp->args[1]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

    <%="cumo_#{c_iter}_max_s_kernel_launch"%>(&a1,sv,&a3,&indexer);
}
<% end %>

/*
  Clip array elements by [min,max].
  If either of min or max is nil, one side is clipped.
  @overload <%=name%>(min,max)
  @param [Cumo::NArray,Numeric] min
  @param [Cumo::NArray,Numeric] max
  @return [Cumo::NArray] result of clip.

  @example
      a = Cumo::Int32.new(10).seq
      # => Cumo::Int32#shape=[10]
      # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

      a.clip(1,8)
      # => Cumo::Int32#shape=[10]
      # [1, 1, 2, 3, 4, 5, 6, 7, 8, 8]

      a.inplace.clip(3,6)
      a
      # => Cumo::Int32#shape=[10]
      # [3, 3, 3, 3, 4, 5, 6, 6, 6, 6]

      b = Cumo::Int32.new(10).seq
      # => Cumo::Int32#shape=[10]
      # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

      b.clip([3,4,1,1,1,4,4,4,4,4], 8)
      # => Cumo::Int32#shape=[10]
      # [3, 4, 2, 3, 4, 5, 6, 7, 8, 8]
*/
static VALUE
<%=c_func(2)%>(VALUE self, VALUE min, VALUE max)
{
    cumo_ndfunc_arg_in_t ain[3] = {{Qnil,0},{cT,0},{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    <% if type_name == 'robject' %>
    cumo_ndfunc_t ndf_min = { <%=c_iter%>_min, CUMO_STRIDE_LOOP, 2, 1, ain, aout };
    cumo_ndfunc_t ndf_max = { <%=c_iter%>_max, CUMO_STRIDE_LOOP, 2, 1, ain, aout };
    cumo_ndfunc_t ndf_both = { <%=c_iter%>, CUMO_STRIDE_LOOP, 3, 1, ain, aout };
    <% else %>
    cumo_ndfunc_t ndf_min = { <%=c_iter%>_min, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 2, 1, ain, aout };
    cumo_ndfunc_t ndf_max = { <%=c_iter%>_max, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 2, 1, ain, aout };
    cumo_ndfunc_t ndf_both = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 3, 1, ain, aout };

    {
        cumo_ndfunc_arg_in_t ain_s[1] = {{Qnil,0}};
        int min_s = RTEST(min) && rb_obj_is_kind_of(min, rb_cNumeric);
        int max_s = RTEST(max) && rb_obj_is_kind_of(max, rb_cNumeric);

        if (min_s && max_s) {
            dtype sv[2];
            cumo_ndfunc_t ndf_s = { <%=c_iter%>_s, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 1, 1, ain_s, aout };
            sv[0] = m_num_to_data(min);
            sv[1] = m_num_to_data(max);
            if (m_gt(sv[0],sv[1])) {rb_raise(cumo_na_eOperationError,"min is greater than max");}
            return cumo_na_ndloop3(&ndf_s, sv, 1, self);
        }
        if (min_s && !RTEST(max)) {
            dtype sv = m_num_to_data(min);
            cumo_ndfunc_t ndf_min_s = { <%=c_iter%>_min_s, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 1, 1, ain_s, aout };
            return cumo_na_ndloop3(&ndf_min_s, &sv, 1, self);
        }
        if (max_s && !RTEST(min)) {
            dtype sv = m_num_to_data(max);
            cumo_ndfunc_t ndf_max_s = { <%=c_iter%>_max_s, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 1, 1, ain_s, aout };
            return cumo_na_ndloop3(&ndf_max_s, &sv, 1, self);
        }
    }
    <% end %>

    if (RTEST(min)) {
        if (RTEST(max)) {
            return cumo_na_ndloop(&ndf_both, 3, self, min, max);
        } else {
            return cumo_na_ndloop(&ndf_min, 2, self, min);
        }
    } else {
        if (RTEST(max)) {
            return cumo_na_ndloop(&ndf_max, 2, self, max);
        }
    }
    rb_raise(rb_eArgError,"min and max are not given");
    return Qnil;
}
