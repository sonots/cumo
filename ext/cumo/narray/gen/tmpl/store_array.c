//<% unless c_iter.include? 'robject' %>
void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, size_t *idx1, dtype* z, uint64_t n);
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, ssize_t s1, dtype* z, uint64_t n);
void <%="cumo_#{c_iter}_index_scalar_kernel_launch"%>(char *p1, size_t *idx1, dtype z, uint64_t n);
void <%="cumo_#{c_iter}_stride_scalar_kernel_launch"%>(char *p1, ssize_t s1, dtype z, uint64_t n);
//<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t i, n;
    size_t i1, n1;
    VALUE  v1;
    char   *p1;
    size_t s1, *idx1;
    VALUE  x;
    double y;
    dtype  z;
    size_t len, c;
    double beg, step;

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    v1 = lp->args[1].value;
    i = 0;

    if (lp->args[1].ptr) {
        if (v1 == Qtrue) {
            // The loop counter is the destination length, but the sub-narray
            // may be shorter. Copy only what the source actually holds, or the
            // kernel would read past its end; the rest is zero-filled below.
            i = lp->args[1].shape[0];
            if (i > n) {
                i = n;
            }
            CUMO_NDL_CNT(lp) = i;
            iter_<%=type_name%>_store_<%=type_name%>(lp);
            CUMO_NDL_CNT(lp) = n;
            //<% if c_iter.include? 'robject' %>
            // The zero fill below walks from here. The kernel the other branch
            // launches takes the offset of the first element left to fill as an
            // argument instead, so advancing for it would count i twice.
            if (idx1) {
                idx1 += i;
            } else {
                p1 += s1 * i;
            }
            //<% end %>
        }
        goto loop_end;
    }

    switch(TYPE(v1)) {
    case T_ARRAY:
        n1 = RARRAY_LEN(v1);
        break;
    case T_NIL:
        n1 = 0;
        break;
    default:
        n1 = 1;
    }

    //<% if c_iter.include? 'robject' %>
    {
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("store_<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

        if (idx1) {
            for (i=i1=0; i1<n1 && i<n; i1++) {
                if (!cumo_na_store_rary_fetch(v1, i1, &x)) break;
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
                if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, rb_cArithSeq)) {
#else
                if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, rb_cEnumerator)) {
#endif
                    cumo_na_step_sequence(x,&len,&beg,&step);
                    for (c=0; c<len && i<n; c++,i++) {
                        y = beg + step * c;
                        z = m_from_double(y);
                        CUMO_SET_DATA_INDEX(p1, idx1, dtype, z);
                    }
                }
                else if (TYPE(x) != T_ARRAY) {
                    z = m_num_to_data(x);
                    CUMO_SET_DATA_INDEX(p1, idx1, dtype, z);
                    i++;
                }
            }
        } else {
            for (i=i1=0; i1<n1 && i<n; i1++) {
                if (!cumo_na_store_rary_fetch(v1, i1, &x)) break;
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
                if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, rb_cArithSeq)) {
#else
                if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, rb_cEnumerator)) {
#endif
                    cumo_na_step_sequence(x,&len,&beg,&step);
                    for (c=0; c<len && i<n; c++,i++) {
                        y = beg + step * c;
                        z = m_from_double(y);
                        CUMO_SET_DATA_STRIDE(p1, s1, dtype, z);
                    }
                }
                else if (TYPE(x) != T_ARRAY) {
                    z = m_num_to_data(x);
                    CUMO_SET_DATA_STRIDE(p1, s1, dtype, z);
                    i++;
                }
            }
        }
    }
    //<% else %>
    {
        // To copy ruby non-contiguous array values into cuda memory asynchronously, we do
        // 1. copy to contiguous heap memory
        // 2. copy to contiguous device memory
        // 3. launch kernel to copy the contiguous device memory into strided (or indexed) narray cuda memory
        // 4. free the contiguous device memory
        //
        // The staging buffer is held by a Ruby object rather than a plain local:
        // the conversion loop below raises for a value which is not a number,
        // and ndloop has no way to free a buffer it does not know about.
        VALUE  buf;
        dtype* host_z = RB_ALLOCV_N(dtype, buf, n);
        for (i=i1=0; i1<n1 && i<n; i1++) {
            if (!cumo_na_store_rary_fetch(v1, i1, &x)) break;
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
            if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, rb_cArithSeq)) {
#else
            if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, rb_cEnumerator)) {
#endif
                cumo_na_step_sequence(x,&len,&beg,&step);
                for (c=0; c<len && i<n; c++,i++) {
                    y = beg + step * c;
                    host_z[i] = m_from_double(y);
                }
            }
            else if (TYPE(x) != T_ARRAY) {
                host_z[i] = m_num_to_data(x);
                i++;
            }
        }

        if (!idx1 && s1 == sizeof(dtype)) {
            // optimization: Since p1 is contiguous, we skip creating another contiguous device memory
            //
            // host_z is pageable, so cudaMemcpyAsync only returns once it has
            // been copied into the driver's staging buffer and the buffer may
            // be released right away.
            cumo_cuda_runtime_check_status(
                cudaMemcpyAsync(p1,host_z,sizeof(dtype)*i,cudaMemcpyHostToDevice,0));
        } else {
            dtype* device_z = (dtype*)cumo_cuda_runtime_malloc(sizeof(dtype) * n);
            cudaError_t status = cudaMemcpyAsync(device_z,host_z,sizeof(dtype)*i,cudaMemcpyHostToDevice,0);
            if (status == 0) {
                if (idx1) {
                    <%="cumo_#{c_iter}_index_kernel_launch"%>(p1,idx1,device_z,i);
                } else {
                    <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,s1,device_z,i);
                }
                // The kernel reads device_z, so wait for it before giving the
                // buffer back. The memory pool hands a freed chunk straight out
                // again, and a host write into that managed memory does not wait
                // for the stream, so the kernel would read the new contents.
                status = cudaStreamSynchronize(0);
            }
            cumo_cuda_runtime_free((void*)device_z);
            cumo_cuda_runtime_check_status(status);
        }
        RB_ALLOCV_END(buf);
    }
    //<% end %>

 loop_end:
    z = m_zero;
    //<% if c_iter.include? 'robject' %>
    {
        if (idx1) {
            for (; i<n; i++) {
                CUMO_SET_DATA_INDEX(p1, idx1, dtype, z);
            }
        } else {
            for (; i<n; i++) {
                CUMO_SET_DATA_STRIDE(p1, s1, dtype, z);
            }
        }
    }
    //<% else %>
    {
        // i may exceed n when the sub-narray is longer than the destination;
        // n-i would then wrap around as size_t. numo's scalar loop is a no-op
        // in that case, so skip the launch instead.
        if (i < n) {
            if (idx1) {
                <%="cumo_#{c_iter}_index_scalar_kernel_launch"%>(p1,idx1+i,z,n-i);
            } else {
                <%="cumo_#{c_iter}_stride_scalar_kernel_launch"%>(p1+s1*i,s1,z,n-i);
            }
        }
    }
    //<% end %>
}

static VALUE
<%=c_func%>(VALUE self, VALUE rary)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{rb_cArray,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 2, 0, ain, 0};

    cumo_na_ndloop_store_rarray(&ndf, self, rary);
    return self;
}
