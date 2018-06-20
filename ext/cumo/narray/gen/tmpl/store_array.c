//<% unless c_iter.include? 'robject' %>
void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, size_t *idx1, dtype* z, uint64_t n);
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, ssize_t s1, dtype* z, uint64_t n);
void <%="cumo_#{c_iter}_index_scalar_kernel_launch"%>(char *p1, size_t *idx1, dtype z, uint64_t n);
void <%="cumo_#{c_iter}_stride_scalar_kernel_launch"%>(char *p1, ssize_t s1, dtype z, uint64_t n);

static void CUDART_CB
<%=c_iter%>_callback(cudaStream_t stream, cudaError_t status, void *data)
{
    xfree(data);
}
//<% end %>

static void
<%=c_iter%>(na_loop_t *const lp)
{
    size_t i, n;
    size_t i1, n1;
    VALUE  v1, *ptr;
    char   *p1;
    size_t s1, *idx1;
    VALUE  x;
    double y;
    dtype  z;
    size_t len, c;
    double beg, step;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    v1 = lp->args[1].value;
    i = 0;

    if (lp->args[1].ptr) {
        if (v1 == Qtrue) {
            iter_<%=type_name%>_store_<%=type_name%>(lp);
            i = lp->args[1].shape[0];
            if (idx1) {
                idx1 += i;
            } else {
                p1 += s1 * i;
            }
        }
        goto loop_end;
    }

    ptr = &v1;

    switch(TYPE(v1)) {
    case T_ARRAY:
        n1 = RARRAY_LEN(v1);
        ptr = RARRAY_PTR(v1);
        break;
    case T_NIL:
        n1 = 0;
        break;
    default:
        n1 = 1;
    }

    //<% if c_iter.include? 'robject' %>
    {
        SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("store_<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

        if (idx1) {
            for (i=i1=0; i1<n1 && i<n; i++,i1++) {
                x = ptr[i1];
                if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, na_cStep)) {
                    na_step_sequence(x,&len,&beg,&step);
                    for (c=0; c<len && i<n; c++,i++) {
                        y = beg + step * c;
                        z = m_from_double(y);
                        SET_DATA_INDEX(p1, idx1, dtype, z);
                    }
                }
                else if (TYPE(x) != T_ARRAY) {
                    z = m_num_to_data(x);
                    SET_DATA_INDEX(p1, idx1, dtype, z);
                }
            }
        } else {
            for (i=i1=0; i1<n1 && i<n; i++,i1++) {
                x = ptr[i1];
                if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, na_cStep)) {
                    na_step_sequence(x,&len,&beg,&step);
                    for (c=0; c<len && i<n; c++,i++) {
                        y = beg + step * c;
                        z = m_from_double(y);
                        SET_DATA_STRIDE(p1, s1, dtype, z);
                    }
                }
                else if (TYPE(x) != T_ARRAY) {
                    z = m_num_to_data(x);
                    SET_DATA_STRIDE(p1, s1, dtype, z);
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
        // 5. run callback to free the heap memory after kernel finishes
        //
        // FYI: We may have to care of cuda stream callback serializes stream execution when we support stream.
        // https://devtalk.nvidia.com/default/topic/822942/why-does-cudastreamaddcallback-serialize-kernel-execution-and-break-concurrency-/
        dtype* host_z = ALLOC_N(dtype, n);
        for (i=i1=0; i1<n1 && i<n; i1++) {
            x = ptr[i1];
            if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, na_cStep)) {
                na_step_sequence(x,&len,&beg,&step);
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
            cudaError_t status = cudaMemcpyAsync(p1,host_z,sizeof(dtype)*i,cudaMemcpyHostToDevice,0);
            if (status == 0) {
                cumo_cuda_runtime_check_status(cudaStreamAddCallback(0,<%=c_iter%>_callback,host_z,0));
            } else {
                xfree(host_z);
            }
            cumo_cuda_runtime_check_status(status);
        } else {
            dtype* device_z = (dtype*)cumo_cuda_runtime_malloc(sizeof(dtype) * n);
            cudaError_t status = cudaMemcpyAsync(device_z,host_z,sizeof(dtype)*i,cudaMemcpyHostToDevice,0);
            if (status == 0) {
                if (idx1) {
                    <%="cumo_#{c_iter}_index_kernel_launch"%>(p1,idx1,device_z,i);
                } else {
                    <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,s1,device_z,i);
                }
                cumo_cuda_runtime_check_status(cudaStreamAddCallback(0,<%=c_iter%>_callback,host_z,0));
            } else {
                xfree(host_z);
            }
            cumo_cuda_runtime_free((void*)device_z);
            cumo_cuda_runtime_check_status(status);
        }
    }
    //<% end %>

 loop_end:
    z = m_zero;
    //<% if c_iter.include? 'robject' %>
    {
        if (idx1) {
            for (; i<n; i++) {
                SET_DATA_INDEX(p1, idx1, dtype, z);
            }
        } else {
            for (; i<n; i++) {
                SET_DATA_STRIDE(p1, s1, dtype, z);
            }
        }
    }
    //<% else %>
    {
        if (idx1) {
            <%="cumo_#{c_iter}_index_scalar_kernel_launch"%>(p1,idx1+i,z,n-i);
        } else {
            <%="cumo_#{c_iter}_stride_scalar_kernel_launch"%>(p1+s1*i,s1,z,n-i);
        }
    }
    //<% end %>
}

static VALUE
<%=c_func%>(VALUE self, VALUE rary)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{rb_cArray,0}};
    ndfunc_t ndf = {<%=c_iter%>, FULL_LOOP, 2, 0, ain, 0};

    na_ndloop_store_rarray(&ndf, self, rary);
    return self;
}
