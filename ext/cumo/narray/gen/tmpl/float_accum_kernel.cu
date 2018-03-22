<% f = File.join(File.dirname(__FILE__), 'real_accum_kernel.cu'); ERB.new(File.read(f)).tap {|erb| erb.filename = f }.result(binding) %>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_mean_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2, uint64_t n)
{
    dtype init = m_zero;
    *p2 = thrust::reduce(thrust::cuda::par, p1_begin, p1_end, init, thrust::plus<dtype>());
    *p2 /= (dtype)n;
}

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_var_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2)
{
    thrust_variance_unary_op<dtype>  unary_op;
    thrust_variance_binary_op<dtype> binary_op;
    thrust_variance_data<dtype> init = {};
    thrust_variance_data<dtype> result;
    result = thrust::transform_reduce(thrust::cuda::par, p1_begin, p1_end, unary_op, init, binary_op);
    *p2 = result.variance();
}

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_stddev_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2)
{
    thrust_variance_unary_op<dtype>  unary_op;
    thrust_variance_binary_op<dtype> binary_op;
    thrust_variance_data<dtype> init = {};
    thrust_variance_data<dtype> result;
    result = thrust::transform_reduce(thrust::cuda::par, p1_begin, p1_end, unary_op, init, binary_op);
    *p2 = m_sqrt(result.variance());
}

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_rms_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2, uint64_t n)
{
    dtype init = m_zero;
    dtype result;
    result = thrust::transform_reduce(thrust::cuda::par, p1_begin, p1_end, thrust_square(), init, thrust::plus<dtype>());
    *p2 = m_sqrt(m_div(result,n));
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_mean_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_mean_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2, n);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_mean_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2, n);
    }
}

void cumo_<%=type_name%>_var_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_var_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_var_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2);
    }
}

void cumo_<%=type_name%>_stddev_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_stddev_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_stddev_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2);
    }
}

void cumo_<%=type_name%>_rms_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_rms_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2, n);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_rms_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2, n);
    }
}
