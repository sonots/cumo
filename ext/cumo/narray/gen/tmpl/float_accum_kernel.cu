<% f = File.join(File.dirname(__FILE__), 'real_accum_kernel.cu'); ERB.new(File.read(f)).tap {|erb| erb.filename = f }.result(binding) %>

dtype <%=type_name%>_mean_kernel_launch(size_t n, char *p, ssize_t stride)
{
    dtype sum = <%=type_name%>_sum_kernel_launch(n, p, stride);
    return sum / (dtype)n;
}

dtype <%=type_name%>_var_kernel_launch(size_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    thrust_variance_unary_op<dtype>  unary_op;
    thrust_variance_binary_op<dtype> binary_op;
    thrust_variance_data<dtype> init = {};
    thrust_variance_data<dtype> result;
    if (stride_idx == 1) {
        result = thrust::transform_reduce(data_begin, data_end, unary_op, init, binary_op);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        result = thrust::transform_reduce(range.begin(), range.end(), unary_op, init, binary_op);
    }
    return result.variance();
}

dtype <%=type_name%>_stddev_kernel_launch(size_t n, char *p, ssize_t stride)
{
    return m_sqrt(<%=type_name%>_var_kernel_launch(n, p, stride));
}

struct thrust_square : public thrust::unary_function<dtype, dtype>
{
    __host__ __device__ dtype operator()(const dtype& x) const { return x * x; }
};
dtype <%=type_name%>_rms_kernel_launch(size_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    dtype init = m_zero;
    dtype result;
    if (stride_idx == 1) {
        result = thrust::transform_reduce(data_begin, data_end, thrust_square(), init, thrust::plus<dtype>());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        result = thrust::transform_reduce(range.begin(), range.end(), thrust_square(), init, thrust::plus<dtype>());
    }
    return m_sqrt(m_div(result,n));
}
