#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

struct cumo_<%=type_name%>_sum_impl {
    __device__ <%=dtype%> Identity() { return m_zero; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, <%=dtype%>& accum) { accum = m_add(next, accum); }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

struct cumo_<%=type_name%>_prod_impl {
    __device__ <%=dtype%> Identity() { return m_one; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, <%=dtype%>& accum) { accum = m_mul(next, accum); }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_mean_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2, uint64_t n)
{
    dtype init = m_zero;
    dtype sum = thrust::reduce(thrust::cuda::par, p1_begin, p1_end, init, cumo_thrust_plus());
    *p2 = c_div_r(sum, n);
}

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_var_kernel(Iterator1 p1_begin, Iterator1 p1_end, rtype* p2)
{
    cumo_thrust_complex_variance_unary_op<dtype, rtype>  unary_op;
    cumo_thrust_complex_variance_binary_op<dtype, rtype> binary_op;
    cumo_thrust_complex_variance_data<dtype, rtype> init = {};
    cumo_thrust_complex_variance_data<dtype, rtype> result;
    result = thrust::transform_reduce(thrust::cuda::par, p1_begin, p1_end, unary_op, init, binary_op);
    *p2 = result.variance();
}

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_stddev_kernel(Iterator1 p1_begin, Iterator1 p1_end, rtype* p2)
{
    cumo_thrust_complex_variance_unary_op<dtype, rtype>  unary_op;
    cumo_thrust_complex_variance_binary_op<dtype, rtype> binary_op;
    cumo_thrust_complex_variance_data<dtype, rtype> init = {};
    cumo_thrust_complex_variance_data<dtype, rtype> result;
    result = thrust::transform_reduce(thrust::cuda::par, p1_begin, p1_end, unary_op, init, binary_op);
    *p2 = r_sqrt(result.variance());
}

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_rms_kernel(Iterator1 p1_begin, Iterator1 p1_end, rtype* p2, uint64_t n)
{
    rtype init = 0;
    rtype result;
    result = thrust::transform_reduce(thrust::cuda::par, p1_begin, p1_end, cumo_thrust_square(), init, thrust::plus<rtype>());
    *p2 = r_sqrt(result/n);
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_sum_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, <%=dtype%>, cumo_<%=type_name%>_sum_impl>(*arg, cumo_<%=type_name%>_sum_impl{});
}

void cumo_<%=type_name%>_prod_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, <%=dtype%>, cumo_<%=type_name%>_prod_impl>(*arg, cumo_<%=type_name%>_prod_impl{});
}

void cumo_<%=type_name%>_mean_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_mean_kernel<<<1,1>>>(data_begin, data_end, (dtype*)p2, n);
    } else {
        cumo_thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_mean_kernel<<<1,1>>>(range.begin(), range.end(), (dtype*)p2, n);
    }
}

void cumo_<%=type_name%>_var_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_var_kernel<<<1,1>>>(data_begin, data_end, (rtype*)p2);
    } else {
        cumo_thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_var_kernel<<<1,1>>>(range.begin(), range.end(), (rtype*)p2);
    }
}

void cumo_<%=type_name%>_stddev_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_stddev_kernel<<<1,1>>>(data_begin, data_end, (rtype*)p2);
    } else {
        cumo_thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_stddev_kernel<<<1,1>>>(range.begin(), range.end(), (rtype*)p2);
    }
}

void cumo_<%=type_name%>_rms_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_rms_kernel<<<1,1>>>(data_begin, data_end, (rtype*)p2, n);
    } else {
        cumo_thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_rms_kernel<<<1,1>>>(range.begin(), range.end(), (rtype*)p2, n);
    }
}

