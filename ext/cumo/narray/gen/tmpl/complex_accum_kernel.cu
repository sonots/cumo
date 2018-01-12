struct thrust_plus : public thrust::binary_function<dtype, dtype, dtype>
{
    __host__ __device__ dtype operator()(dtype x, dtype y) { return m_add(x,y); }
};
dtype <%=type_name%>_sum_kernel_launch(size_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    dtype init = m_zero;
    if (stride_idx == 1) {
        return thrust::reduce(data_begin, data_end, init, thrust_plus());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        return thrust::reduce(range.begin(), range.end(), init, thrust_plus());
    }
}

struct thrust_multiplies : public thrust::binary_function<dtype, dtype, dtype>
{
    __host__ __device__ dtype operator()(dtype x, dtype y) { return m_mul(x,y); }
};
dtype <%=type_name%>_prod_kernel_launch(size_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    dtype init = m_one;
    if (stride_idx == 1) {
        return thrust::reduce(data_begin, data_end, init, thrust_multiplies());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        return thrust::reduce(range.begin(), range.end(), init, thrust_multiplies());
    }
}

dtype <%=type_name%>_mean_kernel_launch(size_t n, char *p, ssize_t stride)
{
    dtype sum = <%=type_name%>_sum_kernel_launch(n, p, stride);
    return c_div_r(sum, n);
}

// ref. https://github.com/thrust/thrust/blob/master/examples/summary_statistics.cu

// structure used to accumulate the moments and other
// statistical properties encountered so far.
struct thrust_complex_variance_data
{
    rtype n;
    dtype mean;
    rtype M2;

    // initialize to the identity element
    void initialize()
    {
        n = M2 = 0;
        mean = c_zero();
    }

    rtype variance()   { return M2 / (n - 1); }
    rtype variance_n() { return M2 / n; }
};

// stats_unary_op is a functor that takes in a value x and
// returns a variace_data whose mean value is initialized to x.
struct thrust_complex_variance_unary_op
{
    __host__ __device__
    thrust_complex_variance_data operator()(const dtype& x) const
    {
         thrust_complex_variance_data result;
         result.n    = 1;
         result.mean = x;
         result.M2   = 0;

         return result;
    }
};

// thrust_variance_binary_op is a functor that accepts two thrust_variance_data
// structs and returns a new thrust_variance_data which are an
// approximation to the thrust_variance for
// all values that have been agregated so far
struct thrust_complex_variance_binary_op
    : public thrust::binary_function<const thrust_complex_variance_data&,
                                     const thrust_complex_variance_data&,
                                           thrust_complex_variance_data >
{
    __host__ __device__
    thrust_complex_variance_data operator()(const thrust_complex_variance_data& x, const thrust_complex_variance_data& y) const
    {
        thrust_complex_variance_data result;

        // precompute some common subexpressions
        rtype n  = x.n + y.n;

        dtype delta = c_sub(y.mean, x.mean);
        rtype delta2 = c_abs_square(delta);

        //Basic number of samples (n)
        result.n = n;

        result.mean = c_mul_r(c_add(x.mean, delta), y.n / n);

        result.M2 = x.M2 + y.M2;
        result.M2 += delta2 * x.n * y.n / n;

        return result;
    }
};

rtype <%=type_name%>_var_kernel_launch(size_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    thrust_complex_variance_unary_op  unary_op;
    thrust_complex_variance_binary_op binary_op;
    thrust_complex_variance_data init = {};
    thrust_complex_variance_data result;
    if (stride_idx == 1) {
        result = thrust::transform_reduce(data_begin, data_end, unary_op, init, binary_op);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        result = thrust::transform_reduce(range.begin(), range.end(), unary_op, init, binary_op);
    }
    return result.variance();
}

rtype <%=type_name%>_stddev_kernel_launch(size_t n, char *p, ssize_t stride)
{
    return r_sqrt(<%=type_name%>_var_kernel_launch(n, p, stride));
}

struct thrust_square : public thrust::unary_function<dtype, dtype>
{
    __host__ __device__ rtype operator()(const dtype& x) const { return c_abs_square(x); }
};
rtype <%=type_name%>_rms_kernel_launch(size_t n, char *p, ssize_t stride)
{
    ssize_t stride_idx = stride / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * stride_idx);
    rtype init = 0;
    rtype result;
    if (stride_idx == 1) {
        result = thrust::transform_reduce(data_begin, data_end, thrust_square(), init, thrust::plus<rtype>());
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, stride_idx);
        result = thrust::transform_reduce(range.begin(), range.end(), thrust_square(), init, thrust::plus<rtype>());
    }
    return r_sqrt(result/n);
}
