#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

struct cumo_<%=type_name%>_sum_impl {
    __device__ <%=dtype%> Identity(int64_t /*index*/) { return m_zero; }
    __device__ <%=dtype%> MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(<%=dtype%> next, <%=dtype%>& accum) { accum = m_add(next, accum); }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

struct cumo_<%=type_name%>_prod_impl {
    __device__ <%=dtype%> Identity(int64_t /*index*/) { return m_one; }
    __device__ <%=dtype%> MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(<%=dtype%> next, <%=dtype%>& accum) { accum = m_mul(next, accum); }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

// Chan's parallel combination of a running count, mean and sum of squared
// deviations, which is what the thrust functor these replace accumulated. The
// mean stays complex and only the deviation collapses to a magnitude. The
// guards are what make Identity neutral: combining two of them would divide
// 0 by 0 and leave the accumulator NaN.
struct cumo_<%=type_name%>_moments_impl {
    struct Moments {
        rtype n;
        dtype mean;
        rtype m2;
    };
    __device__ Moments Identity(int64_t /*index*/) { return {0, m_zero, 0}; }
    __device__ Moments MapIn(dtype in, int64_t /*index*/) { return {1, in, 0}; }
    __device__ void Reduce(Moments next, Moments& accum) {
        if (next.n == 0) { return; }
        if (accum.n == 0) { accum = next; return; }
        rtype n = accum.n + next.n;
        dtype delta = c_sub(next.mean, accum.mean);
        accum.mean = c_add(accum.mean, c_mul_r(delta, next.n / n));
        accum.m2 += next.m2 + c_abs_square(delta) * accum.n * next.n / n;
        accum.n = n;
    }
};

struct cumo_<%=type_name%>_var_impl : cumo_<%=type_name%>_moments_impl {
    __device__ rtype MapOut(Moments accum) { return accum.m2 / (accum.n - 1); }
};

struct cumo_<%=type_name%>_stddev_impl : cumo_<%=type_name%>_moments_impl {
    __device__ rtype MapOut(Moments accum) { return r_sqrt(accum.m2 / (accum.n - 1)); }
};

struct cumo_<%=type_name%>_mean_impl {
    // The reduce axis is the same length for every output, so the divisor is a
    // constant the launcher already knows rather than a count the tree carries.
    rtype n;
    __device__ dtype Identity(int64_t /*index*/) { return m_zero; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = m_add(next, accum); }
    __device__ dtype MapOut(dtype accum) { return c_div_r(accum, n); }
};

struct cumo_<%=type_name%>_rms_impl {
    rtype n;
    __device__ rtype Identity(int64_t /*index*/) { return 0; }
    __device__ rtype MapIn(dtype in, int64_t /*index*/) { return c_abs_square(in); }
    __device__ void Reduce(rtype next, rtype& accum) { accum += next; }
    __device__ rtype MapOut(rtype accum) { return r_sqrt(accum / n); }
};

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_sum_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, <%=dtype%>, cumo_<%=type_name%>_sum_impl>(*arg, cumo_<%=type_name%>_sum_impl{});
}

void cumo_<%=type_name%>_prod_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, <%=dtype%>, cumo_<%=type_name%>_prod_impl>(*arg, cumo_<%=type_name%>_prod_impl{});
}

void cumo_<%=type_name%>_mean_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    rtype n = (rtype)(arg->in_indexer.total_size / arg->out_indexer.total_size);
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_mean_impl>(*arg, cumo_<%=type_name%>_mean_impl{n});
}

void cumo_<%=type_name%>_var_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, rtype, cumo_<%=type_name%>_var_impl>(*arg, cumo_<%=type_name%>_var_impl{});
}

void cumo_<%=type_name%>_stddev_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, rtype, cumo_<%=type_name%>_stddev_impl>(*arg, cumo_<%=type_name%>_stddev_impl{});
}

void cumo_<%=type_name%>_rms_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    rtype n = (rtype)(arg->in_indexer.total_size / arg->out_indexer.total_size);
    cumo_reduce_split<dtype, rtype, cumo_<%=type_name%>_rms_impl>(*arg, cumo_<%=type_name%>_rms_impl{n});
}
