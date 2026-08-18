<%= load_erb('real_accum').result(binding) %>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

// Chan's parallel combination of a running count, mean and sum of squared
// deviations, which is what the thrust functor these replace accumulated. numo
// takes two passes with the exact mean instead, so the last digits can differ.
// The guards are what make Identity neutral: combining two of them would
// divide 0 by 0 and leave the accumulator NaN.
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
        dtype delta = m_sub(next.mean, accum.mean);
        accum.mean = m_add(accum.mean, delta * (next.n / n));
        accum.m2 += next.m2 + delta * delta * accum.n * next.n / n;
        accum.n = n;
    }
};

struct cumo_<%=type_name%>_var_impl : cumo_<%=type_name%>_moments_impl {
    __device__ rtype MapOut(Moments accum) { return accum.m2 / (accum.n - 1); }
};

struct cumo_<%=type_name%>_stddev_impl : cumo_<%=type_name%>_moments_impl {
    __device__ rtype MapOut(Moments accum) { return m_sqrt(accum.m2 / (accum.n - 1)); }
};

struct cumo_<%=type_name%>_mean_impl {
    // The reduce axis is the same length for every output, so the divisor is a
    // constant the launcher already knows rather than a count the tree carries.
    dtype n;
    __device__ dtype Identity(int64_t /*index*/) { return m_zero; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = m_add(next, accum); }
    __device__ dtype MapOut(dtype accum) { return m_div(accum, n); }
};

struct cumo_<%=type_name%>_rms_impl {
    rtype n;
    __device__ rtype Identity(int64_t /*index*/) { return 0; }
    __device__ rtype MapIn(dtype in, int64_t /*index*/) { return m_square(m_abs(in)); }
    __device__ void Reduce(rtype next, rtype& accum) { accum += next; }
    __device__ rtype MapOut(rtype accum) { return m_sqrt(accum / n); }
};

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_mean_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    dtype n = (dtype)(arg->in_indexer.total_size / arg->out_indexer.total_size);
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
