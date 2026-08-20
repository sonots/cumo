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

// The nan-aware forms. A NaN maps to an accumulator that contributes nothing:
// Reduce above already returns early on a zero count, so the moments never see
// it, and mean and rms carry the count of what was left rather than the length
// of the reduce axis.
struct cumo_<%=type_name%>_moments_nan_impl : cumo_<%=type_name%>_moments_impl {
    __device__ Moments MapIn(dtype in, int64_t /*index*/) {
        if (!not_nan(in)) { return {0, m_zero, 0}; }
        return {1, in, 0};
    }
};

struct cumo_<%=type_name%>_var_nan_impl : cumo_<%=type_name%>_moments_nan_impl {
    // Every element being NaN leaves no count. numo divides by count-1 in
    // unsigned arithmetic there, so it answers +0 rather than the -0 that
    // dividing by -1 would give.
    __device__ rtype MapOut(Moments accum) { return accum.n == 0 ? 0 : accum.m2 / (accum.n - 1); }
};

struct cumo_<%=type_name%>_stddev_nan_impl : cumo_<%=type_name%>_moments_nan_impl {
    __device__ rtype MapOut(Moments accum) { return accum.n == 0 ? 0 : m_sqrt(accum.m2 / (accum.n - 1)); }
};

struct cumo_<%=type_name%>_mean_nan_impl {
    struct SumAndCount {
        dtype sum;
        rtype n;
    };
    __device__ SumAndCount Identity(int64_t /*index*/) { return {m_zero, 0}; }
    __device__ SumAndCount MapIn(dtype in, int64_t /*index*/) {
        if (!not_nan(in)) { return {m_zero, 0}; }
        return {in, 1};
    }
    __device__ void Reduce(SumAndCount next, SumAndCount& accum) {
        accum.sum = m_add(next.sum, accum.sum);
        accum.n += next.n;
    }
    __device__ dtype MapOut(SumAndCount accum) { return m_div(accum.sum, m_from_real(accum.n)); }
};

struct cumo_<%=type_name%>_rms_nan_impl {
    struct SumAndCount {
        rtype sum;
        rtype n;
    };
    __device__ SumAndCount Identity(int64_t /*index*/) { return {0, 0}; }
    __device__ SumAndCount MapIn(dtype in, int64_t /*index*/) {
        if (!not_nan(in)) { return {0, 0}; }
        return {m_square(m_abs(in)), 1};
    }
    __device__ void Reduce(SumAndCount next, SumAndCount& accum) {
        accum.sum += next.sum;
        accum.n += next.n;
    }
    __device__ rtype MapOut(SumAndCount accum) { return m_sqrt(accum.sum / accum.n); }
};

<% if is_double_precision %>
// Kahan-Babuska-Neumaier. The compensation term is what makes this associative:
// a partial carries the running sum and everything that sum has lost, and two
// partials combine by adding what their own addition loses to both. numo folds
// the correction back into the sum each step, which a tree cannot do, so the
// last digits differ; the error against an exact sum is not worse.
struct cumo_<%=type_name%>_kahan_sum_impl {
    struct SumAndComp {
        dtype sum;
        dtype comp;
    };
    __device__ SumAndComp Identity(int64_t /*index*/) { return {m_zero, m_zero}; }
    __device__ SumAndComp MapIn(dtype in, int64_t /*index*/) { return {in, m_zero}; }
    __device__ void Reduce(SumAndComp next, SumAndComp& accum) {
        dtype t = accum.sum + next.sum;
        dtype c = (fabs(accum.sum) >= fabs(next.sum))
                ? (accum.sum - t) + next.sum
                : (next.sum - t) + accum.sum;
        accum.comp += next.comp + c;
        accum.sum = t;
    }
    __device__ dtype MapOut(SumAndComp accum) { return accum.sum + accum.comp; }
};

struct cumo_<%=type_name%>_kahan_sum_nan_impl : cumo_<%=type_name%>_kahan_sum_impl {
    __device__ SumAndComp MapIn(dtype in, int64_t /*index*/) {
        if (!not_nan(in)) { return {m_zero, m_zero}; }
        return {in, m_zero};
    }
};
<% end %>

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

void cumo_<%=type_name%>_mean_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_mean_nan_impl>(*arg, cumo_<%=type_name%>_mean_nan_impl{});
}

void cumo_<%=type_name%>_var_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, rtype, cumo_<%=type_name%>_var_nan_impl>(*arg, cumo_<%=type_name%>_var_nan_impl{});
}

void cumo_<%=type_name%>_stddev_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, rtype, cumo_<%=type_name%>_stddev_nan_impl>(*arg, cumo_<%=type_name%>_stddev_nan_impl{});
}

void cumo_<%=type_name%>_rms_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, rtype, cumo_<%=type_name%>_rms_nan_impl>(*arg, cumo_<%=type_name%>_rms_nan_impl{});
}
<% if is_double_precision %>

void cumo_<%=type_name%>_kahan_sum_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_kahan_sum_impl>(*arg, cumo_<%=type_name%>_kahan_sum_impl{});
}

void cumo_<%=type_name%>_kahan_sum_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_kahan_sum_nan_impl>(*arg, cumo_<%=type_name%>_kahan_sum_nan_impl{});
}
<% end %>
