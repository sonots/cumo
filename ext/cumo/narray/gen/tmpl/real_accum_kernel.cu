#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

<% acc = is_half ? 'float' : dtype %>
<% to_acc = is_half ? 'cumo_half2float' : '' %>
<% from_acc = is_half ? 'cumo_float2half' : '' %>
<% acc_zero = is_half ? '0.0f' : 'm_zero' %>


// Reduce also combines two accumulators, so next has to be the accumulator type:
// the integer types widen to 64 bits and half to float, and taking dtype there
// truncates every partial the shared-memory tree merges.
struct cumo_<%=type_name%>_sum_impl {
    __device__ <%=acc%> Identity(int64_t /*index*/) { return <%=acc_zero%>; }
    __device__ <%=acc%> MapIn(dtype in, int64_t /*index*/) { return <%=to_acc%>(in); }
    __device__ void Reduce(<%=acc%> next, <%=acc%>& accum) { accum = accum + next; }
    __device__ <%=dtype%> MapOut(<%=acc%> accum) { return <%=from_acc%>(accum); }
};

struct cumo_<%=type_name%>_prod_impl {
    __device__ <%=dtype%> Identity(int64_t /*index*/) { return m_one; }
    __device__ <%=dtype%> MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(<%=dtype%> next, <%=dtype%>& accum) { accum = m_mul(accum, next); }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

// One set of rules answers for both components, and min, max, minmax and ptp
// all name a whole set rather than a rule each. A pair that mixes two sets --
// the nan-aware min beside the default max -- is then not expressible, which
// is the shape #372 had: #219 changed the identity in min and max and left the
// copies inside the fused pair behind.
struct cumo_<%=type_name%>_extremum_rules {
    struct Min {
<% if is_float %>
        // A NaN loses the comparison, so it is skipped unless the accumulator
        // is still the identity: the reduction then answers NaN only when every
        // element was NaN, and equal elements keep the earlier one as numo does.
        __device__ static dtype Identity() { return (dtype)nan(""); }
        __device__ static void Reduce(dtype next, dtype& accum) { if (m_lt(next, accum) || !not_nan(accum)) { accum = next; } }
<% else %>
        __device__ static dtype Identity() { return DATA_MAX; }
        __device__ static void Reduce(dtype next, dtype& accum) { accum = m_lt(next, accum) ? next : accum; }
<% end %>
    };
    struct Max {
<% if is_float %>
        __device__ static dtype Identity() { return (dtype)nan(""); }
        __device__ static void Reduce(dtype next, dtype& accum) { if (m_lt(accum, next) || !not_nan(accum)) { accum = next; } }
<% else %>
        __device__ static dtype Identity() { return DATA_MIN; }
        __device__ static void Reduce(dtype next, dtype& accum) { accum = m_lt(next, accum) ? accum : next; }
<% end %>
    };
};

template <typename Rule>
struct cumo_<%=type_name%>_extremum_of {
    __device__ dtype Identity(int64_t /*index*/) { return Rule::Identity(); }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { Rule::Reduce(next, accum); }
    __device__ dtype MapOut(dtype accum) { return accum; }
};

// What minmax and ptp accumulate, and how it goes in and out. No Identity and
// no Reduce: those come from the rules, so a form that names none does not
// compile.
struct cumo_<%=type_name%>_minmax_pair {
    struct MinAndMax {
        dtype min;
        dtype max;
    };
    __device__ MinAndMax MapIn(dtype in, int64_t /*index*/) { return {in, in}; }
    __device__ void MapOut(MinAndMax accum, dtype* out_min, dtype* out_max) {
        *out_min = accum.min;
        *out_max = accum.max;
    }
};

// min and max fused, so that minmax reads the input once.
template <typename Rules>
struct cumo_<%=type_name%>_minmax_of : cumo_<%=type_name%>_minmax_pair {
    __device__ MinAndMax Identity(int64_t /*index*/) { return {Rules::Min::Identity(), Rules::Max::Identity()}; }
    __device__ void Reduce(MinAndMax next, MinAndMax& accum) {
        Rules::Min::Reduce(next.min, accum.min);
        Rules::Max::Reduce(next.max, accum.max);
    }
};

// ptp is its minmax with the pair subtracted. The one-argument MapOut hides
// the two-output one it inherits.
template <typename Rules>
struct cumo_<%=type_name%>_ptp_of : cumo_<%=type_name%>_minmax_of<Rules> {
    __device__ dtype MapOut(cumo_<%=type_name%>_minmax_pair::MinAndMax accum) { return m_sub(accum.max, accum.min); }
};

using cumo_<%=type_name%>_min_impl = cumo_<%=type_name%>_extremum_of<cumo_<%=type_name%>_extremum_rules::Min>;
using cumo_<%=type_name%>_max_impl = cumo_<%=type_name%>_extremum_of<cumo_<%=type_name%>_extremum_rules::Max>;
using cumo_<%=type_name%>_minmax_impl = cumo_<%=type_name%>_minmax_of<cumo_<%=type_name%>_extremum_rules>;
using cumo_<%=type_name%>_ptp_impl = cumo_<%=type_name%>_ptp_of<cumo_<%=type_name%>_extremum_rules>;

<% unless is_float %>
// mean, var, stddev and rms answer in double for the integer types, so they
// accumulate in double too rather than in dtype.
struct cumo_<%=type_name%>_moments_impl {
    struct Moments {
        double n;
        double mean;
        double m2;
    };
    __device__ Moments Identity(int64_t /*index*/) { return {0, 0, 0}; }
    __device__ Moments MapIn(dtype in, int64_t /*index*/) { return {1, (double)in, 0}; }
    __device__ void Reduce(Moments next, Moments& accum) {
        if (next.n == 0) { return; }
        if (accum.n == 0) { accum = next; return; }
        double n = accum.n + next.n;
        double delta = next.mean - accum.mean;
        accum.mean += delta * (next.n / n);
        accum.m2 += next.m2 + delta * delta * accum.n * next.n / n;
        accum.n = n;
    }
};

struct cumo_<%=type_name%>_var_impl : cumo_<%=type_name%>_moments_impl {
    __device__ double MapOut(Moments accum) { return accum.m2 / (accum.n - 1); }
};

struct cumo_<%=type_name%>_stddev_impl : cumo_<%=type_name%>_moments_impl {
    __device__ double MapOut(Moments accum) { return sqrt(accum.m2 / (accum.n - 1)); }
};

struct cumo_<%=type_name%>_mean_impl {
    // The reduce axis is the same length for every output, so the divisor is a
    // constant the launcher already knows rather than a count the tree carries.
    double n;
    __device__ double Identity(int64_t /*index*/) { return 0; }
    __device__ double MapIn(dtype in, int64_t /*index*/) { return (double)in; }
    __device__ void Reduce(double next, double& accum) { accum += next; }
    __device__ double MapOut(double accum) { return accum / n; }
};

struct cumo_<%=type_name%>_rms_impl {
    double n;
    __device__ double Identity(int64_t /*index*/) { return 0; }
    __device__ double MapIn(dtype in, int64_t /*index*/) { double x = (double)in; return x * x; }
    __device__ void Reduce(double next, double& accum) { accum += next; }
    __device__ double MapOut(double accum) { return sqrt(accum / n); }
};
<% end %>

<% if is_float %>
// The nan-aware forms of the reductions above. sum, prod and their kin skip a
// NaN, so it maps to the identity and the tree never sees it. min and max go
// the other way: numo answers NaN as soon as one element is NaN, and a NaN
// absorbs in Reduce because it loses every comparison.
//
// min and max take an infinity for their identity rather than the largest
// finite value, so that an element which is itself an infinity still wins.
// numo seeds from the first element and never faces the question.
struct cumo_<%=type_name%>_sum_nan_impl {
    __device__ <%=acc%> Identity(int64_t /*index*/) { return <%=acc_zero%>; }
    __device__ <%=acc%> MapIn(dtype in, int64_t /*index*/) { return not_nan(in) ? <%=to_acc%>(in) : <%=acc_zero%>; }
    __device__ void Reduce(<%=acc%> next, <%=acc%>& accum) { accum = next + accum; }
    __device__ dtype MapOut(<%=acc%> accum) { return <%=from_acc%>(accum); }
};

struct cumo_<%=type_name%>_prod_nan_impl {
    __device__ dtype Identity(int64_t /*index*/) { return m_one; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return not_nan(in) ? in : m_one; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = m_mul(next, accum); }
    __device__ dtype MapOut(dtype accum) { return accum; }
};

struct cumo_<%=type_name%>_extremum_nan_rules {
    struct Min {
        __device__ static dtype Identity() { return (dtype)INFINITY; }
        __device__ static void Reduce(dtype next, dtype& accum) { if (!not_nan(next) || m_lt(next, accum)) { accum = next; } }
    };
    struct Max {
        __device__ static dtype Identity() { return (dtype)(-INFINITY); }
        __device__ static void Reduce(dtype next, dtype& accum) { if (!not_nan(next) || m_lt(accum, next)) { accum = next; } }
    };
};

using cumo_<%=type_name%>_min_nan_impl = cumo_<%=type_name%>_extremum_of<cumo_<%=type_name%>_extremum_nan_rules::Min>;
using cumo_<%=type_name%>_max_nan_impl = cumo_<%=type_name%>_extremum_of<cumo_<%=type_name%>_extremum_nan_rules::Max>;
using cumo_<%=type_name%>_minmax_nan_impl = cumo_<%=type_name%>_minmax_of<cumo_<%=type_name%>_extremum_nan_rules>;
using cumo_<%=type_name%>_ptp_nan_impl = cumo_<%=type_name%>_ptp_of<cumo_<%=type_name%>_extremum_nan_rules>;
<% end %>

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

void cumo_<%=type_name%>_min_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_min_impl>(*arg, cumo_<%=type_name%>_min_impl{});
}

void cumo_<%=type_name%>_max_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_max_impl>(*arg, cumo_<%=type_name%>_max_impl{});
}

void cumo_<%=type_name%>_ptp_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_ptp_impl>(*arg, cumo_<%=type_name%>_ptp_impl{});
}

void cumo_<%=type_name%>_minmax_kernel_launch(cumo_na_reduction_arg_t* arg, cumo_na_iarray_t* out2)
{
    cumo_reduce_pair_split<dtype, dtype, cumo_<%=type_name%>_minmax_impl>(*arg, *out2, cumo_<%=type_name%>_minmax_impl{});
}
<% unless is_float %>

void cumo_<%=type_name%>_mean_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    double n = (double)(arg->in_indexer.total_size / arg->out_indexer.total_size);
    cumo_reduce_split<dtype, double, cumo_<%=type_name%>_mean_impl>(*arg, cumo_<%=type_name%>_mean_impl{n});
}

void cumo_<%=type_name%>_var_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, double, cumo_<%=type_name%>_var_impl>(*arg, cumo_<%=type_name%>_var_impl{});
}

void cumo_<%=type_name%>_stddev_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, double, cumo_<%=type_name%>_stddev_impl>(*arg, cumo_<%=type_name%>_stddev_impl{});
}

void cumo_<%=type_name%>_rms_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    double n = (double)(arg->in_indexer.total_size / arg->out_indexer.total_size);
    cumo_reduce_split<dtype, double, cumo_<%=type_name%>_rms_impl>(*arg, cumo_<%=type_name%>_rms_impl{n});
}
<% end %>
<% if is_float %>

void cumo_<%=type_name%>_sum_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_sum_nan_impl>(*arg, cumo_<%=type_name%>_sum_nan_impl{});
}

void cumo_<%=type_name%>_prod_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_prod_nan_impl>(*arg, cumo_<%=type_name%>_prod_nan_impl{});
}

void cumo_<%=type_name%>_min_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_min_nan_impl>(*arg, cumo_<%=type_name%>_min_nan_impl{});
}

void cumo_<%=type_name%>_max_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_max_nan_impl>(*arg, cumo_<%=type_name%>_max_nan_impl{});
}

void cumo_<%=type_name%>_ptp_nan_kernel_launch(cumo_na_reduction_arg_t* arg)
{
    cumo_reduce_split<dtype, dtype, cumo_<%=type_name%>_ptp_nan_impl>(*arg, cumo_<%=type_name%>_ptp_nan_impl{});
}

void cumo_<%=type_name%>_minmax_nan_kernel_launch(cumo_na_reduction_arg_t* arg, cumo_na_iarray_t* out2)
{
    cumo_reduce_pair_split<dtype, dtype, cumo_<%=type_name%>_minmax_nan_impl>(*arg, *out2, cumo_<%=type_name%>_minmax_nan_impl{});
}
<% end %>
