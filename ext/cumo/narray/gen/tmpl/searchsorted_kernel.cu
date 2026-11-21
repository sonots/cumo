// numpy's order: NaN after every number, and equal to another NaN.
#define CUMO_SEARCHSORTED_LT(x, y) (m_lt(x, y) || (m_isnan(y) && !m_isnan(x)))

__global__ void cumo_<%=type_name%>_searchsorted_kernel(const dtype *a, uint64_t n, const int64_t *sorter, const dtype *v, int64_t *out, uint64_t m, int right, int *oor)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m; i += blockDim.x * gridDim.x) {
        dtype x = v[i];
        uint64_t lo = 0, hi = n;

        while (lo < hi) {
            uint64_t mid = lo + (hi - lo) / 2;
            dtype y;

            if (sorter) {
                int64_t s = sorter[mid];
                if (s < 0 || (uint64_t)s >= n) {
                    *oor = 1;
                    break;
                }
                y = a[s];
            } else {
                y = a[mid];
            }
            if (right ? !CUMO_SEARCHSORTED_LT(x, y) : CUMO_SEARCHSORTED_LT(y, x)) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[i] = (int64_t)lo;
    }
}

#undef CUMO_SEARCHSORTED_LT

void cumo_<%=type_name%>_searchsorted_kernel_launch(const dtype *a, uint64_t n, const int64_t *sorter, const dtype *v, int64_t *out, uint64_t m, int right, int *oor)
{
    size_t grid_dim = cumo_get_grid_dim(m);
    size_t block_dim = cumo_get_block_dim(m);

    cumo_<%=type_name%>_searchsorted_kernel<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(a, n, sorter, v, out, m, right, oor);
    cumo_cuda_runtime_check_kernel_launch();
}
