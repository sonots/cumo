<% if is_int && !is_object %>
typedef double seq_data_t;
<% else %>
typedef dtype seq_data_t;
<% end %>

<% if is_object %>
typedef size_t seq_count_t;
<% else %>
typedef double seq_count_t;
<% end %>

<% unless is_object %>
__global__ void <%="cumo_#{c_iter}_index_kernel"%>(char *p1, size_t* idx1, seq_data_t beg, seq_data_t step, seq_count_t c, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = f_seq(beg,step,c+i);
        *(dtype*)(p1+idx1[i]) = x;
    }
}

__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char *p1, size_t s1, seq_data_t beg, seq_data_t step, seq_count_t c, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = f_seq(beg,step,c+i);
        *(dtype*)(p1+(i*s1)) = x;
    }
}

void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, size_t* idx1, seq_data_t beg, seq_data_t step, seq_count_t c, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_index_kernel"%><<<gridDim, blockDim>>>(p1,idx1,beg,step,c,n);
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, ssize_t s1, seq_data_t beg, seq_data_t step, seq_count_t c, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<gridDim, blockDim>>>(p1,s1,beg,step,c,n);
}
<% end %>
