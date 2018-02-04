<% unless is_object %>
__global__ void <%="#{c_iter}_index_kernel"%>(char *p1, size_t* idx1, seq_data_t beg, seq_data_t step, seq_data_t base, seq_count_t c, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = f_seq(beg,step,c+i);
        *(dtype*)(p1+idx1[i]) = m_pow(base,x);
    }
}

__global__ void <%="#{c_iter}_stride_kernel"%>(char *p1, size_t s1, seq_data_t beg, seq_data_t step, seq_data_t base, seq_count_t c, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = f_seq(beg,step,c+i);
        *(dtype*)(p1+(i*s1)) = m_pow(base,x);
    }
}

void <%="#{c_iter}_index_kernel_launch"%>(char *p1, size_t* idx1, seq_data_t beg, seq_data_t step, seq_data_t base, seq_count_t c, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="#{c_iter}_index_kernel"%><<<gridDim, blockDim>>>(p1,idx1,beg,step,base,c,n);
}

void <%="#{c_iter}_stride_kernel_launch"%>(char *p1, ssize_t s1, seq_data_t beg, seq_data_t step, seq_data_t base, seq_count_t c, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="#{c_iter}_stride_kernel"%><<<gridDim, blockDim>>>(p1,s1,beg,step,base,c,n);
}
<% end %>
