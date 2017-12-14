__global__ void <%="#{c_iter}_index_kernel"%>(char *ptr, size_t *idx, dtype val, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        *(dtype*)(ptr + idx[i]) = val;
    }
}

__global__ void <%="#{c_iter}_stride_kernel"%>(char*ptr, ssize_t step, dtype val, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        *(dtype*)(ptr + (i*step)) = val;
    }
}

void <%="#{c_iter}_kernel_index_launch"%>(char *ptr, size_t *idx, dtype val, size_t N)
{
    size_t gridDim = get_gridDim(N);
    size_t blockDim = get_blockDim(N);
    <%="#{c_iter}_index_kernel"%><<<gridDim, blockDim>>>(ptr,idx,val,N);
}

void <%="#{c_iter}_kernel_stride_launch"%>(char *ptr, ssize_t step, dtype val, size_t N)
{
    size_t gridDim = get_gridDim(N);
    size_t blockDim = get_blockDim(N);
    <%="#{c_iter}_stride_kernel"%><<<gridDim, blockDim>>>(ptr,step,val,N);
}
