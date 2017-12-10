__global__ void <%="#{c_iter}_kernel"%>(dtype *ptr, ssize_t step, dtype val, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x * step; i < N; i += blockDim.x * gridDim.x) {
        ptr[i] = val;
    }
}

void <%="#{c_iter}_kernel_launch"%>(dtype *ptr, ssize_t step, dtype val, size_t N)
{
    size_t maxBlockDim = 128;
    size_t gridDim = (N / maxBlockDim) + 1;
    size_t blockDim = (N > maxBlockDim) ? maxBlockDim : N;
    // ref. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    if (gridDim > 2147483647) gridDim = 2147483647;
    <%="#{c_iter}_kernel"%><<<gridDim, blockDim>>>(ptr,step,val,N);
}
