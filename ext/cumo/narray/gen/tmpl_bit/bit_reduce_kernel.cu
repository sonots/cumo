// The output starts out filled with <%=init_bit%> and an element can only ever
// flip it to the other value, so a store is idempotent and the racing threads of
// a launch need no ordering between them.

__global__ void <%="cumo_#{c_iter}_elementwise_kernel"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, size_t *idx1, CUMO_BIT_DIGIT *a2, size_t p2, ssize_t s2, size_t *idx2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        size_t q2 = cumo_bit_pos(p2,s2,idx2,i);
        CUMO_BIT_DIGIT x, y;
        CUMO_LOAD_BIT(a2, q2, y);
        if (y == <%=init_bit%>) {
            CUMO_LOAD_BIT(a1, cumo_bit_pos(p1,s1,idx1,i), x);
            if (x != <%=init_bit%>) {
                CUMO_STORE_BIT(a2, q2, x);
            }
        }
    }
}

// Every element answers the same output bit, so each block folds its threads
// down to one flag and at most one of them reaches the bit.
__device__ static void <%="cumo_#{c_iter}_store_flip"%>(int flip, CUMO_BIT_DIGIT *a2, size_t p2)
{
    if (__syncthreads_or(flip) && threadIdx.x == 0) {
        CUMO_BIT_DIGIT y;
        CUMO_LOAD_BIT(a2, p2, y);
        if (y == <%=init_bit%>) {
            CUMO_STORE_BIT(a2, p2, (CUMO_BIT_DIGIT)<%=1 - init_bit%>);
        }
    }
}

__global__ void <%="cumo_#{c_iter}_reduce_kernel"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, size_t *idx1, CUMO_BIT_DIGIT *a2, size_t p2, uint64_t n)
{
    int flip = 0;
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n && !flip; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x;
        CUMO_LOAD_BIT(a1, cumo_bit_pos(p1,s1,idx1,i), x);
        flip = (x != <%=init_bit%>);
    }
    <%="cumo_#{c_iter}_store_flip"%>(flip, a2, p2);
}

__global__ void <%="cumo_#{c_iter}_contiguous_reduce_kernel"%>(CUMO_BIT_DIGIT *a1, size_t p1, CUMO_BIT_DIGIT *a2, size_t p2, uint64_t n, uint64_t w1)
{
    const CUMO_BIT_DIGIT init_word = <%=init_bit%> ? CUMO_BALL : (CUMO_BIT_DIGIT)0;
    int flip = 0;
    for (uint64_t w = blockIdx.x * blockDim.x + threadIdx.x; w < w1 && !flip; w += blockDim.x * gridDim.x) {
        flip = ((a1[w] ^ init_word) & cumo_bit_word_mask(p1,n,w)) != 0;
    }
    <%="cumo_#{c_iter}_store_flip"%>(flip, a2, p2);
}

void <%="cumo_#{c_iter}_elementwise_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, size_t *idx1, CUMO_BIT_DIGIT *a2, size_t p2, ssize_t s2, size_t *idx2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_elementwise_kernel"%><<<grid_dim, block_dim>>>(a1,p1,s1,idx1,a2,p2,s2,idx2,n);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_reduce_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, size_t *idx1, CUMO_BIT_DIGIT *a2, size_t p2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_reduce_kernel"%><<<grid_dim, block_dim>>>(a1,p1,s1,idx1,a2,p2,n);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_contiguous_reduce_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, CUMO_BIT_DIGIT *a2, size_t p2, uint64_t n)
{
    uint64_t w1 = (p1 + n + CUMO_NB - 1) / CUMO_NB;
    size_t grid_dim = cumo_get_grid_dim(w1);
    size_t block_dim = cumo_get_block_dim(w1);
    <%="cumo_#{c_iter}_contiguous_reduce_kernel"%><<<grid_dim, block_dim>>>(a1,p1,a2,p2,n,w1);
    cumo_cuda_runtime_check_kernel_launch();
}
