// atomicAdd on double arrived with sm_60. Before that the compare-and-swap loop
// the CUDA programming guide gives is the only way to get one, and the default
// target of the CUDA releases cumo builds against is still below it.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ static inline void <%=c_iter%>_add_double(double *address, double val)
{
    unsigned long long int *p = (unsigned long long int*)address;
    unsigned long long int old = *p, assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}
#else
__device__ static inline void <%=c_iter%>_add_double(double *address, double val) { atomicAdd(address, val); }
#endif

// Counting in shared memory first moves the atomics that pile onto a few bins
// inside each block, at the price of one add per bin per block at the end, so it
// pays once each bin gets a few dozen items and the bins fit in a block's share.
#define CUMO_BINCOUNT_SHARED_BLOCK 256
#define CUMO_BINCOUNT_SHARED_BYTES 49152
#define CUMO_BINCOUNT_ITEMS_PER_BIN 32

// A few bins take every add of a block, and contended double atomics in shared
// memory are slow, so a small histogram is kept once per warp and summed at the
// end. Small means the copies still leave room for as many blocks as before.
#define CUMO_BINCOUNT_COPY_BYTES 8192

static int
cumo_bincount_copies(uint64_t n, size_t bin_bytes)
{
    int copies = 1;
    while (copies * 2 <= CUMO_BINCOUNT_SHARED_BLOCK / 32 &&
           (size_t)n * bin_bytes * (size_t)(copies * 2) <= CUMO_BINCOUNT_COPY_BYTES) {
        copies *= 2;
    }
    return copies;
}

// The grid for the shared memory kernel, as many blocks as the device holds at
// once, or 0 when the global one is the better choice.
static size_t
cumo_bincount_shared_grid(const void *kernel, uint64_t count, uint64_t n, size_t bin_bytes)
{
    int dev, sms, per_sm;
    size_t smem = (size_t)n * bin_bytes, grid, need;

    if (n == 0 || smem > CUMO_BINCOUNT_SHARED_BYTES || count / CUMO_BINCOUNT_ITEMS_PER_BIN < n) {
        return 0;
    }
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess ||
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, CUMO_BINCOUNT_SHARED_BLOCK, smem) != cudaSuccess ||
        per_sm <= 0) {
        // The launch check after this reads the last error, which a failed
        // query would leave behind.
        (void)cudaGetLastError();
        return 0;
    }
    grid = (size_t)sms * (size_t)per_sm;
    need = (count + CUMO_BINCOUNT_SHARED_BLOCK - 1) / CUMO_BINCOUNT_SHARED_BLOCK;
    return (grid < need) ? grid : need;
}

// A kernel cannot raise, so an item that does not fit the output is reported
// through the flag and the value is left for the message. Any offending item
// will do, so the two stores race harmlessly.
#define cumo_check_bincount_item(x, n)  \
    if ((x) >= (n)) {                   \
        *item = (x);                    \
        *oor = 1;                       \
        continue;                       \
    }

<%
[["32","u_int32_t","unsigned int","1u"],
 ["64","u_int64_t","unsigned long long","1ull"]].each do |bits,cnt_type,atom_type,one|
%>
__global__ void <%="cumo_#{c_iter}_#{bits}_init_kernel"%>(char *p2, ssize_t s2, uint64_t n)
{
    for (uint64_t x = blockIdx.x * blockDim.x + threadIdx.x; x < n; x += blockDim.x * gridDim.x) {
        *(<%=cnt_type%>*)(p2 + s2 * x) = 0;
    }
}

__global__ void <%="cumo_#{c_iter}_#{bits}_kernel"%>(char *p1, ssize_t s1, size_t *idx1, uint64_t count, char *p2, ssize_t s2, uint64_t n, int *oor, size_t *item)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        size_t x = (size_t)(*(dtype*)(idx1 ? (p1 + idx1[i]) : (p1 + s1 * i)));
        cumo_check_bincount_item(x, n);
        atomicAdd((<%=atom_type%>*)(p2 + s2 * x), <%=one%>);
    }
}

__global__ void <%="cumo_#{c_iter}_#{bits}_shared_kernel"%>(char *p1, ssize_t s1, size_t *idx1, uint64_t count, char *p2, ssize_t s2, uint64_t n, int *oor, size_t *item, int copies)
{
    extern __shared__ <%=atom_type%> <%="cumo_#{c_iter}_#{bits}_bins"%>[];
    <%=atom_type%> *bins = <%="cumo_#{c_iter}_#{bits}_bins"%>;
    <%=atom_type%> *mine = bins + n * ((threadIdx.x / 32) % copies);
    for (uint64_t x = threadIdx.x; x < n * copies; x += blockDim.x) {
        bins[x] = 0;
    }
    __syncthreads();
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        size_t x = (size_t)(*(dtype*)(idx1 ? (p1 + idx1[i]) : (p1 + s1 * i)));
        cumo_check_bincount_item(x, n);
        atomicAdd(&mine[x], <%=one%>);
    }
    __syncthreads();
    for (uint64_t x = threadIdx.x; x < n; x += blockDim.x) {
        <%=atom_type%> c = 0;
        for (int k = 0; k < copies; k++) {
            c += bins[n * k + x];
        }
        if (c != 0) {
            atomicAdd((<%=atom_type%>*)(p2 + s2 * x), c);
        }
    }
}

void <%="cumo_#{c_iter}_#{bits}_kernel_launch"%>(char *p1, ssize_t s1, size_t *idx1, uint64_t count, char *p2, ssize_t s2, uint64_t n, int *oor, size_t *item)
{
    size_t grid;
    int copies;
    <%="cumo_#{c_iter}_#{bits}_init_kernel"%><<<cumo_get_grid_dim(n), cumo_get_block_dim(n), 0, cumo_cuda_stream()>>>(p2,s2,n);
    cumo_cuda_runtime_check_kernel_launch();
    copies = cumo_bincount_copies(n, sizeof(<%=atom_type%>));
    grid = cumo_bincount_shared_grid((const void*)<%="cumo_#{c_iter}_#{bits}_shared_kernel"%>, count, n, sizeof(<%=atom_type%>) * copies);
    if (grid > 0) {
        <%="cumo_#{c_iter}_#{bits}_shared_kernel"%><<<grid, CUMO_BINCOUNT_SHARED_BLOCK, n * copies * sizeof(<%=atom_type%>), cumo_cuda_stream()>>>(p1,s1,idx1,count,p2,s2,n,oor,item,copies);
    } else {
        <%="cumo_#{c_iter}_#{bits}_kernel"%><<<cumo_get_grid_dim(count), cumo_get_block_dim(count), 0, cumo_cuda_stream()>>>(p1,s1,idx1,count,p2,s2,n,oor,item);
    }
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>

<%
[["sf","float","atomicAdd"],
 ["df","double","#{c_iter}_add_double"]].each do |fn,cnt_type,add|
%>
__global__ void <%="cumo_#{c_iter}_#{fn}_init_kernel"%>(char *p3, ssize_t s3, uint64_t n)
{
    for (uint64_t x = blockIdx.x * blockDim.x + threadIdx.x; x < n; x += blockDim.x * gridDim.x) {
        *(<%=cnt_type%>*)(p3 + s3 * x) = 0;
    }
}

__global__ void <%="cumo_#{c_iter}_#{fn}_kernel"%>(char *p1, ssize_t s1, char *p2, ssize_t s2, uint64_t count, char *p3, ssize_t s3, uint64_t n, int *oor, size_t *item)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        size_t x = (size_t)(*(dtype*)(p1 + s1 * i));
        cumo_check_bincount_item(x, n);
        <%=add%>((<%=cnt_type%>*)(p3 + s3 * x), *(<%=cnt_type%>*)(p2 + s2 * i));
    }
}

__global__ void <%="cumo_#{c_iter}_#{fn}_shared_kernel"%>(char *p1, ssize_t s1, char *p2, ssize_t s2, uint64_t count, char *p3, ssize_t s3, uint64_t n, int *oor, size_t *item, int copies)
{
    extern __shared__ <%=cnt_type%> <%="cumo_#{c_iter}_#{fn}_bins"%>[];
    <%=cnt_type%> *bins = <%="cumo_#{c_iter}_#{fn}_bins"%>;
    <%=cnt_type%> *mine = bins + n * ((threadIdx.x / 32) % copies);
    for (uint64_t x = threadIdx.x; x < n * copies; x += blockDim.x) {
        bins[x] = 0;
    }
    __syncthreads();
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        size_t x = (size_t)(*(dtype*)(p1 + s1 * i));
        cumo_check_bincount_item(x, n);
        <%=add%>(&mine[x], *(<%=cnt_type%>*)(p2 + s2 * i));
    }
    __syncthreads();
    for (uint64_t x = threadIdx.x; x < n; x += blockDim.x) {
        <%=cnt_type%> c = 0;
        for (int k = 0; k < copies; k++) {
            c += bins[n * k + x];
        }
        if (c != 0) {
            <%=add%>((<%=cnt_type%>*)(p3 + s3 * x), c);
        }
    }
}

void <%="cumo_#{c_iter}_#{fn}_kernel_launch"%>(char *p1, ssize_t s1, char *p2, ssize_t s2, uint64_t count, char *p3, ssize_t s3, uint64_t n, int *oor, size_t *item)
{
    size_t grid;
    int copies;
    <%="cumo_#{c_iter}_#{fn}_init_kernel"%><<<cumo_get_grid_dim(n), cumo_get_block_dim(n), 0, cumo_cuda_stream()>>>(p3,s3,n);
    cumo_cuda_runtime_check_kernel_launch();
    copies = cumo_bincount_copies(n, sizeof(<%=cnt_type%>));
    grid = cumo_bincount_shared_grid((const void*)<%="cumo_#{c_iter}_#{fn}_shared_kernel"%>, count, n, sizeof(<%=cnt_type%>) * copies);
    if (grid > 0) {
        <%="cumo_#{c_iter}_#{fn}_shared_kernel"%><<<grid, CUMO_BINCOUNT_SHARED_BLOCK, n * copies * sizeof(<%=cnt_type%>), cumo_cuda_stream()>>>(p1,s1,p2,s2,count,p3,s3,n,oor,item,copies);
    } else {
        <%="cumo_#{c_iter}_#{fn}_kernel"%><<<cumo_get_grid_dim(count), cumo_get_block_dim(count), 0, cumo_cuda_stream()>>>(p1,s1,p2,s2,count,p3,s3,n,oor,item);
    }
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>

#undef cumo_check_bincount_item
