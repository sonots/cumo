#ifndef TEMPLATE_KERNEL_H
#define TEMPLATE_KERNEL_H

#define MAX_BLOCK_DIM 128
#define MAX_GRID_DIM 2147483647 // ref. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

static inline size_t
get_gridDim(size_t N)
{
    size_t gridDim = (N / MAX_BLOCK_DIM) + 1;
    if (gridDim > MAX_GRID_DIM) gridDim = MAX_GRID_DIM;
    return gridDim;
}

static inline size_t
get_blockDim(size_t N)
{
    size_t blockDim = (N > MAX_BLOCK_DIM) ? MAX_BLOCK_DIM : N;
    return blockDim;
}


#endif /* ifndef TEMPLATE_KERNEL_H */
