#ifndef TEMPLATE_KERNEL_H
#define TEMPLATE_KERNEL_H

#define LOAD_BIT( adr, pos, val )                       \
    {                                                   \
        size_t dig = (size_t)(pos) / NB;                \
        int    bit = (size_t)(pos) % NB;                \
        val = (((BIT_DIGIT*)(adr))[dig]>>(bit)) & 1u;   \
    }

#define LOAD_BIT_STEP( adr, pos, step, idx, val )       \
    {                                                   \
        size_t dig; int bit;                            \
        if (idx) {                                      \
            dig = (size_t)((pos) + *(idx)) / NB;        \
            bit = (size_t)((pos) + *(idx)) % NB;        \
            idx++;                                      \
        } else {                                        \
            dig = (size_t)(pos) / NB;                   \
            bit = (size_t)(pos) % NB;                   \
            pos += step;                                \
        }                                               \
        val = (((BIT_DIGIT*)(adr))[dig]>>bit) & 1u;     \
    }

#define STORE_BIT(adr,pos,val)                                     \
    {                                                              \
        size_t dig = (size_t)(pos) / NB;                           \
        int    bit = (size_t)(pos) % NB;                           \
        if (val) {                                                 \
            atomicOr((BIT_DIGIT*)(adr) + (dig), (val)<<(bit));     \
        } else {                                                   \
            atomicAnd((BIT_DIGIT*)(adr) + (dig), ~(1u<<(bit)));    \
        }                                                          \
    }
// val -> val&1 ??

#define STORE_BIT_STEP( adr, pos, step, idx, val )                 \
    {                                                              \
        size_t dig; int bit;                                       \
        if (idx) {                                                 \
            dig = (size_t)((pos) + *(idx)) / NB;                   \
            bit = (size_t)((pos) + *(idx)) % NB;                   \
            idx++;                                                 \
        } else {                                                   \
            dig = (size_t)(pos) / NB;                              \
            bit = (size_t)(pos) % NB;                              \
            pos += step;                                           \
        }                                                          \
        if (val) {                                                 \
            atomicOr((BIT_DIGIT*)(adr) + (dig), (val)<<(bit));     \
        } else {                                                   \
            atomicAnd((BIT_DIGIT*)(adr) + (dig), ~((1u)<<(bit)));  \
        }                                                          \
    }
// val -> val&1 ??

#define MAX_BLOCK_DIM 128
#define MAX_GRID_DIM 2147483647 // ref. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

static inline size_t
get_gridDim(size_t n)
{
    size_t gridDim = (n / MAX_BLOCK_DIM) + 1;
    if (gridDim > MAX_GRID_DIM) gridDim = MAX_GRID_DIM;
    return gridDim;
}

static inline size_t
get_blockDim(size_t n)
{
    size_t blockDim = (n > MAX_BLOCK_DIM) ? MAX_BLOCK_DIM : n;
    return blockDim;
}


#endif /* ifndef TEMPLATE_KERNEL_H */
