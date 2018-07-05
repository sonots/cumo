#ifndef CUMO_TEMPLATE_KERNEL_H
#define CUMO_TEMPLATE_KERNEL_H

#define CUMO_LOAD_BIT( adr, pos, val )                       \
    {                                                   \
        size_t dig = (size_t)(pos) / NB;                \
        int    bit = (size_t)(pos) % NB;                \
        val = (((BIT_DIGIT*)(adr))[dig]>>(bit)) & 1u;   \
    }

#define CUMO_LOAD_BIT_STEP( adr, pos, step, idx, val )       \
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

#define CUMO_STORE_BIT(adr,pos,val)                                     \
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

#define CUMO_STORE_BIT_STEP( adr, pos, step, idx, val )                 \
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

#define CUMO_MAX_BLOCK_DIM 128
#define CUMO_MAX_GRID_DIM 2147483647 // ref. http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

static inline size_t
cumo_get_grid_dim(size_t n)
{
    size_t grid_dim = (n / CUMO_MAX_BLOCK_DIM) + 1;
    if (grid_dim > CUMO_MAX_GRID_DIM) grid_dim = CUMO_MAX_GRID_DIM;
    return grid_dim;
}

static inline size_t
cumo_get_block_dim(size_t n)
{
    size_t block_dim = (n > CUMO_MAX_BLOCK_DIM) ? CUMO_MAX_BLOCK_DIM : n;
    return block_dim;
}


#endif /* ifndef CUMO_TEMPLATE_KERNEL_H */
