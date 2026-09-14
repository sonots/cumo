#ifndef CUMO_TEMPLATE_KERNEL_H
#define CUMO_TEMPLATE_KERNEL_H

#define CUMO_LOAD_BIT( adr, pos, val )                       \
    {                                                   \
        size_t dig = (size_t)(pos) / CUMO_NB;                \
        int    bit = (size_t)(pos) % CUMO_NB;                \
        val = (((CUMO_BIT_DIGIT*)(adr))[dig]>>(bit)) & 1u;   \
    }

#define CUMO_LOAD_BIT_STEP( adr, pos, step, idx, val )       \
    {                                                   \
        size_t dig; int bit;                            \
        if (idx) {                                      \
            dig = (size_t)((pos) + *(idx)) / CUMO_NB;        \
            bit = (size_t)((pos) + *(idx)) % CUMO_NB;        \
            idx++;                                      \
        } else {                                        \
            dig = (size_t)(pos) / CUMO_NB;                   \
            bit = (size_t)(pos) % CUMO_NB;                   \
            pos += step;                                \
        }                                               \
        val = (((CUMO_BIT_DIGIT*)(adr))[dig]>>bit) & 1u;     \
    }

#define CUMO_STORE_BIT(adr,pos,val)                                     \
    {                                                              \
        size_t dig = (size_t)(pos) / CUMO_NB;                           \
        int    bit = (size_t)(pos) % CUMO_NB;                           \
        if (val) {                                                 \
            atomicOr((CUMO_BIT_DIGIT*)(adr) + (dig), (val)<<(bit));     \
        } else {                                                   \
            atomicAnd((CUMO_BIT_DIGIT*)(adr) + (dig), ~(1u<<(bit)));    \
        }                                                          \
    }
// val -> val&1 ??

#define CUMO_STORE_BIT_STEP( adr, pos, step, idx, val )                 \
    {                                                              \
        size_t dig; int bit;                                       \
        if (idx) {                                                 \
            dig = (size_t)((pos) + *(idx)) / CUMO_NB;                   \
            bit = (size_t)((pos) + *(idx)) % CUMO_NB;                   \
            idx++;                                                 \
        } else {                                                   \
            dig = (size_t)(pos) / CUMO_NB;                              \
            bit = (size_t)(pos) % CUMO_NB;                              \
            pos += step;                                           \
        }                                                          \
        if (val) {                                                 \
            atomicOr((CUMO_BIT_DIGIT*)(adr) + (dig), (val)<<(bit));     \
        } else {                                                   \
            atomicAnd((CUMO_BIT_DIGIT*)(adr) + (dig), ~((1u)<<(bit)));  \
        }                                                          \
    }
// val -> val&1 ??

#define CUMO_MAX_BLOCK_DIM 128
// Grid-stride loops step by blockDim.x * gridDim.x, a product of two unsigned
// ints, so the grid is capped where the product of the largest block and the
// grid would reach 2^32 and the step would wrap to zero. The hardware grid
// limit of 2^31-1 is far above this.
#define CUMO_MAX_GRID_DIM (4294967295ul / CUMO_MAX_BLOCK_DIM)

// A transpose has one side strided whichever way it is walked, so it stages a
// square tile in shared memory and reads and writes both sides in rows. The
// extra column spreads a tile's rows over the banks: for a 4-byte element that
// leaves no conflict at all, and it drops an 8-byte one to 2-way and a 16-byte
// one to 4-way. TILE must stay a multiple of ROWS.
#define CUMO_TRANSPOSE_TILE 32
#define CUMO_TRANSPOSE_ROWS 8

// gridDim.y and gridDim.z stop at 65535, unlike gridDim.x.
#define CUMO_MAX_GRID_DIM_Y 65535

// The tile a CUMO_TRANSPOSE_TILE_LOOP stages through. The extra column is what
// spreads its rows over the banks, so the loop reads it back along either axis
// without conflicting with itself.
#define CUMO_TRANSPOSE_TILE_DECL(type, name) \
    __shared__ type name[CUMO_TRANSPOSE_TILE][CUMO_TRANSPOSE_TILE + 1]

// Tiles only pay off once both sides reach one. A shorter side leaves most of
// each warp idle, while the plain loop already reads or writes it in one go.
#define CUMO_TRANSPOSE_TILE_FITS(indexer)                                      \
    ((indexer)->ndim == 2 &&                                                   \
     (indexer)->shape[0] >= CUMO_TRANSPOSE_TILE &&                             \
     (indexer)->shape[1] >= CUMO_TRANSPOSE_TILE)

// Walks a rows-by-cols transpose one tile at a time.
//
// STAGE reads one element of the transposed side, which sits at cumo_tile_src
// of that side's own layout, and lands in cumo_tile_val. The body then runs
// once per output element, at cumo_tile_dst of the row-major layout the output
// and any other operand share.
//
// The body runs inside two nested loops, after a __syncthreads(). A continue
// there skips one output element, a break leaves the current tile but not the
// walk, and a return deadlocks the block on the next __syncthreads().
#define CUMO_TRANSPOSE_TILE_LOOP(tile, rows, cols, STAGE, ...)                 \
    do {                                                                       \
        uint64_t cumo_tile_rows = (rows);                                      \
        uint64_t cumo_tile_cols = (cols);                                      \
        uint64_t cumo_tile_base;                                               \
                                                                               \
        for (cumo_tile_base = (uint64_t)blockIdx.y * CUMO_TRANSPOSE_TILE;      \
             cumo_tile_base < cumo_tile_rows;                                  \
             cumo_tile_base += (uint64_t)gridDim.y * CUMO_TRANSPOSE_TILE) {    \
            uint64_t cumo_tile_x = cumo_tile_base + threadIdx.x;               \
            uint64_t cumo_tile_y =                                             \
                (uint64_t)blockIdx.x * CUMO_TRANSPOSE_TILE + threadIdx.y;      \
            int cumo_tile_j;                                                   \
                                                                               \
            __syncthreads();                                                   \
            for (cumo_tile_j = 0;                                              \
                 threadIdx.y + cumo_tile_j < CUMO_TRANSPOSE_TILE;              \
                 cumo_tile_j += CUMO_TRANSPOSE_ROWS) {                         \
                if (cumo_tile_x < cumo_tile_rows &&                            \
                    cumo_tile_y + cumo_tile_j < cumo_tile_cols) {              \
                    uint64_t cumo_tile_src =                                   \
                        (cumo_tile_y + cumo_tile_j) * cumo_tile_rows + cumo_tile_x; \
                    (tile)[threadIdx.y + cumo_tile_j][threadIdx.x] = STAGE;    \
                }                                                              \
            }                                                                  \
            __syncthreads();                                                   \
            cumo_tile_x = (uint64_t)blockIdx.x * CUMO_TRANSPOSE_TILE + threadIdx.x; \
            cumo_tile_y = cumo_tile_base + threadIdx.y;                        \
            for (cumo_tile_j = 0;                                              \
                 threadIdx.y + cumo_tile_j < CUMO_TRANSPOSE_TILE;              \
                 cumo_tile_j += CUMO_TRANSPOSE_ROWS) {                         \
                if (cumo_tile_x < cumo_tile_cols &&                            \
                    cumo_tile_y + cumo_tile_j < cumo_tile_rows) {              \
                    uint64_t cumo_tile_dst =                                   \
                        (cumo_tile_y + cumo_tile_j) * cumo_tile_cols + cumo_tile_x; \
                    auto cumo_tile_val =                                       \
                        (tile)[threadIdx.x][threadIdx.y + cumo_tile_j];        \
                    __VA_ARGS__                                                \
                }                                                              \
            }                                                                  \
        }                                                                      \
    } while (0)

// Launches such a kernel, which takes its own arguments and then rows and cols.
// gridDim.y walks tile rows on its own, but gridDim.x has to cover every tile
// column: a short grid there drops the columns past it without an error.
#define CUMO_TRANSPOSE_LAUNCH(kernel, rows, cols, ...)                         \
    do {                                                                       \
        uint64_t cumo_launch_rows = (rows);                                    \
        uint64_t cumo_launch_cols = (cols);                                    \
        uint64_t cumo_launch_tiles_y =                                         \
            (cumo_launch_rows + CUMO_TRANSPOSE_TILE - 1) / CUMO_TRANSPOSE_TILE; \
        dim3 cumo_launch_grid(                                                 \
            (unsigned int)((cumo_launch_cols + CUMO_TRANSPOSE_TILE - 1) / CUMO_TRANSPOSE_TILE), \
            (unsigned int)(cumo_launch_tiles_y > CUMO_MAX_GRID_DIM_Y           \
                           ? CUMO_MAX_GRID_DIM_Y : cumo_launch_tiles_y));      \
        dim3 cumo_launch_block(CUMO_TRANSPOSE_TILE, CUMO_TRANSPOSE_ROWS);      \
                                                                               \
        kernel<<<cumo_launch_grid, cumo_launch_block>>>(                       \
            __VA_ARGS__, cumo_launch_rows, cumo_launch_cols);                  \
    } while (0)

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
    // A launch of zero threads is rejected as an invalid configuration, and an
    // empty array reaches here with n == 0. Every kernel is bounded by n, so a
    // single thread does no work.
    size_t block_dim = (n > CUMO_MAX_BLOCK_DIM) ? CUMO_MAX_BLOCK_DIM : n;
    return (block_dim == 0) ? 1 : block_dim;
}

#if defined(__cplusplus)
extern "C" {
#endif

// Raises the error a kernel launch reported, if any. Defined in cuda/runtime.c
// because raising needs ruby.h, which the .cu translation units do not include.
void cumo_cuda_runtime_check_kernel_launch(void);

// The forms to call while scratch is held, which free what they are given
// before they raise. Pass every buffer that is still outstanding, including the
// caller's: a raise here leaves through longjmp and nothing below it runs.
void cumo_cuda_runtime_check_kernel_launch_holding(char *p0, char *p1, char *p2, char *p3, char *p4);
void cumo_cuda_runtime_check_taken_status_holding(int status, char *p0, char *p1, char *p2, char *p3, char *p4);

#if defined(__cplusplus)
static inline void cumo_check_launch_holding(void *p0 = 0, void *p1 = 0, void *p2 = 0, void *p3 = 0, void *p4 = 0)
{
    cumo_cuda_runtime_check_kernel_launch_holding((char*)p0, (char*)p1, (char*)p2, (char*)p3, (char*)p4);
}
static inline void cumo_check_status_holding(cudaError_t status, void *p0 = 0, void *p1 = 0, void *p2 = 0, void *p3 = 0, void *p4 = 0)
{
    cumo_cuda_runtime_check_taken_status_holding((int)status, (char*)p0, (char*)p1, (char*)p2, (char*)p3, (char*)p4);
}
#endif

// Scratch memory for a kernel that needs somewhere to put partial results.
// Declared here rather than including cuda/memory_pool.h, which needs ruby.h.
char* cumo_cuda_runtime_malloc(size_t size);
void cumo_cuda_runtime_free(char *ptr);

#if defined(__cplusplus)
}
#endif


#endif /* ifndef CUMO_TEMPLATE_KERNEL_H */
