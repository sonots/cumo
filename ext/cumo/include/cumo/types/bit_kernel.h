#ifndef CUMO_BIT_KERNEL_H
#define CUMO_BIT_KERNEL_H

typedef CUMO_BIT_DIGIT dtype;
typedef CUMO_BIT_DIGIT rtype;

#define m_zero 0
#define m_one  1

#define m_abs(x)     (x)
#define m_sign(x)    (((x)==0) ? 0:1)

#define m_from_double(x) (((x)==0) ? 0 : 1)
#define m_from_real(x) (((x)==0) ? 0 : 1)
#define m_from_sint(x) (((x)==0) ? 0 : 1)
#define m_from_int32(x) (((x)==0) ? 0 : 1)
#define m_from_int64(x) (((x)==0) ? 0 : 1)
#define m_from_uint32(x) (((x)==0) ? 0 : 1)
#define m_from_uint64(x) (((x)==0) ? 0 : 1)
#define m_data_to_num(x) INT2FIX(x)
#define m_sprintf(s,x)   sprintf(s,"%1d",(int)(x))

#define m_copy(x)  (x)
#define m_not(x)   (~(x))
#define m_and(x,y) ((x)&(y))
#define m_or(x,y)  ((x)|(y))
#define m_xor(x,y) ((x)^(y))
#define m_eq(x,y)  (~((x)^(y)))
#define m_count_true(x)  ((x)!=0)
#define m_count_true_cpu(x)  m_count_true(x)
#define m_count_false(x) ((x)==0)
#define m_count_false_cpu(x) m_count_false(x)

// Bit position of element i of an operand that is either indexed or strided.
__device__ static inline size_t
cumo_bit_pos(size_t p, ssize_t s, const size_t *idx, uint64_t i)
{
    return idx ? (p + idx[i]) : (p + i * s);
}

// The CUMO_NB bits starting o bits after word w of a, so a contiguous bit loop
// can give a whole word to each thread. nw bounds the operand, and a shift that
// would reach past it yields zeroes; those bits are outside the store mask.
__device__ static inline CUMO_BIT_DIGIT
cumo_bit_gather_word(const CUMO_BIT_DIGIT *a, ssize_t o, uint64_t nw, uint64_t w)
{
    CUMO_BIT_DIGIT x = 0;
    if (o >= 0) {
        if (w < nw)               x  = a[w] >> o;
        if (o > 0 && w + 1 < nw)  x |= a[w+1] << (CUMO_NB - o);
    } else {
        if (w < nw)               x  = a[w] << -o;
        if (w > 0)                x |= a[w-1] >> (CUMO_NB + o);
    }
    return x;
}

// Threads per block of the chunk scan below. The warp-shuffle scan is written
// for a block that is a whole number of warps.
#define CUMO_BIT_CHUNK_BLOCK 128

// Exclusive prefix sum of v over the block, leaving the block total in *total.
// Every thread of the block must reach it, and a caller that runs it more than
// once has to put a barrier between the runs because the shared state is reused.
__device__ static inline uint64_t
cumo_bit_block_exscan(uint64_t v, uint64_t *total)
{
    __shared__ uint64_t warp_sum[CUMO_BIT_CHUNK_BLOCK / 32];
    __shared__ uint64_t block_sum;
    unsigned int lane = threadIdx.x & 31u;
    unsigned int warp = threadIdx.x >> 5;
    uint64_t x = v;

    for (unsigned int d = 1; d < 32u; d <<= 1) {
        uint64_t y = __shfl_up_sync(0xffffffffu, x, d);
        if (lane >= d) x += y;
    }
    if (lane == 31u) warp_sum[warp] = x;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint64_t acc = 0;
        for (unsigned int w = 0; w < CUMO_BIT_CHUNK_BLOCK / 32; ++w) {
            uint64_t s = warp_sum[w];
            warp_sum[w] = acc;
            acc += s;
        }
        block_sum = acc;
    }
    __syncthreads();
    *total = block_sum;
    return warp_sum[warp] + x - v;
}

// Elements [c*CUMO_NB, c*CUMO_NB+CUMO_NB) of a loop over n elements, as one word
// whose bit k is element c*CUMO_NB+k. A contiguous loop straddles two words of
// the operand, which cumo_bit_gather_word recombines; any other loop gathers
// its bits one element at a time. invert answers the complement, masked to the
// elements the loop covers.
__device__ static inline CUMO_BIT_DIGIT
cumo_bit_chunk(const CUMO_BIT_DIGIT *a, size_t p, ssize_t s, const size_t *idx, uint64_t n, uint64_t nw, uint64_t c, int contiguous, int invert)
{
    uint64_t base = c * CUMO_NB;
    uint64_t kend = (n - base < CUMO_NB) ? (n - base) : CUMO_NB;
    CUMO_BIT_DIGIT z;

    if (contiguous) {
        z = cumo_bit_gather_word(a, (ssize_t)p, nw, c);
    } else {
        uint64_t k;
        z = 0;
        for (k = 0; k < kend; ++k) {
            CUMO_BIT_DIGIT x;
            CUMO_LOAD_BIT(a, cumo_bit_pos(p, s, idx, base + k), x);
            z |= x << k;
        }
    }
    if (invert) z = ~z;
    return z & CUMO_SLB(kend);
}

// The trailing loop dimensions along which a Bit operand runs bit by bit. When
// the output takes a whole word from one thread, the input can be gathered a
// word at a time over those rather than a bit at a time through the indexer,
// whose decomposition costs two runtime divisions per element. A column slice
// is a run per row with the rows apart, which is the case this is for.
typedef struct {
    int ok;
    int outer_ndim;      // loop dimensions [0, outer_ndim) address a row
    int64_t inner_len;   // elements in one row
} cumo_bit_run_t;

static inline cumo_bit_run_t
cumo_bit_make_run(const cumo_na_bit_iarray_stridx_t* a, const cumo_na_indexer_t* indexer)
{
    cumo_bit_run_t run;
    int k = indexer->ndim;
    ssize_t acc = 1;

    run.ok = 0;
    run.outer_ndim = indexer->ndim;
    run.inner_len = 1;
    while (k > 0 &&
           CUMO_SDX_IS_STRIDE(a->stridx[k-1]) &&
           CUMO_SDX_GET_STRIDE(a->stridx[k-1]) == acc) {
        acc *= (ssize_t)indexer->shape[k-1];
        --k;
    }
    // A row shorter than a word would leave the gather below assembling one out
    // of several rows, which is what the bit-at-a-time path already does.
    if (k < indexer->ndim && acc >= (ssize_t)CUMO_NB) {
        run.ok = 1;
        run.outer_ndim = k;
        run.inner_len = (int64_t)acc;
    }
    return run;
}

// Bit position of the first element of the given row.
__device__ static inline size_t
cumo_bit_run_pos(const cumo_na_bit_iarray_stridx_t* a, const cumo_na_indexer_t* indexer, cumo_bit_run_t run, int64_t row)
{
    size_t pos = a->pos;
    for (int j = run.outer_ndim; --j >= 0;) {
        int64_t n = (int64_t)indexer->shape[j];
        int64_t k = row % n;
        row /= n;
        if (CUMO_SDX_IS_INDEX(a->stridx[j])) {
            pos += CUMO_SDX_GET_INDEX(a->stridx[j])[k];
        } else {
            pos += CUMO_SDX_GET_STRIDE(a->stridx[j]) * k;
        }
    }
    return pos;
}

// Elements [w*CUMO_NB, w*CUMO_NB+CUMO_NB) of a loop over n elements, as one
// word. A word straddles two rows unless the row length is a multiple of
// CUMO_NB, so it is taken in as many pieces as it has rows -- one, almost
// always two at most.
__device__ static inline CUMO_BIT_DIGIT
cumo_bit_run_word(const cumo_na_bit_iarray_stridx_t* a, const cumo_na_indexer_t* indexer, cumo_bit_run_t run, uint64_t w, uint64_t n)
{
    const ssize_t nb = (ssize_t)CUMO_NB;
    uint64_t base = w * CUMO_NB;
    CUMO_BIT_DIGIT z = 0;
    uint64_t k = 0;

    while (k < CUMO_NB && base + k < n) {
        int64_t i = (int64_t)(base + k);
        int64_t row = i / run.inner_len;
        int64_t col = i - row * run.inner_len;
        uint64_t take = (uint64_t)(run.inner_len - col);
        if (take > CUMO_NB - k)   take = CUMO_NB - k;
        if (base + k + take > n)  take = n - (base + k);

        size_t p = cumo_bit_run_pos(a, indexer, run, row) + (size_t)col;
        ssize_t o = (ssize_t)(p % CUMO_NB);
        uint64_t nw = (uint64_t)((o + (ssize_t)take + nb - 1) / nb);
        CUMO_BIT_DIGIT piece = cumo_bit_gather_word(a->ptr + p / CUMO_NB, o, nw, 0);

        z |= (piece & CUMO_SLB(take)) << k;
        k += take;
    }
    return z;
}

// Stores the k-th index of a where result, whose element size the caller picked
// from the size of the operand.
__device__ static inline void
cumo_bit_store_index(char *a, size_t elmsz, uint64_t k, size_t v)
{
    if (elmsz == sizeof(uint32_t)) {
        ((uint32_t*)a)[k] = (uint32_t)v;
    } else {
        ((uint64_t*)a)[k] = (uint64_t)v;
    }
}

// Last, because the reductions there fold with cumo_bit_gather_word above.
#include "cumo/bit_reduce_kernel.h"

#endif // CUMO_BIT_KERNEL_H
