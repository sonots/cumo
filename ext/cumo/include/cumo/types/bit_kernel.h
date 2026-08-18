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

// Stores z as word w of a loop covering bits [p, p+n) of a. Only one thread
// owns a word, so the partial words at either end need no atomic.
__device__ static inline void
cumo_bit_store_word(CUMO_BIT_DIGIT *a, uint64_t w, CUMO_BIT_DIGIT z, size_t p, uint64_t n)
{
    size_t lb = (w == 0) ? p : 0;
    size_t hb = ((w + 1) * CUMO_NB <= p + n) ? CUMO_NB : (p + n - w * CUMO_NB);
    if (lb == 0 && hb == CUMO_NB) {
        a[w] = z;
    } else {
        CUMO_BIT_DIGIT mask = CUMO_SLB(hb) & ~CUMO_SLB(lb);
        a[w] = (z & mask) | (a[w] & ~mask);
    }
}

#endif // CUMO_BIT_KERNEL_H
