#ifndef CUMO_TEMPLATE_H
#define CUMO_TEMPLATE_H

#define CUMO_INIT_COUNTER( lp, c )                   \
    {   c = (lp)->n[0]; }

#define CUMO_NDL_CNT(lp) ((lp)->n[0])
#define CUMO_NDL_ARG(lp,i) ((lp)->args[i])
#define CUMO_NDL_PTR(lp,i) ((lp)->args[i].ptr + (lp)->args[i].iter[0].pos)
#define CUMO_NDL_STEP(lp,i) ((lp)->args[i].iter[0].step)
#define CUMO_NDL_IDX(lp,i) ((lp)->args[i].iter[0].idx)
#define CUMO_NDL_ESZ(lp,i) ((lp)->args[i].elmsz)
#define CUMO_NDL_SHAPE(lp,i) ((lp)->args[i].shape)

#define CUMO_NDL_ARG_STEP(arg,idim) ((arg).iter[idim].step)
#define CUMO_NDL_ARG_IDX(arg,idim) ((arg).iter[idim].idx)
#define CUMO_NDL_ARG_SHAPE(arg,idim) ((arg).shape[idim])

#define CUMO_INIT_PTR( lp, i, pt, st )                               \
    {                                                           \
        pt = ((lp)->args[i]).ptr + ((lp)->args[i].iter[0]).pos;         \
        st = ((lp)->args[i].iter[0]).step;                              \
    }

#define CUMO_INIT_PTR_IDX( lp, i, pt, st, id )                       \
    {                                                           \
        pt = ((lp)->args[i]).ptr + ((lp)->args[i].iter[0]).pos;         \
        st = ((lp)->args[i].iter[0]).step;                              \
        id = ((lp)->args[i].iter[0]).idx;                               \
    }

#define CUMO_INIT_ELMSIZE( lp, i, es )                               \
    {                                                           \
        es = ((lp)->args[i]).elmsz;                             \
    }

// pos is the first position in iteration order, so a step below zero walks down
// from there and rebasing the pointer onto pos would underflow the position of
// every element past the first word. A step of zero comes with an index, whose
// entries only ever move forward.
#define CUMO_INIT_PTR_BIT( lp, i, ad, ps, st )               \
    {                                                   \
        ps = ((lp)->args[i].iter[0]).pos;                       \
        st = ((lp)->args[i].iter[0]).step;                      \
        ad = (CUMO_BIT_DIGIT*)(((lp)->args[i]).ptr);            \
        if (st >= 0) {                                       \
            ad += ps/CUMO_NB;                                \
            ps %= CUMO_NB;                                   \
        }                                                    \
    }

#define CUMO_INIT_PTR_BIT_IDX( lp, i, ad, ps, st, id )       \
    {                                                   \
        ps = ((lp)->args[i].iter[0]).pos;                       \
        st = ((lp)->args[i].iter[0]).step;                      \
        id = ((lp)->args[i].iter[0]).idx;                       \
        ad = (CUMO_BIT_DIGIT*)(((lp)->args[i]).ptr);            \
        if (st >= 0) {                                       \
            ad += ps/CUMO_NB;                                \
            ps %= CUMO_NB;                                   \
        }                                                    \
    }

// A where compaction takes three launches, so below this many elements walking
// the bits on the host costs less. Measured on an RTX 5070 Ti.
#define CUMO_BIT_WHERE_MIN_KERNEL_SIZE 8192

// A cumulative reduction along a short axis runs its scan once per row, and a
// parallel scan needs scratch and several launches to start up.
#define CUMO_CUM_MIN_KERNEL_SIZE 8192

#define CUMO_GET_DATA( ptr, type, val )                 \
    {                                              \
        val = *(type*)(ptr);                       \
    }

#define CUMO_SET_DATA( ptr, type, val )                 \
    {                                              \
        *(type*)(ptr) = val;                       \
    }

#define CUMO_GET_DATA_STRIDE( ptr, step, type, val )    \
    {                                              \
        val = *(type*)(ptr);                       \
        ptr += step;                               \
    }

#define CUMO_GET_DATA_INDEX( ptr, idx, type, val )     \
    {                                           \
        val = *(type*)(ptr + *idx);             \
        idx++;                                  \
    }

#define CUMO_SET_DATA_STRIDE( ptr, step, type, val ) \
    {                                           \
        *(type*)(ptr) = val;                    \
        ptr += step;                            \
    }

#define CUMO_SET_DATA_INDEX( ptr, idx, type, val )   \
    {                                           \
        *(type*)(ptr + *idx) = val;             \
        idx++;                                  \
    }

#define CUMO_LOAD_BIT( adr, pos, val )                       \
    {                                                   \
        size_t dig = (pos) / CUMO_NB;                        \
        int    bit = (pos) % CUMO_NB;                        \
        val = (((CUMO_BIT_DIGIT*)(adr))[dig]>>(bit)) & 1u;   \
    }

#define CUMO_LOAD_BIT_STEP( adr, pos, step, idx, val )       \
    {                                                   \
        size_t dig; int bit;                            \
        if (idx) {                                      \
            dig = ((pos) + *(idx)) / CUMO_NB;                \
            bit = ((pos) + *(idx)) % CUMO_NB;                \
            idx++;                                      \
        } else {                                        \
            dig = (pos) / CUMO_NB;                           \
            bit = (pos) % CUMO_NB;                           \
            pos += step;                                \
        }                                               \
        val = (((CUMO_BIT_DIGIT*)(adr))[dig]>>bit) & 1u;     \
    }

#define CUMO_STORE_BIT(adr,pos,val)                  \
    {                                           \
        size_t dig = (pos) / CUMO_NB;                \
        int    bit = (pos) % CUMO_NB;                \
        ((CUMO_BIT_DIGIT*)(adr))[dig] =              \
            (((CUMO_BIT_DIGIT*)(adr))[dig] & ~(1u<<(bit))) | (((val)&1u)<<(bit)); \
    }

#define CUMO_STORE_BIT_STEP( adr, pos, step, idx, val )\
    {                                           \
        size_t dig; int bit;                    \
        if (idx) {                              \
            dig = ((pos) + *(idx)) / CUMO_NB;        \
            bit = ((pos) + *(idx)) % CUMO_NB;        \
            idx++;                              \
        } else {                                \
            dig = (pos) / CUMO_NB;                   \
            bit = (pos) % CUMO_NB;                   \
            pos += step;                        \
        }                                       \
        ((CUMO_BIT_DIGIT*)(adr))[dig] =              \
            (((CUMO_BIT_DIGIT*)(adr))[dig] & ~(1u<<(bit))) | (((val)&1u)<<(bit)); \
    }

static inline int
cumo_is_aligned(const void *ptr, const size_t alignment)
{
    return ((size_t)(ptr) & ((alignment)-1)) == 0;
}

static inline int
cumo_is_aligned_step(const ssize_t step, const size_t alignment)
{
    return ((step) & ((alignment)-1)) == 0;
}

// The store_array loops run Ruby code for every element -- m_num_to_data calls
// to_f / to_int, cumo_na_step_sequence calls to_int on a Range's endpoints --
// and that can reallocate or shrink the array being read, so neither its buffer
// nor its length survives one iteration.
static inline bool
cumo_na_store_rary_fetch(VALUE ary, size_t i, VALUE *x)
{
    if (!RB_TYPE_P(ary, T_ARRAY)) {
        *x = ary;
        return true;
    }
    if (i >= (size_t)RARRAY_LEN(ary)) {
        return false;
    }
    *x = RARRAY_AREF(ary, (long)i);
    return true;
}

#endif /* ifndef CUMO_TEMPLATE_H */
