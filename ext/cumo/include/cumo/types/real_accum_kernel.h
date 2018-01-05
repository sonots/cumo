#ifndef CUMO_REAL_ACCUM_KERNEL_H
#define CUMO_REAL_ACCUM_KERNEL_H

#define not_nan(x) ((x)==(x))

#define m_mulsum(x,y,z) {z = m_add(m_mul(x,y),z);}
#define m_mulsum_nan(x,y,z) {          \
        if(not_nan(x) && not_nan(y)) { \
            z = m_add(m_mul(x,y),z);   \
        }}

#define m_cumsum(x,y) {(x)=m_add(x,y);}
#define m_cumsum_nan(x,y) {      \
        if (!not_nan(x)) {       \
            (x) = (y);           \
        } else if (not_nan(y)) { \
            (x) = m_add(x,y);    \
        }}

#define m_cumprod(x,y) {(x)=m_mul(x,y);}
#define m_cumprod_nan(x,y) {     \
        if (!not_nan(x)) {       \
            (x) = (y);           \
        } else if (not_nan(y)) { \
            (x) = m_mul(x,y);    \
        }}

#endif // CUMO_REAL_ACCUM_KERNEL_H
