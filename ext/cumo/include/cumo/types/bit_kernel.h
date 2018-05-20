#ifndef CUMO_BIT_KERNEL_H
#define CUMO_BIT_KERNEL_H

typedef BIT_DIGIT dtype;
typedef BIT_DIGIT rtype;

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

#endif // CUMO_BIT_KERNEL_H
