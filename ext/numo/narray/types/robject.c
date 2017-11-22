
#line 1 "gen/tmpl/lib.c"
/*
  types/robject.c
  Ruby/Numo::GSL - GSL wrapper for Ruby/Numo::NArray

  created on: 2017-03-11
  Copyright (C) 2017 Masahiro Tanaka
*/

#include <ruby.h>
#include <assert.h>
#include "numo/narray.h"
#include "numo/template.h"
#include "SFMT.h"

#define m_map(x) m_num_to_data(rb_yield(m_data_to_num(x)))

static ID id_ne;
static ID id_lt;
static ID id_left_shift;
static ID id_le;
static ID id_ufo;
static ID id_eq;
static ID id_gt;
static ID id_ge;
static ID id_right_shift;
static ID id_abs;
static ID id_bit_and;
static ID id_bit_not;
static ID id_bit_or;
static ID id_bit_xor;
static ID id_cast;
static ID id_ceil;
static ID id_copysign;
static ID id_divmod;
static ID id_eq;
static ID id_finite_p;
static ID id_floor;
static ID id_infinite_p;
static ID id_minus;
static ID id_mulsum;
static ID id_nan_p;
static ID id_ne;
static ID id_nearly_eq;
static ID id_pow;
static ID id_reciprocal;
static ID id_round;
static ID id_square;
static ID id_truncate;

#line 22 "gen/tmpl/lib.c"
#include <numo/types/robject.h>

#line 25 "gen/tmpl/lib.c"
VALUE cT;
extern VALUE cRT;


#line 1 "gen/tmpl/class.c"
/*
  class definition: Numo::RObject
*/

VALUE cT;

static VALUE robject_store(VALUE,VALUE);












#line 1 "gen/tmpl/alloc_func.c"
static size_t
robject_memsize(const void* ptr)
{
    size_t size = sizeof(narray_data_t);
    const narray_data_t *na = (const narray_data_t*)ptr;

    assert(na->base.type == NARRAY_DATA_T);

    if (na->ptr != NULL) {
  
#line 13 "gen/tmpl/alloc_func.c"
        size += na->base.size * sizeof(dtype);
  
    }
    if (na->base.size > 0) {
        if (na->base.shape != NULL && na->base.shape != &(na->base.size)) {
            size += sizeof(size_t) * na->base.ndim;
        }
    }
    return size;
}

static void
robject_free(void* ptr)
{
    narray_data_t *na = (narray_data_t*)ptr;

    assert(na->base.type == NARRAY_DATA_T);

    if (na->ptr != NULL) {
        xfree(na->ptr);
        na->ptr = NULL;
    }
    if (na->base.size > 0) {
        if (na->base.shape != NULL && na->base.shape != &(na->base.size)) {
            xfree(na->base.shape);
            na->base.shape = NULL;
        }
    }
    xfree(na);
}

static narray_type_info_t robject_info = {
  
#line 50 "gen/tmpl/alloc_func.c"
    0,             // element_bits
    sizeof(dtype), // element_bytes
    sizeof(dtype), // element_stride (in bytes)
  
};

#line 57 "gen/tmpl/alloc_func.c"
static void
robject_gc_mark(void *ptr)
{
    size_t n, i;
    VALUE *a;
    narray_data_t *na = ptr;

    if (na->ptr) {
        a = (VALUE*)(na->ptr);
        n = na->base.size;
        for (i=0; i<n; i++) {
            rb_gc_mark(a[i]);
        }
    }
}

const rb_data_type_t robject_data_type = {
    "Numo::RObject",
    {robject_gc_mark, robject_free, robject_memsize,},
    &na_data_type,
    &robject_info,
    0, // flags
};


#line 93 "gen/tmpl/alloc_func.c"
VALUE
robject_s_alloc_func(VALUE klass)
{
    narray_data_t *na = ALLOC(narray_data_t);

    na->base.ndim = 0;
    na->base.type = NARRAY_DATA_T;
    na->base.flag[0] = NA_FL0_INIT;
    na->base.flag[1] = NA_FL1_INIT;
    na->base.size = 0;
    na->base.shape = NULL;
    na->base.reduce = INT2FIX(0);
    na->ptr = NULL;
    return TypedData_Wrap_Struct(klass, &robject_data_type, (void*)na);
}


#line 1 "gen/tmpl/allocate.c"
static VALUE
robject_allocate(VALUE self)
{
    narray_t *na;
    char *ptr;

    GetNArray(self,na);

    switch(NA_TYPE(na)) {
    case NARRAY_DATA_T:
        ptr = NA_DATA_PTR(na);
        if (na->size > 0 && ptr == NULL) {
            ptr = xmalloc(sizeof(dtype) * na->size);
            
            {   size_t i;
                VALUE *a = (VALUE*)ptr;
                for (i=na->size; i--;) {
                    *a++ = Qnil;
                }
            }
            
            NA_DATA_PTR(na) = ptr;
        }
        break;
    case NARRAY_VIEW_T:
        rb_funcall(NA_VIEW_DATA(na), rb_intern("allocate"), 0);
        break;
    case NARRAY_FILEMAP_T:
        //ptr = ((narray_filemap_t*)na)->ptr;
        // to be implemented
    default:
        rb_bug("invalid narray type : %d",NA_TYPE(na));
    }
    return self;
}


#line 1 "gen/tmpl/extract.c"
/*
  Extract an element only if self is a dimensionless NArray.
  @overload extract
  @return [Numeric,Numo::NArray]
  --- Extract element value as Ruby Object if self is a dimensionless NArray,
  otherwise returns self.
*/
static VALUE
robject_extract(VALUE self)
{
    volatile VALUE v;
    char *ptr;
    narray_t *na;
    GetNArray(self,na);

    if (na->ndim==0) {
        ptr = na_get_pointer_for_read(self) + na_get_offset(self);
        v = m_extract(ptr);
        na_release_lock(self);
        return v;
    }
    return self;
}


#line 1 "gen/tmpl/new_dim0.c"
static VALUE
robject_new_dim0(dtype x)
{
    VALUE v;
    dtype *ptr;

    v = nary_new(cT, 0, NULL);
    ptr = (dtype*)(char*)na_get_pointer_for_write(v);
    *ptr = x;
    na_release_lock(v);
    return v;
}


#line 1 "gen/tmpl/store.c"

#line 1 "gen/tmpl/store_numeric.c"
static VALUE
robject_store_numeric(VALUE self, VALUE obj)
{
    dtype x;
    x = m_num_to_data(obj);
    obj = robject_new_dim0(x);
    robject_store(self,obj);
    return self;
}


#line 1 "gen/tmpl/store_bit.c"
static void
iter_robject_store_bit(na_loop_t *const lp)
{
    size_t     i;
    char      *p1;
    size_t     p2;
    ssize_t    s1, s2;
    size_t    *idx1, *idx2;
    BIT_DIGIT *a2, x;
    dtype      y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_BIT_IDX(lp, 1, a2, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                LOAD_BIT(a2, p2+*idx2, x); idx2++;
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                LOAD_BIT(a2, p2+*idx2, x); idx2++;
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                LOAD_BIT(a2, p2, x); p2 += s2;
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                LOAD_BIT(a2, p2, x); p2 += s2;
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_bit(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = {iter_robject_store_bit, FULL_LOOP, 2,0, ain,0};

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_dfloat(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    double x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,double,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,double,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,double,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,double,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_dfloat(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_dfloat, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_sfloat(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    float x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,float,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,float,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,float,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,float,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_sfloat(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_sfloat, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_int64(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    int64_t x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,int64_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,int64_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,int64_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,int64_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_int64(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_int64, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_int32(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    int32_t x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,int32_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,int32_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,int32_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,int32_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_int32(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_int32, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_int16(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    int16_t x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,int16_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,int16_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,int16_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,int16_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_int16(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_int16, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_int8(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    int8_t x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,int8_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,int8_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,int8_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,int8_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_int8(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_int8, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_uint64(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    u_int64_t x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,u_int64_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,u_int64_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,u_int64_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,u_int64_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_uint64(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_uint64, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_uint32(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    u_int32_t x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,u_int32_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,u_int32_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,u_int32_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,u_int32_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_uint32(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_uint32, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_uint16(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    u_int16_t x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,u_int16_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,u_int16_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,u_int16_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,u_int16_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_uint16(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_uint16, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_uint8(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    u_int8_t x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,u_int8_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,u_int8_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,u_int8_t,x);
                y = m_from_real(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,u_int8_t,x);
                y = m_from_real(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_uint8(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_uint8, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_robject_store_robject(na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    VALUE x;
    dtype y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,VALUE,x);
                y = m_num_to_data(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p2,idx2,VALUE,x);
                y = m_num_to_data(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,VALUE,x);
                y = m_num_to_data(x);
                SET_DATA_INDEX(p1,idx1,dtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p2,s2,VALUE,x);
                y = m_num_to_data(x);
                SET_DATA_STRIDE(p1,s1,dtype,y);
            }
        }
    }
}


static VALUE
robject_store_robject(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_robject_store_robject, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_array.c"
static void
iter_robject_store_array(na_loop_t *const lp)
{
    size_t i, n;
    size_t i1, n1;
    VALUE  v1, *ptr;
    char   *p1;
    size_t s1, *idx1;
    VALUE  x;
    double y;
    dtype  z;
    size_t len, c;
    double beg, step;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    v1 = lp->args[1].value;
    i = 0;

    if (lp->args[1].ptr) {
        if (v1 == Qtrue) {
            iter_robject_store_robject(lp);
            i = lp->args[1].shape[0];
            if (idx1) {
                idx1 += i;
            } else {
                p1 += s1 * i;
            }
        }
        goto loop_end;
    }

    ptr = &v1;

    switch(TYPE(v1)) {
    case T_ARRAY:
        n1 = RARRAY_LEN(v1);
        ptr = RARRAY_PTR(v1);
        break;
    case T_NIL:
        n1 = 0;
        break;
    default:
        n1 = 1;
    }

    if (idx1) {
        for (i=i1=0; i1<n1 && i<n; i++,i1++) {
            x = ptr[i1];
            if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, na_cStep)) {
                nary_step_sequence(x,&len,&beg,&step);
                for (c=0; c<len && i<n; c++,i++) {
                    y = beg + step * c;
                    z = m_from_double(y);
                    SET_DATA_INDEX(p1, idx1, dtype, z);
                }
            }
            else if (TYPE(x) != T_ARRAY) {
                z = m_num_to_data(x);
                SET_DATA_INDEX(p1, idx1, dtype, z);
            }
        }
    } else {
        for (i=i1=0; i1<n1 && i<n; i++,i1++) {
            x = ptr[i1];
            if (rb_obj_is_kind_of(x, rb_cRange) || rb_obj_is_kind_of(x, na_cStep)) {
                nary_step_sequence(x,&len,&beg,&step);
                for (c=0; c<len && i<n; c++,i++) {
                    y = beg + step * c;
                    z = m_from_double(y);
                    SET_DATA_STRIDE(p1, s1, dtype, z);
                }
            }
            else if (TYPE(x) != T_ARRAY) {
                z = m_num_to_data(x);
                SET_DATA_STRIDE(p1, s1, dtype, z);
            }
        }
    }

 loop_end:
    z = m_zero;
    if (idx1) {
        for (; i<n; i++) {
            SET_DATA_INDEX(p1, idx1, dtype, z);
        }
    } else {
        for (; i<n; i++) {
            SET_DATA_STRIDE(p1, s1, dtype, z);
        }
    }
}

static VALUE
robject_store_array(VALUE self, VALUE rary)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{rb_cArray,0}};
    ndfunc_t ndf = {iter_robject_store_array, FULL_LOOP, 2, 0, ain, 0};

    na_ndloop_store_rarray(&ndf, self, rary);
    return self;
}

#line 5 "gen/tmpl/store.c"
/*
  Store elements to Numo::RObject from other.
  @overload store(other)
  @param [Object] other
  @return [Numo::RObject] self
*/
static VALUE
robject_store(VALUE self, VALUE obj)
{
    VALUE r, klass;

    klass = CLASS_OF(obj);

    
    if (klass==numo_cRObject) {
        robject_store_robject(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (IS_INTEGER_CLASS(klass) || klass==rb_cFloat || klass==rb_cComplex) {
        robject_store_numeric(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cBit) {
        robject_store_bit(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cDFloat) {
        robject_store_dfloat(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cSFloat) {
        robject_store_sfloat(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cInt64) {
        robject_store_int64(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cInt32) {
        robject_store_int32(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cInt16) {
        robject_store_int16(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cInt8) {
        robject_store_int8(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cUInt64) {
        robject_store_uint64(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cUInt32) {
        robject_store_uint32(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cUInt16) {
        robject_store_uint16(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cUInt8) {
        robject_store_uint8(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==rb_cArray) {
        robject_store_array(self,obj);
        return self;
    }
    

    if (IsNArray(obj)) {
        r = rb_funcall(obj, rb_intern("coerce_cast"), 1, cT);
        if (CLASS_OF(r)==cT) {
            robject_store(self,r);
            return self;
        }
    }

    
    robject_store_numeric(self,obj);
    
#line 40 "gen/tmpl/store.c"
    return self;
}


#line 1 "gen/tmpl/extract_data.c"
/*
  Convert a data value of obj (with a single element) to dtype.
*/
static dtype
robject_extract_data(VALUE obj)
{
    narray_t *na;
    dtype  x;
    char  *ptr;
    size_t pos;
    VALUE  r, klass;

    if (IsNArray(obj)) {
        GetNArray(obj,na);
        if (na->size != 1) {
            rb_raise(nary_eShapeError,"narray size should be 1");
        }
        klass = CLASS_OF(obj);
        ptr = na_get_pointer_for_read(obj);
        pos = na_get_offset(obj);
        
        if (klass==numo_cRObject) {
            x = m_num_to_data(*(VALUE*)(ptr+pos));
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cBit) {
            {BIT_DIGIT b; LOAD_BIT(ptr,pos,b); x = m_from_real(b);};
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cDFloat) {
            x = m_from_real(*(double*)(ptr+pos));
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cSFloat) {
            x = m_from_real(*(float*)(ptr+pos));
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cInt64) {
            x = m_from_real(*(int64_t*)(ptr+pos));
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cInt32) {
            x = m_from_real(*(int32_t*)(ptr+pos));
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cInt16) {
            x = m_from_real(*(int16_t*)(ptr+pos));
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cInt8) {
            x = m_from_real(*(int8_t*)(ptr+pos));
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cUInt64) {
            x = m_from_real(*(u_int64_t*)(ptr+pos));
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cUInt32) {
            x = m_from_real(*(u_int32_t*)(ptr+pos));
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cUInt16) {
            x = m_from_real(*(u_int16_t*)(ptr+pos));
            return x;
        }
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cUInt8) {
            x = m_from_real(*(u_int8_t*)(ptr+pos));
            return x;
        }
        

        // coerce
        r = rb_funcall(obj, rb_intern("coerce_cast"), 1, cT);
        if (CLASS_OF(r)==cT) {
            return robject_extract_data(r);
        }
        
        return obj;
        
#line 40 "gen/tmpl/extract_data.c"
    }
    if (TYPE(obj)==T_ARRAY) {
        if (RARRAY_LEN(obj) != 1) {
            rb_raise(nary_eShapeError,"array size should be 1");
        }
        return m_num_to_data(RARRAY_AREF(obj,0));
    }
    return m_num_to_data(obj);
}


#line 1 "gen/tmpl/cast_array.c"
static VALUE
robject_cast_array(VALUE rary)
{
    VALUE nary;
    narray_t *na;

    nary = na_s_new_like(cT, rary);
    GetNArray(nary,na);
    if (na->size > 0) {
        robject_store_array(nary,rary);
    }
    return nary;
}


#line 1 "gen/tmpl/cast.c"
#line 5 "gen/tmpl/cast.c"
/*
  Cast object to Numo::RObject.
  @overload [](elements)
  @overload cast(array)
  @param [Numeric,Array] elements
  @param [Array] array
  @return [Numo::RObject]
*/
static VALUE
robject_s_cast(VALUE type, VALUE obj)
{
    VALUE v;
    narray_t *na;
    dtype x;

    if (CLASS_OF(obj)==cT) {
        return obj;
    }
    if (RTEST(rb_obj_is_kind_of(obj,rb_cNumeric))) {
        x = m_num_to_data(obj);
        return robject_new_dim0(x);
    }
    if (RTEST(rb_obj_is_kind_of(obj,rb_cArray))) {
        return robject_cast_array(obj);
    }
    if (IsNArray(obj)) {
        GetNArray(obj,na);
        v = nary_new(cT, NA_NDIM(na), NA_SHAPE(na));
        if (NA_SIZE(na) > 0) {
            robject_store(v,obj);
        }
        return v;
    }
    
    return robject_new_dim0(obj);
    
#line 44 "gen/tmpl/cast.c"
}


#line 1 "gen/tmpl/aref.c"
/*
  Array element referenece or slice view.
  @overload [](dim0,...,dimL)
  @param [Numeric,Range,etc] dim0,...,dimL  Multi-dimensional Index.
  @return [Numeric,NArray::RObject] Element object or NArray view.

  --- Returns the element at +dim0+, +dim1+, ... are Numeric indices
  for each dimension, or returns a NArray View as a sliced subarray if
  +dim0+, +dim1+, ... includes other than Numeric index, e.g., Range
  or Array or true.

  @example
      a = Numo::DFloat.new(4,5).seq
      => Numo::DFloat#shape=[4,5]
      [[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9],
       [10, 11, 12, 13, 14],
       [15, 16, 17, 18, 19]]

      a[1,1]
      => 6.0

      a[1..3,1]
      => Numo::DFloat#shape=[3]
      [6, 11, 16]

      a[1,[1,3,4]]
      => Numo::DFloat#shape=[3]
      [6, 8, 9]

      a[true,2].fill(99)
      a
      => Numo::DFloat#shape=[4,5]
      [[0, 1, 99, 3, 4],
       [5, 6, 99, 8, 9],
       [10, 11, 99, 13, 14],
       [15, 16, 99, 18, 19]]
 */
static VALUE
robject_aref(int argc, VALUE *argv, VALUE self)
{
    int nd;
    size_t pos;
    char *ptr;

    nd = na_get_result_dimension(self, argc, argv, sizeof(dtype), &pos);
    if (nd) {
        return na_aref_main(argc, argv, self, 0, nd);
    } else {
        ptr = na_get_pointer_for_read(self) + pos;
        return m_extract(ptr);
    }
}


#line 1 "gen/tmpl/aset.c"
/*
  Array element(s) set.
  @overload []=(dim0,..,dimL,val)
  @param [Numeric,Range,etc] dim0,..,dimL  Multi-dimensional Index.
  @param [Numeric,Numo::NArray,etc] val  Value(s) to be set to self.
  @return [Numeric] returns val (last argument).

  --- Replace element(s) at +dim0+, +dim1+, ... (index/range/array/true
  for each dimention). Broadcasting mechanism is applied.

  @example
      a = Numo::DFloat.new(3,4).seq
      => Numo::DFloat#shape=[3,4]
      [[0, 1, 2, 3],
       [4, 5, 6, 7],
       [8, 9, 10, 11]]

      a[1,2]=99
      a
      => Numo::DFloat#shape=[3,4]
      [[0, 1, 2, 3],
       [4, 5, 99, 7],
       [8, 9, 10, 11]]

      a[1,[0,2]] = [101,102]
      a
      => Numo::DFloat#shape=[3,4]
      [[0, 1, 2, 3],
       [101, 5, 102, 7],
       [8, 9, 10, 11]]

      a[1,true]=99
      a
      => Numo::DFloat#shape=[3,4]
      [[0, 1, 2, 3],
       [99, 99, 99, 99],
       [8, 9, 10, 11]]

*/
static VALUE
robject_aset(int argc, VALUE *argv, VALUE self)
{
    int nd;
    size_t pos;
    char *ptr;
    VALUE a;
    dtype x;

    argc--;
    if (argc==0) {
        robject_store(self, argv[argc]);
    } else {
        nd = na_get_result_dimension(self, argc, argv, sizeof(dtype), &pos);
        if (nd) {
            a = na_aref_main(argc, argv, self, 0, nd);
            robject_store(a, argv[argc]);
        } else {
            x = robject_extract_data(argv[argc]);
            ptr = na_get_pointer_for_read_write(self) + pos;
            *(dtype*)ptr = x;
        }

    }
    return argv[argc];
}


#line 1 "gen/tmpl/coerce_cast.c"
/*
  return NArray with cast to the type of self.
  @overload coerce_cast(type)
  @return [nil]
*/
static VALUE
robject_coerce_cast(VALUE self, VALUE type)
{
    return Qnil;
}


#line 1 "gen/tmpl/to_a.c"
void
iter_robject_to_a(na_loop_t *const lp)
{
    size_t i, s1;
    char *p1;
    size_t *idx1;
    dtype x;
    volatile VALUE a, y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    a = rb_ary_new2(i);
    rb_ary_push(lp->args[1].value, a);
    if (idx1) {
        for (; i--;) {
            GET_DATA_INDEX(p1,idx1,dtype,x);
            y = m_data_to_num(x);
            rb_ary_push(a,y);
        }
    } else {
        for (; i--;) {
            GET_DATA_STRIDE(p1,s1,dtype,x);
            y = m_data_to_num(x);
            rb_ary_push(a,y);
        }
    }
}

/*
  Convert self to Array.
  @overload to_a
  @return [Array]
*/
static VALUE
robject_to_a(VALUE self)
{
    ndfunc_arg_in_t ain[3] = {{Qnil,0},{sym_loop_opt},{sym_option}};
    ndfunc_arg_out_t aout[1] = {{rb_cArray,0}}; // dummy?
    ndfunc_t ndf = { iter_robject_to_a, FULL_LOOP_NIP, 3, 1, ain, aout };
    return na_ndloop_cast_narray_to_rarray(&ndf, self, Qnil);
}


#line 1 "gen/tmpl/fill.c"
static void
iter_robject_fill(na_loop_t *const lp)
{
    size_t   i;
    char    *p1;
    ssize_t  s1;
    size_t  *idx1;
    VALUE    x = lp->option;
    dtype    y;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    y = m_num_to_data(x);
    if (idx1) {
        for (; i--;) {
            SET_DATA_INDEX(p1,idx1,dtype,y);
        }
    } else {
        for (; i--;) {
            SET_DATA_STRIDE(p1,s1,dtype,y);
        }
    }
}

/*
  Fill elements with other.
  @overload fill other
  @param [Numeric] other
  @return [Numo::RObject] self.
*/
static VALUE
robject_fill(VALUE self, VALUE val)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{sym_option}};
    ndfunc_t ndf = { iter_robject_fill, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, val);
    return self;
}


#line 1 "gen/tmpl/format.c"
static VALUE
format_robject(VALUE fmt, dtype* x)
{
    // fix-me
    char s[48];
    int n;

    if (NIL_P(fmt)) {
        n = m_sprintf(s,*x);
        return rb_str_new(s,n);
    }
    return rb_funcall(fmt, '%', 1, m_data_to_num(*x));
}

static void
iter_robject_format(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1;
    dtype *x;
    VALUE y;
    VALUE fmt = lp->option;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR(lp, 1, p2, s2);
    if (idx1) {
        for (; i--;) {
            x = (dtype*)(p1+*idx1); idx1++;
            y = format_robject(fmt, x);
            SET_DATA_STRIDE(p2, s2, VALUE, y);
        }
    } else {
        for (; i--;) {
            x = (dtype*)p1;         p1+=s1;
            y = format_robject(fmt, x);
            SET_DATA_STRIDE(p2, s2, VALUE, y);
        }
    }
}

/*
  Format elements into strings.
  @overload format format
  @param [String] format
  @return [Numo::RObject] array of formated strings.
*/
static VALUE
robject_format(int argc, VALUE *argv, VALUE self)
{
    VALUE fmt=Qnil;

    ndfunc_arg_in_t ain[2] = {{Qnil,0},{sym_option}};
    ndfunc_arg_out_t aout[1] = {{numo_cRObject,0}};
    ndfunc_t ndf = { iter_robject_format, FULL_LOOP_NIP, 2, 1, ain, aout };

    rb_scan_args(argc, argv, "01", &fmt);
    return na_ndloop(&ndf, 2, self, fmt);
}


#line 1 "gen/tmpl/format_to_a.c"
static void
iter_robject_format_to_a(na_loop_t *const lp)
{
    size_t  i;
    char   *p1;
    ssize_t s1;
    size_t *idx1;
    dtype *x;
    VALUE y;
    volatile VALUE a;
    VALUE fmt = lp->option;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    a = rb_ary_new2(i);
    rb_ary_push(lp->args[1].value, a);
    if (idx1) {
        for (; i--;) {
            x = (dtype*)(p1 + *idx1);  idx1++;
            y = format_robject(fmt, x);
            rb_ary_push(a,y);
        }
    } else {
        for (; i--;) {
            x = (dtype*)p1;  p1+=s1;
            y = format_robject(fmt, x);
            rb_ary_push(a,y);
        }
    }
}

/*
  Format elements into strings.
  @overload format_to_a format
  @param [String] format
  @return [Array] array of formated strings.
*/
static VALUE
robject_format_to_a(int argc, VALUE *argv, VALUE self)
{
    volatile VALUE fmt=Qnil;
    ndfunc_arg_in_t ain[3] = {{Qnil,0},{sym_loop_opt},{sym_option}};
    ndfunc_arg_out_t aout[1] = {{rb_cArray,0}}; // dummy?
    ndfunc_t ndf = { iter_robject_format_to_a, FULL_LOOP_NIP, 3, 1, ain, aout };

    rb_scan_args(argc, argv, "01", &fmt);
    return na_ndloop_cast_narray_to_rarray(&ndf, self, fmt);
}


#line 1 "gen/tmpl/inspect.c"
static VALUE
iter_robject_inspect(char *ptr, size_t pos, VALUE fmt)
{
#line 5 "gen/tmpl/inspect.c"
    return rb_inspect(*(VALUE*)(ptr+pos));
#line 9 "gen/tmpl/inspect.c"
}

/*
  Returns a string containing a human-readable representation of NArray.
  @overload inspect
  @return [String]
*/
VALUE
robject_inspect(VALUE ary)
{
    return na_ndloop_inspect(ary, iter_robject_inspect, Qnil);
}


#line 1 "gen/tmpl/each.c"
void
iter_robject_each(na_loop_t *const lp)
{
    size_t i, s1;
    char *p1;
    size_t *idx1;
    dtype x;
    VALUE y;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    if (idx1) {
        for (; i--;) {
            GET_DATA_INDEX(p1,idx1,dtype,x);
            y = m_data_to_num(x);
            rb_yield(y);
        }
    } else {
        for (; i--;) {
            GET_DATA_STRIDE(p1,s1,dtype,x);
            y = m_data_to_num(x);
            rb_yield(y);
        }
    }
}

/*
  Calls the given block once for each element in self,
  passing that element as a parameter.
  @overload each
  @return [Numo::NArray] self
  For a block {|x| ... }
  @yield [x]  x is element of NArray.
*/
static VALUE
robject_each(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    ndfunc_t ndf = {iter_robject_each, FULL_LOOP_NIP, 1,0, ain,0};

    na_ndloop(&ndf, 1, self);
    return self;
}


#line 1 "gen/tmpl/unary.c"
static void
iter_robject_map(na_loop_t *const lp)
{
    size_t  i, n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx1) {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_map(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_map(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_map(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            //
#line 49 "gen/tmpl/unary.c"
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_map(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
#line 64 "gen/tmpl/unary.c"
        }
    }
}

/*
  Unary map.
  @overload map
  @return [Numo::RObject] map of self.
*/
static VALUE
robject_map(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_map, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/each_with_index.c"
static inline void
yield_each_with_index(dtype x, size_t *c, VALUE *a, int nd, int md)
{
    int j;

    a[0] = m_data_to_num(x);
    for (j=0; j<=nd; j++) {
        a[j+1] = SIZET2NUM(c[j]);
    }
    rb_yield(rb_ary_new4(md,a));
}


void
iter_robject_each_with_index(na_loop_t *const lp)
{
    size_t i, s1;
    char *p1;
    size_t *idx1;
    dtype x;
    VALUE *a;
    size_t *c;
    int nd, md;

    c = (size_t*)(lp->opt_ptr);
    nd = lp->ndim - 1;
    md = lp->ndim + 1;
    a = ALLOCA_N(VALUE,md);

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    c[nd] = 0;
    if (idx1) {
        for (; i--;) {
            GET_DATA_INDEX(p1,idx1,dtype,x);
            yield_each_with_index(x,c,a,nd,md);
            c[nd]++;
        }
    } else {
        for (; i--;) {
            GET_DATA_STRIDE(p1,s1,dtype,x);
            yield_each_with_index(x,c,a,nd,md);
            c[nd]++;
        }
    }
}

/*
  Invokes the given block once for each element of self,
  passing that element and indices along each axis as parameters.
  @overload each_with_index
  @return [Numo::NArray] self
  For a block {|x,i,j,...| ... }
  @yield [x,i,j,...]  x is an element, i,j,... are multidimensional indices.
*/
static VALUE
robject_each_with_index(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    ndfunc_t ndf = {iter_robject_each_with_index, FULL_LOOP_NIP, 1,0, ain,0};

    na_ndloop_with_index(&ndf, 1, self);
    return self;
}


#line 1 "gen/tmpl/map_with_index.c"
static inline dtype
yield_map_with_index(dtype x, size_t *c, VALUE *a, int nd, int md)
{
    int j;
    VALUE y;

    a[0] = m_data_to_num(x);
    for (j=0; j<=nd; j++) {
        a[j+1] = SIZET2NUM(c[j]);
    }
    y = rb_yield(rb_ary_new4(md,a));
    return m_num_to_data(y);
}

void
iter_robject_map_with_index(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype x;
    VALUE *a;
    size_t *c;
    int nd, md;

    c = (size_t*)(lp->opt_ptr);
    nd = lp->ndim - 1;
    md = lp->ndim + 1;
    a = ALLOCA_N(VALUE,md);

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    c[nd] = 0;
    if (idx1) {
        if (idx2) {
            for (; i--;) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = yield_map_with_index(x,c,a,nd,md);
                SET_DATA_INDEX(p2,idx2,dtype,x);
                c[nd]++;
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = yield_map_with_index(x,c,a,nd,md);
                SET_DATA_STRIDE(p2,s2,dtype,x);
                c[nd]++;
            }
        }
    } else {
        if (idx2) {
            for (; i--;) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = yield_map_with_index(x,c,a,nd,md);
                SET_DATA_INDEX(p2,idx2,dtype,x);
                c[nd]++;
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = yield_map_with_index(x,c,a,nd,md);
                SET_DATA_STRIDE(p2,s2,dtype,x);
                c[nd]++;
            }
        }
    }
}

/*
  Invokes the given block once for each element of self,
  passing that element and indices along each axis as parameters.
  Creates a new NArray containing the values returned by the block.
  Inplace option is allowed, i.e., `nary.inplace.map` overwrites `nary`.

  @overload map_with_index

  For a block {|x,i,j,...| ... }
  @yield [x,i,j,...]  x is an element, i,j,... are multidimensional indices.

  @return [Numo::NArray] mapped array

*/
static VALUE
robject_map_with_index(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_map_with_index, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop_with_index(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary2.c"
static void
iter_robject_abs(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;
    rtype y;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    if (idx1) {
        if (idx2) {
            for (; i--;) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                y = m_abs(x);
                SET_DATA_INDEX(p2,idx2,rtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                y = m_abs(x);
                SET_DATA_STRIDE(p2,s2,rtype,y);
            }
        }
    } else {
        if (idx2) {
            for (; i--;) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                y = m_abs(x);
                SET_DATA_INDEX(p2,idx2,rtype,y);
            }
        } else {
            for (; i--;) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                y = m_abs(x);
                SET_DATA_STRIDE(p2,s2,rtype,y);
            }
        }
    }
}


/*
  abs of self.
  @overload abs
  @return [Numo::RObject] abs of self.
*/
static VALUE
robject_abs(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cRT,0}};
    ndfunc_t ndf = { iter_robject_abs, FULL_LOOP, 1, 1, ain, aout };

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_robject_add(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
#line 42 "gen/tmpl/binary.c"
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_add(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
#line 62 "gen/tmpl/binary.c"
}
#undef check_intdivzero

static VALUE
robject_add_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_add, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary add.
  @overload + other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self + other
*/
static VALUE
robject_add(VALUE self, VALUE other)
{
    
    return robject_add_self(self, other);
    
#line 97 "gen/tmpl/binary.c"
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_robject_sub(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
#line 42 "gen/tmpl/binary.c"
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_sub(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
#line 62 "gen/tmpl/binary.c"
}
#undef check_intdivzero

static VALUE
robject_sub_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_sub, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary sub.
  @overload - other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self - other
*/
static VALUE
robject_sub(VALUE self, VALUE other)
{
    
    return robject_sub_self(self, other);
    
#line 97 "gen/tmpl/binary.c"
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_robject_mul(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
#line 42 "gen/tmpl/binary.c"
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_mul(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
#line 62 "gen/tmpl/binary.c"
}
#undef check_intdivzero

static VALUE
robject_mul_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_mul, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary mul.
  @overload * other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self * other
*/
static VALUE
robject_mul(VALUE self, VALUE other)
{
    
    return robject_mul_self(self, other);
    
#line 97 "gen/tmpl/binary.c"
}


#line 1 "gen/tmpl/binary.c"
#line 2 "gen/tmpl/binary.c"
#define check_intdivzero(y)              \
    if ((y)==0) {                        \
        lp->err_type = rb_eZeroDivError; \
        return;                          \
    }

#line 11 "gen/tmpl/binary.c"
static void
iter_robject_div(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
#line 42 "gen/tmpl/binary.c"
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_div(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
#line 62 "gen/tmpl/binary.c"
}
#undef check_intdivzero

static VALUE
robject_div_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_div, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary div.
  @overload / other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self / other
*/
static VALUE
robject_div(VALUE self, VALUE other)
{
    
    return robject_div_self(self, other);
    
#line 97 "gen/tmpl/binary.c"
}


#line 1 "gen/tmpl/binary.c"
#line 2 "gen/tmpl/binary.c"
#define check_intdivzero(y)              \
    if ((y)==0) {                        \
        lp->err_type = rb_eZeroDivError; \
        return;                          \
    }

#line 11 "gen/tmpl/binary.c"
static void
iter_robject_mod(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
#line 42 "gen/tmpl/binary.c"
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_mod(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
#line 62 "gen/tmpl/binary.c"
}
#undef check_intdivzero

static VALUE
robject_mod_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_mod, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary mod.
  @overload % other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self % other
*/
static VALUE
robject_mod(VALUE self, VALUE other)
{
    
    return robject_mod_self(self, other);
    
#line 97 "gen/tmpl/binary.c"
}


#line 1 "gen/tmpl/binary2.c"
static void
iter_robject_divmod(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3, *p4;
    ssize_t  s1, s2, s3, s4;
    dtype    x, y, a, b;
    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);
    INIT_PTR(lp, 3, p4, s4);
    for (i=n; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
#line 17 "gen/tmpl/binary2.c"
        if (y==0) {
            lp->err_type = rb_eZeroDivError;
            return;
        }
#line 22 "gen/tmpl/binary2.c"
        m_divmod(x,y,a,b);
        SET_DATA_STRIDE(p3,s3,dtype,a);
        SET_DATA_STRIDE(p4,s4,dtype,b);
    }
}

static VALUE
robject_divmod_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[2] = {{cT,0},{cT,0}};
    ndfunc_t ndf = { iter_robject_divmod, STRIDE_LOOP, 2, 2, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary divmod.
  @overload divmod other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] divmod of self and other.
*/
static VALUE
robject_divmod(VALUE self, VALUE other)
{
    
    return robject_divmod_self(self, other);
    
#line 59 "gen/tmpl/binary2.c"
}


#line 1 "gen/tmpl/pow.c"
static void
iter_robject_pow(na_loop_t *const lp)
{
    size_t  i;
    char    *p1, *p2, *p3;
    ssize_t s1, s2, s3;
    dtype    x, y;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        x = m_pow(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,x);
    }
}

static void
iter_robject_pow_int32(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2, *p3;
    ssize_t s1, s2, s3;
    dtype   x;
    int32_t y;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,int32_t,y);
        x = m_pow_int(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,x);
    }
}

static VALUE
robject_pow_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_in_t ain_i[2] = {{cT,0},{numo_cInt32,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_pow, STRIDE_LOOP, 2, 1, ain, aout };
    ndfunc_t ndf_i = { iter_robject_pow_int32, STRIDE_LOOP, 2, 1, ain_i, aout };

    // fixme : use na.integer?
    if (FIXNUM_P(other) || rb_obj_is_kind_of(other,numo_cInt32)) {
        return na_ndloop(&ndf_i, 2, self, other);
    } else {
        return na_ndloop(&ndf, 2, self, other);
    }
}

/*
  Binary power.
  @overload ** other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self to the other-th power.
*/
static VALUE
robject_pow(VALUE self, VALUE other)
{
    
    return robject_pow_self(self,other);
    
#line 78 "gen/tmpl/pow.c"
}


#line 1 "gen/tmpl/unary.c"
static void
iter_robject_minus(na_loop_t *const lp)
{
    size_t  i, n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx1) {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_minus(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_minus(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_minus(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            //
#line 49 "gen/tmpl/unary.c"
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_minus(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
#line 64 "gen/tmpl/unary.c"
        }
    }
}

/*
  Unary minus.
  @overload -@
  @return [Numo::RObject] minus of self.
*/
static VALUE
robject_minus(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_minus, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary.c"
static void
iter_robject_reciprocal(na_loop_t *const lp)
{
    size_t  i, n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx1) {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_reciprocal(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_reciprocal(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_reciprocal(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            //
#line 49 "gen/tmpl/unary.c"
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_reciprocal(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
#line 64 "gen/tmpl/unary.c"
        }
    }
}

/*
  Unary reciprocal.
  @overload reciprocal
  @return [Numo::RObject] reciprocal of self.
*/
static VALUE
robject_reciprocal(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_reciprocal, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary.c"
static void
iter_robject_sign(na_loop_t *const lp)
{
    size_t  i, n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx1) {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_sign(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_sign(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_sign(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            //
#line 49 "gen/tmpl/unary.c"
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_sign(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
#line 64 "gen/tmpl/unary.c"
        }
    }
}

/*
  Unary sign.
  @overload sign
  @return [Numo::RObject] sign of self.
*/
static VALUE
robject_sign(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_sign, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary.c"
static void
iter_robject_square(na_loop_t *const lp)
{
    size_t  i, n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx1) {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_square(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_square(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_square(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            //
#line 49 "gen/tmpl/unary.c"
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_square(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
#line 64 "gen/tmpl/unary.c"
        }
    }
}

/*
  Unary square.
  @overload square
  @return [Numo::RObject] square of self.
*/
static VALUE
robject_square(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_square, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}





#line 1 "gen/tmpl/cond_binary.c"
static void
iter_robject_eq(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    BIT_DIGIT *a3;
    size_t  p3;
    ssize_t s1, s2, s3;
    dtype   x, y;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR_BIT(lp, 2, a3, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        b = (m_eq(x,y)) ? 1:0;
        STORE_BIT(a3,p3,b);
        p3+=s3;
    }
}

static VALUE
robject_eq_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_eq, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison eq other.
  @overload eq other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self eq other.
*/
static VALUE
robject_eq(VALUE self, VALUE other)
{
    
    return robject_eq_self(self, other);
    
#line 55 "gen/tmpl/cond_binary.c"
}


#line 1 "gen/tmpl/cond_binary.c"
static void
iter_robject_ne(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    BIT_DIGIT *a3;
    size_t  p3;
    ssize_t s1, s2, s3;
    dtype   x, y;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR_BIT(lp, 2, a3, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        b = (m_ne(x,y)) ? 1:0;
        STORE_BIT(a3,p3,b);
        p3+=s3;
    }
}

static VALUE
robject_ne_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_ne, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison ne other.
  @overload ne other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self ne other.
*/
static VALUE
robject_ne(VALUE self, VALUE other)
{
    
    return robject_ne_self(self, other);
    
#line 55 "gen/tmpl/cond_binary.c"
}


#line 1 "gen/tmpl/cond_binary.c"
static void
iter_robject_nearly_eq(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    BIT_DIGIT *a3;
    size_t  p3;
    ssize_t s1, s2, s3;
    dtype   x, y;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR_BIT(lp, 2, a3, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        b = (m_nearly_eq(x,y)) ? 1:0;
        STORE_BIT(a3,p3,b);
        p3+=s3;
    }
}

static VALUE
robject_nearly_eq_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_nearly_eq, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison nearly_eq other.
  @overload nearly_eq other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self nearly_eq other.
*/
static VALUE
robject_nearly_eq(VALUE self, VALUE other)
{
    
    return robject_nearly_eq_self(self, other);
    
#line 55 "gen/tmpl/cond_binary.c"
}



#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_robject_bit_and(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
#line 42 "gen/tmpl/binary.c"
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_bit_and(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
#line 62 "gen/tmpl/binary.c"
}
#undef check_intdivzero

static VALUE
robject_bit_and_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_bit_and, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary bit_and.
  @overload & other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self & other
*/
static VALUE
robject_bit_and(VALUE self, VALUE other)
{
    
    return robject_bit_and_self(self, other);
    
#line 97 "gen/tmpl/binary.c"
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_robject_bit_or(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
#line 42 "gen/tmpl/binary.c"
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_bit_or(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
#line 62 "gen/tmpl/binary.c"
}
#undef check_intdivzero

static VALUE
robject_bit_or_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_bit_or, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary bit_or.
  @overload | other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self | other
*/
static VALUE
robject_bit_or(VALUE self, VALUE other)
{
    
    return robject_bit_or_self(self, other);
    
#line 97 "gen/tmpl/binary.c"
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_robject_bit_xor(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
#line 42 "gen/tmpl/binary.c"
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_bit_xor(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
#line 62 "gen/tmpl/binary.c"
}
#undef check_intdivzero

static VALUE
robject_bit_xor_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_bit_xor, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary bit_xor.
  @overload ^ other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self ^ other
*/
static VALUE
robject_bit_xor(VALUE self, VALUE other)
{
    
    return robject_bit_xor_self(self, other);
    
#line 97 "gen/tmpl/binary.c"
}


#line 1 "gen/tmpl/unary.c"
static void
iter_robject_bit_not(na_loop_t *const lp)
{
    size_t  i, n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx1) {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_bit_not(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_bit_not(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_bit_not(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            //
#line 49 "gen/tmpl/unary.c"
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_bit_not(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
#line 64 "gen/tmpl/unary.c"
        }
    }
}

/*
  Unary bit_not.
  @overload ~
  @return [Numo::RObject] bit_not of self.
*/
static VALUE
robject_bit_not(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_bit_not, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_robject_left_shift(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
#line 42 "gen/tmpl/binary.c"
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_left_shift(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
#line 62 "gen/tmpl/binary.c"
}
#undef check_intdivzero

static VALUE
robject_left_shift_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_left_shift, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary left_shift.
  @overload << other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self << other
*/
static VALUE
robject_left_shift(VALUE self, VALUE other)
{
    
    return robject_left_shift_self(self, other);
    
#line 97 "gen/tmpl/binary.c"
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_robject_right_shift(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
#line 42 "gen/tmpl/binary.c"
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_right_shift(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
#line 62 "gen/tmpl/binary.c"
}
#undef check_intdivzero

static VALUE
robject_right_shift_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_right_shift, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary right_shift.
  @overload >> other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self >> other
*/
static VALUE
robject_right_shift(VALUE self, VALUE other)
{
    
    return robject_right_shift_self(self, other);
    
#line 97 "gen/tmpl/binary.c"
}


#line 1 "gen/tmpl/unary.c"
static void
iter_robject_floor(na_loop_t *const lp)
{
    size_t  i, n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx1) {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_floor(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_floor(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_floor(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            //
#line 49 "gen/tmpl/unary.c"
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_floor(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
#line 64 "gen/tmpl/unary.c"
        }
    }
}

/*
  Unary floor.
  @overload floor
  @return [Numo::RObject] floor of self.
*/
static VALUE
robject_floor(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_floor, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary.c"
static void
iter_robject_round(na_loop_t *const lp)
{
    size_t  i, n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx1) {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_round(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_round(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_round(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            //
#line 49 "gen/tmpl/unary.c"
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_round(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
#line 64 "gen/tmpl/unary.c"
        }
    }
}

/*
  Unary round.
  @overload round
  @return [Numo::RObject] round of self.
*/
static VALUE
robject_round(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_round, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary.c"
static void
iter_robject_ceil(na_loop_t *const lp)
{
    size_t  i, n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx1) {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_ceil(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_ceil(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_ceil(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            //
#line 49 "gen/tmpl/unary.c"
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_ceil(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
#line 64 "gen/tmpl/unary.c"
        }
    }
}

/*
  Unary ceil.
  @overload ceil
  @return [Numo::RObject] ceil of self.
*/
static VALUE
robject_ceil(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_ceil, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary.c"
static void
iter_robject_trunc(na_loop_t *const lp)
{
    size_t  i, n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx1) {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_trunc(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            for (i=0; i<n; i++) {
                GET_DATA_INDEX(p1,idx1,dtype,x);
                x = m_trunc(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_trunc(x);
                SET_DATA_INDEX(p2,idx2,dtype,x);
            }
        } else {
            //
#line 49 "gen/tmpl/unary.c"
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_trunc(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
#line 64 "gen/tmpl/unary.c"
        }
    }
}

/*
  Unary trunc.
  @overload trunc
  @return [Numo::RObject] trunc of self.
*/
static VALUE
robject_trunc(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_robject_trunc, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/cond_binary.c"
static void
iter_robject_gt(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    BIT_DIGIT *a3;
    size_t  p3;
    ssize_t s1, s2, s3;
    dtype   x, y;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR_BIT(lp, 2, a3, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        b = (m_gt(x,y)) ? 1:0;
        STORE_BIT(a3,p3,b);
        p3+=s3;
    }
}

static VALUE
robject_gt_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_gt, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison gt other.
  @overload gt other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self gt other.
*/
static VALUE
robject_gt(VALUE self, VALUE other)
{
    
    return robject_gt_self(self, other);
    
#line 55 "gen/tmpl/cond_binary.c"
}


#line 1 "gen/tmpl/cond_binary.c"
static void
iter_robject_ge(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    BIT_DIGIT *a3;
    size_t  p3;
    ssize_t s1, s2, s3;
    dtype   x, y;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR_BIT(lp, 2, a3, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        b = (m_ge(x,y)) ? 1:0;
        STORE_BIT(a3,p3,b);
        p3+=s3;
    }
}

static VALUE
robject_ge_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_ge, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison ge other.
  @overload ge other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self ge other.
*/
static VALUE
robject_ge(VALUE self, VALUE other)
{
    
    return robject_ge_self(self, other);
    
#line 55 "gen/tmpl/cond_binary.c"
}


#line 1 "gen/tmpl/cond_binary.c"
static void
iter_robject_lt(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    BIT_DIGIT *a3;
    size_t  p3;
    ssize_t s1, s2, s3;
    dtype   x, y;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR_BIT(lp, 2, a3, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        b = (m_lt(x,y)) ? 1:0;
        STORE_BIT(a3,p3,b);
        p3+=s3;
    }
}

static VALUE
robject_lt_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_lt, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison lt other.
  @overload lt other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self lt other.
*/
static VALUE
robject_lt(VALUE self, VALUE other)
{
    
    return robject_lt_self(self, other);
    
#line 55 "gen/tmpl/cond_binary.c"
}


#line 1 "gen/tmpl/cond_binary.c"
static void
iter_robject_le(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    BIT_DIGIT *a3;
    size_t  p3;
    ssize_t s1, s2, s3;
    dtype   x, y;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR_BIT(lp, 2, a3, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        b = (m_le(x,y)) ? 1:0;
        STORE_BIT(a3,p3,b);
        p3+=s3;
    }
}

static VALUE
robject_le_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_le, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison le other.
  @overload le other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self le other.
*/
static VALUE
robject_le(VALUE self, VALUE other)
{
    
    return robject_le_self(self, other);
    
#line 55 "gen/tmpl/cond_binary.c"
}






#line 1 "gen/tmpl/clip.c"
static void
iter_robject_clip(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2, *p3, *p4;
    ssize_t s1, s2, s3, s4;
    dtype   x, min, max;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);
    INIT_PTR(lp, 3, p4, s4);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,min);
        GET_DATA_STRIDE(p3,s3,dtype,max);
        if (m_gt(min,max)) {rb_raise(nary_eOperationError,"min is greater than max");}
        if (m_lt(x,min)) {x=min;}
        if (m_gt(x,max)) {x=max;}
        SET_DATA_STRIDE(p4,s4,dtype,x);
    }
}

static void
iter_robject_clip_min(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2, *p3;
    ssize_t s1, s2, s3;
    dtype   x, min;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,min);
        if (m_lt(x,min)) {x=min;}
        SET_DATA_STRIDE(p3,s3,dtype,x);
    }
}

static void
iter_robject_clip_max(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2, *p3;
    ssize_t s1, s2, s3;
    dtype   x, max;
    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,max);
        if (m_gt(x,max)) {x=max;}
        SET_DATA_STRIDE(p3,s3,dtype,x);
    }
}

/*
  Clip array elements by [min,max].
  If either of min or max is nil, one side is clipped.
  @overload clip(min,max)
  @param [Numo::NArray,Numeric] min
  @param [Numo::NArray,Numeric] max
  @return [Numo::NArray] result of clip.

  @example
      a = Numo::Int32.new(10).seq
      p a.clip(1,8)
      # Numo::Int32#shape=[10]
      # [1, 1, 2, 3, 4, 5, 6, 7, 8, 8]

      p a
      # Numo::Int32#shape=[10]
      # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

      p a.inplace.clip(3,6)
      # Numo::Int32(view)#shape=[10]
      # [3, 3, 3, 3, 4, 5, 6, 6, 6, 6]

      p a
      # Numo::Int32#shape=[10]
      # [3, 3, 3, 3, 4, 5, 6, 6, 6, 6]

      p a = Numo::Int32.new(10).seq
      # Numo::Int32#shape=[10]
      # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

      p a.clip([3,4,1,1,1,4,4,4,4,4], 8)
      # Numo::Int32#shape=[10]
      # [3, 4, 2, 3, 4, 5, 6, 7, 8, 8]
*/
static VALUE
robject_clip(VALUE self, VALUE min, VALUE max)
{
    ndfunc_arg_in_t ain[3] = {{Qnil,0},{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf_min = { iter_robject_clip_min, STRIDE_LOOP, 2, 1, ain, aout };
    ndfunc_t ndf_max = { iter_robject_clip_max, STRIDE_LOOP, 2, 1, ain, aout };
    ndfunc_t ndf_both = { iter_robject_clip, STRIDE_LOOP, 3, 1, ain, aout };

    if (RTEST(min)) {
        if (RTEST(max)) {
            return na_ndloop(&ndf_both, 3, self, min, max);
        } else {
            return na_ndloop(&ndf_min, 2, self, min);
        }
    } else {
        if (RTEST(max)) {
            return na_ndloop(&ndf_max, 2, self, max);
        }
    }
    rb_raise(rb_eArgError,"min and max are not given");
    return Qnil;
}


#line 1 "gen/tmpl/cond_unary.c"
static void
iter_robject_isnan(na_loop_t *const lp)
{
    size_t    i;
    char     *p1;
    BIT_DIGIT *a2;
    size_t    p2;
    ssize_t   s1, s2;
    size_t   *idx1;
    dtype     x;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_BIT(lp, 1, a2, p2, s2);
    if (idx1) {
        for (; i--;) {
            GET_DATA_INDEX(p1,idx1,dtype,x);
            b = (m_isnan(x)) ? 1:0;
            STORE_BIT(a2,p2,b);
            p2+=s2;
        }
    } else {
        for (; i--;) {
            GET_DATA_STRIDE(p1,s1,dtype,x);
            b = (m_isnan(x)) ? 1:0;
            STORE_BIT(a2,p2,b);
            p2+=s2;
        }
    }
}

/*
  Condition of isnan.
  @overload isnan
  @return [Numo::Bit] Condition of isnan.
*/
static VALUE
robject_isnan(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_isnan, FULL_LOOP, 1, 1, ain, aout };

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/cond_unary.c"
static void
iter_robject_isinf(na_loop_t *const lp)
{
    size_t    i;
    char     *p1;
    BIT_DIGIT *a2;
    size_t    p2;
    ssize_t   s1, s2;
    size_t   *idx1;
    dtype     x;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_BIT(lp, 1, a2, p2, s2);
    if (idx1) {
        for (; i--;) {
            GET_DATA_INDEX(p1,idx1,dtype,x);
            b = (m_isinf(x)) ? 1:0;
            STORE_BIT(a2,p2,b);
            p2+=s2;
        }
    } else {
        for (; i--;) {
            GET_DATA_STRIDE(p1,s1,dtype,x);
            b = (m_isinf(x)) ? 1:0;
            STORE_BIT(a2,p2,b);
            p2+=s2;
        }
    }
}

/*
  Condition of isinf.
  @overload isinf
  @return [Numo::Bit] Condition of isinf.
*/
static VALUE
robject_isinf(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_isinf, FULL_LOOP, 1, 1, ain, aout };

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/cond_unary.c"
static void
iter_robject_isposinf(na_loop_t *const lp)
{
    size_t    i;
    char     *p1;
    BIT_DIGIT *a2;
    size_t    p2;
    ssize_t   s1, s2;
    size_t   *idx1;
    dtype     x;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_BIT(lp, 1, a2, p2, s2);
    if (idx1) {
        for (; i--;) {
            GET_DATA_INDEX(p1,idx1,dtype,x);
            b = (m_isposinf(x)) ? 1:0;
            STORE_BIT(a2,p2,b);
            p2+=s2;
        }
    } else {
        for (; i--;) {
            GET_DATA_STRIDE(p1,s1,dtype,x);
            b = (m_isposinf(x)) ? 1:0;
            STORE_BIT(a2,p2,b);
            p2+=s2;
        }
    }
}

/*
  Condition of isposinf.
  @overload isposinf
  @return [Numo::Bit] Condition of isposinf.
*/
static VALUE
robject_isposinf(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_isposinf, FULL_LOOP, 1, 1, ain, aout };

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/cond_unary.c"
static void
iter_robject_isneginf(na_loop_t *const lp)
{
    size_t    i;
    char     *p1;
    BIT_DIGIT *a2;
    size_t    p2;
    ssize_t   s1, s2;
    size_t   *idx1;
    dtype     x;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_BIT(lp, 1, a2, p2, s2);
    if (idx1) {
        for (; i--;) {
            GET_DATA_INDEX(p1,idx1,dtype,x);
            b = (m_isneginf(x)) ? 1:0;
            STORE_BIT(a2,p2,b);
            p2+=s2;
        }
    } else {
        for (; i--;) {
            GET_DATA_STRIDE(p1,s1,dtype,x);
            b = (m_isneginf(x)) ? 1:0;
            STORE_BIT(a2,p2,b);
            p2+=s2;
        }
    }
}

/*
  Condition of isneginf.
  @overload isneginf
  @return [Numo::Bit] Condition of isneginf.
*/
static VALUE
robject_isneginf(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_isneginf, FULL_LOOP, 1, 1, ain, aout };

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/cond_unary.c"
static void
iter_robject_isfinite(na_loop_t *const lp)
{
    size_t    i;
    char     *p1;
    BIT_DIGIT *a2;
    size_t    p2;
    ssize_t   s1, s2;
    size_t   *idx1;
    dtype     x;
    BIT_DIGIT b;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_BIT(lp, 1, a2, p2, s2);
    if (idx1) {
        for (; i--;) {
            GET_DATA_INDEX(p1,idx1,dtype,x);
            b = (m_isfinite(x)) ? 1:0;
            STORE_BIT(a2,p2,b);
            p2+=s2;
        }
    } else {
        for (; i--;) {
            GET_DATA_STRIDE(p1,s1,dtype,x);
            b = (m_isfinite(x)) ? 1:0;
            STORE_BIT(a2,p2,b);
            p2+=s2;
        }
    }
}

/*
  Condition of isfinite.
  @overload isfinite
  @return [Numo::Bit] Condition of isfinite.
*/
static VALUE
robject_isfinite(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_robject_isfinite, FULL_LOOP, 1, 1, ain, aout };

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_sum(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_sum(n,p1,s1);
}
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_sum_nan(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_sum_nan(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  sum of self.
  @overload sum(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN for sum/mean etc, or, return NaN for min/max etc).
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::RObject] returns result of sum.
*/
static VALUE
robject_sum(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_sum, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_sum_nan);
  
#line 42 "gen/tmpl/accum.c"
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return robject_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_prod(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_prod(n,p1,s1);
}
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_prod_nan(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_prod_nan(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  prod of self.
  @overload prod(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN for sum/mean etc, or, return NaN for min/max etc).
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::RObject] returns result of prod.
*/
static VALUE
robject_prod(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_prod, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_prod_nan);
  
#line 42 "gen/tmpl/accum.c"
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return robject_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_mean(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_mean(n,p1,s1);
}
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_mean_nan(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_mean_nan(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  mean of self.
  @overload mean(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN for sum/mean etc, or, return NaN for min/max etc).
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::RObject] returns result of mean.
*/
static VALUE
robject_mean(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_mean, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_mean_nan);
  
#line 42 "gen/tmpl/accum.c"
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return robject_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_stddev(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(rtype*)p2 = f_stddev(n,p1,s1);
}
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_stddev_nan(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(rtype*)p2 = f_stddev_nan(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  stddev of self.
  @overload stddev(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN for sum/mean etc, or, return NaN for min/max etc).
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::RObject] returns result of stddev.
*/
static VALUE
robject_stddev(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cRT,0}};
    ndfunc_t ndf = { iter_robject_stddev, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_stddev_nan);
  
#line 42 "gen/tmpl/accum.c"
    v =  na_ndloop(&ndf, 2, self, reduce);
  
#line 46 "gen/tmpl/accum.c"
    return rb_funcall(v,rb_intern("extract"),0);
  
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_var(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(rtype*)p2 = f_var(n,p1,s1);
}
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_var_nan(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(rtype*)p2 = f_var_nan(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  var of self.
  @overload var(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN for sum/mean etc, or, return NaN for min/max etc).
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::RObject] returns result of var.
*/
static VALUE
robject_var(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cRT,0}};
    ndfunc_t ndf = { iter_robject_var, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_var_nan);
  
#line 42 "gen/tmpl/accum.c"
    v =  na_ndloop(&ndf, 2, self, reduce);
  
#line 46 "gen/tmpl/accum.c"
    return rb_funcall(v,rb_intern("extract"),0);
  
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_rms(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(rtype*)p2 = f_rms(n,p1,s1);
}
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_rms_nan(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(rtype*)p2 = f_rms_nan(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  rms of self.
  @overload rms(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN for sum/mean etc, or, return NaN for min/max etc).
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::RObject] returns result of rms.
*/
static VALUE
robject_rms(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cRT,0}};
    ndfunc_t ndf = { iter_robject_rms, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_rms_nan);
  
#line 42 "gen/tmpl/accum.c"
    v =  na_ndloop(&ndf, 2, self, reduce);
  
#line 46 "gen/tmpl/accum.c"
    return rb_funcall(v,rb_intern("extract"),0);
  
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_min(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_min(n,p1,s1);
}
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_min_nan(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_min_nan(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  min of self.
  @overload min(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN for sum/mean etc, or, return NaN for min/max etc).
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::RObject] returns result of min.
*/
static VALUE
robject_min(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_min, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_min_nan);
  
#line 42 "gen/tmpl/accum.c"
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return robject_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_max(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_max(n,p1,s1);
}
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_max_nan(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_max_nan(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  max of self.
  @overload max(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN for sum/mean etc, or, return NaN for min/max etc).
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::RObject] returns result of max.
*/
static VALUE
robject_max(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_max, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_max_nan);
  
#line 42 "gen/tmpl/accum.c"
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return robject_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_ptp(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_ptp(n,p1,s1);
}
#line 2 "gen/tmpl/accum.c"
static void
iter_robject_ptp_nan(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_ptp_nan(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  ptp of self.
  @overload ptp(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN for sum/mean etc, or, return NaN for min/max etc).
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::RObject] returns result of ptp.
*/
static VALUE
robject_ptp(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_ptp, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_ptp_nan);
  
#line 42 "gen/tmpl/accum.c"
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return robject_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum_index.c"

#line 3 "gen/tmpl/accum_index.c"
#define idx_t int64_t
static void
iter_robject_max_index_index64(na_loop_t *const lp)
{
    size_t   n, idx;
    char    *d_ptr, *i_ptr, *o_ptr;
    ssize_t  d_step, i_step;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, d_ptr, d_step);

    idx = f_max_index(n,d_ptr,d_step);

    INIT_PTR(lp, 1, i_ptr, i_step);
    o_ptr = NDL_PTR(lp,2);
    *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
}
#undef idx_t

#line 3 "gen/tmpl/accum_index.c"
#define idx_t int32_t
static void
iter_robject_max_index_index32(na_loop_t *const lp)
{
    size_t   n, idx;
    char    *d_ptr, *i_ptr, *o_ptr;
    ssize_t  d_step, i_step;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, d_ptr, d_step);

    idx = f_max_index(n,d_ptr,d_step);

    INIT_PTR(lp, 1, i_ptr, i_step);
    o_ptr = NDL_PTR(lp,2);
    *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
}
#undef idx_t

#line 3 "gen/tmpl/accum_index.c"
#define idx_t int64_t
static void
iter_robject_max_index_index64_nan(na_loop_t *const lp)
{
    size_t   n, idx;
    char    *d_ptr, *i_ptr, *o_ptr;
    ssize_t  d_step, i_step;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, d_ptr, d_step);

    idx = f_max_index_nan(n,d_ptr,d_step);

    INIT_PTR(lp, 1, i_ptr, i_step);
    o_ptr = NDL_PTR(lp,2);
    *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
}
#undef idx_t

#line 3 "gen/tmpl/accum_index.c"
#define idx_t int32_t
static void
iter_robject_max_index_index32_nan(na_loop_t *const lp)
{
    size_t   n, idx;
    char    *d_ptr, *i_ptr, *o_ptr;
    ssize_t  d_step, i_step;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, d_ptr, d_step);

    idx = f_max_index_nan(n,d_ptr,d_step);

    INIT_PTR(lp, 1, i_ptr, i_step);
    o_ptr = NDL_PTR(lp,2);
    *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
}
#undef idx_t

#line 23 "gen/tmpl/accum_index.c"
/*
  max_index. Return an index of result.
  @overload max_index(axis:nil, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (return NaN posision if exist).
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @return [Integer,Numo::Int] returns result index of max_index.
  @example
      Numo::NArray[3,4,1,2].min_index => 3
 */
static VALUE
robject_max_index(int argc, VALUE *argv, VALUE self)
{
    narray_t *na;
    VALUE idx, reduce;
    ndfunc_arg_in_t ain[3] = {{Qnil,0},{Qnil,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{0,0,0}};
    ndfunc_t ndf = {0, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE|NDF_EXTRACT, 3,1, ain,aout};

    GetNArray(self,na);
    if (na->ndim==0) {
        return INT2FIX(0);
    }
    if (na->size > (~(u_int32_t)0)) {
        aout[0].type = numo_cInt64;
        idx = nary_new(numo_cInt64, na->ndim, na->shape);
        ndf.func = iter_robject_max_index_index64;
      
        reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_max_index_index64_nan);
      
#line 58 "gen/tmpl/accum_index.c"
    } else {
        aout[0].type = numo_cInt32;
        idx = nary_new(numo_cInt32, na->ndim, na->shape);
        ndf.func = iter_robject_max_index_index32;
      
        reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_max_index_index32_nan);
      
#line 67 "gen/tmpl/accum_index.c"
    }
    rb_funcall(idx, rb_intern("seq"), 0);

    return na_ndloop(&ndf, 3, self, idx, reduce);
}


#line 1 "gen/tmpl/accum_index.c"

#line 3 "gen/tmpl/accum_index.c"
#define idx_t int64_t
static void
iter_robject_min_index_index64(na_loop_t *const lp)
{
    size_t   n, idx;
    char    *d_ptr, *i_ptr, *o_ptr;
    ssize_t  d_step, i_step;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, d_ptr, d_step);

    idx = f_min_index(n,d_ptr,d_step);

    INIT_PTR(lp, 1, i_ptr, i_step);
    o_ptr = NDL_PTR(lp,2);
    *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
}
#undef idx_t

#line 3 "gen/tmpl/accum_index.c"
#define idx_t int32_t
static void
iter_robject_min_index_index32(na_loop_t *const lp)
{
    size_t   n, idx;
    char    *d_ptr, *i_ptr, *o_ptr;
    ssize_t  d_step, i_step;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, d_ptr, d_step);

    idx = f_min_index(n,d_ptr,d_step);

    INIT_PTR(lp, 1, i_ptr, i_step);
    o_ptr = NDL_PTR(lp,2);
    *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
}
#undef idx_t

#line 3 "gen/tmpl/accum_index.c"
#define idx_t int64_t
static void
iter_robject_min_index_index64_nan(na_loop_t *const lp)
{
    size_t   n, idx;
    char    *d_ptr, *i_ptr, *o_ptr;
    ssize_t  d_step, i_step;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, d_ptr, d_step);

    idx = f_min_index_nan(n,d_ptr,d_step);

    INIT_PTR(lp, 1, i_ptr, i_step);
    o_ptr = NDL_PTR(lp,2);
    *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
}
#undef idx_t

#line 3 "gen/tmpl/accum_index.c"
#define idx_t int32_t
static void
iter_robject_min_index_index32_nan(na_loop_t *const lp)
{
    size_t   n, idx;
    char    *d_ptr, *i_ptr, *o_ptr;
    ssize_t  d_step, i_step;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, d_ptr, d_step);

    idx = f_min_index_nan(n,d_ptr,d_step);

    INIT_PTR(lp, 1, i_ptr, i_step);
    o_ptr = NDL_PTR(lp,2);
    *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
}
#undef idx_t

#line 23 "gen/tmpl/accum_index.c"
/*
  min_index. Return an index of result.
  @overload min_index(axis:nil, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (return NaN posision if exist).
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @return [Integer,Numo::Int] returns result index of min_index.
  @example
      Numo::NArray[3,4,1,2].min_index => 3
 */
static VALUE
robject_min_index(int argc, VALUE *argv, VALUE self)
{
    narray_t *na;
    VALUE idx, reduce;
    ndfunc_arg_in_t ain[3] = {{Qnil,0},{Qnil,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{0,0,0}};
    ndfunc_t ndf = {0, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE|NDF_EXTRACT, 3,1, ain,aout};

    GetNArray(self,na);
    if (na->ndim==0) {
        return INT2FIX(0);
    }
    if (na->size > (~(u_int32_t)0)) {
        aout[0].type = numo_cInt64;
        idx = nary_new(numo_cInt64, na->ndim, na->shape);
        ndf.func = iter_robject_min_index_index64;
      
        reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_min_index_index64_nan);
      
#line 58 "gen/tmpl/accum_index.c"
    } else {
        aout[0].type = numo_cInt32;
        idx = nary_new(numo_cInt32, na->ndim, na->shape);
        ndf.func = iter_robject_min_index_index32;
      
        reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_min_index_index32_nan);
      
#line 67 "gen/tmpl/accum_index.c"
    }
    rb_funcall(idx, rb_intern("seq"), 0);

    return na_ndloop(&ndf, 3, self, idx, reduce);
}


#line 1 "gen/tmpl/minmax.c"
#line 2 "gen/tmpl/minmax.c"
static void
iter_robject_minmax(na_loop_t *const lp)
{
    size_t   n;
    char    *p1;
    ssize_t  s1;
    dtype    xmin,xmax;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);

    f_minmax(n,p1,s1,&xmin,&xmax);

    *(dtype*)(lp->args[1].ptr + lp->args[1].iter[0].pos) = xmin;
    *(dtype*)(lp->args[2].ptr + lp->args[2].iter[0].pos) = xmax;
}
#line 2 "gen/tmpl/minmax.c"
static void
iter_robject_minmax_nan(na_loop_t *const lp)
{
    size_t   n;
    char    *p1;
    ssize_t  s1;
    dtype    xmin,xmax;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);

    f_minmax_nan(n,p1,s1,&xmin,&xmax);

    *(dtype*)(lp->args[1].ptr + lp->args[1].iter[0].pos) = xmin;
    *(dtype*)(lp->args[2].ptr + lp->args[2].iter[0].pos) = xmax;
}

#line 20 "gen/tmpl/minmax.c"
/*
  minmax of self.
  @overload minmax(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (return NaN if exist).
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::RObject,Numo::RObject] min and max of self.
*/
static VALUE
robject_minmax(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[2] = {{cT,0},{cT,0}};
    ndfunc_t ndf = {iter_robject_minmax, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE|NDF_EXTRACT, 2,2, ain,aout};

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_minmax_nan);
  
#line 45 "gen/tmpl/minmax.c"
    return na_ndloop(&ndf, 2, self, reduce);
}


#line 1 "gen/tmpl/cum.c"
#line 2 "gen/tmpl/cum.c"
static void
iter_robject_cumsum(na_loop_t *const lp)
{
    size_t   i;
    char    *p1, *p2;
    ssize_t  s1, s2;
    dtype    x, y;

    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    //printf("i=%lu p1=%lx s1=%lu p2=%lx s2=%lu\n",i,(size_t)p1,s1,(size_t)p2,s2);

    GET_DATA_STRIDE(p1,s1,dtype,x);
    SET_DATA_STRIDE(p2,s2,dtype,x);
    //printf("i=%lu x=%f\n",i,x);
    for (i--; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,y);
        m_cumsum(x,y);
        SET_DATA_STRIDE(p2,s2,dtype,x);
        //printf("i=%lu x=%f\n",i,x);
    }
}
#line 2 "gen/tmpl/cum.c"
static void
iter_robject_cumsum_nan(na_loop_t *const lp)
{
    size_t   i;
    char    *p1, *p2;
    ssize_t  s1, s2;
    dtype    x, y;

    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    //printf("i=%lu p1=%lx s1=%lu p2=%lx s2=%lu\n",i,(size_t)p1,s1,(size_t)p2,s2);

    GET_DATA_STRIDE(p1,s1,dtype,x);
    SET_DATA_STRIDE(p2,s2,dtype,x);
    //printf("i=%lu x=%f\n",i,x);
    for (i--; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,y);
        m_cumsum_nan(x,y);
        SET_DATA_STRIDE(p2,s2,dtype,x);
        //printf("i=%lu x=%f\n",i,x);
    }
}

#line 27 "gen/tmpl/cum.c"
/*
  cumsum of self.
  @overload cumsum(axis:nil, nan:false)
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN if exists).
  @return [Numo::RObject] cumsum of self.
*/
static VALUE
robject_cumsum(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_cumsum, STRIDE_LOOP|NDF_FLAT_REDUCE|NDF_CUM,
                     2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_cumsum_nan);
  
#line 48 "gen/tmpl/cum.c"
    return na_ndloop(&ndf, 2, self, reduce);
}


#line 1 "gen/tmpl/cum.c"
#line 2 "gen/tmpl/cum.c"
static void
iter_robject_cumprod(na_loop_t *const lp)
{
    size_t   i;
    char    *p1, *p2;
    ssize_t  s1, s2;
    dtype    x, y;

    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    //printf("i=%lu p1=%lx s1=%lu p2=%lx s2=%lu\n",i,(size_t)p1,s1,(size_t)p2,s2);

    GET_DATA_STRIDE(p1,s1,dtype,x);
    SET_DATA_STRIDE(p2,s2,dtype,x);
    //printf("i=%lu x=%f\n",i,x);
    for (i--; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,y);
        m_cumprod(x,y);
        SET_DATA_STRIDE(p2,s2,dtype,x);
        //printf("i=%lu x=%f\n",i,x);
    }
}
#line 2 "gen/tmpl/cum.c"
static void
iter_robject_cumprod_nan(na_loop_t *const lp)
{
    size_t   i;
    char    *p1, *p2;
    ssize_t  s1, s2;
    dtype    x, y;

    INIT_COUNTER(lp, i);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    //printf("i=%lu p1=%lx s1=%lu p2=%lx s2=%lu\n",i,(size_t)p1,s1,(size_t)p2,s2);

    GET_DATA_STRIDE(p1,s1,dtype,x);
    SET_DATA_STRIDE(p2,s2,dtype,x);
    //printf("i=%lu x=%f\n",i,x);
    for (i--; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,y);
        m_cumprod_nan(x,y);
        SET_DATA_STRIDE(p2,s2,dtype,x);
        //printf("i=%lu x=%f\n",i,x);
    }
}

#line 27 "gen/tmpl/cum.c"
/*
  cumprod of self.
  @overload cumprod(axis:nil, nan:false)
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN if exists).
  @return [Numo::RObject] cumprod of self.
*/
static VALUE
robject_cumprod(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_cumprod, STRIDE_LOOP|NDF_FLAT_REDUCE|NDF_CUM,
                     2, 1, ain, aout };

  
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_robject_cumprod_nan);
  
#line 48 "gen/tmpl/cum.c"
    return na_ndloop(&ndf, 2, self, reduce);
}


#line 1 "gen/tmpl/accum_binary.c"
//
static void
iter_robject_mulsum(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    if (s3==0) {
        dtype z;
        // Reduce loop
        GET_DATA(p3,dtype,z);
        for (i=0; i<n; i++) {
            dtype x, y;
            GET_DATA_STRIDE(p1,s1,dtype,x);
            GET_DATA_STRIDE(p2,s2,dtype,y);
            m_mulsum(x,y,z);
        }
        SET_DATA(p3,dtype,z);
        return;
    } else {
        for (i=0; i<n; i++) {
            dtype x, y, z;
            GET_DATA_STRIDE(p1,s1,dtype,x);
            GET_DATA_STRIDE(p2,s2,dtype,y);
            GET_DATA(p3,dtype,z);
            m_mulsum(x,y,z);
            SET_DATA_STRIDE(p3,s3,dtype,z);
        }
    }
}
//
#line 2 "gen/tmpl/accum_binary.c"
static void
iter_robject_mulsum_nan(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    if (s3==0) {
        dtype z;
        // Reduce loop
        GET_DATA(p3,dtype,z);
        for (i=0; i<n; i++) {
            dtype x, y;
            GET_DATA_STRIDE(p1,s1,dtype,x);
            GET_DATA_STRIDE(p2,s2,dtype,y);
            m_mulsum_nan(x,y,z);
        }
        SET_DATA(p3,dtype,z);
        return;
    } else {
        for (i=0; i<n; i++) {
            dtype x, y, z;
            GET_DATA_STRIDE(p1,s1,dtype,x);
            GET_DATA_STRIDE(p2,s2,dtype,y);
            GET_DATA(p3,dtype,z);
            m_mulsum_nan(x,y,z);
            SET_DATA_STRIDE(p3,s3,dtype,z);
        }
    }
}
//

static VALUE
robject_mulsum_self(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    VALUE naryv[2];
    ndfunc_arg_in_t ain[4] = {{cT,0},{cT,0},{sym_reduce,0},{sym_init,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_mulsum, STRIDE_LOOP_NIP, 4, 1, ain, aout };

    if (argc < 1) {
        rb_raise(rb_eArgError,"wrong number of arguments (%d for >=1)",argc);
    }
    // should fix below: [self.ndim,other.ndim].max or?
    naryv[0] = self;
    naryv[1] = argv[0];
    //
    reduce = na_reduce_dimension(argc-1, argv+1, 2, naryv, &ndf, iter_robject_mulsum_nan);
    //

#line 60 "gen/tmpl/accum_binary.c"
    v =  na_ndloop(&ndf, 4, self, argv[0], reduce, m_mulsum_init);
    return robject_extract(v);
}

/*
  Binary mulsum.

  @overload mulsum(other, axis:nil, keepdims:false, nan:false)
  @param [Numo::NArray,Numeric] other
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @param [TrueClass] nan (keyword) If true, apply NaN-aware algorithm (avoid NaN if exists).
  @return [Numo::NArray] mulsum of self and other.
*/
static VALUE
robject_mulsum(int argc, VALUE *argv, VALUE self)
{
    //
#line 86 "gen/tmpl/accum_binary.c"
    if (argc < 1) {
        rb_raise(rb_eArgError,"wrong number of arguments (%d for >=1)",argc);
    }
    //
    return robject_mulsum_self(argc, argv, self);
    //
#line 100 "gen/tmpl/accum_binary.c"
}


#line 1 "gen/tmpl/seq.c"
#line 4 "gen/tmpl/seq.c"
typedef dtype seq_data_t;

#line 8 "gen/tmpl/seq.c"
typedef size_t seq_count_t;

#line 13 "gen/tmpl/seq.c"
typedef struct {
    seq_data_t beg;
    seq_data_t step;
    seq_count_t count;
} seq_opt_t;

static void
iter_robject_seq(na_loop_t *const lp)
{
    size_t  i;
    char   *p1;
    ssize_t s1;
    size_t *idx1;
    dtype   x;
    seq_data_t beg, step;
    seq_count_t c;
    seq_opt_t *g;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    g = (seq_opt_t*)(lp->opt_ptr);
    beg  = g->beg;
    step = g->step;
    c    = g->count;
    if (idx1) {
        for (; i--;) {
            x = f_seq(beg,step,c++);
            *(dtype*)(p1+*idx1) = x;
            idx1++;
        }
    } else {
        for (; i--;) {
            x = f_seq(beg,step,c++);
            *(dtype*)(p1) = x;
            p1 += s1;
        }
    }
    g->count = c;
}

/*
  Set linear sequence of numbers to self. The sequence is obtained from
     beg+i*step
  where i is 1-dimensional index.
  @overload seq([beg,[step]])
  @param [Numeric] beg  begining of sequence. (default=0)
  @param [Numeric] step  step of sequence. (default=1)
  @return [Numo::RObject] self.
  @example
    Numo::DFloat.new(6).seq(1,-0.2)
    => Numo::DFloat#shape=[6]
       [1, 0.8, 0.6, 0.4, 0.2, 0]
    Numo::DComplex.new(6).seq(1,-0.2+0.2i)
    => Numo::DComplex#shape=[6]
       [1+0i, 0.8+0.2i, 0.6+0.4i, 0.4+0.6i, 0.2+0.8i, 0+1i]
*/
static VALUE
robject_seq(int argc, VALUE *args, VALUE self)
{
    seq_opt_t *g;
    VALUE vbeg=Qnil, vstep=Qnil;
    ndfunc_arg_in_t ain[1] = {{OVERWRITE,0}};
    ndfunc_t ndf = {iter_robject_seq, FULL_LOOP, 1,0, ain,0};

    g = ALLOCA_N(seq_opt_t,1);
    g->beg = m_zero;
    g->step = m_one;
    g->count = 0;
    rb_scan_args(argc, args, "02", &vbeg, &vstep);
#line 86 "gen/tmpl/seq.c"
    if (vbeg!=Qnil) {g->beg = m_num_to_data(vbeg);}
    if (vstep!=Qnil) {g->step = m_num_to_data(vstep);}

#line 90 "gen/tmpl/seq.c"
    na_ndloop3(&ndf, g, 1, self);
    return self;
}


#line 1 "gen/tmpl/logseq.c"
typedef struct {
    seq_data_t beg;
    seq_data_t step;
    seq_data_t base;
    seq_count_t count;
} logseq_opt_t;

static void
iter_robject_logseq(na_loop_t *const lp)
{
    size_t  i;
    char   *p1;
    ssize_t s1;
    size_t *idx1;
    dtype   x;
    seq_data_t beg, step, base;
    seq_count_t c;
    logseq_opt_t *g;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    g = (logseq_opt_t*)(lp->opt_ptr);
    beg  = g->beg;
    step = g->step;
    base = g->base;
    c    = g->count;
    if (idx1) {
        for (; i--;) {
            x = f_seq(beg,step,c++);
            *(dtype*)(p1+*idx1) = m_pow(base,x);
            idx1++;
        }
    } else {
        for (; i--;) {
            x = f_seq(beg,step,c++);
            *(dtype*)(p1) = m_pow(base,x);
            p1 += s1;
        }
    }
    g->count = c;
}

/*
  Set logarithmic sequence of numbers to self. The sequence is obtained from
     base**(beg+i*step)
  where i is 1-dimensional index.
  Applicable classes: DFloat, SFloat, DComplex, SCopmplex.

  @overload logseq(beg,step,[base])
  @param [Numeric] beg  The begining of sequence.
  @param [Numeric] step  The step of sequence.
  @param [Numeric] base  The base of log space. (default=10)
  @return [Numo::RObject] self.

  @example
    Numo::DFloat.new(5).logseq(4,-1,2)
    => Numo::DFloat#shape=[5]
      [16, 8, 4, 2, 1]
    Numo::DComplex.new(5).logseq(0,1i*Math::PI/3,Math::E)
    => Numo::DComplex#shape=[5]
      [1+7.26156e-310i, 0.5+0.866025i, -0.5+0.866025i, -1+1.22465e-16i, ...]
*/
static VALUE
robject_logseq(int argc, VALUE *args, VALUE self)
{
    logseq_opt_t *g;
    VALUE vbeg, vstep, vbase;
    ndfunc_arg_in_t ain[1] = {{OVERWRITE,0}};
    ndfunc_t ndf = {iter_robject_logseq, FULL_LOOP, 1,0, ain,0};

    g = ALLOCA_N(logseq_opt_t,1);
    rb_scan_args(argc, args, "21", &vbeg, &vstep, &vbase);
    g->beg = m_num_to_data(vbeg);
    g->step = m_num_to_data(vstep);
    if (vbase==Qnil) {
        g->base = m_from_real(10);
    } else {
        g->base = m_num_to_data(vbase);
    }
    na_ndloop3(&ndf, g, 1, self);
    return self;
}


#line 1 "gen/tmpl/eye.c"
static void
iter_robject_eye(na_loop_t *const lp)
{
    size_t   n0, n1;
    size_t   i0, i1;
    ssize_t  s0, s1;
    char    *p0, *p1;
    char    *g;
    ssize_t kofs;
    dtype   data;

    g = (char*)(lp->opt_ptr);
    kofs = *(ssize_t*)g;
    data = *(dtype*)(g+sizeof(ssize_t));

    n0 = lp->args[0].shape[0];
    n1 = lp->args[0].shape[1];
    s0 = lp->args[0].iter[0].step;
    s1 = lp->args[0].iter[1].step;
    p0 = NDL_PTR(lp,0);

    for (i0=0; i0 < n0; i0++) {
        p1 = p0;
        for (i1=0; i1 < n1; i1++) {
            *(dtype*)p1 = (i0+kofs==i1) ? data : m_zero;
            p1 += s1;
        }
        p0 += s0;
    }
}

/*
  Eye: Set a value to diagonal components, set 0 to non-diagonal components.
  @overload eye([element,offset])
  @param [Numeric] element  Diagonal element to be stored. Default is 1.
  @param [Integer] offset Diagonal offset from the main diagonal.  The
      default is 0. k>0 for diagonals above the main diagonal, and k<0
      for diagonals below the main diagonal.
  @return [Numo::RObject] eye of self.
*/
static VALUE
robject_eye(int argc, VALUE *argv, VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{OVERWRITE,2}};
    ndfunc_t ndf = {iter_robject_eye, NO_LOOP, 1,0, ain,0};
    ssize_t kofs;
    dtype data;
    char *g;
    int nd;
    narray_t *na;

    // check arguments
    if (argc > 2) {
        rb_raise(rb_eArgError,"too many arguments (%d for 0..2)",argc);
    } else if (argc == 2) {
        data = m_num_to_data(argv[0]);
        kofs = NUM2SSIZET(argv[1]);
    } else if (argc == 1) {
        data = m_num_to_data(argv[0]);
        kofs = 0;
    } else {
        data = m_one;
        kofs = 0;
    }

    GetNArray(self,na);
    nd = na->ndim;
    if (nd < 2) {
        rb_raise(nary_eDimensionError,"less than 2-d array");
    }

    // Diagonal offset from the main diagonal.
    if (kofs >= 0) {
        if ((size_t)(kofs) >= na->shape[nd-1]) {
            rb_raise(rb_eArgError,"invalid diagonal offset(%"SZF"d) for "
                     "last dimension size(%"SZF"d)",kofs,na->shape[nd-1]);
        }
    } else {
        if ((size_t)(-kofs) >= na->shape[nd-2]) {
            rb_raise(rb_eArgError,"invalid diagonal offset(%"SZF"d) for "
                     "last-1 dimension size(%"SZF"d)",kofs,na->shape[nd-2]);
        }
    }

    g = ALLOCA_N(char,sizeof(ssize_t)+sizeof(dtype));
    *(ssize_t*)g = kofs;
    *(dtype*)(g+sizeof(ssize_t)) = data;

    na_ndloop3(&ndf, g, 1, self);
    return self;
}



#line 1 "gen/tmpl/rand.c"


#line 69 "gen/tmpl/rand.c"
typedef struct {
    dtype low;
    dtype max;
} rand_opt_t;

static void
iter_robject_rand(na_loop_t *const lp)
{
    size_t   i;
    char    *p1;
    ssize_t  s1;
    size_t  *idx1;
    dtype    x;
    rand_opt_t *g;
    dtype    low;
    dtype max;
    

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    g = (rand_opt_t*)(lp->opt_ptr);
    low = g->low;
    max = g->max;
    

    if (idx1) {
        for (; i--;) {
            x = m_add(m_rand(max),low);
            SET_DATA_INDEX(p1,idx1,dtype,x);
        }
    } else {
        for (; i--;) {
            x = m_add(m_rand(max),low);
            SET_DATA_STRIDE(p1,s1,dtype,x);
        }
    }
}


/*
  Generate uniformly distributed random numbers on self narray.
  @overload rand([[low],high])
  @param [Numeric] low  lower inclusive boundary of random numbers. (default=0)
  @param [Numeric] high  upper exclusive boundary of random numbers. (default=1 or 1+1i for complex types)
  @return [Numo::RObject] self.
  @example
    Numo::DFloat.new(6).rand
    => Numo::DFloat#shape=[6]
       [0.0617545, 0.373067, 0.794815, 0.201042, 0.116041, 0.344032]
    Numo::DComplex.new(6).rand(5+5i)
    => Numo::DComplex#shape=[6]
       [2.69974+3.68908i, 0.825443+0.254414i, 0.540323+0.34354i, 4.52061+2.39322i, ...]
    Numo::Int32.new(6).rand(2,5)
    => Numo::Int32#shape=[6]
       [4, 3, 3, 2, 4, 2]
*/
static VALUE
robject_rand(int argc, VALUE *args, VALUE self)
{
    rand_opt_t g;
    VALUE v1=Qnil, v2=Qnil;
    dtype high;
    ndfunc_arg_in_t ain[1] = {{OVERWRITE,0}};
    ndfunc_t ndf = {iter_robject_rand, FULL_LOOP, 1,0, ain,0};

    
#line 140 "gen/tmpl/rand.c"
    rb_scan_args(argc, args, "02", &v1, &v2);
    if (v2==Qnil) {
        g.low = m_zero;
        if (v1==Qnil) {
            
#line 147 "gen/tmpl/rand.c"
            g.max = high = m_one;
            
        } else {
            g.max = high = m_num_to_data(v1);
        }
    
    } else {
        g.low = m_num_to_data(v1);
        high = m_num_to_data(v2);
        g.max = m_sub(high,g.low);
    }
    
#line 163 "gen/tmpl/rand.c"
    na_ndloop3(&ndf, &g, 1, self);
    return self;
}


#line 1 "gen/tmpl/poly.c"
static void
iter_robject_poly(na_loop_t *const lp)
{
    size_t  i;
    dtype  x, y, a;

    x = *(dtype*)(lp->args[0].ptr + lp->args[0].iter[0].pos);
    i = lp->narg - 2;
    y = *(dtype*)(lp->args[i].ptr + lp->args[i].iter[0].pos);
    for (; --i;) {
        y = m_mul(x,y);
        a = *(dtype*)(lp->args[i].ptr + lp->args[i].iter[0].pos);
        y = m_add(y,a);
    }
    i = lp->narg - 1;
    *(dtype*)(lp->args[i].ptr + lp->args[i].iter[0].pos) = y;
}

/*
  Polynomial.: a0 + a1*x + a2*x**2 + a3*x**3 + ... + an*x**n
  @overload poly a0, a1, ...
  @param [Numo::NArray,Numeric] a0
  @param [Numo::NArray,Numeric] a1 , ...
  @return [Numo::RObject]
*/
static VALUE
robject_poly(VALUE self, VALUE args)
{
    int argc, i;
    VALUE *argv;
    volatile VALUE v, a;
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_robject_poly, NO_LOOP, 0, 1, 0, aout };

    argc = RARRAY_LEN(args);
    ndf.nin = argc+1;
    ndf.ain = ALLOCA_N(ndfunc_arg_in_t,argc+1);
    for (i=0; i<argc+1; i++) {
        ndf.ain[i].type = cT;
    }
    argv = ALLOCA_N(VALUE,argc+1);
    argv[0] = self;
    for (i=0; i<argc; i++) {
        argv[i+1] = RARRAY_PTR(args)[i];
    }
    a = rb_ary_new4(argc+1, argv);
    v = na_ndloop2(&ndf, a);
    return robject_extract(v);
}



#line 32 "gen/tmpl/lib.c"
void
Init_numo_robject(void)
{
    VALUE hCast, mNumo;

    mNumo = rb_define_module("Numo");

    
    id_ne = rb_intern("!=");
    id_lt = rb_intern("<");
    id_left_shift = rb_intern("<<");
    id_le = rb_intern("<=");
    id_ufo = rb_intern("<=>");
    id_eq = rb_intern("==");
    id_gt = rb_intern(">");
    id_ge = rb_intern(">=");
    id_right_shift = rb_intern(">>");
    id_abs = rb_intern("abs");
    id_bit_and = rb_intern("bit_and");
    id_bit_not = rb_intern("bit_not");
    id_bit_or = rb_intern("bit_or");
    id_bit_xor = rb_intern("bit_xor");
    id_cast = rb_intern("cast");
    id_ceil = rb_intern("ceil");
    id_copysign = rb_intern("copysign");
    id_divmod = rb_intern("divmod");
    id_eq = rb_intern("eq");
    id_finite_p = rb_intern("finite?");
    id_floor = rb_intern("floor");
    id_infinite_p = rb_intern("infinite?");
    id_minus = rb_intern("minus");
    id_mulsum = rb_intern("mulsum");
    id_nan_p = rb_intern("nan?");
    id_ne = rb_intern("ne");
    id_nearly_eq = rb_intern("nearly_eq");
    id_pow = rb_intern("pow");
    id_reciprocal = rb_intern("reciprocal");
    id_round = rb_intern("round");
    id_square = rb_intern("square");
    id_truncate = rb_intern("truncate");


#line 1 "gen/tmpl/init_class.c"
    /*
      Document-class: Numo::RObject
      
    */
    cT = rb_define_class_under(mNumo, "RObject", cNArray);

  

#line 12 "gen/tmpl/init_class.c"
    hCast = rb_hash_new();
    rb_define_const(cT, "UPCAST", hCast);
    rb_hash_aset(hCast, rb_cArray,   cT);
    
    #ifdef RUBY_INTEGER_UNIFICATION
    rb_hash_aset(hCast, rb_cInteger, cT);
    #else
    rb_hash_aset(hCast, rb_cFixnum, cT);
    rb_hash_aset(hCast, rb_cBignum, cT);
    #endif
    rb_hash_aset(hCast, rb_cFloat, cT);
    rb_hash_aset(hCast, rb_cComplex, cT);
    rb_hash_aset(hCast, numo_cDComplex, numo_cRObject);
    rb_hash_aset(hCast, numo_cSComplex, numo_cRObject);
    rb_hash_aset(hCast, numo_cDFloat, numo_cRObject);
    rb_hash_aset(hCast, numo_cSFloat, numo_cRObject);
    rb_hash_aset(hCast, numo_cInt64, numo_cRObject);
    rb_hash_aset(hCast, numo_cInt32, numo_cRObject);
    rb_hash_aset(hCast, numo_cInt16, numo_cRObject);
    rb_hash_aset(hCast, numo_cInt8, numo_cRObject);
    rb_hash_aset(hCast, numo_cUInt64, numo_cRObject);
    rb_hash_aset(hCast, numo_cUInt32, numo_cRObject);
    rb_hash_aset(hCast, numo_cUInt16, numo_cRObject);
    rb_hash_aset(hCast, numo_cUInt8, numo_cRObject);

    
    /**/
    rb_define_const(cT,"ELEMENT_BIT_SIZE",INT2FIX(sizeof(dtype)*8));
    /**/
    rb_define_const(cT,"ELEMENT_BYTE_SIZE",INT2FIX(sizeof(dtype)));
    /**/
    rb_define_const(cT,"CONTIGUOUS_STRIDE",INT2FIX(sizeof(dtype)));
    rb_undef_method(rb_singleton_class(cT),"from_binary");
    rb_undef_method(cT,"to_binary");
    rb_undef_method(cT,"swap_byte");
    rb_undef_method(cT,"to_network");
    rb_undef_method(cT,"to_vacs");
    rb_undef_method(cT,"to_host");
    rb_undef_method(cT,"to_swapped");
    rb_define_alloc_func(cT, robject_s_alloc_func);
    rb_define_method(cT, "allocate", robject_allocate, 0);
    rb_define_method(cT, "extract", robject_extract, 0);
    
    rb_define_method(cT, "store", robject_store, 1);
    
    
    rb_define_singleton_method(cT, "cast", robject_s_cast, 1);
    rb_define_method(cT, "[]", robject_aref, -1);
    rb_define_method(cT, "[]=", robject_aset, -1);
    rb_define_method(cT, "coerce_cast", robject_coerce_cast, 1);
    rb_define_method(cT, "to_a", robject_to_a, 0);
    rb_define_method(cT, "fill", robject_fill, 1);
    rb_define_method(cT, "format", robject_format, -1);
    rb_define_method(cT, "format_to_a", robject_format_to_a, -1);
    rb_define_method(cT, "inspect", robject_inspect, 0);
    rb_define_method(cT, "each", robject_each, 0);
    rb_define_method(cT, "map", robject_map, 0);
    rb_define_method(cT, "each_with_index", robject_each_with_index, 0);
    rb_define_method(cT, "map_with_index", robject_map_with_index, 0);
    rb_define_method(cT, "abs", robject_abs, 0);
    rb_define_method(cT, "+", robject_add, 1);
    rb_define_method(cT, "-", robject_sub, 1);
    rb_define_method(cT, "*", robject_mul, 1);
    rb_define_method(cT, "/", robject_div, 1);
    rb_define_method(cT, "%", robject_mod, 1);
    rb_define_method(cT, "divmod", robject_divmod, 1);
    rb_define_method(cT, "**", robject_pow, 1);
    rb_define_method(cT, "-@", robject_minus, 0);
    rb_define_method(cT, "reciprocal", robject_reciprocal, 0);
    rb_define_method(cT, "sign", robject_sign, 0);
    rb_define_method(cT, "square", robject_square, 0);
    rb_define_alias(cT, "conj", "view");
    rb_define_alias(cT, "im", "view");
    rb_define_alias(cT, "conjugate", "conj");
    rb_define_method(cT, "eq", robject_eq, 1);
    rb_define_method(cT, "ne", robject_ne, 1);
    rb_define_method(cT, "nearly_eq", robject_nearly_eq, 1);
    rb_define_alias(cT, "close_to", "nearly_eq");
    rb_define_method(cT, "&", robject_bit_and, 1);
    rb_define_method(cT, "|", robject_bit_or, 1);
    rb_define_method(cT, "^", robject_bit_xor, 1);
    rb_define_method(cT, "~", robject_bit_not, 0);
    rb_define_method(cT, "<<", robject_left_shift, 1);
    rb_define_method(cT, ">>", robject_right_shift, 1);
    rb_define_method(cT, "floor", robject_floor, 0);
    rb_define_method(cT, "round", robject_round, 0);
    rb_define_method(cT, "ceil", robject_ceil, 0);
    rb_define_method(cT, "trunc", robject_trunc, 0);
    rb_define_method(cT, "gt", robject_gt, 1);
    rb_define_method(cT, "ge", robject_ge, 1);
    rb_define_method(cT, "lt", robject_lt, 1);
    rb_define_method(cT, "le", robject_le, 1);
    rb_define_alias(cT, ">", "gt");
    rb_define_alias(cT, ">=", "ge");
    rb_define_alias(cT, "<", "lt");
    rb_define_alias(cT, "<=", "le");
    rb_define_method(cT, "clip", robject_clip, 2);
    rb_define_method(cT, "isnan", robject_isnan, 0);
    rb_define_method(cT, "isinf", robject_isinf, 0);
    rb_define_method(cT, "isposinf", robject_isposinf, 0);
    rb_define_method(cT, "isneginf", robject_isneginf, 0);
    rb_define_method(cT, "isfinite", robject_isfinite, 0);
    rb_define_method(cT, "sum", robject_sum, -1);
    rb_define_method(cT, "prod", robject_prod, -1);
    rb_define_method(cT, "mean", robject_mean, -1);
    rb_define_method(cT, "stddev", robject_stddev, -1);
    rb_define_method(cT, "var", robject_var, -1);
    rb_define_method(cT, "rms", robject_rms, -1);
    rb_define_method(cT, "min", robject_min, -1);
    rb_define_method(cT, "max", robject_max, -1);
    rb_define_method(cT, "ptp", robject_ptp, -1);
    rb_define_method(cT, "max_index", robject_max_index, -1);
    rb_define_method(cT, "min_index", robject_min_index, -1);
    rb_define_method(cT, "minmax", robject_minmax, -1);
    rb_define_method(cT, "cumsum", robject_cumsum, -1);
    rb_define_method(cT, "cumprod", robject_cumprod, -1);
    rb_define_method(cT, "mulsum", robject_mulsum, -1);
    rb_define_method(cT, "seq", robject_seq, -1);
    rb_define_method(cT, "logseq", robject_logseq, -1);
    rb_define_method(cT, "eye", robject_eye, -1);
    rb_define_alias(cT, "indgen", "seq");
    rb_define_method(cT, "rand", robject_rand, -1);
    rb_define_method(cT, "poly", robject_poly, -2);
#line 20 "gen/tmpl/init_class.c"
    rb_define_singleton_method(cT, "[]", robject_s_cast, -2);
#line 45 "gen/tmpl/lib.c"
}
