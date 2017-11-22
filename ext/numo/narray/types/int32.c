
#line 1 "gen/tmpl/lib.c"
/*
  types/int32.c
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

static ID id_left_shift;
static ID id_right_shift;
static ID id_cast;
static ID id_divmod;
static ID id_eq;
static ID id_ge;
static ID id_gt;
static ID id_le;
static ID id_lt;
static ID id_minlength;
static ID id_mulsum;
static ID id_ne;
static ID id_pow;

#line 22 "gen/tmpl/lib.c"
#include <numo/types/int32.h>

#line 25 "gen/tmpl/lib.c"
VALUE cT;
extern VALUE cRT;


#line 1 "gen/tmpl/class.c"
/*
  class definition: Numo::Int32
*/

VALUE cT;

static VALUE int32_store(VALUE,VALUE);







#line 1 "gen/tmpl/alloc_func.c"
static size_t
int32_memsize(const void* ptr)
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
int32_free(void* ptr)
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

static narray_type_info_t int32_info = {
  
#line 50 "gen/tmpl/alloc_func.c"
    0,             // element_bits
    sizeof(dtype), // element_bytes
    sizeof(dtype), // element_stride (in bytes)
  
};


#line 83 "gen/tmpl/alloc_func.c"
const rb_data_type_t int32_data_type = {
    "Numo::Int32",
    {0, int32_free, int32_memsize,},
    &na_data_type,
    &int32_info,
    0, // flags
};


#line 93 "gen/tmpl/alloc_func.c"
VALUE
int32_s_alloc_func(VALUE klass)
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
    return TypedData_Wrap_Struct(klass, &int32_data_type, (void*)na);
}


#line 1 "gen/tmpl/allocate.c"
static VALUE
int32_allocate(VALUE self)
{
    narray_t *na;
    char *ptr;

    GetNArray(self,na);

    switch(NA_TYPE(na)) {
    case NARRAY_DATA_T:
        ptr = NA_DATA_PTR(na);
        if (na->size > 0 && ptr == NULL) {
            ptr = xmalloc(sizeof(dtype) * na->size);
            
#line 22 "gen/tmpl/allocate.c"
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
int32_extract(VALUE self)
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
int32_new_dim0(dtype x)
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
int32_store_numeric(VALUE self, VALUE obj)
{
    dtype x;
    x = m_num_to_data(obj);
    obj = int32_new_dim0(x);
    int32_store(self,obj);
    return self;
}


#line 1 "gen/tmpl/store_bit.c"
static void
iter_int32_store_bit(na_loop_t *const lp)
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
int32_store_bit(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = {iter_int32_store_bit, FULL_LOOP, 2,0, ain,0};

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_dfloat(na_loop_t *const lp)
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
int32_store_dfloat(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_dfloat, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_sfloat(na_loop_t *const lp)
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
int32_store_sfloat(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_sfloat, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_int64(na_loop_t *const lp)
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
int32_store_int64(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_int64, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_int32(na_loop_t *const lp)
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
int32_store_int32(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_int32, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_int16(na_loop_t *const lp)
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
int32_store_int16(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_int16, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_int8(na_loop_t *const lp)
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
int32_store_int8(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_int8, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_uint64(na_loop_t *const lp)
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
int32_store_uint64(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_uint64, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_uint32(na_loop_t *const lp)
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
int32_store_uint32(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_uint32, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_uint16(na_loop_t *const lp)
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
int32_store_uint16(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_uint16, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_uint8(na_loop_t *const lp)
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
int32_store_uint8(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_uint8, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_from.c"
static void
iter_int32_store_robject(na_loop_t *const lp)
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
int32_store_robject(VALUE self, VALUE obj)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    ndfunc_t ndf = { iter_int32_store_robject, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, obj);
    return self;
}


#line 1 "gen/tmpl/store_array.c"
static void
iter_int32_store_array(na_loop_t *const lp)
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
            iter_int32_store_int32(lp);
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
int32_store_array(VALUE self, VALUE rary)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{rb_cArray,0}};
    ndfunc_t ndf = {iter_int32_store_array, FULL_LOOP, 2, 0, ain, 0};

    na_ndloop_store_rarray(&ndf, self, rary);
    return self;
}

#line 5 "gen/tmpl/store.c"
/*
  Store elements to Numo::Int32 from other.
  @overload store(other)
  @param [Object] other
  @return [Numo::Int32] self
*/
static VALUE
int32_store(VALUE self, VALUE obj)
{
    VALUE r, klass;

    klass = CLASS_OF(obj);

    
    if (klass==numo_cInt32) {
        int32_store_int32(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (IS_INTEGER_CLASS(klass) || klass==rb_cFloat || klass==rb_cComplex) {
        int32_store_numeric(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cBit) {
        int32_store_bit(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cDFloat) {
        int32_store_dfloat(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cSFloat) {
        int32_store_sfloat(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cInt64) {
        int32_store_int64(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cInt16) {
        int32_store_int16(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cInt8) {
        int32_store_int8(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cUInt64) {
        int32_store_uint64(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cUInt32) {
        int32_store_uint32(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cUInt16) {
        int32_store_uint16(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cUInt8) {
        int32_store_uint8(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==numo_cRObject) {
        int32_store_robject(self,obj);
        return self;
    }
    
#line 19 "gen/tmpl/store.c"
    if (klass==rb_cArray) {
        int32_store_array(self,obj);
        return self;
    }
    

    if (IsNArray(obj)) {
        r = rb_funcall(obj, rb_intern("coerce_cast"), 1, cT);
        if (CLASS_OF(r)==cT) {
            int32_store(self,r);
            return self;
        }
    }

    
#line 36 "gen/tmpl/store.c"
    rb_raise(nary_eCastError, "unknown conversion from %s to %s",
             rb_class2name(CLASS_OF(obj)),
             rb_class2name(CLASS_OF(self)));
    
    return self;
}


#line 1 "gen/tmpl/extract_data.c"
/*
  Convert a data value of obj (with a single element) to dtype.
*/
static dtype
int32_extract_data(VALUE obj)
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
        
        if (klass==numo_cInt32) {
            x = m_from_real(*(int32_t*)(ptr+pos));
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
        
#line 22 "gen/tmpl/extract_data.c"
        if (klass==numo_cRObject) {
            x = m_num_to_data(*(VALUE*)(ptr+pos));
            return x;
        }
        

        // coerce
        r = rb_funcall(obj, rb_intern("coerce_cast"), 1, cT);
        if (CLASS_OF(r)==cT) {
            return int32_extract_data(r);
        }
        
#line 36 "gen/tmpl/extract_data.c"
        rb_raise(nary_eCastError, "unknown conversion from %s to %s",
                 rb_class2name(CLASS_OF(obj)),
                 rb_class2name(cT));
        
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
int32_cast_array(VALUE rary)
{
    VALUE nary;
    narray_t *na;

    nary = na_s_new_like(cT, rary);
    GetNArray(nary,na);
    if (na->size > 0) {
        int32_store_array(nary,rary);
    }
    return nary;
}


#line 1 "gen/tmpl/cast.c"
#line 5 "gen/tmpl/cast.c"
/*
  Cast object to Numo::Int32.
  @overload [](elements)
  @overload cast(array)
  @param [Numeric,Array] elements
  @param [Array] array
  @return [Numo::Int32]
*/
static VALUE
int32_s_cast(VALUE type, VALUE obj)
{
    VALUE v;
    narray_t *na;
    dtype x;

    if (CLASS_OF(obj)==cT) {
        return obj;
    }
    if (RTEST(rb_obj_is_kind_of(obj,rb_cNumeric))) {
        x = m_num_to_data(obj);
        return int32_new_dim0(x);
    }
    if (RTEST(rb_obj_is_kind_of(obj,rb_cArray))) {
        return int32_cast_array(obj);
    }
    if (IsNArray(obj)) {
        GetNArray(obj,na);
        v = nary_new(cT, NA_NDIM(na), NA_SHAPE(na));
        if (NA_SIZE(na) > 0) {
            int32_store(v,obj);
        }
        return v;
    }
    
#line 41 "gen/tmpl/cast.c"
    rb_raise(nary_eCastError,"cannot cast to %s",rb_class2name(type));
    return Qnil;
    
}


#line 1 "gen/tmpl/aref.c"
/*
  Array element referenece or slice view.
  @overload [](dim0,...,dimL)
  @param [Numeric,Range,etc] dim0,...,dimL  Multi-dimensional Index.
  @return [Numeric,NArray::Int32] Element object or NArray view.

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
int32_aref(int argc, VALUE *argv, VALUE self)
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
int32_aset(int argc, VALUE *argv, VALUE self)
{
    int nd;
    size_t pos;
    char *ptr;
    VALUE a;
    dtype x;

    argc--;
    if (argc==0) {
        int32_store(self, argv[argc]);
    } else {
        nd = na_get_result_dimension(self, argc, argv, sizeof(dtype), &pos);
        if (nd) {
            a = na_aref_main(argc, argv, self, 0, nd);
            int32_store(a, argv[argc]);
        } else {
            x = int32_extract_data(argv[argc]);
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
int32_coerce_cast(VALUE self, VALUE type)
{
    return Qnil;
}


#line 1 "gen/tmpl/to_a.c"
void
iter_int32_to_a(na_loop_t *const lp)
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
int32_to_a(VALUE self)
{
    ndfunc_arg_in_t ain[3] = {{Qnil,0},{sym_loop_opt},{sym_option}};
    ndfunc_arg_out_t aout[1] = {{rb_cArray,0}}; // dummy?
    ndfunc_t ndf = { iter_int32_to_a, FULL_LOOP_NIP, 3, 1, ain, aout };
    return na_ndloop_cast_narray_to_rarray(&ndf, self, Qnil);
}


#line 1 "gen/tmpl/fill.c"
static void
iter_int32_fill(na_loop_t *const lp)
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
  @return [Numo::Int32] self.
*/
static VALUE
int32_fill(VALUE self, VALUE val)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{sym_option}};
    ndfunc_t ndf = { iter_int32_fill, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, val);
    return self;
}


#line 1 "gen/tmpl/format.c"
static VALUE
format_int32(VALUE fmt, dtype* x)
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
iter_int32_format(na_loop_t *const lp)
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
            y = format_int32(fmt, x);
            SET_DATA_STRIDE(p2, s2, VALUE, y);
        }
    } else {
        for (; i--;) {
            x = (dtype*)p1;         p1+=s1;
            y = format_int32(fmt, x);
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
int32_format(int argc, VALUE *argv, VALUE self)
{
    VALUE fmt=Qnil;

    ndfunc_arg_in_t ain[2] = {{Qnil,0},{sym_option}};
    ndfunc_arg_out_t aout[1] = {{numo_cRObject,0}};
    ndfunc_t ndf = { iter_int32_format, FULL_LOOP_NIP, 2, 1, ain, aout };

    rb_scan_args(argc, argv, "01", &fmt);
    return na_ndloop(&ndf, 2, self, fmt);
}


#line 1 "gen/tmpl/format_to_a.c"
static void
iter_int32_format_to_a(na_loop_t *const lp)
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
            y = format_int32(fmt, x);
            rb_ary_push(a,y);
        }
    } else {
        for (; i--;) {
            x = (dtype*)p1;  p1+=s1;
            y = format_int32(fmt, x);
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
int32_format_to_a(int argc, VALUE *argv, VALUE self)
{
    volatile VALUE fmt=Qnil;
    ndfunc_arg_in_t ain[3] = {{Qnil,0},{sym_loop_opt},{sym_option}};
    ndfunc_arg_out_t aout[1] = {{rb_cArray,0}}; // dummy?
    ndfunc_t ndf = { iter_int32_format_to_a, FULL_LOOP_NIP, 3, 1, ain, aout };

    rb_scan_args(argc, argv, "01", &fmt);
    return na_ndloop_cast_narray_to_rarray(&ndf, self, fmt);
}


#line 1 "gen/tmpl/inspect.c"
static VALUE
iter_int32_inspect(char *ptr, size_t pos, VALUE fmt)
{
#line 7 "gen/tmpl/inspect.c"
    return format_int32(fmt, (dtype*)(ptr+pos));
#line 9 "gen/tmpl/inspect.c"
}

/*
  Returns a string containing a human-readable representation of NArray.
  @overload inspect
  @return [String]
*/
VALUE
int32_inspect(VALUE ary)
{
    return na_ndloop_inspect(ary, iter_int32_inspect, Qnil);
}


#line 1 "gen/tmpl/each.c"
void
iter_int32_each(na_loop_t *const lp)
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
int32_each(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    ndfunc_t ndf = {iter_int32_each, FULL_LOOP_NIP, 1,0, ain,0};

    na_ndloop(&ndf, 1, self);
    return self;
}


#line 1 "gen/tmpl/unary.c"
static void
iter_int32_map(na_loop_t *const lp)
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
            if (is_aligned(p1,sizeof(dtype)) &&
                is_aligned(p2,sizeof(dtype)) ) {
                if (s1 == sizeof(dtype) &&
                    s2 == sizeof(dtype) ) {
                    for (i=0; i<n; i++) {
                        ((dtype*)p2)[i] = m_map(((dtype*)p1)[i]);
                    }
                    return;
                }
                if (is_aligned_step(s1,sizeof(dtype)) &&
                    is_aligned_step(s2,sizeof(dtype)) ) {
                    //
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_map(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
                }
            }
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_map(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
            //
        }
    }
}

/*
  Unary map.
  @overload map
  @return [Numo::Int32] map of self.
*/
static VALUE
int32_map(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_int32_map, FULL_LOOP, 1,1, ain,aout};

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
iter_int32_each_with_index(na_loop_t *const lp)
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
int32_each_with_index(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    ndfunc_t ndf = {iter_int32_each_with_index, FULL_LOOP_NIP, 1,0, ain,0};

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
iter_int32_map_with_index(na_loop_t *const lp)
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
int32_map_with_index(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{Qnil,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_int32_map_with_index, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop_with_index(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary2.c"
static void
iter_int32_abs(na_loop_t *const lp)
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
  @return [Numo::Int32] abs of self.
*/
static VALUE
int32_abs(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cRT,0}};
    ndfunc_t ndf = { iter_int32_abs, FULL_LOOP, 1, 1, ain, aout };

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_int32_add(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                ((dtype*)p3)[i] = m_add(((dtype*)p1)[i],((dtype*)p2)[i]);
            }
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_add(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
        }
    }
    for (i=0; i<n; i++) {
        dtype x, y, z;
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        check_intdivzero(y);
        z = m_add(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,z);
    }
    //
}
#undef check_intdivzero

static VALUE
int32_add_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_add, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary add.
  @overload + other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self + other
*/
static VALUE
int32_add(VALUE self, VALUE other)
{
    
#line 87 "gen/tmpl/binary.c"
    VALUE klass, v;

    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_add_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, '+', 1, other);
    }
    
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_int32_sub(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                ((dtype*)p3)[i] = m_sub(((dtype*)p1)[i],((dtype*)p2)[i]);
            }
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_sub(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
        }
    }
    for (i=0; i<n; i++) {
        dtype x, y, z;
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        check_intdivzero(y);
        z = m_sub(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,z);
    }
    //
}
#undef check_intdivzero

static VALUE
int32_sub_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_sub, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary sub.
  @overload - other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self - other
*/
static VALUE
int32_sub(VALUE self, VALUE other)
{
    
#line 87 "gen/tmpl/binary.c"
    VALUE klass, v;

    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_sub_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, '-', 1, other);
    }
    
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_int32_mul(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                ((dtype*)p3)[i] = m_mul(((dtype*)p1)[i],((dtype*)p2)[i]);
            }
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_mul(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
        }
    }
    for (i=0; i<n; i++) {
        dtype x, y, z;
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        check_intdivzero(y);
        z = m_mul(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,z);
    }
    //
}
#undef check_intdivzero

static VALUE
int32_mul_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_mul, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary mul.
  @overload * other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self * other
*/
static VALUE
int32_mul(VALUE self, VALUE other)
{
    
#line 87 "gen/tmpl/binary.c"
    VALUE klass, v;

    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_mul_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, '*', 1, other);
    }
    
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
iter_int32_div(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                ((dtype*)p3)[i] = m_div(((dtype*)p1)[i],((dtype*)p2)[i]);
            }
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_div(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
        }
    }
    for (i=0; i<n; i++) {
        dtype x, y, z;
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        check_intdivzero(y);
        z = m_div(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,z);
    }
    //
}
#undef check_intdivzero

static VALUE
int32_div_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_div, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary div.
  @overload / other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self / other
*/
static VALUE
int32_div(VALUE self, VALUE other)
{
    
#line 87 "gen/tmpl/binary.c"
    VALUE klass, v;

    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_div_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, '/', 1, other);
    }
    
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
iter_int32_mod(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                ((dtype*)p3)[i] = m_mod(((dtype*)p1)[i],((dtype*)p2)[i]);
            }
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_mod(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
        }
    }
    for (i=0; i<n; i++) {
        dtype x, y, z;
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        check_intdivzero(y);
        z = m_mod(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,z);
    }
    //
}
#undef check_intdivzero

static VALUE
int32_mod_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_mod, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary mod.
  @overload % other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self % other
*/
static VALUE
int32_mod(VALUE self, VALUE other)
{
    
#line 87 "gen/tmpl/binary.c"
    VALUE klass, v;

    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_mod_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, '%', 1, other);
    }
    
}


#line 1 "gen/tmpl/binary2.c"
static void
iter_int32_divmod(na_loop_t *const lp)
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
int32_divmod_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[2] = {{cT,0},{cT,0}};
    ndfunc_t ndf = { iter_int32_divmod, STRIDE_LOOP, 2, 2, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary divmod.
  @overload divmod other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] divmod of self and other.
*/
static VALUE
int32_divmod(VALUE self, VALUE other)
{
    
#line 50 "gen/tmpl/binary2.c"
    VALUE klass, v;
    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_divmod_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, id_divmod, 1, other);
    }
    
}


#line 1 "gen/tmpl/pow.c"
static void
iter_int32_pow(na_loop_t *const lp)
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
iter_int32_pow_int32(na_loop_t *const lp)
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
int32_pow_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_in_t ain_i[2] = {{cT,0},{numo_cInt32,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_pow, STRIDE_LOOP, 2, 1, ain, aout };
    ndfunc_t ndf_i = { iter_int32_pow_int32, STRIDE_LOOP, 2, 1, ain_i, aout };

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
int32_pow(VALUE self, VALUE other)
{
    
#line 69 "gen/tmpl/pow.c"
    VALUE klass, v;
    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_pow_self(self,other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, id_pow, 1, other);
    }
    
}


#line 1 "gen/tmpl/unary.c"
static void
iter_int32_minus(na_loop_t *const lp)
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
            if (is_aligned(p1,sizeof(dtype)) &&
                is_aligned(p2,sizeof(dtype)) ) {
                if (s1 == sizeof(dtype) &&
                    s2 == sizeof(dtype) ) {
                    for (i=0; i<n; i++) {
                        ((dtype*)p2)[i] = m_minus(((dtype*)p1)[i]);
                    }
                    return;
                }
                if (is_aligned_step(s1,sizeof(dtype)) &&
                    is_aligned_step(s2,sizeof(dtype)) ) {
                    //
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_minus(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
                }
            }
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_minus(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
            //
        }
    }
}

/*
  Unary minus.
  @overload -@
  @return [Numo::Int32] minus of self.
*/
static VALUE
int32_minus(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_int32_minus, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary.c"
static void
iter_int32_reciprocal(na_loop_t *const lp)
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
            if (is_aligned(p1,sizeof(dtype)) &&
                is_aligned(p2,sizeof(dtype)) ) {
                if (s1 == sizeof(dtype) &&
                    s2 == sizeof(dtype) ) {
                    for (i=0; i<n; i++) {
                        ((dtype*)p2)[i] = m_reciprocal(((dtype*)p1)[i]);
                    }
                    return;
                }
                if (is_aligned_step(s1,sizeof(dtype)) &&
                    is_aligned_step(s2,sizeof(dtype)) ) {
                    //
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_reciprocal(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
                }
            }
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_reciprocal(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
            //
        }
    }
}

/*
  Unary reciprocal.
  @overload reciprocal
  @return [Numo::Int32] reciprocal of self.
*/
static VALUE
int32_reciprocal(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_int32_reciprocal, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary.c"
static void
iter_int32_sign(na_loop_t *const lp)
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
            if (is_aligned(p1,sizeof(dtype)) &&
                is_aligned(p2,sizeof(dtype)) ) {
                if (s1 == sizeof(dtype) &&
                    s2 == sizeof(dtype) ) {
                    for (i=0; i<n; i++) {
                        ((dtype*)p2)[i] = m_sign(((dtype*)p1)[i]);
                    }
                    return;
                }
                if (is_aligned_step(s1,sizeof(dtype)) &&
                    is_aligned_step(s2,sizeof(dtype)) ) {
                    //
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_sign(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
                }
            }
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_sign(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
            //
        }
    }
}

/*
  Unary sign.
  @overload sign
  @return [Numo::Int32] sign of self.
*/
static VALUE
int32_sign(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_int32_sign, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/unary.c"
static void
iter_int32_square(na_loop_t *const lp)
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
            if (is_aligned(p1,sizeof(dtype)) &&
                is_aligned(p2,sizeof(dtype)) ) {
                if (s1 == sizeof(dtype) &&
                    s2 == sizeof(dtype) ) {
                    for (i=0; i<n; i++) {
                        ((dtype*)p2)[i] = m_square(((dtype*)p1)[i]);
                    }
                    return;
                }
                if (is_aligned_step(s1,sizeof(dtype)) &&
                    is_aligned_step(s2,sizeof(dtype)) ) {
                    //
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_square(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
                }
            }
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_square(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
            //
        }
    }
}

/*
  Unary square.
  @overload square
  @return [Numo::Int32] square of self.
*/
static VALUE
int32_square(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_int32_square, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}





#line 1 "gen/tmpl/cond_binary.c"
static void
iter_int32_eq(na_loop_t *const lp)
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
int32_eq_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_int32_eq, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison eq other.
  @overload eq other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self eq other.
*/
static VALUE
int32_eq(VALUE self, VALUE other)
{
    
#line 46 "gen/tmpl/cond_binary.c"
    VALUE klass, v;
    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_eq_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, id_eq, 1, other);
    }
    
}


#line 1 "gen/tmpl/cond_binary.c"
static void
iter_int32_ne(na_loop_t *const lp)
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
int32_ne_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_int32_ne, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison ne other.
  @overload ne other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self ne other.
*/
static VALUE
int32_ne(VALUE self, VALUE other)
{
    
#line 46 "gen/tmpl/cond_binary.c"
    VALUE klass, v;
    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_ne_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, id_ne, 1, other);
    }
    
}




#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_int32_bit_and(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                ((dtype*)p3)[i] = m_bit_and(((dtype*)p1)[i],((dtype*)p2)[i]);
            }
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_bit_and(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
        }
    }
    for (i=0; i<n; i++) {
        dtype x, y, z;
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        check_intdivzero(y);
        z = m_bit_and(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,z);
    }
    //
}
#undef check_intdivzero

static VALUE
int32_bit_and_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_bit_and, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary bit_and.
  @overload & other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self & other
*/
static VALUE
int32_bit_and(VALUE self, VALUE other)
{
    
#line 87 "gen/tmpl/binary.c"
    VALUE klass, v;

    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_bit_and_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, '&', 1, other);
    }
    
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_int32_bit_or(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                ((dtype*)p3)[i] = m_bit_or(((dtype*)p1)[i],((dtype*)p2)[i]);
            }
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_bit_or(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
        }
    }
    for (i=0; i<n; i++) {
        dtype x, y, z;
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        check_intdivzero(y);
        z = m_bit_or(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,z);
    }
    //
}
#undef check_intdivzero

static VALUE
int32_bit_or_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_bit_or, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary bit_or.
  @overload | other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self | other
*/
static VALUE
int32_bit_or(VALUE self, VALUE other)
{
    
#line 87 "gen/tmpl/binary.c"
    VALUE klass, v;

    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_bit_or_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, '|', 1, other);
    }
    
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_int32_bit_xor(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                ((dtype*)p3)[i] = m_bit_xor(((dtype*)p1)[i],((dtype*)p2)[i]);
            }
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_bit_xor(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
        }
    }
    for (i=0; i<n; i++) {
        dtype x, y, z;
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        check_intdivzero(y);
        z = m_bit_xor(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,z);
    }
    //
}
#undef check_intdivzero

static VALUE
int32_bit_xor_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_bit_xor, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary bit_xor.
  @overload ^ other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self ^ other
*/
static VALUE
int32_bit_xor(VALUE self, VALUE other)
{
    
#line 87 "gen/tmpl/binary.c"
    VALUE klass, v;

    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_bit_xor_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, '^', 1, other);
    }
    
}


#line 1 "gen/tmpl/unary.c"
static void
iter_int32_bit_not(na_loop_t *const lp)
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
            if (is_aligned(p1,sizeof(dtype)) &&
                is_aligned(p2,sizeof(dtype)) ) {
                if (s1 == sizeof(dtype) &&
                    s2 == sizeof(dtype) ) {
                    for (i=0; i<n; i++) {
                        ((dtype*)p2)[i] = m_bit_not(((dtype*)p1)[i]);
                    }
                    return;
                }
                if (is_aligned_step(s1,sizeof(dtype)) &&
                    is_aligned_step(s2,sizeof(dtype)) ) {
                    //
                    for (i=0; i<n; i++) {
                        *(dtype*)p2 = m_bit_not(*(dtype*)p1);
                        p1 += s1;
                        p2 += s2;
                    }
                    return;
                    //
                }
            }
            for (i=0; i<n; i++) {
                GET_DATA_STRIDE(p1,s1,dtype,x);
                x = m_bit_not(x);
                SET_DATA_STRIDE(p2,s2,dtype,x);
            }
            //
        }
    }
}

/*
  Unary bit_not.
  @overload ~
  @return [Numo::Int32] bit_not of self.
*/
static VALUE
int32_bit_not(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {iter_int32_bit_not, FULL_LOOP, 1,1, ain,aout};

    return na_ndloop(&ndf, 1, self);
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_int32_left_shift(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                ((dtype*)p3)[i] = m_left_shift(((dtype*)p1)[i],((dtype*)p2)[i]);
            }
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_left_shift(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
        }
    }
    for (i=0; i<n; i++) {
        dtype x, y, z;
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        check_intdivzero(y);
        z = m_left_shift(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,z);
    }
    //
}
#undef check_intdivzero

static VALUE
int32_left_shift_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_left_shift, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary left_shift.
  @overload << other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self << other
*/
static VALUE
int32_left_shift(VALUE self, VALUE other)
{
    
#line 87 "gen/tmpl/binary.c"
    VALUE klass, v;

    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_left_shift_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, id_left_shift, 1, other);
    }
    
}


#line 1 "gen/tmpl/binary.c"
#line 8 "gen/tmpl/binary.c"
#define check_intdivzero(y) {}

#line 11 "gen/tmpl/binary.c"
static void
iter_int32_right_shift(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                ((dtype*)p3)[i] = m_right_shift(((dtype*)p1)[i],((dtype*)p2)[i]);
            }
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //
            for (i=0; i<n; i++) {
                check_intdivzero(*(dtype*)p2);
                *(dtype*)p3 = m_right_shift(*(dtype*)p1,*(dtype*)p2);
                p1 += s1;
                p2 += s2;
                p3 += s3;
            }
            return;
            //
        }
    }
    for (i=0; i<n; i++) {
        dtype x, y, z;
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,dtype,y);
        check_intdivzero(y);
        z = m_right_shift(x,y);
        SET_DATA_STRIDE(p3,s3,dtype,z);
    }
    //
}
#undef check_intdivzero

static VALUE
int32_right_shift_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_right_shift, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary right_shift.
  @overload >> other
  @param [Numo::NArray,Numeric] other
  @return [Numo::NArray] self >> other
*/
static VALUE
int32_right_shift(VALUE self, VALUE other)
{
    
#line 87 "gen/tmpl/binary.c"
    VALUE klass, v;

    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_right_shift_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, id_right_shift, 1, other);
    }
    
}







#line 1 "gen/tmpl/cond_binary.c"
static void
iter_int32_gt(na_loop_t *const lp)
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
int32_gt_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_int32_gt, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison gt other.
  @overload gt other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self gt other.
*/
static VALUE
int32_gt(VALUE self, VALUE other)
{
    
#line 46 "gen/tmpl/cond_binary.c"
    VALUE klass, v;
    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_gt_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, id_gt, 1, other);
    }
    
}


#line 1 "gen/tmpl/cond_binary.c"
static void
iter_int32_ge(na_loop_t *const lp)
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
int32_ge_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_int32_ge, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison ge other.
  @overload ge other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self ge other.
*/
static VALUE
int32_ge(VALUE self, VALUE other)
{
    
#line 46 "gen/tmpl/cond_binary.c"
    VALUE klass, v;
    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_ge_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, id_ge, 1, other);
    }
    
}


#line 1 "gen/tmpl/cond_binary.c"
static void
iter_int32_lt(na_loop_t *const lp)
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
int32_lt_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_int32_lt, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison lt other.
  @overload lt other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self lt other.
*/
static VALUE
int32_lt(VALUE self, VALUE other)
{
    
#line 46 "gen/tmpl/cond_binary.c"
    VALUE klass, v;
    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_lt_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, id_lt, 1, other);
    }
    
}


#line 1 "gen/tmpl/cond_binary.c"
static void
iter_int32_le(na_loop_t *const lp)
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
int32_le_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{numo_cBit,0}};
    ndfunc_t ndf = { iter_int32_le, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Comparison le other.
  @overload le other
  @param [Numo::NArray,Numeric] other
  @return [Numo::Bit] result of self le other.
*/
static VALUE
int32_le(VALUE self, VALUE other)
{
    
#line 46 "gen/tmpl/cond_binary.c"
    VALUE klass, v;
    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return int32_le_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, id_le, 1, other);
    }
    
}






#line 1 "gen/tmpl/clip.c"
static void
iter_int32_clip(na_loop_t *const lp)
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
iter_int32_clip_min(na_loop_t *const lp)
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
iter_int32_clip_max(na_loop_t *const lp)
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
int32_clip(VALUE self, VALUE min, VALUE max)
{
    ndfunc_arg_in_t ain[3] = {{Qnil,0},{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf_min = { iter_int32_clip_min, STRIDE_LOOP, 2, 1, ain, aout };
    ndfunc_t ndf_max = { iter_int32_clip_max, STRIDE_LOOP, 2, 1, ain, aout };
    ndfunc_t ndf_both = { iter_int32_clip, STRIDE_LOOP, 3, 1, ain, aout };

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


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_int32_sum(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_sum(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  sum of self.
  @overload sum(axis:nil, keepdims:false)
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::Int32] returns result of sum.
*/
static VALUE
int32_sum(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_sum, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
#line 40 "gen/tmpl/accum.c"
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return int32_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_int32_prod(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_prod(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  prod of self.
  @overload prod(axis:nil, keepdims:false)
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::Int32] returns result of prod.
*/
static VALUE
int32_prod(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_prod, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
#line 40 "gen/tmpl/accum.c"
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return int32_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_int32_min(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_min(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  min of self.
  @overload min(axis:nil, keepdims:false)
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::Int32] returns result of min.
*/
static VALUE
int32_min(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_min, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
#line 40 "gen/tmpl/accum.c"
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return int32_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_int32_max(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_max(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  max of self.
  @overload max(axis:nil, keepdims:false)
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::Int32] returns result of max.
*/
static VALUE
int32_max(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_max, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
#line 40 "gen/tmpl/accum.c"
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return int32_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum.c"
#line 2 "gen/tmpl/accum.c"
static void
iter_int32_ptp(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    *(dtype*)p2 = f_ptp(n,p1,s1);
}

#line 17 "gen/tmpl/accum.c"
/*
  ptp of self.
  @overload ptp(axis:nil, keepdims:false)
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::Int32] returns result of ptp.
*/
static VALUE
int32_ptp(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_ptp, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  
#line 40 "gen/tmpl/accum.c"
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  
    v =  na_ndloop(&ndf, 2, self, reduce);
  
    return int32_extract(v);
  
#line 48 "gen/tmpl/accum.c"
}


#line 1 "gen/tmpl/accum_index.c"

#line 3 "gen/tmpl/accum_index.c"
#define idx_t int64_t
static void
iter_int32_max_index_index64(na_loop_t *const lp)
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
iter_int32_max_index_index32(na_loop_t *const lp)
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

#line 23 "gen/tmpl/accum_index.c"
/*
  max_index. Return an index of result.
  @overload max_index(axis:nil)
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @return [Integer,Numo::Int] returns result index of max_index.
  @example
      Numo::NArray[3,4,1,2].min_index => 3
 */
static VALUE
int32_max_index(int argc, VALUE *argv, VALUE self)
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
        ndf.func = iter_int32_max_index_index64;
      
#line 56 "gen/tmpl/accum_index.c"
        reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
      
    } else {
        aout[0].type = numo_cInt32;
        idx = nary_new(numo_cInt32, na->ndim, na->shape);
        ndf.func = iter_int32_max_index_index32;
      
#line 65 "gen/tmpl/accum_index.c"
        reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
      
    }
    rb_funcall(idx, rb_intern("seq"), 0);

    return na_ndloop(&ndf, 3, self, idx, reduce);
}


#line 1 "gen/tmpl/accum_index.c"

#line 3 "gen/tmpl/accum_index.c"
#define idx_t int64_t
static void
iter_int32_min_index_index64(na_loop_t *const lp)
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
iter_int32_min_index_index32(na_loop_t *const lp)
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

#line 23 "gen/tmpl/accum_index.c"
/*
  min_index. Return an index of result.
  @overload min_index(axis:nil)
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @return [Integer,Numo::Int] returns result index of min_index.
  @example
      Numo::NArray[3,4,1,2].min_index => 3
 */
static VALUE
int32_min_index(int argc, VALUE *argv, VALUE self)
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
        ndf.func = iter_int32_min_index_index64;
      
#line 56 "gen/tmpl/accum_index.c"
        reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
      
    } else {
        aout[0].type = numo_cInt32;
        idx = nary_new(numo_cInt32, na->ndim, na->shape);
        ndf.func = iter_int32_min_index_index32;
      
#line 65 "gen/tmpl/accum_index.c"
        reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
      
    }
    rb_funcall(idx, rb_intern("seq"), 0);

    return na_ndloop(&ndf, 3, self, idx, reduce);
}


#line 1 "gen/tmpl/minmax.c"
#line 2 "gen/tmpl/minmax.c"
static void
iter_int32_minmax(na_loop_t *const lp)
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

#line 20 "gen/tmpl/minmax.c"
/*
  minmax of self.
  @overload minmax(axis:nil, keepdims:false)
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::Int32,Numo::Int32] min and max of self.
*/
static VALUE
int32_minmax(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[2] = {{cT,0},{cT,0}};
    ndfunc_t ndf = {iter_int32_minmax, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE|NDF_EXTRACT, 2,2, ain,aout};

  
#line 43 "gen/tmpl/minmax.c"
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  
    return na_ndloop(&ndf, 2, self, reduce);
}


#line 1 "gen/tmpl/bincount.c"
// ------- Integer count without weights -------

#line 7 "gen/tmpl/bincount.c"
static void
iter_int32_bincount_32(na_loop_t *const lp)
{
    size_t   i, x, n;
    char    *p1, *p2;
    ssize_t  s1, s2;
    size_t  *idx1;

    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR(lp, 1, p2, s2);
    i = lp->args[0].shape[0];
    n = lp->args[1].shape[0];

    // initialize
    for (x=0; x < n; x++) {
        *(u_int32_t*)(p2 + s2*x) = 0;
    }

    if (idx1) {
        for (; i--;) {
            GET_DATA_INDEX(p1,idx1,dtype,x);
            (*(u_int32_t*)(p2 + s2*x))++;
        }
    } else {
        for (; i--;) {
            GET_DATA_STRIDE(p1,s1,dtype,x);
            (*(u_int32_t*)(p2 + s2*x))++;
        }
    }
}

static VALUE
int32_bincount_32(VALUE self, size_t length)
{
    size_t shape_out[1] = {length};
    ndfunc_arg_in_t ain[1] = {{cT,1}};
    ndfunc_arg_out_t aout[1] = {{numo_cUInt32,1,shape_out}};
    ndfunc_t ndf = {iter_int32_bincount_32, NO_LOOP|NDF_STRIDE_LOOP|NDF_INDEX_LOOP,
                    1, 1, ain, aout};

    return na_ndloop(&ndf, 1, self);
}

#line 7 "gen/tmpl/bincount.c"
static void
iter_int32_bincount_64(na_loop_t *const lp)
{
    size_t   i, x, n;
    char    *p1, *p2;
    ssize_t  s1, s2;
    size_t  *idx1;

    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR(lp, 1, p2, s2);
    i = lp->args[0].shape[0];
    n = lp->args[1].shape[0];

    // initialize
    for (x=0; x < n; x++) {
        *(u_int64_t*)(p2 + s2*x) = 0;
    }

    if (idx1) {
        for (; i--;) {
            GET_DATA_INDEX(p1,idx1,dtype,x);
            (*(u_int64_t*)(p2 + s2*x))++;
        }
    } else {
        for (; i--;) {
            GET_DATA_STRIDE(p1,s1,dtype,x);
            (*(u_int64_t*)(p2 + s2*x))++;
        }
    }
}

static VALUE
int32_bincount_64(VALUE self, size_t length)
{
    size_t shape_out[1] = {length};
    ndfunc_arg_in_t ain[1] = {{cT,1}};
    ndfunc_arg_out_t aout[1] = {{numo_cUInt64,1,shape_out}};
    ndfunc_t ndf = {iter_int32_bincount_64, NO_LOOP|NDF_STRIDE_LOOP|NDF_INDEX_LOOP,
                    1, 1, ain, aout};

    return na_ndloop(&ndf, 1, self);
}
#line 50 "gen/tmpl/bincount.c"
// ------- end of Integer count without weights -------

// ------- Float count with weights -------

#line 59 "gen/tmpl/bincount.c"
static void
iter_int32_bincount_sf(na_loop_t *const lp)
{
    float w;
    size_t   i, x, n, m;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);
    i = lp->args[0].shape[0];
    m = lp->args[1].shape[0];
    n = lp->args[2].shape[0];

    if (i != m) {
        rb_raise(nary_eShapeError,
                 "size mismatch along last axis between self and weight");
    }

    // initialize
    for (x=0; x < n; x++) {
        *(float*)(p3 + s3*x) = 0;
    }
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,float,w);
        (*(float*)(p3 + s3*x)) += w;
    }
}

static VALUE
int32_bincount_sf(VALUE self, VALUE weight, size_t length)
{
    size_t shape_out[1] = {length};
    ndfunc_arg_in_t ain[2] = {{cT,1},{numo_cSFloat,1}};
    ndfunc_arg_out_t aout[1] = {{numo_cSFloat,1,shape_out}};
    ndfunc_t ndf = {iter_int32_bincount_sf, NO_LOOP|NDF_STRIDE_LOOP,
                    2, 1, ain, aout};

    return na_ndloop(&ndf, 2, self, weight);
}

#line 59 "gen/tmpl/bincount.c"
static void
iter_int32_bincount_df(na_loop_t *const lp)
{
    double w;
    size_t   i, x, n, m;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);
    i = lp->args[0].shape[0];
    m = lp->args[1].shape[0];
    n = lp->args[2].shape[0];

    if (i != m) {
        rb_raise(nary_eShapeError,
                 "size mismatch along last axis between self and weight");
    }

    // initialize
    for (x=0; x < n; x++) {
        *(double*)(p3 + s3*x) = 0;
    }
    for (; i--;) {
        GET_DATA_STRIDE(p1,s1,dtype,x);
        GET_DATA_STRIDE(p2,s2,double,w);
        (*(double*)(p3 + s3*x)) += w;
    }
}

static VALUE
int32_bincount_df(VALUE self, VALUE weight, size_t length)
{
    size_t shape_out[1] = {length};
    ndfunc_arg_in_t ain[2] = {{cT,1},{numo_cDFloat,1}};
    ndfunc_arg_out_t aout[1] = {{numo_cDFloat,1,shape_out}};
    ndfunc_t ndf = {iter_int32_bincount_df, NO_LOOP|NDF_STRIDE_LOOP,
                    2, 1, ain, aout};

    return na_ndloop(&ndf, 2, self, weight);
}
#line 102 "gen/tmpl/bincount.c"
// ------- end of Float count with weights -------

/*
  Count the number of occurrences of each non-negative integer value.
  Only Integer-types has this method.

  @overload bincount([weight], minlength:nil)
  @param [SFloat or DFloat or Array] weight (optional) Array of
    float values. Its size along last axis should be same as that of self.
  @param [Integer] minlength (keyword, optional) Minimum size along
    last axis for the output array.
  @return [UInt32 or UInt64 or SFloat or DFloat]
    Returns Float NArray if weight array is supplied,
    otherwise returns UInt32 or UInt64 depending on the size along last axis.
  @example
    Numo::Int32[0..4].bincount
    => Numo::UInt32#shape=[5]
       [1, 1, 1, 1, 1]

    Numo::Int32[0, 1, 1, 3, 2, 1, 7].bincount
    => Numo::UInt32#shape=[8]
       [1, 3, 1, 1, 0, 0, 0, 1]

    x = Numo::Int32[0, 1, 1, 3, 2, 1, 7, 23]
    x.bincount.size == x.max+1
    => true

    w = Numo::DFloat[0.3, 0.5, 0.2, 0.7, 1.0, -0.6]
    x = Numo::Int32[0, 1, 1, 2, 2, 2]
    x.bincount(w)
    => Numo::DFloat#shape=[3]
       [0.3, 0.7, 1.1]

*/
static VALUE
int32_bincount(int argc, VALUE *argv, VALUE self)
{
    VALUE weight=Qnil, kw=Qnil;
    VALUE opts[1] = {Qundef};
    VALUE v, wclass;
    ID table[1] = {id_minlength};
    size_t length, minlength;

    rb_scan_args(argc, argv, "01:", &weight, &kw);
    rb_get_kwargs(kw, table, 0, 1, opts);

  
#line 151 "gen/tmpl/bincount.c"
    v = int32_minmax(0,0,self);
    if (m_num_to_data(RARRAY_AREF(v,0)) < 0) {
        rb_raise(rb_eArgError,"array items must be non-netagive");
    }
    v = RARRAY_AREF(v,1);
  
    length = NUM2SIZET(v) + 1;

    if (opts[0] != Qundef) {
        minlength = NUM2SIZET(opts[0]);
        if (minlength > length) {
            length = minlength;
        }
    }

    if (NIL_P(weight)) {
        if (length > 4294967295ul) {
            return int32_bincount_64(self, length);
        } else {
            return int32_bincount_32(self, length);
        }
    } else {
        wclass = CLASS_OF(weight);
        if (wclass == numo_cSFloat) {
            return int32_bincount_sf(self, weight, length);
        } else {
            return int32_bincount_df(self, weight, length);
        }
    }
}


#line 1 "gen/tmpl/cum.c"
#line 2 "gen/tmpl/cum.c"
static void
iter_int32_cumsum(na_loop_t *const lp)
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

#line 27 "gen/tmpl/cum.c"
/*
  cumsum of self.
  @overload cumsum(axis:nil, nan:false)
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN if exists).
  @return [Numo::Int32] cumsum of self.
*/
static VALUE
int32_cumsum(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_cumsum, STRIDE_LOOP|NDF_FLAT_REDUCE|NDF_CUM,
                     2, 1, ain, aout };

  
#line 46 "gen/tmpl/cum.c"
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  
    return na_ndloop(&ndf, 2, self, reduce);
}


#line 1 "gen/tmpl/cum.c"
#line 2 "gen/tmpl/cum.c"
static void
iter_int32_cumprod(na_loop_t *const lp)
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

#line 27 "gen/tmpl/cum.c"
/*
  cumprod of self.
  @overload cumprod(axis:nil, nan:false)
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN if exists).
  @return [Numo::Int32] cumprod of self.
*/
static VALUE
int32_cumprod(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_cumprod, STRIDE_LOOP|NDF_FLAT_REDUCE|NDF_CUM,
                     2, 1, ain, aout };

  
#line 46 "gen/tmpl/cum.c"
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  
    return na_ndloop(&ndf, 2, self, reduce);
}


#line 1 "gen/tmpl/accum_binary.c"
//
static void
iter_int32_mulsum(na_loop_t *const lp)
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

static VALUE
int32_mulsum_self(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    VALUE naryv[2];
    ndfunc_arg_in_t ain[4] = {{cT,0},{cT,0},{sym_reduce,0},{sym_init,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_mulsum, STRIDE_LOOP_NIP, 4, 1, ain, aout };

    if (argc < 1) {
        rb_raise(rb_eArgError,"wrong number of arguments (%d for >=1)",argc);
    }
    // should fix below: [self.ndim,other.ndim].max or?
    naryv[0] = self;
    naryv[1] = argv[0];
    //
#line 57 "gen/tmpl/accum_binary.c"
    reduce = na_reduce_dimension(argc-1, argv+1, 2, naryv, &ndf, 0);
    //

    v =  na_ndloop(&ndf, 4, self, argv[0], reduce, m_mulsum_init);
    return int32_extract(v);
}

/*
  Binary mulsum.

  @overload mulsum(other, axis:nil, keepdims:false)
  @param [Numo::NArray,Numeric] other
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::NArray] mulsum of self and other.
*/
static VALUE
int32_mulsum(int argc, VALUE *argv, VALUE self)
{
    //
    VALUE klass, v;
    //
    if (argc < 1) {
        rb_raise(rb_eArgError,"wrong number of arguments (%d for >=1)",argc);
    }
    //
#line 92 "gen/tmpl/accum_binary.c"
    klass = na_upcast(CLASS_OF(self),CLASS_OF(argv[0]));
    if (klass==cT) {
        return int32_mulsum_self(argc, argv, self);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall2(v, rb_intern("mulsum"), argc, argv);
    }
    //
}


#line 1 "gen/tmpl/seq.c"
#line 2 "gen/tmpl/seq.c"
typedef double seq_data_t;

#line 10 "gen/tmpl/seq.c"
typedef double seq_count_t;

#line 13 "gen/tmpl/seq.c"
typedef struct {
    seq_data_t beg;
    seq_data_t step;
    seq_count_t count;
} seq_opt_t;

static void
iter_int32_seq(na_loop_t *const lp)
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
  @return [Numo::Int32] self.
  @example
    Numo::DFloat.new(6).seq(1,-0.2)
    => Numo::DFloat#shape=[6]
       [1, 0.8, 0.6, 0.4, 0.2, 0]
    Numo::DComplex.new(6).seq(1,-0.2+0.2i)
    => Numo::DComplex#shape=[6]
       [1+0i, 0.8+0.2i, 0.6+0.4i, 0.4+0.6i, 0.2+0.8i, 0+1i]
*/
static VALUE
int32_seq(int argc, VALUE *args, VALUE self)
{
    seq_opt_t *g;
    VALUE vbeg=Qnil, vstep=Qnil;
    ndfunc_arg_in_t ain[1] = {{OVERWRITE,0}};
    ndfunc_t ndf = {iter_int32_seq, FULL_LOOP, 1,0, ain,0};

    g = ALLOCA_N(seq_opt_t,1);
    g->beg = m_zero;
    g->step = m_one;
    g->count = 0;
    rb_scan_args(argc, args, "02", &vbeg, &vstep);
#line 83 "gen/tmpl/seq.c"
    if (vbeg!=Qnil) {g->beg = NUM2DBL(vbeg);}
    if (vstep!=Qnil) {g->step = NUM2DBL(vstep);}

#line 90 "gen/tmpl/seq.c"
    na_ndloop3(&ndf, g, 1, self);
    return self;
}


#line 1 "gen/tmpl/eye.c"
static void
iter_int32_eye(na_loop_t *const lp)
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
  @return [Numo::Int32] eye of self.
*/
static VALUE
int32_eye(int argc, VALUE *argv, VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{OVERWRITE,2}};
    ndfunc_t ndf = {iter_int32_eye, NO_LOOP, 1,0, ain,0};
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


#line 14 "gen/tmpl/rand.c"
#define HWID (sizeof(dtype)*4)

static int msb_pos(uint32_t a)
{
    int width = HWID;
    int pos = 0;
    uint32_t mask = (((dtype)1 << HWID)-1) << HWID;

    if (a==0) {return -1;}

    while (width) {
        if (a & mask) {
            pos += width;
        } else {
            mask >>= width;
        }
        width >>= 1;
        mask &= mask << width;
    }
    return pos;
}

/* generates a random number on [0,max) */
#line 50 "gen/tmpl/rand.c"
inline static dtype m_rand(uint32_t max, int shift)
{
    uint32_t x;
    do {
        x = gen_rand32();
        x >>= shift;
    } while (x >= max);
    return x;
}


#line 69 "gen/tmpl/rand.c"
typedef struct {
    dtype low;
    uint32_t max;
} rand_opt_t;

static void
iter_int32_rand(na_loop_t *const lp)
{
    size_t   i;
    char    *p1;
    ssize_t  s1;
    size_t  *idx1;
    dtype    x;
    rand_opt_t *g;
    dtype    low;
    uint32_t max;
    int shift;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    g = (rand_opt_t*)(lp->opt_ptr);
    low = g->low;
    max = g->max;
    shift = 31 - msb_pos(max);

    if (idx1) {
        for (; i--;) {
            x = m_add(m_rand(max,shift),low);
            SET_DATA_INDEX(p1,idx1,dtype,x);
        }
    } else {
        for (; i--;) {
            x = m_add(m_rand(max,shift),low);
            SET_DATA_STRIDE(p1,s1,dtype,x);
        }
    }
}


/*
  Generate uniformly distributed random numbers on self narray.
  @overload rand([[low],high])
  @param [Numeric] low  lower inclusive boundary of random numbers. (default=0)
  @param [Numeric] high  upper exclusive boundary of random numbers. (default=1 or 1+1i for complex types)
  @return [Numo::Int32] self.
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
int32_rand(int argc, VALUE *args, VALUE self)
{
    rand_opt_t g;
    VALUE v1=Qnil, v2=Qnil;
    dtype high;
    ndfunc_arg_in_t ain[1] = {{OVERWRITE,0}};
    ndfunc_t ndf = {iter_int32_rand, FULL_LOOP, 1,0, ain,0};

    
    rb_scan_args(argc, args, "11", &v1, &v2);
    if (v2==Qnil) {
        g.low = m_zero;
        g.max = high = m_num_to_data(v1);
    
#line 153 "gen/tmpl/rand.c"
    } else {
        g.low = m_num_to_data(v1);
        high = m_num_to_data(v2);
        g.max = m_sub(high,g.low);
    }
    
    if (high <= g.low) {
        rb_raise(rb_eArgError,"high must be larger than low");
    }
    
    na_ndloop3(&ndf, &g, 1, self);
    return self;
}


#line 1 "gen/tmpl/poly.c"
static void
iter_int32_poly(na_loop_t *const lp)
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
  @return [Numo::Int32]
*/
static VALUE
int32_poly(VALUE self, VALUE args)
{
    int argc, i;
    VALUE *argv;
    volatile VALUE v, a;
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { iter_int32_poly, NO_LOOP, 0, 1, 0, aout };

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
    return int32_extract(v);
}


#line 1 "gen/tmpl/qsort.c"
/*
  qsort.c
  Numerical Array Extension for Ruby
    modified by Masahiro TANAKA
*/

/*
 *      qsort.c: standard quicksort algorithm
 *
 *      Modifications from vanilla NetBSD source:
 *        Add do ... while() macro fix
 *        Remove __inline, _DIAGASSERTs, __P
 *        Remove ill-considered "swap_cnt" switch to insertion sort,
 *        in favor of a simple check for presorted input.
 *
 *      CAUTION: if you change this file, see also qsort_arg.c
 *
 *      $PostgreSQL: pgsql/src/port/qsort.c,v 1.12 2006/10/19 20:56:22 tgl Exp $
 */

/*      $NetBSD: qsort.c,v 1.13 2003/08/07 16:43:42 agc Exp $   */

/*-
 * Copyright (c) 1992, 1993
 *      The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *        may be used to endorse or promote products derived from this software
 *        without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.      IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef QSORT_INCL
#define QSORT_INCL
#define Min(x, y)               ((x) < (y) ? (x) : (y))

#define swap(type,a,b) \
    do {type tmp=*(type*)(a); *(type*)(a)=*(type*)(b); *(type*)(b)=tmp;} while(0)

#define vecswap(type, a, b, n) if ((n)>0) swap(type,(a),(b))

#define MED3(a,b,c)                                     \
    (cmpgt(b,a) ?                                       \
     (cmpgt(c,b) ? b : (cmpgt(c,a) ? c : a))            \
     : (cmpgt(b,c) ? b : (cmpgt(c,a) ? a : c)))
#endif

#undef qsort_dtype
#define qsort_dtype dtype
#undef qsort_cast
#define qsort_cast *(dtype*)

#line 79 "gen/tmpl/qsort.c"
void
int32_qsort(void *a, size_t n, ssize_t es)
{
    char *pa, *pb, *pc, *pd, *pl, *pm, *pn;
    int  d, r, presorted;

 loop:
    if (n < 7) {
        for (pm = (char *) a + es; pm < (char *) a + n * es; pm += es)
            for (pl = pm; pl > (char *) a && cmpgt(pl - es, pl);
                 pl -= es)
                swap(qsort_dtype, pl, pl - es);
        return;
    }
    presorted = 1;
    for (pm = (char *) a + es; pm < (char *) a + n * es; pm += es) {
        if (cmpgt(pm - es, pm)) {
            presorted = 0;
            break;
        }
    }
    if (presorted)
        return;
    pm = (char *) a + (n / 2) * es;
    if (n > 7) {
        pl = (char *) a;
        pn = (char *) a + (n - 1) * es;
        if (n > 40) {
            d = (n / 8) * es;
            pl = MED3(pl, pl + d, pl + 2 * d);
            pm = MED3(pm - d, pm, pm + d);
            pn = MED3(pn - 2 * d, pn - d, pn);
        }
        pm = MED3(pl, pm, pn);
    }
    swap(qsort_dtype, a, pm);
    pa = pb = (char *) a + es;
    pc = pd = (char *) a + (n - 1) * es;
    for (;;) {
        while (pb <= pc && (r = cmp(pb, a)) <= 0) {
            if (r == 0) {
                swap(qsort_dtype, pa, pb);
                pa += es;
            }
            pb += es;
        }
        while (pb <= pc && (r = cmp(pc, a)) >= 0) {
            if (r == 0) {
                swap(qsort_dtype, pc, pd);
                pd -= es;
            }
            pc -= es;
        }
        if (pb > pc)
            break;
        swap(qsort_dtype, pb, pc);
        pb += es;
        pc -= es;
    }
    pn = (char *) a + n * es;
    r = Min(pa - (char *) a, pb - pa);
    vecswap(qsort_dtype, a, pb - r, r);
    r = Min(pd - pc, pn - pd - es);
    vecswap(qsort_dtype, pb, pn - r, r);
    if ((r = pb - pa) > es)
        int32_qsort(a, r / es, es);
    if ((r = pd - pc) > es) {
        a = pn - r;
        n = r / es;
        goto loop;
    }
}


#line 1 "gen/tmpl/sort.c"
#line 2 "gen/tmpl/sort.c"
static void
iter_int32_sort(na_loop_t *const lp)
{
    size_t n;
    char *ptr;
    ssize_t step;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, ptr, step);
    int32_qsort(ptr, n, step);
}

#line 15 "gen/tmpl/sort.c"
/*
  sort of self.
  @overload sort(axis:nil)
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @return [Numo::Int32] returns result of sort.
  @example
      Numo::DFloat[3,4,1,2].sort => Numo::DFloat[1,2,3,4]
*/
static VALUE
int32_sort(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{sym_reduce,0}};
    ndfunc_t ndf = {0, STRIDE_LOOP|NDF_FLAT_REDUCE, 2,0, ain,0};

    if (!TEST_INPLACE(self)) {
        self = na_copy(self);
    }
  
#line 42 "gen/tmpl/sort.c"
    ndf.func = iter_int32_sort;
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  
    na_ndloop(&ndf, 2, self, reduce);
    return self;
}


#line 1 "gen/tmpl/qsort.c"
/*
  qsort.c
  Numerical Array Extension for Ruby
    modified by Masahiro TANAKA
*/

/*
 *      qsort.c: standard quicksort algorithm
 *
 *      Modifications from vanilla NetBSD source:
 *        Add do ... while() macro fix
 *        Remove __inline, _DIAGASSERTs, __P
 *        Remove ill-considered "swap_cnt" switch to insertion sort,
 *        in favor of a simple check for presorted input.
 *
 *      CAUTION: if you change this file, see also qsort_arg.c
 *
 *      $PostgreSQL: pgsql/src/port/qsort.c,v 1.12 2006/10/19 20:56:22 tgl Exp $
 */

/*      $NetBSD: qsort.c,v 1.13 2003/08/07 16:43:42 agc Exp $   */

/*-
 * Copyright (c) 1992, 1993
 *      The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *        may be used to endorse or promote products derived from this software
 *        without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.      IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef QSORT_INCL
#define QSORT_INCL
#define Min(x, y)               ((x) < (y) ? (x) : (y))

#define swap(type,a,b) \
    do {type tmp=*(type*)(a); *(type*)(a)=*(type*)(b); *(type*)(b)=tmp;} while(0)

#define vecswap(type, a, b, n) if ((n)>0) swap(type,(a),(b))

#define MED3(a,b,c)                                     \
    (cmpgt(b,a) ?                                       \
     (cmpgt(c,b) ? b : (cmpgt(c,a) ? c : a))            \
     : (cmpgt(b,c) ? b : (cmpgt(c,a) ? a : c)))
#endif

#undef qsort_dtype
#define qsort_dtype dtype*
#undef qsort_cast
#define qsort_cast **(dtype**)

#line 79 "gen/tmpl/qsort.c"
void
int32_index_qsort(void *a, size_t n, ssize_t es)
{
    char *pa, *pb, *pc, *pd, *pl, *pm, *pn;
    int  d, r, presorted;

 loop:
    if (n < 7) {
        for (pm = (char *) a + es; pm < (char *) a + n * es; pm += es)
            for (pl = pm; pl > (char *) a && cmpgt(pl - es, pl);
                 pl -= es)
                swap(qsort_dtype, pl, pl - es);
        return;
    }
    presorted = 1;
    for (pm = (char *) a + es; pm < (char *) a + n * es; pm += es) {
        if (cmpgt(pm - es, pm)) {
            presorted = 0;
            break;
        }
    }
    if (presorted)
        return;
    pm = (char *) a + (n / 2) * es;
    if (n > 7) {
        pl = (char *) a;
        pn = (char *) a + (n - 1) * es;
        if (n > 40) {
            d = (n / 8) * es;
            pl = MED3(pl, pl + d, pl + 2 * d);
            pm = MED3(pm - d, pm, pm + d);
            pn = MED3(pn - 2 * d, pn - d, pn);
        }
        pm = MED3(pl, pm, pn);
    }
    swap(qsort_dtype, a, pm);
    pa = pb = (char *) a + es;
    pc = pd = (char *) a + (n - 1) * es;
    for (;;) {
        while (pb <= pc && (r = cmp(pb, a)) <= 0) {
            if (r == 0) {
                swap(qsort_dtype, pa, pb);
                pa += es;
            }
            pb += es;
        }
        while (pb <= pc && (r = cmp(pc, a)) >= 0) {
            if (r == 0) {
                swap(qsort_dtype, pc, pd);
                pd -= es;
            }
            pc -= es;
        }
        if (pb > pc)
            break;
        swap(qsort_dtype, pb, pc);
        pb += es;
        pc -= es;
    }
    pn = (char *) a + n * es;
    r = Min(pa - (char *) a, pb - pa);
    vecswap(qsort_dtype, a, pb - r, r);
    r = Min(pd - pc, pn - pd - es);
    vecswap(qsort_dtype, pb, pn - r, r);
    if ((r = pb - pa) > es)
        int32_index_qsort(a, r / es, es);
    if ((r = pd - pc) > es) {
        a = pn - r;
        n = r / es;
        goto loop;
    }
}


#line 1 "gen/tmpl/sort_index.c"

#line 3 "gen/tmpl/sort_index.c"
#define idx_t int64_t
static void
int32_index64_qsort(na_loop_t *const lp)
{
    size_t   i, n, idx;
    char    *d_ptr, *i_ptr, *o_ptr;
    ssize_t  d_step, i_step, o_step;
    char   **ptr;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, d_ptr, d_step);
    INIT_PTR(lp, 1, i_ptr, i_step);
    INIT_PTR(lp, 2, o_ptr, o_step);

    ptr = (char**)(lp->opt_ptr);

    //printf("(ptr=%lx, d_ptr=%lx,d_step=%ld, i_ptr=%lx,i_step=%ld, o_ptr=%lx,o_step=%ld)\n",(size_t)ptr,(size_t)d_ptr,(ssize_t)d_step,(size_t)i_ptr,(ssize_t)i_step,(size_t)o_ptr,(ssize_t)o_step);

    if (n==1) {
        *(idx_t*)o_ptr = *(idx_t*)(i_ptr);
        return;
    }

    for (i=0; i<n; i++) {
        ptr[i] = d_ptr + d_step * i;
        //printf("(%ld,%.3f)",i,*(double*)ptr[i]);
    }

    int32_index_qsort(ptr, n, sizeof(dtype*));

    //d_ptr = lp->args[0].ptr;
    //printf("(d_ptr=%lx)\n",(size_t)d_ptr);

    for (i=0; i<n; i++) {
        idx = (ptr[i] - d_ptr) / d_step;
        *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
        //printf("(idx[%ld]=%ld,%d)",i,idx,*(idx_t*)o_ptr);
        o_ptr += o_step;
    }
    //printf("\n");
}
#undef idx_t

#line 3 "gen/tmpl/sort_index.c"
#define idx_t int32_t
static void
int32_index32_qsort(na_loop_t *const lp)
{
    size_t   i, n, idx;
    char    *d_ptr, *i_ptr, *o_ptr;
    ssize_t  d_step, i_step, o_step;
    char   **ptr;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, d_ptr, d_step);
    INIT_PTR(lp, 1, i_ptr, i_step);
    INIT_PTR(lp, 2, o_ptr, o_step);

    ptr = (char**)(lp->opt_ptr);

    //printf("(ptr=%lx, d_ptr=%lx,d_step=%ld, i_ptr=%lx,i_step=%ld, o_ptr=%lx,o_step=%ld)\n",(size_t)ptr,(size_t)d_ptr,(ssize_t)d_step,(size_t)i_ptr,(ssize_t)i_step,(size_t)o_ptr,(ssize_t)o_step);

    if (n==1) {
        *(idx_t*)o_ptr = *(idx_t*)(i_ptr);
        return;
    }

    for (i=0; i<n; i++) {
        ptr[i] = d_ptr + d_step * i;
        //printf("(%ld,%.3f)",i,*(double*)ptr[i]);
    }

    int32_index_qsort(ptr, n, sizeof(dtype*));

    //d_ptr = lp->args[0].ptr;
    //printf("(d_ptr=%lx)\n",(size_t)d_ptr);

    for (i=0; i<n; i++) {
        idx = (ptr[i] - d_ptr) / d_step;
        *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
        //printf("(idx[%ld]=%ld,%d)",i,idx,*(idx_t*)o_ptr);
        o_ptr += o_step;
    }
    //printf("\n");
}
#undef idx_t

#line 47 "gen/tmpl/sort_index.c"
/*
  sort_index. Returns an index array of sort result.
  @overload sort_index(axis:nil)
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @return [Integer,Numo::Int] returns result index of sort_index.
  @example
      Numo::NArray[3,4,1,2].sort_index => Numo::Int32[2,3,0,1]
*/
static VALUE
int32_sort_index(int argc, VALUE *argv, VALUE self)
{
    size_t size;
    narray_t *na;
    VALUE idx, tmp, reduce, res;
    char *buf;
    ndfunc_arg_in_t ain[3] = {{cT,0},{0,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{0,0,0}};
    ndfunc_t ndf = {0, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE|NDF_CUM, 3,1, ain,aout};

    GetNArray(self,na);
    if (na->ndim==0) {
        return INT2FIX(0);
    }
    if (na->size > (~(u_int32_t)0)) {
        ain[1].type =
        aout[0].type = numo_cInt64;
        idx = nary_new(numo_cInt64, na->ndim, na->shape);
       
#line 84 "gen/tmpl/sort_index.c"
         ndf.func = int32_index64_qsort;
         reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
       
    } else {
        ain[1].type =
        aout[0].type = numo_cInt32;
        idx = nary_new(numo_cInt32, na->ndim, na->shape);
       
#line 96 "gen/tmpl/sort_index.c"
         ndf.func = int32_index32_qsort;
         reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
       
    }
    rb_funcall(idx, rb_intern("seq"), 0);

    size = na->size*sizeof(void*); // max capa
    buf = rb_alloc_tmp_buffer(&tmp, size);
    res = na_ndloop3(&ndf, buf, 3, self, idx, reduce);
    rb_free_tmp_buffer(&tmp);
    return res;
}


#line 1 "gen/tmpl/median.c"
#line 2 "gen/tmpl/median.c"
static void
iter_int32_median(na_loop_t *const lp)
{
    size_t n;
    char *p1, *p2;
    dtype *buf;

    INIT_COUNTER(lp, n);
    p1 = (lp->args[0]).ptr + (lp->args[0].iter[0]).pos;
    p2 = (lp->args[1]).ptr + (lp->args[1].iter[0]).pos;
    buf = (dtype*)p1;

    int32_qsort(buf, n, sizeof(dtype));

    

#line 22 "gen/tmpl/median.c"
    if (n==0) {
        *(dtype*)p2 = buf[0];
    }
    else if (n%2==0) {
        *(dtype*)p2 = (buf[n/2-1]+buf[n/2])/2;
    }
    else {
        *(dtype*)p2 = buf[(n-1)/2];
    }
}

#line 34 "gen/tmpl/median.c"
/*
  median of self.
  @overload median(axis:nil, keepdims:false)
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Numo::Int32] returns median of self.
*/

static VALUE
int32_median(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{INT2FIX(0),0}};
    ndfunc_t ndf = {0, NDF_HAS_LOOP|NDF_FLAT_REDUCE, 2,1, ain,aout};

    self = na_copy(self); // as temporary buffer
  
#line 60 "gen/tmpl/median.c"
    ndf.func = iter_int32_median;
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  
    return na_ndloop(&ndf, 2, self, reduce);
}



#line 32 "gen/tmpl/lib.c"
void
Init_numo_int32(void)
{
    VALUE hCast, mNumo;

    mNumo = rb_define_module("Numo");

    
    id_left_shift = rb_intern("<<");
    id_right_shift = rb_intern(">>");
    id_cast = rb_intern("cast");
    id_divmod = rb_intern("divmod");
    id_eq = rb_intern("eq");
    id_ge = rb_intern("ge");
    id_gt = rb_intern("gt");
    id_le = rb_intern("le");
    id_lt = rb_intern("lt");
    id_minlength = rb_intern("minlength");
    id_mulsum = rb_intern("mulsum");
    id_ne = rb_intern("ne");
    id_pow = rb_intern("pow");


#line 1 "gen/tmpl/init_class.c"
    /*
      Document-class: Numo::Int32
      
    */
    cT = rb_define_class_under(mNumo, "Int32", cNArray);

  

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
    rb_hash_aset(hCast, rb_cFloat, numo_cDFloat);
    rb_hash_aset(hCast, rb_cComplex, numo_cDComplex);
    rb_hash_aset(hCast, numo_cRObject, numo_cRObject);
    rb_hash_aset(hCast, numo_cDComplex, numo_cDComplex);
    rb_hash_aset(hCast, numo_cSComplex, numo_cSComplex);
    rb_hash_aset(hCast, numo_cDFloat, numo_cDFloat);
    rb_hash_aset(hCast, numo_cSFloat, numo_cSFloat);
    rb_hash_aset(hCast, numo_cInt64, numo_cInt64);
    rb_hash_aset(hCast, numo_cInt32, cT);
    rb_hash_aset(hCast, numo_cInt16, cT);
    rb_hash_aset(hCast, numo_cInt8, cT);
    rb_hash_aset(hCast, numo_cUInt64, numo_cInt64);
    rb_hash_aset(hCast, numo_cUInt32, cT);
    rb_hash_aset(hCast, numo_cUInt16, cT);
    rb_hash_aset(hCast, numo_cUInt8, cT);

    
    /**/
    rb_define_const(cT,"ELEMENT_BIT_SIZE",INT2FIX(sizeof(dtype)*8));
    /**/
    rb_define_const(cT,"ELEMENT_BYTE_SIZE",INT2FIX(sizeof(dtype)));
    /**/
    rb_define_const(cT,"CONTIGUOUS_STRIDE",INT2FIX(sizeof(dtype)));
    /**/
    rb_define_const(cT,"MAX",M_MAX);
    /**/
    rb_define_const(cT,"MIN",M_MIN);
    rb_define_alloc_func(cT, int32_s_alloc_func);
    rb_define_method(cT, "allocate", int32_allocate, 0);
    rb_define_method(cT, "extract", int32_extract, 0);
    
    rb_define_method(cT, "store", int32_store, 1);
    
    
    rb_define_singleton_method(cT, "cast", int32_s_cast, 1);
    rb_define_method(cT, "[]", int32_aref, -1);
    rb_define_method(cT, "[]=", int32_aset, -1);
    rb_define_method(cT, "coerce_cast", int32_coerce_cast, 1);
    rb_define_method(cT, "to_a", int32_to_a, 0);
    rb_define_method(cT, "fill", int32_fill, 1);
    rb_define_method(cT, "format", int32_format, -1);
    rb_define_method(cT, "format_to_a", int32_format_to_a, -1);
    rb_define_method(cT, "inspect", int32_inspect, 0);
    rb_define_method(cT, "each", int32_each, 0);
    rb_define_method(cT, "map", int32_map, 0);
    rb_define_method(cT, "each_with_index", int32_each_with_index, 0);
    rb_define_method(cT, "map_with_index", int32_map_with_index, 0);
    rb_define_method(cT, "abs", int32_abs, 0);
    rb_define_method(cT, "+", int32_add, 1);
    rb_define_method(cT, "-", int32_sub, 1);
    rb_define_method(cT, "*", int32_mul, 1);
    rb_define_method(cT, "/", int32_div, 1);
    rb_define_method(cT, "%", int32_mod, 1);
    rb_define_method(cT, "divmod", int32_divmod, 1);
    rb_define_method(cT, "**", int32_pow, 1);
    rb_define_method(cT, "-@", int32_minus, 0);
    rb_define_method(cT, "reciprocal", int32_reciprocal, 0);
    rb_define_method(cT, "sign", int32_sign, 0);
    rb_define_method(cT, "square", int32_square, 0);
    rb_define_alias(cT, "conj", "view");
    rb_define_alias(cT, "im", "view");
    rb_define_alias(cT, "conjugate", "conj");
    rb_define_method(cT, "eq", int32_eq, 1);
    rb_define_method(cT, "ne", int32_ne, 1);
    rb_define_alias(cT, "nearly_eq", "eq");
    rb_define_alias(cT, "close_to", "nearly_eq");
    rb_define_method(cT, "&", int32_bit_and, 1);
    rb_define_method(cT, "|", int32_bit_or, 1);
    rb_define_method(cT, "^", int32_bit_xor, 1);
    rb_define_method(cT, "~", int32_bit_not, 0);
    rb_define_method(cT, "<<", int32_left_shift, 1);
    rb_define_method(cT, ">>", int32_right_shift, 1);
    rb_define_alias(cT, "floor", "view");
    rb_define_alias(cT, "round", "view");
    rb_define_alias(cT, "ceil", "view");
    rb_define_alias(cT, "trunc", "view");
    rb_define_alias(cT, "rint", "view");
    rb_define_method(cT, "gt", int32_gt, 1);
    rb_define_method(cT, "ge", int32_ge, 1);
    rb_define_method(cT, "lt", int32_lt, 1);
    rb_define_method(cT, "le", int32_le, 1);
    rb_define_alias(cT, ">", "gt");
    rb_define_alias(cT, ">=", "ge");
    rb_define_alias(cT, "<", "lt");
    rb_define_alias(cT, "<=", "le");
    rb_define_method(cT, "clip", int32_clip, 2);
    rb_define_method(cT, "sum", int32_sum, -1);
    rb_define_method(cT, "prod", int32_prod, -1);
    rb_define_method(cT, "min", int32_min, -1);
    rb_define_method(cT, "max", int32_max, -1);
    rb_define_method(cT, "ptp", int32_ptp, -1);
    rb_define_method(cT, "max_index", int32_max_index, -1);
    rb_define_method(cT, "min_index", int32_min_index, -1);
    rb_define_method(cT, "minmax", int32_minmax, -1);
    rb_define_method(cT, "bincount", int32_bincount, -1);
    rb_define_method(cT, "cumsum", int32_cumsum, -1);
    rb_define_method(cT, "cumprod", int32_cumprod, -1);
    rb_define_method(cT, "mulsum", int32_mulsum, -1);
    rb_define_method(cT, "seq", int32_seq, -1);
    rb_define_method(cT, "eye", int32_eye, -1);
    rb_define_alias(cT, "indgen", "seq");
    rb_define_method(cT, "rand", int32_rand, -1);
    rb_define_method(cT, "poly", int32_poly, -2);
    
    rb_define_method(cT, "sort", int32_sort, -1);
    
    rb_define_method(cT, "sort_index", int32_sort_index, -1);
    rb_define_method(cT, "median", int32_median, -1);
#line 20 "gen/tmpl/init_class.c"
    rb_define_singleton_method(cT, "[]", int32_s_cast, -2);
#line 45 "gen/tmpl/lib.c"
}
