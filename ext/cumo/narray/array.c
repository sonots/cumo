#include <ruby.h>
#include "cumo/narray.h"

// mdai: Multi-Dimensional Array Investigation
typedef struct {
  size_t shape;
  VALUE  val;
} cumo_na_mdai_item_t;

typedef struct {
    int   capa;
    cumo_na_mdai_item_t *item;
    int   type;    // Ruby numeric type - investigated separately
    VALUE na_type;  // NArray type
    VALUE int_max;
} cumo_na_mdai_t;

// Order of Ruby object.
enum { CUMO_NA_NONE, CUMO_NA_BIT, CUMO_NA_INT32, CUMO_NA_INT64, CUMO_NA_RATIONAL,
       CUMO_NA_DFLOAT, CUMO_NA_DCOMPLEX, CUMO_NA_ROBJ, CUMO_NA_NTYPES };

static ID cumo_id_begin;
static ID cumo_id_end;
static ID cumo_id_step;
static ID cumo_id_abs;
static ID cumo_id_cast;
static ID cumo_id_le;
static ID cumo_id_Complex;


static VALUE
 cumo_na_object_type(int type, VALUE v)
{
    static VALUE int32_max = Qnil;
    if (NIL_P(int32_max))
        int32_max = ULONG2NUM(2147483647);

    switch(TYPE(v)) {

    case T_TRUE:
    case T_FALSE:
        if (type<CUMO_NA_BIT)
            return CUMO_NA_BIT;
        return type;

#if SIZEOF_LONG == 4
    case T_FIXNUM:
        if (type<CUMO_NA_INT32)
            return CUMO_NA_INT32;
        return type;
    case T_BIGNUM:
        if (type<CUMO_NA_INT64) {
            v = rb_funcall(v,cumo_id_abs,0);
            if (RTEST(rb_funcall(v,cumo_id_le,1,int32_max))) {
                if (type<CUMO_NA_INT32)
                    return CUMO_NA_INT32;
            } else {
                return CUMO_NA_INT64;
            }
        }
        return type;

#elif SIZEOF_LONG == 8
    case T_FIXNUM:
        if (type<CUMO_NA_INT64) {
            long x = NUM2LONG(v);
            if (x<0) x=-x;
            if (x<=2147483647) {
                if (type<CUMO_NA_INT32)
                    return CUMO_NA_INT32;
            } else {
                return CUMO_NA_INT64;
            }
        }
        return type;
    case T_BIGNUM:
        if (type<CUMO_NA_INT64)
            return CUMO_NA_INT64;
        return type;
#else
    case T_FIXNUM:
    case T_BIGNUM:
        if (type<CUMO_NA_INT64) {
            v = rb_funcall(v,cumo_id_abs,0);
            if (RTEST(rb_funcall(v,cumo_id_le,1,int32_max))) {
                if (type<CUMO_NA_INT32)
                    return CUMO_NA_INT32;
            } else {
                return CUMO_NA_INT64;
            }
        }
        return type;
#endif

    case T_FLOAT:
        if (type<CUMO_NA_DFLOAT)
            return CUMO_NA_DFLOAT;
        return type;

    case T_NIL:
        return type;

    default:
        if (rb_obj_class(v) == rb_const_get( rb_cObject, cumo_id_Complex )) {
            return CUMO_NA_DCOMPLEX;
        }
    }
    return CUMO_NA_ROBJ;
}


#define MDAI_ATTR_TYPE(tp,v,attr)                               \
    {tp = cumo_na_object_type(tp,rb_funcall(v,cumo_id_##attr,0));}

static int cumo_na_mdai_object_type(int type, VALUE v)
{
    if (rb_obj_is_kind_of(v, rb_cRange)) {
        MDAI_ATTR_TYPE(type,v,begin);
        MDAI_ATTR_TYPE(type,v,end);
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
    } else if (rb_obj_is_kind_of(v, rb_cArithSeq)) {
        MDAI_ATTR_TYPE(type,v,begin);
        MDAI_ATTR_TYPE(type,v,end);
        MDAI_ATTR_TYPE(type,v,step);
#endif
    } else {
        type = cumo_na_object_type(type,v);
    }
    return type;
}


static cumo_na_mdai_t *
cumo_na_mdai_alloc(VALUE ary)
{
    int i, n=4;
    cumo_na_mdai_t *mdai;

    mdai = ALLOC(cumo_na_mdai_t);
    mdai->capa = n;
    mdai->item = ALLOC_N( cumo_na_mdai_item_t, n );
    for (i=0; i<n; i++) {
        mdai->item[i].shape = 0;
        mdai->item[i].val = Qnil;
    }
    mdai->item[0].val = ary;
    mdai->type = CUMO_NA_NONE;
    mdai->na_type = Qnil;

    return mdai;
}

static void
cumo_na_mdai_realloc(cumo_na_mdai_t *mdai, int n_extra)
{
    int i, n;

    i = mdai->capa;
    mdai->capa += n_extra;
    n = mdai->capa;
    REALLOC_N( mdai->item, cumo_na_mdai_item_t, n );
    for (; i<n; i++) {
        mdai->item[i].shape = 0;
        mdai->item[i].val = Qnil;
    }
}

static void
cumo_na_mdai_free(void *ptr)
{
    cumo_na_mdai_t *mdai = (cumo_na_mdai_t*)ptr;
    xfree(mdai->item);
    xfree(mdai);
}


/* investigate ndim, shape, type of Array */
static int
cumo_na_mdai_investigate(cumo_na_mdai_t *mdai, int ndim)
{
    ssize_t i;
    int j;
    size_t len, length;
    double dbeg, dstep;
    VALUE  v;
    VALUE  val;

    val = mdai->item[ndim-1].val;
    len = RARRAY_LEN(val);

    for (i=0; i < RARRAY_LEN(val); i++) {
        v = RARRAY_AREF(val,i);

        if (TYPE(v) == T_ARRAY) {
            /* check recursive array */
            for (j=0; j<ndim; j++) {
                if (mdai->item[j].val == v)
                    rb_raise(rb_eStandardError,
                             "cannot convert from a recursive Array to NArray");
            }
            if ( ndim >= mdai->capa ) {
                cumo_na_mdai_realloc(mdai,4);
            }
            mdai->item[ndim].val = v;
            if ( cumo_na_mdai_investigate(mdai,ndim+1) ) {
                len--; /* Array is empty */
            }
        }
        else
#ifdef HAVE_RB_ARITHMETIC_SEQUENCE_EXTRACT
        if (rb_obj_is_kind_of(v, rb_cRange) || rb_obj_is_kind_of(v, rb_cArithSeq)) {
#else
        if (rb_obj_is_kind_of(v, rb_cRange) || rb_obj_is_kind_of(v, rb_cEnumerator)) {
#endif
            cumo_na_step_sequence(v,&length,&dbeg,&dstep);
            len += length-1;
            mdai->type = cumo_na_mdai_object_type(mdai->type, v);
        }
        else if (CumoIsNArray(v)) {
            int r;
            cumo_narray_t *na;
            CumoGetNArray(v,na);
            if ( na->ndim == 0 ) {
                len--; /* NArray is empty */
            } else {
                if ( ndim+na->ndim > mdai->capa ) {
                    cumo_na_mdai_realloc(mdai,((na->ndim-1)/4+1)*4);
                }
                for ( j=0,r=ndim; j < na->ndim  ; j++,r++ ) {
                    if ( mdai->item[r].shape < na->shape[j] )
                        mdai->item[r].shape = na->shape[j];
                }
            }
            // type
            if (NIL_P(mdai->na_type)) {
                mdai->na_type = rb_obj_class(v);
            } else {
                mdai->na_type = cumo_na_upcast(rb_obj_class(v), mdai->na_type);
            }
        } else {
            mdai->type = cumo_na_mdai_object_type(mdai->type, v);
        }
    }

    if (len==0) return 1; /* this array is empty */
    if (mdai->item[ndim-1].shape < len) {
        mdai->item[ndim-1].shape = len;
    }
    return 0;
}


static inline int
cumo_na_mdai_ndim(cumo_na_mdai_t *mdai)
{
    int i;
    // Dimension
    for (i=0; i < mdai->capa && mdai->item[i].shape > 0; i++) ;
    return i;
}

static inline void
cumo_na_mdai_shape(cumo_na_mdai_t *mdai, int ndim, size_t *shape)
{
    int i;
    for (i=0; i<ndim; i++) {
        shape[i] = mdai->item[i].shape;
    }
}

static VALUE
cumo_na_mdai_dtype_numeric(int type)
{
    VALUE tp;
    // DataType
    switch(type) {
    case CUMO_NA_BIT:
        tp = cumo_cBit;
        break;
    case CUMO_NA_INT32:
        tp = cumo_cInt32;
        break;
    case CUMO_NA_INT64:
        tp = cumo_cInt64;
        break;
    case CUMO_NA_DFLOAT:
        tp = cumo_cDFloat;
        break;
    case CUMO_NA_DCOMPLEX:
        tp = cumo_cDComplex;
        break;
    case CUMO_NA_ROBJ:
        tp = cumo_cRObject;
        break;
    default:
        tp = Qnil;
    }
    return tp;
}

static VALUE
cumo_na_mdai_dtype(cumo_na_mdai_t *mdai)
{
    VALUE tp;

    tp = cumo_na_mdai_dtype_numeric(mdai->type);

    if (!NIL_P(mdai->na_type)) {
        if (NIL_P(tp)) {
            tp = mdai->na_type;
        } else {
            tp = cumo_na_upcast(mdai->na_type,tp);
        }
    }
    return tp;
}


static inline VALUE
update_type(VALUE *ptype, VALUE dtype)
{
    if (ptype) {
        if (*ptype == cNArray || !RTEST(*ptype)) {
            *ptype = dtype;
        } else {
            dtype = *ptype;
        }
    }
    return dtype;
}

static inline void
check_subclass_of_narray(VALUE dtype)
{
    if (RTEST(rb_obj_is_kind_of(dtype, rb_cClass))) {
        if (RTEST(rb_funcall(dtype, cumo_id_le, 1, cNArray))) {
            return;
        }
    }
    rb_raise(cumo_na_eCastError, "cannot convert to NArray");
}


static size_t
cumo_na_mdai_memsize(const void *ptr)
{
    const cumo_na_mdai_t *mdai = (const cumo_na_mdai_t*)ptr;

    return sizeof(cumo_na_mdai_t) + mdai->capa * sizeof(cumo_na_mdai_item_t);
}

static const rb_data_type_t mdai_data_type = {
    "Cumo::NArray/mdai",
    {NULL, cumo_na_mdai_free, cumo_na_mdai_memsize,},
    0, 0, RUBY_TYPED_FREE_IMMEDIATELY|RUBY_TYPED_WB_PROTECTED
};


static void
cumo_na_composition3_ary(VALUE ary, VALUE *ptype, VALUE *pshape, VALUE *pnary)
{
    VALUE vmdai;
    cumo_na_mdai_t *mdai;
    int i, ndim;
    size_t *shape;
    VALUE dtype, dshape;

    mdai = cumo_na_mdai_alloc(ary);
    vmdai = TypedData_Wrap_Struct(rb_cObject, &mdai_data_type, (void*)mdai);
    if ( cumo_na_mdai_investigate(mdai, 1) ) {
        // empty
        dtype = update_type(ptype, cumo_cInt32);
        if (pshape) {
            *pshape = rb_ary_new3(1, INT2FIX(0));
        }
        if (pnary) {
            check_subclass_of_narray(dtype);
            shape = ALLOCA_N(size_t, 1);
            shape[0] = 0;
            *pnary = cumo_na_new(dtype, 1, shape);
        }
    } else {
        ndim = cumo_na_mdai_ndim(mdai);
        shape = ALLOCA_N(size_t, ndim);
        cumo_na_mdai_shape(mdai, ndim, shape);
        dtype = update_type(ptype, cumo_na_mdai_dtype(mdai));
        if (pshape) {
            dshape = rb_ary_new2(ndim);
            for (i=0; i<ndim; i++) {
                rb_ary_push(dshape, SIZET2NUM(shape[i]));
            }
            *pshape = dshape;
        }
        if (pnary) {
            check_subclass_of_narray(dtype);
            *pnary = cumo_na_new(dtype, ndim, shape);
        }
    }
    RB_GC_GUARD(vmdai);
}


static void
cumo_na_composition3(VALUE obj, VALUE *ptype, VALUE *pshape, VALUE *pnary)
{
    VALUE dtype, dshape;

    if (TYPE(obj) == T_ARRAY) {
        cumo_na_composition3_ary(obj, ptype, pshape, pnary);
    }
    else if (RTEST(rb_obj_is_kind_of(obj,rb_cNumeric))) {
        dtype = cumo_na_mdai_dtype_numeric(cumo_na_mdai_object_type(CUMO_NA_NONE, obj));
        dtype = update_type(ptype, dtype);
        if (pshape) {
            *pshape = rb_ary_new();
        }
        if (pnary) {
            check_subclass_of_narray(dtype);
            *pnary = cumo_na_new(dtype, 0, 0);
        }
    }
    else if (CumoIsNArray(obj)) {
        int i, ndim;
        cumo_narray_t *na;
        CumoGetNArray(obj,na);
        ndim = na->ndim;
        dtype = update_type(ptype, rb_obj_class(obj));
        if (pshape) {
            dshape = rb_ary_new2(ndim);
            for (i=0; i<ndim; i++) {
                rb_ary_push(dshape, SIZET2NUM(na->shape[i]));
            }
            *pshape = dshape;
        }
        if (pnary) {
            *pnary = cumo_na_new(dtype, ndim, na->shape);
        }
    } else {
        rb_raise(rb_eTypeError,"invalid type for NArray: %s",
                 rb_class2name(rb_obj_class(obj)));
    }
}


static VALUE
cumo_na_s_array_shape(VALUE mod, VALUE ary)
{
    VALUE shape;

    if (TYPE(ary) != T_ARRAY) {
        // 0-dimension
        return rb_ary_new();
    }
    cumo_na_composition3(ary, 0, &shape, 0);
    return shape;
}


/*
  Generate new unallocated NArray instance with shape and type defined from obj.
  Cumo::NArray.new_like(obj) returns instance whose type is defined from obj.
  Cumo::DFloat.new_like(obj) returns DFloat instance.

  @overload new_like(obj)
  @param [Numeric,Array,Cumo::NArray] obj
  @return [Cumo::NArray]
  @example
    Cumo::NArray.new_like([[1,2,3],[4,5,6]])
    => Cumo::Int32#shape=[2,3](empty)
    Cumo::DFloat.new_like([[1,2],[3,4]])
    => Cumo::DFloat#shape=[2,2](empty)
    Cumo::NArray.new_like([1,2i,3])
    => Cumo::DComplex#shape=[3](empty)
*/
VALUE
cumo_na_s_new_like(VALUE type, VALUE obj)
{
    VALUE newary;

    cumo_na_composition3(obj, &type, 0, &newary);
    return newary;
}


VALUE
cumo_na_ary_composition_dtype(VALUE ary)
{
    VALUE type = Qnil;

    cumo_na_composition3(ary, &type, 0, 0);
    return type;
}

static VALUE
cumo_na_s_array_type(VALUE mod, VALUE ary)
{
    return cumo_na_ary_composition_dtype(ary);
}


/*
  Generate NArray object. NArray datatype is automatically selected.
  @overload [](elements)
  @param [Numeric,Array] elements
  @return [NArray]
*/
static VALUE
cumo_na_s_bracket(VALUE klass, VALUE ary)
{
    VALUE dtype=Qnil;

    if (TYPE(ary)!=T_ARRAY) {
        rb_bug("Argument is not array");
    }
    dtype = cumo_na_ary_composition_dtype(ary);
    check_subclass_of_narray(dtype);
    return rb_funcall(dtype, cumo_id_cast, 1, ary);
}


//VALUE
//nst_check_compatibility(VALUE self, VALUE ary);


/* investigate ndim, shape, type of Array */
/*
static int
cumo_na_mdai_for_struct(cumo_na_mdai_t *mdai, int ndim)
{
    size_t i;
    int j, r;
    size_t len;
    VALUE  v;
    VALUE  val;
    cumo_narray_t *na;

    //fprintf(stderr,"ndim=%d\n",ndim);    rb_p(mdai->na_type);
    if (ndim>4) { abort(); }
    val = mdai->item[ndim].val;

    //fpintf(stderr,"val = ");    rb_p(val);

    if (rb_obj_class(val) == mdai->na_type) {
        CumoGetNArray(val,na);
        if ( ndim+na->ndim > mdai->capa ) {
            abort();
            cumo_na_mdai_realloc(mdai,((na->ndim-1)/4+1)*4);
        }
        for ( j=0,r=ndim; j < na->ndim; j++,r++ ) {
            if ( mdai->item[r].shape < na->shape[j] )
                mdai->item[r].shape = na->shape[j];
        }
        return 1;
    }

    if (TYPE(val) == T_ARRAY) {
        // check recursive array
        for (j=0; j<ndim-1; j++) {
            if (mdai->item[j].val == val)
                rb_raise(rb_eStandardError,
                         "cannot convert from a recursive Array to NArray");
        }
        //fprintf(stderr,"check:");        rb_p(val);
        // val is a Struct recort
        if (RTEST( nst_check_compatibility(mdai->na_type, val) )) {
            //fputs("compati\n",stderr);
            return 1;
        }
        // otherwise, multi-dimention
        if (ndim >= mdai->capa) {
            //fprintf(stderr,"exeed capa\n");            abort();
            cumo_na_mdai_realloc(mdai,4);
        }
        // finally, multidimension-check
        len = RARRAY_LEN(val);
        for (i=0; i < len; i++) {
            v = RARRAY_AREF(val,i);
            if (TYPE(v) != T_ARRAY) {
                //abort();
                return 0;
            }
        }
        for (i=0; i < len; i++) {
            v = RARRAY_AREF(val,i);
            //fprintf(stderr,"check:");            rb_p(v);
            mdai->item[ndim+1].val = v;
            if ( cumo_na_mdai_for_struct( mdai, ndim+1 ) == 0 ) {
                //fprintf(stderr,"not struct:");                rb_p(v);
                //abort();
                return 0;
            }
        }
        if (mdai->item[ndim].shape < len) {
            mdai->item[ndim].shape = len;
        }
        return 1;
    }

    //fprintf(stderr,"invalid for struct:");    rb_p(val);    abort();
    return 0;
}
*/


/*
VALUE
cumo_na_ary_composition_for_struct(VALUE nstruct, VALUE ary)
{
    volatile VALUE vmdai, vnc;
    cumo_na_mdai_t *mdai;
    cumo_na_compose_t *nc;

    mdai = cumo_na_mdai_alloc(ary);
    mdai->na_type = nstruct;
    vmdai = TypedData_Wrap_Struct(rb_cObject, &mdai_data_type, (void*)mdai);
    cumo_na_mdai_for_struct(mdai, 0);
    nc = cumo_na_compose_alloc();
    vnc = WrapCompose(nc);
    cumo_na_mdai_result(mdai, nc);
    //fprintf(stderr,"nc->ndim=%d\n",nc->ndim);
    rb_gc_force_recycle(vmdai);
    return vnc;
}
*/



void
Init_cumo_na_array(void)
{
    rb_define_singleton_method(cNArray, "array_shape", cumo_na_s_array_shape, 1);
    rb_define_singleton_method(cNArray, "array_type", cumo_na_s_array_type, 1);
    rb_define_singleton_method(cNArray, "new_like", cumo_na_s_new_like, 1);

    rb_define_singleton_method(cNArray, "[]", cumo_na_s_bracket, -2);

    cumo_id_begin   = rb_intern("begin");
    cumo_id_end     = rb_intern("end");
    cumo_id_step    = rb_intern("step");
    cumo_id_cast    = rb_intern("cast");
    cumo_id_abs     = rb_intern("abs");
    cumo_id_le      = rb_intern("<=");
    cumo_id_Complex = rb_intern("Complex");
}
