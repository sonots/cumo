#include <ruby.h>
#include "cumo/narray.h"

VALUE cumo_mNMath;
extern VALUE cumo_mDFloatMath, cumo_mDComplexMath;
extern VALUE cumo_mSFloatMath, cumo_mSComplexMath;
static ID cumo_id_send;
static ID cumo_id_UPCAST;
static ID cumo_id_DISPATCH;
static ID cumo_id_extract;

static VALUE
cumo_na_type_s_upcast(VALUE type1, VALUE type2)
{
    VALUE upcast_hash;
    VALUE result_type;

    if (type1==type2) return type1;
    upcast_hash = rb_const_get(type1, cumo_id_UPCAST);
    result_type = rb_hash_aref(upcast_hash, type2);
    if (NIL_P(result_type)) {
        if (TYPE(type2)==T_CLASS) {
            if ( RTEST(rb_class_inherited_p(type2,cNArray)) ) {
                upcast_hash = rb_const_get(type2, cumo_id_UPCAST);
                result_type = rb_hash_aref(upcast_hash, type1);
            }
        }
    }
    return result_type;
}


static VALUE cumo_na_math_cast2(VALUE type1, VALUE type2)
{
    if ( RTEST(rb_class_inherited_p( type1, cNArray )) ){
	return cumo_na_type_s_upcast( type1, type2 );
    }
    if ( RTEST(rb_class_inherited_p( type2, cNArray )) ){
	return cumo_na_type_s_upcast( type2, type1 );
    }
    if ( RTEST(rb_class_inherited_p( type1, rb_cNumeric )) &&
	 RTEST(rb_class_inherited_p( type2, rb_cNumeric )) ){
	if ( RTEST(rb_class_inherited_p( type1, rb_cComplex)) ||
	     RTEST(rb_class_inherited_p( type2, rb_cComplex )) ){
	    return rb_cComplex;
	}
	return rb_cFloat;
    }
    return type2;
}


VALUE cumo_na_ary_composition_dtype(VALUE);

static VALUE cumo_na_mathcast(int argc, VALUE *argv)
{
    VALUE type, type2;
    int i;

    type = cumo_na_ary_composition_dtype(argv[0]);
    for (i=1; i<argc; i++) {
        type2 = cumo_na_ary_composition_dtype(argv[i]);
        type = cumo_na_math_cast2(type, type2);
        if (NIL_P(type)) {
            rb_raise(rb_eTypeError,"includes unknown DataType for upcast");
        }
    }
    return type;
}


/*
  Dispatches method to Math module of upcasted type,
  eg, Cumo::DFloat::Math.
  @overload method_missing(name,x,...)
  @param [Symbol] name  method name.
  @param [NArray,Numeric] x  input array.
  @return [NArray] result.
*/
static VALUE cumo_na_math_method_missing(int argc, VALUE *argv, VALUE mod)
{
    VALUE type, ans, typemod, hash;
    if (argc>1) {
	type = cumo_na_mathcast(argc-1,argv+1);

	hash = rb_const_get(mod, cumo_id_DISPATCH);
	typemod = rb_hash_aref( hash, type );
	if (NIL_P(typemod)) {
	    rb_raise(rb_eTypeError,"%s is unknown for Cumo::NMath",
		     rb_class2name(type));
	}

	ans = rb_funcall2(typemod,cumo_id_send,argc,argv);

	if (!RTEST(rb_class_inherited_p(type,cNArray)) &&
	    IsNArray(ans) ) {
	    ans = rb_funcall(ans,cumo_id_extract,0);
	}
	return ans;
    }
    rb_raise(rb_eArgError,"argument or method missing");
    return Qnil;
}


void
Init_cumo_na_math()
{
    VALUE hCast;

    cumo_mNMath = rb_define_module_under(mCumo, "NMath");
    rb_define_singleton_method(cumo_mNMath, "method_missing", cumo_na_math_method_missing, -1);

    hCast = rb_hash_new();
    rb_define_const(cumo_mNMath, "DISPATCH", hCast);
    rb_hash_aset(hCast, cumo_cInt64,    cumo_mDFloatMath);
    rb_hash_aset(hCast, cumo_cInt32,    cumo_mDFloatMath);
    rb_hash_aset(hCast, cumo_cInt16,    cumo_mDFloatMath);
    rb_hash_aset(hCast, cumo_cInt8,     cumo_mDFloatMath);
    rb_hash_aset(hCast, cumo_cUInt64,   cumo_mDFloatMath);
    rb_hash_aset(hCast, cumo_cUInt32,   cumo_mDFloatMath);
    rb_hash_aset(hCast, cumo_cUInt16,   cumo_mDFloatMath);
    rb_hash_aset(hCast, cumo_cUInt8,    cumo_mDFloatMath);
    rb_hash_aset(hCast, cumo_cDFloat,   cumo_mDFloatMath);
    rb_hash_aset(hCast, cumo_cDFloat,   cumo_mDFloatMath);
    rb_hash_aset(hCast, cumo_cDComplex, cumo_mDComplexMath);
    rb_hash_aset(hCast, cumo_cSFloat,   cumo_mSFloatMath);
    rb_hash_aset(hCast, cumo_cSComplex, cumo_mSComplexMath);
#ifdef RUBY_INTEGER_UNIFICATION
    rb_hash_aset(hCast, rb_cInteger, rb_mMath);
#else
    rb_hash_aset(hCast, rb_cFixnum,  rb_mMath);
    rb_hash_aset(hCast, rb_cBignum,  rb_mMath);
#endif
    rb_hash_aset(hCast, rb_cFloat,   rb_mMath);
    rb_hash_aset(hCast, rb_cComplex, cumo_mDComplexMath);

    cumo_id_send     = rb_intern("send");
    cumo_id_UPCAST   = rb_intern("UPCAST");
    cumo_id_DISPATCH = rb_intern("DISPATCH");
    cumo_id_extract  = rb_intern("extract");
}
