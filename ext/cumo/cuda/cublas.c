#include "cumo/cuda/cublas.h"

#include <assert.h>
#include <ruby.h>
#include "cumo/narray.h"
#include "cumo/template.h"

static void *blas_handle = 0;
static char *blas_prefix = 0;

VALUE
cumo_cublas_option_value(VALUE order, VALUE default_value)
{
    switch(TYPE(order)) {
    case T_NIL:
    case T_UNDEF:
        return default_value;
    }
    return order;
}

enum CBLAS_ORDER
cumo_cublas_option_order(VALUE order)
{
    int opt;
    char *ptr;

    switch(TYPE(order)) {
    case T_NIL:
    case T_UNDEF:
    case T_FALSE:
        return CblasRowMajor;
    case T_TRUE:
        return CblasColMajor;
    case T_FIXNUM:
        opt = FIX2INT(order);
        if (opt >= CblasRowMajor && opt <= CblasColMajor) {
            return opt;
        }
        break;
    case T_SYMBOL:
        order = rb_sym2str(order);
    case T_STRING:
        ptr = RSTRING_PTR(order);
        if (RSTRING_LEN(order) > 0) {
            switch(ptr[0]){
            case 'R': case 'r':
                return CblasRowMajor;
            case 'C': case 'c':
                return CblasColMajor;
            }
        }
        break;
    }
    rb_raise(rb_eArgError,"invalid value for CBLAS_ORDER");
    return 0;
}

enum CBLAS_TRANSPOSE
cumo_cublas_option_trans(VALUE trans)
{
    int opt;
    char *ptr;

    switch(TYPE(trans)) {
    case T_NIL:
    case T_UNDEF:
    case T_FALSE:
        return CblasNoTrans;
    case T_TRUE:
        return CblasTrans;
    case T_FIXNUM:
        opt = FIX2INT(trans);
        if (opt >= CblasNoTrans && opt <= CblasConjTrans) {
            return opt;
        }
        break;
    case T_SYMBOL:
        trans = rb_sym2str(trans);
    case T_STRING:
        ptr = RSTRING_PTR(trans);
        if (RSTRING_LEN(trans) > 0) {
            switch(ptr[0]){
            case 'N': case 'n':
                return CblasNoTrans;
            case 'T': case 't':
                return CblasTrans;
            case 'C': case 'c':
                return CblasConjTrans;
            }
        }
        break;
    }
    rb_raise(rb_eArgError,"invalid value for CBLAS_TRANSPOSE");
    return 0;
}

enum CBLAS_UPLO
cumo_cublas_option_uplo(VALUE uplo)
{
    int opt;
    char *ptr;

    switch(TYPE(uplo)) {
    case T_NIL:
    case T_UNDEF:
    case T_FALSE:
        return CblasUpper;
    case T_TRUE:
        return CblasLower;
    case T_FIXNUM:
        opt = FIX2INT(uplo);
        switch(opt){
        case CblasUpper:
        case CblasLower:
            return opt;
        }
        break;
    case T_SYMBOL:
        uplo = rb_sym2str(uplo);
    case T_STRING:
        ptr = RSTRING_PTR(uplo);
        if (RSTRING_LEN(uplo) > 0) {
            switch(ptr[0]){
            case 'U': case 'u':
                return CblasUpper;
            case 'L': case 'l':
                return CblasLower;
            }
        }
        break;
    }
    rb_raise(rb_eArgError,"invalid value for CBLAS_UPLO");
    return 0;
}

enum CBLAS_DIAG
cumo_cublas_option_diag(VALUE diag)
{
    int opt;
    char *ptr;

    switch(TYPE(diag)) {
    case T_NIL:
    case T_UNDEF:
    case T_FALSE:
        return CblasNonUnit;
    case T_TRUE:
        return CblasUnit;
    case T_FIXNUM:
        opt = FIX2INT(diag);
        switch(opt){
        case CblasNonUnit:
        case CblasUnit:
            return opt;
        }
        break;
    case T_SYMBOL:
        diag = rb_sym2str(diag);
    case T_STRING:
        ptr = RSTRING_PTR(diag);
        if (RSTRING_LEN(diag) > 0) {
            switch(ptr[0]){
            case 'N': case 'n':
                return CblasNonUnit;
            case 'U': case 'u':
                return CblasUnit;
            }
        }
        break;
    }
    rb_raise(rb_eArgError,"invalid value for CBLAS_DIAG");
    return 0;
}

enum CBLAS_SIDE
cumo_cublas_option_side(VALUE side)
{
    int opt;
    char *ptr;

    switch(TYPE(side)) {
    case T_NIL:
    case T_UNDEF:
    case T_FALSE:
        return CblasLeft;
    case T_TRUE:
        return CblasRight;
    case T_FIXNUM:
        opt = FIX2INT(side);
        switch(opt){
        case CblasLeft:
        case CblasRight:
            return opt;
        }
        break;
    case T_SYMBOL:
        side = rb_sym2str(side);
    case T_STRING:
        ptr = RSTRING_PTR(side);
        if (RSTRING_LEN(side) > 0) {
            switch(ptr[0]){
            case 'L': case 'l':
                return CblasLeft;
            case 'R': case 'r':
                return CblasRight;
            }
        }
        break;
    }
    rb_raise(rb_eArgError,"invalid value for CBLAS_SIDE");
    return 0;
}

void
cumo_cublas_check_func(void **func, const char *name)
{
    char *s, *error;

    if (*func==0) {
        if (blas_handle==0) {
            rb_raise(rb_eRuntimeError,"BLAS library is not loaded");
        }
        if (blas_prefix==0) {
            rb_raise(rb_eRuntimeError,"CBLAS prefix is not set");
        }
        s = alloca(strlen(blas_prefix)+strlen(name)+1);
        strcpy(s,blas_prefix);
        strcat(s,name);
        dlerror();
        *func = dlsym(blas_handle, s);
        error = dlerror();
        if (error != NULL) {
            rb_raise(rb_eRuntimeError, "%s", error);
        }
    }
}

/*
  module definition: Numo::Linalg
*/
static VALUE mLinalg;

/*
  module definition: Numo::Linalg::Blas
*/
static VALUE mBlas;


static VALUE
blas_s_dlopen(int argc, VALUE *argv, VALUE mod)
{
    int i, f;
    VALUE lib, flag;
    char *error;
    void *handle;

    i = rb_scan_args(argc, argv, "11", &lib, &flag);
    if (i==2) {
        f = NUM2INT(flag);
    } else {
        f = RTLD_LAZY | RTLD_LOCAL;
    }
    dlerror();
    handle = dlopen(StringValueCStr(lib), f);
    error = dlerror();
    if (error != NULL) {
        rb_raise(rb_eRuntimeError, "%s", error);
    }
    blas_handle = handle;
    return Qnil;
}


static VALUE
blas_s_prefix_set(VALUE mod, VALUE prefix)
{
    long len;

    if (TYPE(prefix) != T_STRING) {
        rb_raise(rb_eTypeError,"argument must be string");
    }
    if (blas_prefix) {
        free(blas_prefix);
    }
    len = RSTRING_LEN(prefix);
    blas_prefix = malloc(len+1);
    strcpy(blas_prefix, StringValueCStr(prefix));
    return prefix;
}


void Init_cumo_linalg_blas_s();
void Init_cumo_linalg_blas_d();
void Init_cumo_linalg_blas_c();
void Init_cumo_linalg_blas_z();

void
Init_blas(void)
{
    VALUE mN;

    mN = rb_define_module("Numo");
    /*
      Document-module: Numo::Linalg
    */
    mLinalg = rb_define_module_under(mN, "Linalg");
    mBlas = rb_define_module_under(mLinalg, "Blas");

    rb_define_module_function(mBlas, "dlopen", blas_s_dlopen, -1);
    rb_define_module_function(mBlas, "prefix=", blas_s_prefix_set, 1);

    blas_prefix = malloc(strlen("cublas_")+1); // default prefix
    strcpy(blas_prefix,"cublas_");

    Init_cumo_linalg_blas_s();
    Init_cumo_linalg_blas_d();
    Init_cumo_linalg_blas_c();
    Init_cumo_linalg_blas_z();
}


