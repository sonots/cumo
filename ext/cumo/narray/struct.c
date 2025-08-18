#include <ruby.h>
#include "cumo/narray.h"
#include "cumo/template.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"

#define cT cumo_cStruct
VALUE cT;

static VALUE
nst_allocate(VALUE self)
{
    cumo_narray_t *na;
    void *ptr;
    VALUE velmsz;

    CumoGetNArray(self,na);

    switch(CUMO_NA_TYPE(na)) {
    case CUMO_NARRAY_DATA_T:
        ptr = CUMO_NA_DATA_PTR(na);
        if (na->size > 0 && ptr == NULL) {
            velmsz = rb_const_get(rb_obj_class(self), rb_intern("element_byte_size"));
            ptr = cumo_cuda_runtime_malloc(NUM2SIZET(velmsz) * na->size);
            CUMO_NA_DATA_PTR(na) = ptr;
        }
        break;
    case CUMO_NARRAY_VIEW_T:
        rb_funcall(CUMO_NA_VIEW_DATA(na), rb_intern("allocate"), 0);
        break;
    case CUMO_NARRAY_FILEMAP_T:
        //ptr = ((cumo_narray_filemap_t*)na)->ptr;
        // to be implemented
    default:
        rb_bug("invalid narray type : %d",CUMO_NA_TYPE(na));
    }
    return self;
}


static inline VALUE
nst_definitions(VALUE nst)
{
    return rb_const_get(nst, rb_intern("DEFINITIONS"));
}

static VALUE
nst_definition(VALUE nst, VALUE idx)
{
    long i;
    VALUE def = nst_definitions(rb_obj_class(nst));
    long  len = RARRAY_LEN(def);

    if (TYPE(idx) == T_STRING || TYPE(idx) == T_SYMBOL) {
        ID id  = rb_to_id(idx);
        for (i=0; i<len; i++) {
            VALUE key = RARRAY_AREF(RARRAY_AREF(def,i),0);
            if (SYM2ID(key) == id) {
                return RARRAY_AREF(def,i);
            }
        }
    } else if (rb_obj_is_kind_of(idx,rb_cNumeric)) {
        i = NUM2LONG(idx);
        if (i<-len || i>=len) {
            rb_raise(rb_eIndexError,"offset %ld out of range of struct(size:%ld)", i, len);
        }
        return RARRAY_AREF(def,i);
    }
    return Qnil;
}



void cumo_na_copy_array_structure(VALUE self, VALUE view);

static VALUE
cumo_na_make_view_struct(VALUE self, VALUE dtype, VALUE offset)
{
    size_t n;
    int j, k, ndim;
    size_t *shape;
    size_t *idx1, *idx2;
    ssize_t stride;
    cumo_stridx_t *stridx;
    cumo_narray_t *na, *nt;
    cumo_narray_view_t *na1, *na2;
    VALUE klass;
    volatile VALUE view;

    CumoGetNArray(self,na);

    // build from Cumo::Struct
    if (rb_obj_is_kind_of(dtype,cNArray)) {
	CumoGetNArray(dtype,nt);
        ndim = na->ndim + nt->ndim;
        shape = ALLOCA_N(size_t,ndim);
        // struct dimensions
        for (j=0; j<na->ndim; j++) {
            shape[j] = na->shape[j];
        }
        // member dimension
        for (j=na->ndim,k=0; j<ndim; j++,k++) {
            shape[j] = nt->shape[k];
        }
        klass = rb_obj_class(dtype);
        stridx = ALLOC_N(cumo_stridx_t, ndim);
        stride = cumo_na_dtype_element_stride(klass);
        for (j=ndim,k=nt->ndim; k; ) {
            CUMO_SDX_SET_STRIDE(stridx[--j],stride);
            stride *= nt->shape[--k];
        }
    } else {
        ndim = na->ndim;
        shape = ALLOCA_N(size_t,ndim);
        for (j=0; j<ndim; j++) {
            shape[j] = na->shape[j];
        }
        klass = rb_obj_class(self);
        if (TYPE(dtype)==T_CLASS) {
            if (RTEST(rb_class_inherited_p(dtype,cNArray))) {
                klass = dtype;
            }
        }
        stridx = ALLOC_N(cumo_stridx_t, ndim);
    }

    view = cumo_na_s_allocate_view(klass);
    cumo_na_copy_flags(self, view);
    CumoGetNArrayView(view, na2);
    cumo_na_setup_shape((cumo_narray_t*)na2, ndim, shape);
    na2->stridx = stridx;

    switch(na->type) {
    case CUMO_NARRAY_DATA_T:
    case CUMO_NARRAY_FILEMAP_T:
        stride = cumo_na_element_stride(self);
        for (j=na->ndim; j--;) {
            CUMO_SDX_SET_STRIDE(na2->stridx[j], stride);
            stride *= na->shape[j];
        }
        na2->offset = 0;
        na2->data = self;
        break;
    case CUMO_NARRAY_VIEW_T:
        CumoGetNArrayView(self, na1);
        for (j=na1->base.ndim; j--; ) {
            if (CUMO_SDX_IS_INDEX(na1->stridx[j])) {
                n = na1->base.shape[j];
                idx1 = CUMO_SDX_GET_INDEX(na1->stridx[j]);
                // idx2 = ALLOC_N(size_t, na1->base.shape[j]);
                // for (i=0; i<n; i++) {
                //     idx2[i] = idx1[i];
                // }
                idx2 = (size_t*)cumo_cuda_runtime_malloc(sizeof(size_t)*n);
                cumo_cuda_runtime_check_status(cudaMemcpyAsync(idx2,idx1,sizeof(size_t)*n,cudaMemcpyDeviceToDevice,0));
                CUMO_SDX_SET_INDEX(na2->stridx[j],idx2);
            } else {
                na2->stridx[j] = na1->stridx[j];
            }
        }
        na2->offset = na1->offset;
        na2->data = na1->data;
        break;
    }

    if (RTEST(offset)) {
        na2->offset += NUM2SIZET(offset);
    }

    return view;
}


static VALUE
nst_field_view(VALUE self, VALUE idx)
{
    VALUE def, type, ofs;

    def = nst_definition(self, idx);
    if (!RTEST(def)) {
        idx = rb_funcall(idx, rb_intern("to_s"), 0);
        rb_raise(rb_eTypeError, "Invalid field: '%s' for struct %s",
                 StringValuePtr(idx), rb_class2name(rb_obj_class(self)));
    }
    type = RARRAY_AREF(def,1);
    ofs  = RARRAY_AREF(def,2);
    return cumo_na_make_view_struct(self, type, ofs);
}

static VALUE
nst_field(VALUE self, VALUE idx)
{
    VALUE obj;
    cumo_narray_view_t *nv;

    obj = nst_field_view(self,idx);
    CumoGetNArrayView(obj,nv);
    if (nv->base.ndim==0) {
        obj = rb_funcall(obj,rb_intern("extract_cpu"),0);
    }
    return obj;
}

static VALUE
nst_field_set(VALUE self, VALUE idx, VALUE other)
{
    VALUE obj;

    obj = nst_field_view(self,idx);
    rb_funcall(obj,rb_intern("store"),1,other);
    return other;
}


static VALUE
nst_method_missing(int argc, VALUE *argv, VALUE self)
{
    VALUE s, tag, obj;

    if (argc == 2) {
        s = rb_sym_to_s(argv[0]);
        if (RSTRING_PTR(s)[RSTRING_LEN(s)-1] == '=') {
            tag = rb_str_intern(rb_str_new(RSTRING_PTR(s), RSTRING_LEN(s)-1));
            obj = nst_field(self, tag);
            if (RTEST(obj)) {
                rb_funcall(obj, rb_intern("store"), 1, argv[1]);
                return argv[1];
            }
        }
        return rb_call_super(argc,argv);
    }
    if (argc == 1) {
        obj = nst_field(self,argv[0]);
        if (RTEST(obj)) {
            return obj;
        }
    }
    return rb_call_super(argc,argv);
}


/*
  Foo = Cumo::Struct.new {
    int8     :byte
    float64  :float, [2,2]
    dcomplex :compl
  }
 */
static VALUE
nst_s_new(int argc, VALUE *argv, VALUE klass)
{
    VALUE name=Qnil, rest, size;
    VALUE st, members;
    ID id;

    rb_scan_args(argc, argv, "0*", &rest);
    if (RARRAY_LEN(rest)>0) {
        name = RARRAY_AREF(rest,0);
        if (!NIL_P(name)) {
            VALUE tmp = rb_check_string_type(name);
            if (!NIL_P(tmp)) {
                rb_ary_shift(rest);
            } else {
                name = Qnil;
            }
        }
    }

    if (NIL_P(name)) {
        st = rb_define_class_id(name, klass);
        rb_funcall(klass, rb_intern("inherited"), 1, st);
    }
    else {
        char *cname = StringValuePtr(name);
        id = rb_intern(cname);
        if (!rb_is_const_id(id)) {
            rb_name_error(id, "identifier %s needs to be constant", cname);
        }
        if (rb_const_defined_at(klass, id)) {
            rb_warn("redefining constant Struct::%s", cname);
            rb_mod_remove_const(klass, ID2SYM(id));
        }
        st = rb_define_class_under(klass, rb_id2name(id), klass);
    }

    rb_iv_set(st, "__members__", rb_ary_new());
    rb_iv_set(st, "__offset__", INT2FIX(0));

    if (rb_block_given_p()) {
        rb_mod_module_eval(0, 0, st);
    }

    size = rb_iv_get(st, "__offset__");
    members = rb_iv_get(st, "__members__");
    //printf("size=%d\n",NUM2INT(size));
    rb_define_const(st, "CONTIGUOUS_STRIDE", size);
    rb_define_const(st, "ELEMENT_BYTE_SIZE", size);
    rb_define_const(st, "ELEMENT_BIT_SIZE",  rb_funcall(size,'*',1,INT2FIX(8)));

    OBJ_FREEZE(members);
    rb_define_const(st, "DEFINITIONS", members);

    rb_define_singleton_method(st, "new", rb_class_new_instance, -1);
    //rb_define_singleton_method(st, "[]", rb_class_new_instance, -1);
    rb_define_method(st, "allocate", nst_allocate, 0);

    return st;
}


static VALUE
nstruct_add_type(VALUE type, int argc, VALUE *argv, VALUE nst)
{
    VALUE ofs, size;
    ID id;
    int i;
    VALUE name=Qnil;
    size_t *shape=NULL;
    int ndim=0;
    ssize_t stride;
    cumo_narray_view_t *nt;
    int j;

    for (i=0; i<argc; i++) {
        switch(TYPE(argv[i])) {
        case T_STRING:
        case T_SYMBOL:
            if (NIL_P(name)) {
                name = argv[i];
                break;
            }
            rb_raise(rb_eArgError,"multiple name in struct definition");
        case T_ARRAY:
            if (shape) {
                rb_raise(rb_eArgError,"multiple shape in struct definition");
            }
            ndim = RARRAY_LEN(argv[i]);
            if (ndim > CUMO_NA_MAX_DIMENSION) {
                rb_raise(rb_eArgError,"too large number of dimensions");
            }
            if (ndim == 0) {
                rb_raise(rb_eArgError,"array is empty");
            }
            shape = ALLOCA_N(size_t, ndim);
            cumo_na_array_to_internal_shape(Qnil, argv[i], shape);
            break;
        }
    }

    id = rb_to_id(name);
    name = ID2SYM(id);
    if (rb_obj_is_kind_of(type,cNArray)) {
        cumo_narray_t *na;
        CumoGetNArray(type,na);
        type = rb_obj_class(type);
        ndim = na->ndim;
        shape = na->shape;
    }
    type = cumo_na_view_new(type,ndim,shape);
    CumoGetNArrayView(type,nt);

    nt->stridx = ALLOC_N(cumo_stridx_t,ndim);
    stride = cumo_na_dtype_element_stride(rb_obj_class(type));
    for (j=ndim; j--; ) {
        CUMO_SDX_SET_STRIDE(nt->stridx[j], stride);
        stride *= shape[j];
    }

    ofs  = rb_iv_get(nst, "__offset__");
    nt->offset = NUM2SIZET(ofs);

    size = rb_funcall(type, rb_intern("byte_size"), 0);
    rb_iv_set(nst, "__offset__", rb_funcall(ofs,'+',1,size));
    rb_ary_push(rb_iv_get(nst,"__members__"),
                rb_ary_new3(4,name,type,ofs,size));  // <- field definition
    return Qnil;
}


static VALUE
nst_extract(VALUE self)
{
    return self;
}


static void
iter_nstruct_to_a(cumo_na_loop_t *const lp)
{
    long    i, len;
    VALUE   opt, types, defs, def;
    VALUE   elmt, velm, vary;
    size_t  ofs, pos;
    cumo_narray_view_t *ne;

    opt = lp->option;
    types = RARRAY_AREF(opt,0);
    defs = RARRAY_AREF(opt,1);
    pos = lp->args[0].iter[0].pos;

    len = RARRAY_LEN(types);
    vary = rb_ary_new2(len);

    for (i=0; i<len; i++) {
        def  = RARRAY_AREF(defs,i);
        ofs  = NUM2SIZET(RARRAY_AREF(def,2));
        //ofs  = NUM2SIZET(RARRAY_AREF(ofsts,i));
        elmt = RARRAY_AREF(types,i);
        CumoGetNArrayView(elmt,ne);
        ne->offset = pos + ofs;
        if (ne->base.ndim==0) {
            velm = rb_funcall(elmt,rb_intern("extract_cpu"),0);
        } else {
            velm = rb_funcall(elmt,rb_intern("to_a"),0);
        }
        rb_ary_push(vary, velm);
    }
    rb_ary_push(lp->args[1].value, vary);
}

static VALUE
cumo_na_original_data(VALUE self)
{
    cumo_narray_t *na;
    cumo_narray_view_t *nv;

    CumoGetNArray(self,na);
    switch(na->type) {
    case CUMO_NARRAY_VIEW_T:
        CumoGetNArrayView(self, nv);
        return nv->data;
    }
    return self;
}

static VALUE
nst_create_member_views(VALUE self)
{
    VALUE defs, def, types, type, elmt;
    long  i, len;
    cumo_narray_view_t *ne;

    defs = nst_definitions(rb_obj_class(self));
    len = RARRAY_LEN(defs);
    types = rb_ary_new2(len);
    //ofsts = rb_ary_new2(len);
    for (i=0; i<len; i++) {
        def  = RARRAY_AREF(defs,i);
        type = RARRAY_AREF(def,1);
        //ofst = RARRAY_AREF(def,2);
        elmt = cumo_na_make_view(type);
        rb_ary_push(types, elmt);
        //rb_ary_push(ofsts, ofst);
        CumoGetNArrayView(elmt,ne);
        ne->data = cumo_na_original_data(self);
    }
    return rb_assoc_new(types,defs);
}

static VALUE
cumo_na_struct_to_a(VALUE self)
{
    volatile VALUE opt;
    cumo_ndfunc_arg_in_t ain[3] = {{Qnil,0},{cumo_sym_loop_opt},{cumo_sym_option}};
    cumo_ndfunc_arg_out_t aout[1] = {{rb_cArray,0}}; // dummy?
    cumo_ndfunc_t ndf = {iter_nstruct_to_a, CUMO_NO_LOOP, 3, 1, ain, aout};

    opt = nst_create_member_views(self);
    return cumo_na_ndloop_cast_narray_to_rarray(&ndf, self, opt);
}



/*
static size_t
check_array(VALUE item) {
    cumo_narray_t *na;

    if (TYPE(item) == T_ARRAY) {
        return 1;
    }
    if (RTEST(rb_obj_is_kind_of(item, cNArray))) {
        CumoGetNArray(item,na);
        if (na->ndim == 1) {
            return 1;
        } else {
            return 0;
        }
    }
    return 0;
}
*/

/*
static size_t
check_array_1d(VALUE item, size_t size) {
    cumo_narray_t *na;
    size_t i, len;

    if (TYPE(item) == T_ARRAY) {
        len = RARRAY_LEN(item);
        if (size != len) {
            return 0;
        }
        for (i=0; i<len; i++) {
            if (!check_array(RARRAY_AREF(item,i))) {
                return 0;
            }
        }
        return 1;
    }
    if (RTEST(rb_obj_is_kind_of(item, cNArray))) {
        CumoGetNArray(item,na);
        if (na->ndim == 1 && na->size == size) {
            return 1;
        } else {
            return 0;
        }
    }
    return 0;
}
*/

/*
VALUE
nst_check_compatibility(VALUE nst, VALUE ary)
{
    VALUE defs, def, type, item;
    long len, i;
    cumo_narray_t *nt;

    if (TYPE(ary) != T_ARRAY) {
        if (nst==rb_obj_class(ary)) { // same Struct
            return Qtrue;
        }
        return Qfalse;
    }
    defs = nst_definitions(nst);
    len = RARRAY_LEN(defs);

    if (len != RARRAY_LEN(ary)) {
        //puts("pass2");
        return Qfalse;
    }
    for (i=0; i<len; i++) {
        def  = RARRAY_AREF(defs,i);
        type = RARRAY_AREF(def,1);
        CumoGetNArray(type,nt);
        item = RARRAY_AREF(ary,i);
        if (nt->ndim == 0) {
            if (check_array(item)) {
                //puts("pass3");
                return Qfalse;
            }
        } else if (nt->ndim == 1) {
            if (!check_array_1d(item, nt->size)) {
                //puts("pass4");
                return Qfalse;
            }
        } else {
            // multi-dimension member
            //volatile VALUE vnc;
            //cumo_na_compose_t *nc;
            VALUE vnc;
            cumo_narray_t *nc;
            int j;

            //vnc = cumo_na_ary_composition(item);
            //Data_Get_Struct(vnc, cumo_na_compose_t, nc);
            vnc = cumo_na_s_new_like(cNArray, item);
            CumoGetNArray(vnc,nc);
            if (nt->ndim != nc->ndim) {
                return Qfalse;
            }
            for (j=0; j<nc->ndim; j++) {
                if (nc->shape[j] != nt->shape[j]) {
                    return Qfalse;
                }
            }
            return Qtrue;
        }
    }
    return Qtrue;
}
*/


VALUE cumo_na_ary_composition_for_struct(VALUE nstruct, VALUE ary);

// ------
static void
iter_nstruct_from_a(cumo_na_loop_t *const lp)
{
    long  i, len;
    VALUE ary;
    VALUE types, defs, def;
    VALUE elmt, item;
    size_t ofs;
    cumo_narray_view_t *ne;

    types = RARRAY_AREF(lp->option,0);
    defs = RARRAY_AREF(lp->option,1);

    len = RARRAY_LEN(types);
    ary = lp->args[1].value;
    //rb_p(rb_obj_class(ary));rb_p(ary);

    for (i=0; i<len; i++) {
        def  = RARRAY_AREF(defs,i);
        ofs  = NUM2SIZET(RARRAY_AREF(def,2));
        elmt = RARRAY_AREF(types,i);
        CumoGetNArrayView(elmt,ne);
        ne->offset = lp->args[0].iter[0].pos + ofs;
        item = RARRAY_AREF(ary,i);
        //rb_p(ary);
        //rb_p(item);
        //rb_p(elmt);
        //abort();
        rb_funcall(elmt, rb_intern("store"), 1, item);
    }
}

static VALUE
cumo_na_struct_cast_array(VALUE klass, VALUE rary)
{
    //volatile VALUE vnc, nary;
    VALUE nary;
    cumo_narray_t *na;
    //cumo_na_compose_t *nc;
    VALUE opt;
    cumo_ndfunc_arg_in_t ain[3] = {{CUMO_OVERWRITE,0},{rb_cArray,0},{cumo_sym_option}};
    cumo_ndfunc_t ndf = {iter_nstruct_from_a, CUMO_NO_LOOP, 3, 0, ain, 0};

    //fprintf(stderr,"rary:");rb_p(rary);
    //fprintf(stderr,"class_of(rary):");rb_p(rb_obj_class(rary));

    //vnc = cumo_na_ary_composition_for_struct(klass, rary);
    //Data_Get_Struct(vnc, cumo_na_compose_t, nc);
    //nary = cumo_na_new(klass, nc->ndim, nc->shape);
    nary = cumo_na_s_new_like(klass, rary);
    CumoGetNArray(nary,na);
    //fprintf(stderr,"na->size=%lu\n",na->size);
    //fprintf(stderr,"na->ndim=%d\n",na->ndim);
    if (na->size>0) {
        opt = nst_create_member_views(nary);
        rb_funcall(nary, rb_intern("allocate"), 0);
        cumo_na_ndloop_store_rarray2(&ndf, nary, rary, opt);
    }
    return nary;
}

static inline VALUE
cumo_na_struct_s_cast(VALUE klass, VALUE rary)
{
    return cumo_na_struct_cast_array(klass, rary);
}



static void
iter_struct_store_struct(cumo_na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;
    size_t  elmsz;
    char   *x, *y;

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    CUMO_INIT_ELMSIZE(lp, 0, elmsz);
    if (idx2) {
        if (idx1) {
            for (; i--;) {
                x = (char*)(p1+*idx1); idx1++;
                y = (char*)(p2+*idx2); idx2++;
                memcpy(x,y,elmsz);
            }
        } else {
            for (; i--;) {
                x = (char*)p1;         p1+=s1;
                y = (char*)(p2+*idx2); idx2++;
                memcpy(x,y,elmsz);
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                x = (char*)(p1+*idx1); idx1++;
                y = (char*)p2;         p2+=s2;
                memcpy(x,y,elmsz);
            }
        } else {
            for (; i--;) {
                x = (char*)p1;         p1+=s1;
                y = (char*)p2;         p2+=s2;
                memcpy(x,y,elmsz);
            }
        }
    }
}


static VALUE
cumo_na_struct_store_struct(VALUE self, VALUE obj)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{Qnil,0}};
    cumo_ndfunc_t ndf = {iter_struct_store_struct, CUMO_FULL_LOOP, 2, 0, ain, 0};

    cumo_na_ndloop(&ndf, 2, self, obj);
    return self;
}




static inline VALUE
cumo_na_struct_store_array(VALUE self, VALUE obj)
{
    return cumo_na_struct_store_struct(self, cumo_na_struct_cast_array(rb_obj_class(self),obj));
}

/*
  Store elements to Cumo::Struct from other.
  @overload store(other)
  @param [Object] other
  @return [Cumo::Struct] self
*/
static VALUE
cumo_na_struct_store(VALUE self, VALUE obj)
{
    if (TYPE(obj)==T_ARRAY) {
        cumo_na_struct_store_array(self,obj);
        return self;
    }
    if (rb_obj_class(self) == rb_obj_class(obj)) {
        cumo_na_struct_store_struct(self,obj);
        return self;
    }
    rb_raise(cumo_na_eCastError, "unknown conversion from %s to %s",
             rb_class2name(rb_obj_class(obj)),
             rb_class2name(rb_obj_class(self)));
    return self;
}



static VALUE
//iter_struct_inspect(cumo_na_loop_t *const lp)
iter_struct_inspect(char *ptr, size_t pos, VALUE opt)
{
    VALUE   types, defs, def, name, elmt, vary, v, x;
    size_t  ofs;
    long    i, len;
    cumo_narray_view_t *ne;

    types = RARRAY_AREF(opt,0);
    defs = RARRAY_AREF(opt,1);

    len = RARRAY_LEN(types);
    vary = rb_ary_new2(len);

    for (i=0; i<len; i++) {
        def  = RARRAY_AREF(defs,i);
        name = RARRAY_AREF(def,0);
        ofs  = NUM2SIZET(RARRAY_AREF(def,2));
        elmt = RARRAY_AREF(types,i);
        CumoGetNArrayView(elmt,ne);
        ne->offset = pos + ofs;
        v = rb_str_concat(rb_sym_to_s(name), rb_str_new2(": "));
        x = rb_funcall(elmt, rb_intern("format_to_a"), 0);        // <-- fix me
        if (ne->base.ndim==0) {
            x = rb_funcall(x, rb_intern("first"), 0);
        }
        x = rb_funcall(x, rb_intern("to_s"), 0);
        v = rb_str_concat(v, x);
        rb_ary_push(vary, v);
    }
    v = rb_ary_join(vary, rb_str_new2(", "));
    v = rb_str_concat(rb_str_new2("["), v);
    v = rb_str_concat(v, rb_str_new2("]"));
    return v;
}

/*
  Returns a string containing a human-readable representation of NArray.
  @overload inspect
  @return [String]
*/
static VALUE
cumo_na_struct_inspect(VALUE ary)
{
    VALUE opt;
    opt = nst_create_member_views(ary);
    return cumo_na_ndloop_inspect(ary, iter_struct_inspect, opt);
}


static VALUE
nst_s_add_type(int argc, VALUE *argv, VALUE mod)
{
    if (argc==0)
        rb_raise(rb_eArgError,
                 "wrong number of arguments (%d for 1)", argc);
    nstruct_add_type(argv[0],argc-1,argv+1,mod);
    return Qnil;
}




#define NST_TYPEDEF(tpname,tpclass)                 \
static VALUE                                        \
nst_s_##tpname(int argc, VALUE *argv, VALUE mod)  \
{   nstruct_add_type(tpclass,argc,argv,mod);        \
    return Qnil;                                    \
}

NST_TYPEDEF(int8,cumo_cInt8)
NST_TYPEDEF(int16,cumo_cInt16)
NST_TYPEDEF(int32,cumo_cInt32)
NST_TYPEDEF(int64,cumo_cInt64)
NST_TYPEDEF(uint8,cumo_cUInt8)
NST_TYPEDEF(uint16,cumo_cUInt16)
NST_TYPEDEF(uint32,cumo_cUInt32)
NST_TYPEDEF(uint64,cumo_cUInt64)
NST_TYPEDEF(dfloat,cumo_cDFloat)
NST_TYPEDEF(dcomplex,cumo_cDComplex)
NST_TYPEDEF(sfloat,cumo_cSFloat)
NST_TYPEDEF(scomplex,cumo_cSComplex)


#define rb_define_singleton_alias(klass,name1,name2) \
    rb_define_alias(rb_singleton_class(klass),name1,name2)

void
Init_cumo_na_struct()
{
    cT = rb_define_class_under(mCumo, "Struct", cumo_cNArray);
    //cNStMember = rb_define_class_under(cT, "Member", rb_cObject);

    //rb_define_alloc_func(cNStMember, nst_member_s_allocate);
    //rb_define_method(cNStMember, "initialize", nst_member_initialize, -1);

    //rb_undef_alloc_func(cT);
    rb_define_singleton_method(cT, "new", nst_s_new, -1);
    rb_define_singleton_method(cT, "add_type", nst_s_add_type, -1);
    rb_define_singleton_method(cT, "int8",   nst_s_int8,   -1);
    rb_define_singleton_method(cT, "int16",  nst_s_int16,  -1);
    rb_define_singleton_method(cT, "int32",  nst_s_int32,  -1);
    rb_define_singleton_method(cT, "int64",  nst_s_int64,  -1);
    rb_define_singleton_method(cT, "uint8",  nst_s_uint8,  -1);
    rb_define_singleton_method(cT, "uint16", nst_s_uint16, -1);
    rb_define_singleton_method(cT, "uint32", nst_s_uint32, -1);
    rb_define_singleton_method(cT, "uint64", nst_s_uint64, -1);
    rb_define_singleton_method(cT, "sfloat",   nst_s_sfloat, -1);
    rb_define_singleton_alias (cT, "float32", "sfloat");
    rb_define_singleton_method(cT, "scomplex", nst_s_scomplex, -1);
    rb_define_singleton_alias (cT, "complex64", "scomplex");
    rb_define_singleton_method(cT, "dfloat",   nst_s_dfloat, -1);
    rb_define_singleton_alias (cT, "float64", "dfloat");
    rb_define_singleton_method(cT, "dcomplex", nst_s_dcomplex, -1);
    rb_define_singleton_alias (cT, "complex128", "dcomplex");

    rb_define_method(cT, "definition", nst_definition, 1);
    rb_define_method(cT, "definitions", nst_definitions, 0);
    rb_define_method(cT, "field", nst_field, 1);
    rb_define_method(cT, "field_set", nst_field_set, 2);
    rb_define_method(cT, "extract", nst_extract, 0);
    rb_define_method(cT, "method_missing", nst_method_missing, -1);

    //rb_define_method(cT, "fill", cumo_na_nstruct_fill, 1);

    //rb_define_method(cT, "debug_print", cumo_na_nstruct_debug_print, 0);

    rb_define_method(cT, "to_a", cumo_na_struct_to_a, 0);

    rb_define_method(cT, "store", cumo_na_struct_store, 1);

    rb_define_method(cT, "inspect", cumo_na_struct_inspect, 0);

    rb_define_singleton_method(cT, "cast", cumo_na_struct_s_cast, 1);
    rb_define_singleton_method(cT, "[]", cumo_na_struct_s_cast, -2);

    //rb_define_method(cT, "initialize", rb_struct_initialize, -2);
    //rb_define_method(cT, "initialize_copy", rb_struct_init_copy, 1);
}
