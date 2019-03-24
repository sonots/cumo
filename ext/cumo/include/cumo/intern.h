#ifndef CUMO_INTERN_H
#define CUMO_INTERN_H

void cumo_debug_breakpoint(void);

VALUE cumo_na_new(VALUE elem, int ndim, size_t *shape);
VALUE cumo_na_view_new(VALUE elem, int ndim, size_t *shape);
VALUE cumo_na_debug_info(VALUE);

VALUE cumo_na_make_view(VALUE self);

VALUE cumo_na_s_allocate(VALUE klass);
VALUE cumo_na_s_allocate_view(VALUE klass);
VALUE cumo_na_s_new_like(VALUE type, VALUE obj);

void cumo_na_alloc_shape(cumo_narray_t *na, int ndim);
void cumo_na_array_to_internal_shape(VALUE self, VALUE ary, size_t *shape);
void cumo_na_index_arg_to_internal_order(int argc, VALUE *argv, VALUE self);
void cumo_na_setup_shape(cumo_narray_t *na, int ndim, size_t *shape);

unsigned int cumo_na_element_stride(VALUE nary);
size_t cumo_na_dtype_element_stride(VALUE klass);

char *cumo_na_get_pointer(VALUE);
char *cumo_na_get_pointer_for_write(VALUE);
char *cumo_na_get_pointer_for_read(VALUE);
char *cumo_na_get_pointer_for_read_write(VALUE);
size_t cumo_na_get_offset(VALUE self);
char* cumo_na_get_offset_pointer(VALUE);
char* cumo_na_get_offset_pointer_for_write(VALUE);
char* cumo_na_get_offset_pointer_for_read(VALUE);
char* cumo_na_get_offset_pointer_for_read_write(VALUE);

void cumo_na_copy_flags(VALUE src, VALUE dst);

VALUE cumo_na_check_ladder(VALUE self, int start_dim);
VALUE cumo_na_check_contiguous(VALUE self);
VALUE cumo_na_as_contiguous_array(VALUE a);

VALUE cumo_na_flatten_dim(VALUE self, int sd);

VALUE cumo_na_flatten(VALUE);

VALUE cumo_na_copy(VALUE);

VALUE cumo_na_store(VALUE self, VALUE src);

VALUE cumo_na_upcast(VALUE type1, VALUE type2);

void cumo_na_release_lock(VALUE); // currently do nothing

// used in reduce methods
VALUE cumo_na_reduce_dimension(int argc, VALUE *argv, int naryc, VALUE *naryv,
                            cumo_ndfunc_t *ndf, cumo_na_iter_func_t nan_iter);

VALUE cumo_na_reduce_options(VALUE axes, VALUE *opts, int naryc, VALUE *naryv,
                          cumo_ndfunc_t *ndf);

// ndloop
VALUE cumo_na_ndloop(cumo_ndfunc_t *nf, int argc, ...);
VALUE cumo_na_ndloop2(cumo_ndfunc_t *nf, VALUE args);
VALUE cumo_na_ndloop3(cumo_ndfunc_t *nf, void *ptr, int argc, ...);
VALUE cumo_na_ndloop4(cumo_ndfunc_t *nf, void *ptr, VALUE args);

VALUE cumo_na_ndloop_cast_narray_to_rarray(cumo_ndfunc_t *nf, VALUE nary, VALUE fmt);
VALUE cumo_na_ndloop_store_rarray(cumo_ndfunc_t *nf, VALUE nary, VALUE rary);
VALUE cumo_na_ndloop_store_rarray2(cumo_ndfunc_t *nf, VALUE nary, VALUE rary, VALUE opt);
VALUE cumo_na_ndloop_inspect(VALUE nary, cumo_na_text_func_t func, VALUE opt);
VALUE cumo_na_ndloop_with_index(cumo_ndfunc_t *nf, int argc, ...);

VALUE cumo_na_info_str(VALUE);

bool cumo_na_test_reduce(VALUE reduce, int dim);

void cumo_na_step_array_index(VALUE self, size_t ary_size, size_t *plen, ssize_t *pbeg, ssize_t *pstep);
void cumo_na_step_sequence(VALUE self, size_t *plen, double *pbeg, double *pstep);
void cumo_na_parse_enumerator_step(VALUE enum_obj, VALUE *pstep);

// used in aref, aset
int cumo_na_get_result_dimension(VALUE self, int argc, VALUE *argv, ssize_t stride, size_t *pos_idx);
VALUE cumo_na_aref_main(int nidx, VALUE *idx, VALUE self, int keep_dim, int result_nd, size_t pos);

// defined in array, used in math
VALUE cumo_na_ary_composition_dtype(VALUE ary);

#include "ruby/version.h"

#if RUBY_API_VERSION_CODE == 20100 // 2.1.0
int rb_get_kwargs(VALUE keyword_hash, const ID *table, int required, int optional, VALUE *);
VALUE rb_extract_keywords(VALUE *orighash);
#endif


#endif /* ifndef CUMO_INTERN_H */
