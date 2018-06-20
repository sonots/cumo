#ifndef CUMO_INTERN_H
#define CUMO_INTERN_H

void cumo_debug_breakpoint(void);

/* Add cumo_ prefix to avoid C symbol collisions with Numo without modifying C implementations */

#define rb_narray_new cumo_na_new
#define na_new cumo_na_new
VALUE cumo_na_new(VALUE elem, int ndim, size_t *shape);
#define rb_narray_view_new cumo_na_view_new
#define na_view_new cumo_na_view_new
VALUE cumo_na_view_new(VALUE elem, int ndim, size_t *shape);
#define rb_narray_debug_info cumo_na_debug_info
#define na_debug_info cumo_na_debug_info
VALUE cumo_na_debug_info(VALUE);

#define na_make_view cumo_na_make_view
VALUE cumo_na_make_view(VALUE self);

#define na_s_allocate cumo_na_s_allocate
VALUE cumo_na_s_allocate(VALUE klass);
#define na_s_allocate_view cumo_na_s_allocate_view
VALUE cumo_na_s_allocate_view(VALUE klass);
#define na_s_new_like cumo_na_s_new_like
VALUE cumo_na_s_new_like(VALUE type, VALUE obj);

#define na_alloc_shape cumo_na_alloc_shape
void cumo_na_alloc_shape(narray_t *na, int ndim);
#define na_array_to_internal_shape cumo_na_array_to_internal_shape
void cumo_na_array_to_internal_shape(VALUE self, VALUE ary, size_t *shape);
#define na_index_arg_to_internal_order cumo_na_index_arg_to_internal_order
void cumo_na_index_arg_to_internal_order(int argc, VALUE *argv, VALUE self);
#define na_setup_shape cumo_na_setup_shape
void cumo_na_setup_shape(narray_t *na, int ndim, size_t *shape);

#define na_get_elmsz cumo_na_element_stride
#define na_element_stride cumo_na_element_stride
//#define na_element_stride cumo_na_element_stride
unsigned int cumo_na_element_stride(VALUE nary);
#define na_dtype_elmsz cumo_na_dtype_element_stride
size_t cumo_na_dtype_element_stride(VALUE klass);

#define na_get_pointer cumo_na_get_pointer
char *cumo_na_get_pointer(VALUE);
#define na_get_pointer_for_write cumo_na_get_pointer_for_write
char *cumo_na_get_pointer_for_write(VALUE);
#define na_get_pointer_for_read cumo_na_get_pointer_for_read
char *cumo_na_get_pointer_for_read(VALUE);
#define na_get_pointer_for_read_write cumo_na_get_pointer_for_read_write
char *cumo_na_get_pointer_for_read_write(VALUE);
#define na_get_offset cumo_na_get_offset
size_t cumo_na_get_offset(VALUE self);

#define na_copy_flags cumo_na_copy_flags
void cumo_na_copy_flags(VALUE src, VALUE dst);

#define na_check_ladder cumo_na_check_ladder
VALUE cumo_na_check_ladder(VALUE self, int start_dim);
#define na_check_contiguous cumo_na_check_contiguous
VALUE cumo_na_check_contiguous(VALUE self);

#define na_flatten_dim cumo_na_flatten_dim
VALUE cumo_na_flatten_dim(VALUE self, int sd);

#define na_flatten cumo_na_flatten
VALUE cumo_na_flatten(VALUE);

#define na_copy cumo_na_dup
VALUE cumo_na_dup(VALUE);

#define na_store cumo_na_store
VALUE cumo_na_store(VALUE self, VALUE src);

#define na_upcast cumo_na_upcast
VALUE cumo_na_upcast(VALUE type1, VALUE type2);

#define na_release_lock cumo_na_release_lock
void cumo_na_release_lock(VALUE); // currently do nothing

// used in reduce methods
#define na_reduce_dimension cumo_na_reduce_dimension
#define na_reduce_dimension cumo_na_reduce_dimension
VALUE cumo_na_reduce_dimension(int argc, VALUE *argv, int naryc, VALUE *naryv,
                            ndfunc_t *ndf, na_iter_func_t nan_iter);

#define na_reduce_options cumo_na_reduce_options
#define na_reduce_options cumo_na_reduce_options
VALUE cumo_na_reduce_options(VALUE axes, VALUE *opts, int naryc, VALUE *naryv,
                          ndfunc_t *ndf);

// ndloop
#define na_ndloop cumo_na_ndloop
VALUE cumo_na_ndloop(ndfunc_t *nf, int argc, ...);
#define na_ndloop2 cumo_na_ndloop2
VALUE cumo_na_ndloop2(ndfunc_t *nf, VALUE args);
#define na_ndloop3 cumo_na_ndloop3
VALUE cumo_na_ndloop3(ndfunc_t *nf, void *ptr, int argc, ...);
#define na_ndloop4 cumo_na_ndloop4
VALUE cumo_na_ndloop4(ndfunc_t *nf, void *ptr, VALUE args);

#define na_ndloop_cast_narray_to_rarray cumo_na_ndloop_cast_narray_to_rarray
VALUE cumo_na_ndloop_cast_narray_to_rarray(ndfunc_t *nf, VALUE nary, VALUE fmt);
#define na_ndloop_store_rarray cumo_na_ndloop_store_rarray
VALUE cumo_na_ndloop_store_rarray(ndfunc_t *nf, VALUE nary, VALUE rary);
#define na_ndloop_store_rarray2 cumo_na_ndloop_store_rarray2
VALUE cumo_na_ndloop_store_rarray2(ndfunc_t *nf, VALUE nary, VALUE rary, VALUE opt);
#define na_ndloop_inspect cumo_na_ndloop_inspect
VALUE cumo_na_ndloop_inspect(VALUE nary, na_text_func_t func, VALUE opt);
#define na_ndloop_with_index cumo_na_ndloop_with_index
VALUE cumo_na_ndloop_with_index(ndfunc_t *nf, int argc, ...);

#define na_info_str cumo_na_info_str
VALUE cumo_na_info_str(VALUE);

#define na_test_reduce cumo_na_test_reduce
bool cumo_na_test_reduce(VALUE reduce, int dim);

#define na_step_array_index cumo_na_step_array_index
void cumo_na_step_array_index(VALUE self, size_t ary_size, size_t *plen, ssize_t *pbeg, ssize_t *pstep);
#define na_step_sequence cumo_na_step_sequence
void cumo_na_step_sequence(VALUE self, size_t *plen, double *pbeg, double *pstep);

// used in aref, aset
#define na_get_result_dimension cumo_na_get_result_dimension
int cumo_na_get_result_dimension(VALUE self, int argc, VALUE *argv, ssize_t stride, size_t *pos_idx);
#define na_aref_main cumo_na_aref_main
VALUE cumo_na_aref_main(int nidx, VALUE *idx, VALUE self, int keep_dim, int result_nd, size_t pos);

// defined in array, used in math
#define na_ary_composition_dtype cumo_na_ary_composition_dtype
VALUE cumo_na_ary_composition_dtype(VALUE ary);

#include "ruby/version.h"

#if RUBY_API_VERSION_CODE == 20100 // 2.1.0
int rb_get_kwargs(VALUE keyword_hash, const ID *table, int required, int optional, VALUE *);
VALUE rb_extract_keywords(VALUE *orighash);
#endif


#endif /* ifndef CUMO_INTERN_H */
