#ifndef CUMO_ROW_METHOD_H
#define CUMO_ROW_METHOD_H

#include <ruby.h>
#include "cumo/narray.h"
#include "cumo/check.h"

/* Take every pointer, then measure, then run these. Taking one runs allocate,
 * which is Ruby and free to resize an operand, replace its buffer, or free one
 * whose pointer was already taken, and the last of those leaves a size that
 * still matches. Re-taking a pointer answers a different one only in that case.
 */
#define CUMO_ROW_CHECK_READ_BUFFER(a, ptr, what)                        \
    do {                                                                \
        if (cumo_na_get_offset_pointer_for_read(a) != (ptr)) {          \
            rb_raise(rb_eRuntimeError,                                  \
                     "the buffer of %s was replaced while it was measured", \
                     (what));                                           \
        }                                                               \
    } while (0)

#define CUMO_ROW_CHECK_WRITE_BUFFER(a, ptr, what)                       \
    do {                                                                \
        if (cumo_na_get_offset_pointer_for_write(a) != (ptr)) {         \
            rb_raise(rb_eRuntimeError,                                  \
                     "the buffer of %s was replaced while it was measured", \
                     (what));                                           \
        }                                                               \
    } while (0)

#define CUMO_ROW_CHECK_SAME_SHAPE(nout, nin)                            \
    do {                                                                \
        cumo_narray_t *cumo_row_out = (nout);                           \
        cumo_narray_t *cumo_row_in = (nin);                             \
        int cumo_row_i;                                                 \
        if (cumo_row_out->ndim != cumo_row_in->ndim) {                  \
            rb_raise(cumo_na_eShapeError,                               \
                     "self or the result was reshaped while it was measured"); \
        }                                                               \
        for (cumo_row_i = 0; cumo_row_i < cumo_row_in->ndim; ++cumo_row_i) { \
            if (cumo_row_out->shape[cumo_row_i] != cumo_row_in->shape[cumo_row_i]) { \
                rb_raise(cumo_na_eShapeError,                           \
                         "self or the result was reshaped while it was measured"); \
            }                                                           \
        }                                                               \
    } while (0)

/* What a row method holds while it runs. The caller declares one of these, so
 * every VALUE below sits in the caller's frame and the conservative GC keeps
 * looking at it until the launch is queued. Taking its address is what forces
 * it into memory rather than a register.
 *
 * cont is the array cumo_na_as_contiguous_array answered, which for a
 * non-contiguous operand is a dup nothing else refers to. Nobody reads it after
 * the launch; holding it is the whole of its job, and dropping the field would
 * let that dup be collected while the kernel is still reading its buffer.
 */
typedef struct {
    VALUE value;       /* in: the operand as given */
    const char *name;  /* in: what to call it in a message */
    VALUE cont;        /* out: held, not read */
    char *ptr;         /* out */
} cumo_row_operand_t;

typedef struct {
    VALUE self_cont;   /* out: held, not read, for the same reason */
    char *x_ptr;       /* out */
    char *y_ptr;       /* out */
    size_t rows;       /* out */
    size_t cols;       /* out */
    cumo_row_operand_t ops[2];
} cumo_row_args_t;

/* The preamble layer_norm, rms_norm and softmax share, in the order the last
 * two fixes to it settled on: take every pointer, because taking one runs
 * allocate and allocate is Ruby, then measure what is left, then check that
 * none of it moved. It is one function rather than three copies because that
 * order is the whole of what makes it correct.
 *
 * Answers the output array, shaped like self, and fills in a. Every operand is
 * checked to be one-dimensional and as long as the last axis. An empty self
 * answers an empty array and leaves a->rows zero, which every launcher here
 * already takes as nothing to do.
 */
static inline VALUE
cumo_row_prepare(VALUE self, VALUE klass, const char *what,
                 cumo_row_args_t *a, int n_ops)
{
    cumo_narray_t *nx, *ny, *nop;
    VALUE y;
    int i;

    CUMO_CHECK_NARRAY_TYPE(self, klass, "self");
    for (i = 0; i < n_ops; ++i) {
        CUMO_CHECK_NARRAY_TYPE(a->ops[i].value, klass, a->ops[i].name);
    }

    a->self_cont = cumo_na_as_contiguous_array(self);
    for (i = 0; i < n_ops; ++i) {
        a->ops[i].cont = cumo_na_as_contiguous_array(a->ops[i].value);
    }

    CumoGetNArray(a->self_cont, nx);
    if (nx->ndim < 1) {
        rb_raise(cumo_na_eShapeError, "%s needs an axis to run along", what);
    }
    y = cumo_na_new(klass, nx->ndim, nx->shape);

    a->x_ptr = cumo_na_get_offset_pointer_for_read(a->self_cont);
    for (i = 0; i < n_ops; ++i) {
        a->ops[i].ptr = cumo_na_get_offset_pointer_for_read(a->ops[i].cont);
    }
    a->y_ptr = cumo_na_get_offset_pointer_for_write(y);

    /* cumo_na_as_contiguous_array answers what dup gave it, which a class is
     * free to define, so what these carry is not what was checked above. */
    CUMO_CHECK_NARRAY_TYPE(a->self_cont, klass, "self");
    for (i = 0; i < n_ops; ++i) {
        CUMO_CHECK_NARRAY_TYPE(a->ops[i].cont, klass, a->ops[i].name);
    }
    CumoGetNArray(a->self_cont, nx);
    CumoGetNArray(y, ny);

    if (nx->ndim < 1) {
        rb_raise(cumo_na_eShapeError, "%s needs an axis to run along", what);
    }
    /* Before the operands are measured, so that a reshaped self is reported as
     * one rather than as a gamma of the wrong length. */
    CUMO_ROW_CHECK_SAME_SHAPE(ny, nx);

    a->cols = nx->shape[nx->ndim - 1];
    for (i = 0; i < n_ops; ++i) {
        CumoGetNArray(a->ops[i].cont, nop);
        if (nop->ndim != 1 || nop->shape[0] != a->cols) {
            rb_raise(cumo_na_eShapeError, "%s must be 1-dimensional and %"SZF"u long",
                     a->ops[i].name, a->cols);
        }
    }
    CUMO_ROW_CHECK_READ_BUFFER(a->self_cont, a->x_ptr, "self");
    for (i = 0; i < n_ops; ++i) {
        CUMO_ROW_CHECK_READ_BUFFER(a->ops[i].cont, a->ops[i].ptr, a->ops[i].name);
    }
    CUMO_ROW_CHECK_WRITE_BUFFER(y, a->y_ptr, "the result");

    a->rows = nx->size == 0 ? 0 : nx->size / a->cols;
    return y;
}

#endif /* CUMO_ROW_METHOD_H */
