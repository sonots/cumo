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

#endif /* CUMO_ROW_METHOD_H */
