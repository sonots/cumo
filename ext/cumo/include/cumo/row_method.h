#ifndef CUMO_ROW_METHOD_H
#define CUMO_ROW_METHOD_H

#include <ruby.h>
#include "cumo/narray.h"
#include "cumo/check.h"

/* A method that hands raw pointers to a kernel of its own, rather than going
 * through ndloop, stands on one invariant: once every pointer is taken, no Ruby
 * runs before the launch, so what was measured is what the kernel sees.
 *
 * Taking a pointer runs allocate, which is Ruby. It is free to resize an
 * operand, hand back a buffer of its own, or free one whose pointer was already
 * taken, and the last of those leaves a pointer that still looks right and a
 * size that still matches. Re-taking a pointer costs nothing while the buffer is
 * where it was left, and answers a different one when it is not, which is what
 * tells that case from the rest.
 *
 * So: take every pointer, then measure, then run these over every one of them.
 */
#define CUMO_ROW_CHECK_READ_BUFFER(a, ptr, what)                        \
    if (cumo_na_get_offset_pointer_for_read(a) != (ptr)) {              \
        rb_raise(rb_eRuntimeError,                                      \
                 "the buffer of %s was replaced while it was measured", \
                 (what));                                               \
    }

#define CUMO_ROW_CHECK_WRITE_BUFFER(a, ptr, what)                       \
    if (cumo_na_get_offset_pointer_for_write(a) != (ptr)) {             \
        rb_raise(rb_eRuntimeError,                                      \
                 "the buffer of %s was replaced while it was measured", \
                 (what));                                               \
    }

#endif /* CUMO_ROW_METHOD_H */
