#ifndef CUMO_CHECK_H
#define CUMO_CHECK_H

#include <ruby.h>
#include "cumo/narray.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

// Two of these often sit side by side, so the message has to say which array
// was turned down as well as what it is.
static inline void
cumo_na_check_narray_type(VALUE x, VALUE type, const char *name)
{
    if (rb_obj_class(x) != type) {
        rb_raise(rb_eTypeError, "%s must be %" PRIsVALUE ", not %" PRIsVALUE,
                 name, type, rb_obj_class(x));
    }
}

#define CUMO_CHECK_NARRAY_TYPE(x,t,name) cumo_na_check_narray_type((x),(t),(name))

#define CUMO_CHECK_SIZE_EQ(sz1,sz2)                  \
    if ((sz1) != (sz2)) {                            \
        rb_raise(cumo_na_eShapeError,                \
                 "size mismatch: %d != %d",          \
                 (int)(sz1), (int)(sz2));            \
    }

#define CUMO_CHECK_DIM_EQ(nd1,nd2)                   \
    if ((nd1) != (nd2)) {                            \
        rb_raise(cumo_na_eShapeError,                \
                 "dimension mismatch: %d != %d",     \
                 (int)(nd1), (int)(nd2));            \
    }

static inline VALUE
cumo_option_value(VALUE value, VALUE default_value)
{
    switch(TYPE(value)) {
    case T_NIL:
    case T_UNDEF:
        return default_value;
    }
    return value;
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* CUMO_CHECK_H */
