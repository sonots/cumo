#ifndef CUMO_ATOMIC_H
#define CUMO_ATOMIC_H

#include <stddef.h>

// The compiler builtins rather than ruby/atomic.h: the size_t fetch-add
// there arrived after the oldest Ruby cumo builds with, and this header is
// read without ruby.h as well.
static inline size_t
cumo_atomic_size_fetch_add(size_t *var, size_t n)
{
    return __atomic_fetch_add(var, n, __ATOMIC_SEQ_CST);
}

static inline size_t
cumo_atomic_size_load(const size_t *var)
{
    return __atomic_load_n(var, __ATOMIC_SEQ_CST);
}

static inline void
cumo_atomic_size_store(size_t *var, size_t value)
{
    __atomic_store_n(var, value, __ATOMIC_SEQ_CST);
}

#endif /* ifndef CUMO_ATOMIC_H */
