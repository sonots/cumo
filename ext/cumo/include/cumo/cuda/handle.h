#ifndef CUMO_CUDA_HANDLE_H
#define CUMO_CUDA_HANDLE_H
#include <ruby.h>
#include <ruby/st.h>
#include <ruby/thread_native.h>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

// The driver and NVRTC hand their handles to Ruby as plain Integers, so a
// made-up or already destroyed one arrives back here as a pointer the CUDA API
// dereferences without checking. Keeping the live ones is what lets those be
// told apart from a valid handle.
//
// The lock is here because a Ractor gets its own GVL, so two of them can reach
// the table at once.
typedef struct {
    st_table *table;
    rb_nativethread_lock_t lock;
} cumo_cuda_handle_set_t;

static inline void
cumo_cuda_handle_set_init(cumo_cuda_handle_set_t *set)
{
    set->table = st_init_numtable();
    rb_nativethread_lock_initialize(&set->lock);
}

static inline void
cumo_cuda_handle_set_add(cumo_cuda_handle_set_t *set, size_t handle)
{
    rb_nativethread_lock_lock(&set->lock);
    st_insert(set->table, (st_data_t)handle, (st_data_t)1);
    rb_nativethread_lock_unlock(&set->lock);
}

// Raises unless the Integer names a handle this process created and has not
// destroyed. NUM2SIZET rejects non-Integers first.
static inline size_t
cumo_cuda_handle_get(cumo_cuda_handle_set_t *set, VALUE v, const char *what)
{
    size_t handle = NUM2SIZET(v);
    int found;

    rb_nativethread_lock_lock(&set->lock);
    found = st_lookup(set->table, (st_data_t)handle, 0);
    rb_nativethread_lock_unlock(&set->lock);

    if (!found) {
        rb_raise(rb_eArgError, "not a live %s", what);
    }
    return handle;
}

// Same, and forgets the handle, so a second destroy raises instead of handing
// the driver a pointer it has already released.
static inline size_t
cumo_cuda_handle_take(cumo_cuda_handle_set_t *set, VALUE v, const char *what)
{
    size_t handle = NUM2SIZET(v);
    st_data_t key = (st_data_t)handle;
    int found;

    rb_nativethread_lock_lock(&set->lock);
    found = st_delete(set->table, &key, 0);
    rb_nativethread_lock_unlock(&set->lock);

    if (!found) {
        rb_raise(rb_eArgError, "not a live %s", what);
    }
    return handle;
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_CUDA_HANDLE_H */
