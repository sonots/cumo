#include <ruby.h>
#include <cuda_runtime.h>
#include "memory_pool_impl.hpp"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"

#include <cstdlib>
#include <string>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

static cumo::internal::MemoryPool pool{};
static bool memory_pool_enabled;

VALUE cumo_cuda_eOutOfMemoryError;

char*
cumo_cuda_runtime_malloc(size_t size)
{
    if (memory_pool_enabled) {
        try {
            // TODO(sonots): Get current CUDA stream and pass it
            return reinterpret_cast<char*>(pool.Malloc(size));
        } catch (const cumo::internal::CUDARuntimeError& e) {
            cumo_cuda_runtime_check_status(e.status());
        } catch (const cumo::internal::OutOfMemoryError& e) {
            rb_raise(cumo_cuda_eOutOfMemoryError, "%s", e.what());
        }
    } else {
        void *ptr = 0;
        cumo_cuda_runtime_check_status(cudaMallocManaged(&ptr, size, cudaMemAttachGlobal));
        return reinterpret_cast<char*>(ptr);
    }
    return 0; // should not reach here
}

void
cumo_cuda_runtime_free(char *ptr)
{
    if (memory_pool_enabled) {
        try {
            // TODO(sonots): Get current CUDA stream and pass it
            pool.Free(reinterpret_cast<intptr_t>(ptr));
        } catch (const cumo::internal::CUDARuntimeError& e) {
            cumo_cuda_runtime_check_status(e.status());
        }
    } else {
        cumo_cuda_runtime_check_status(cudaFree((void*)ptr));
    }
}

/*
  Enable memory pool.

  @return [Boolean] Returns previous state (true if enabled)
 */
static VALUE
rb_memory_pool_enable(VALUE self)
{
    VALUE ret = (memory_pool_enabled ? Qtrue : Qfalse);
    memory_pool_enabled = true;
    return ret;
}

/*
  Disable memory pool.

  @return [Boolean] Returns previous state (true if enabled)
 */
static VALUE
rb_memory_pool_disable(VALUE self)
{
    VALUE ret = (memory_pool_enabled ? Qtrue : Qfalse);
    memory_pool_enabled = false;
    return ret;
}

/*
  Returns whether memory pool is enabled or not.

  @return [Boolean] Returns the state (true if enabled)
 */
static VALUE
rb_memory_pool_enabled_p(VALUE self)
{
    return (memory_pool_enabled ? Qtrue : Qfalse);
}

/*
  Free all **non-split** chunks in all arenas.
 */
static VALUE
rb_memory_pool_free_all_blocks(int argc, VALUE* argv, VALUE self)
{
    try {
        if (argc < 1) {
            pool.FreeAllBlocks();
        } else {
            // TODO(sonots): FIX if we create a Stream object
            cudaStream_t stream_ptr = (cudaStream_t)NUM2SIZET(argv[0]);
            pool.FreeAllBlocks(stream_ptr);
        }
    } catch (const cumo::internal::CUDARuntimeError& e) {
        cumo_cuda_runtime_check_status(e.status());
    }
    return Qnil;
}

/*
  Count the total number of free blocks.

  @return [Integer] The total number of free blocks.
 */
static VALUE
rb_memory_pool_n_free_blocks(VALUE self)
{
    return SIZET2NUM(pool.GetNumFreeBlocks());
}

/*
  Get the total number of bytes used.

  @return [Integer] The total number of bytes used.
 */
static VALUE
rb_memory_pool_used_bytes(VALUE self)
{
    return SIZET2NUM(pool.GetUsedBytes());
}

/*
  Get the total number of bytes acquired but not used in the pool.

  @return [Integer] The total number of bytes acquired but not used in the pool.
 */
static VALUE
rb_memory_pool_free_bytes(VALUE self)
{
    return SIZET2NUM(pool.GetFreeBytes());
}

/*
  Get the total number of bytes acquired in the pool.

  @return [Integer] The total number of bytes acquired in the pool.
 */
static VALUE
rb_memory_pool_total_bytes(VALUE self)
{
    return SIZET2NUM(pool.GetTotalBytes());
}

#define METHOD VALUE(*)(ANYARGS)

void
Init_cumo_cuda_memory_pool()
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");
    VALUE mMemoryPool = rb_define_module_under(mCUDA, "MemoryPool");
    cumo_cuda_eOutOfMemoryError = rb_define_class_under(mCUDA, "OutOfMemoryError", rb_eStandardError);
    
    rb_define_singleton_method(mMemoryPool, "enable", RUBY_METHOD_FUNC(rb_memory_pool_enable), 0);
    rb_define_singleton_method(mMemoryPool, "disable", RUBY_METHOD_FUNC(rb_memory_pool_disable), 0);
    rb_define_singleton_method(mMemoryPool, "enabled?", RUBY_METHOD_FUNC(rb_memory_pool_enabled_p), 0);
    rb_define_singleton_method(mMemoryPool, "free_all_blocks", RUBY_METHOD_FUNC(rb_memory_pool_free_all_blocks), -1);
    rb_define_singleton_method(mMemoryPool, "n_free_blocks", RUBY_METHOD_FUNC(rb_memory_pool_n_free_blocks), 0);
    rb_define_singleton_method(mMemoryPool, "used_bytes", RUBY_METHOD_FUNC(rb_memory_pool_used_bytes), 0);
    rb_define_singleton_method(mMemoryPool, "free_bytes", RUBY_METHOD_FUNC(rb_memory_pool_free_bytes), 0);
    rb_define_singleton_method(mMemoryPool, "total_bytes", RUBY_METHOD_FUNC(rb_memory_pool_total_bytes), 0);

    // default is false, yet
    const char* env = std::getenv("CUMO_MEMORY_POOL");
    memory_pool_enabled = (env != nullptr && std::string(env) != "OFF" && std::string(env) != "0" && std::string(env) != "NO");
}

#undef METHOD

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
