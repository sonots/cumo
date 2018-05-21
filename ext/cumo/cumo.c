#define CUMO_C
#include <ruby.h>
#include <assert.h>
#include <stdlib.h>
#include "cumo.h"
#include "cumo/narray.h"

void Init_cumo();
void Init_cumo_narray();
void Init_cumo_nary_data();
void Init_cumo_nary_ndloop();
void Init_cumo_nary_step();
void Init_cumo_nary_index();
void Init_cumo_bit();
void Init_cumo_int8();
void Init_cumo_int16();
void Init_cumo_int32();
void Init_cumo_int64();
void Init_cumo_uint8();
void Init_cumo_uint16();
void Init_cumo_uint32();
void Init_cumo_uint64();
void Init_cumo_sfloat();
void Init_cumo_scomplex();
void Init_cumo_dfloat();
void Init_cumo_dcomplex();
void Init_cumo_robject();
void Init_cumo_nary_math();
void Init_cumo_nary_rand();
void Init_cumo_nary_array();
void Init_cumo_nary_struct();
void Init_cumo_cuda_driver();
void Init_cumo_cuda_memory_pool();
void Init_cumo_cuda_runtime();
void Init_cumo_cuda_nvrtc();

void
cumo_debug_breakpoint(void)
{
    /* */
}

static bool cumo_compatible_mode_enabled;

bool cumo_compatible_mode_enabled_p()
{
    return cumo_compatible_mode_enabled;
}

/*
  Enable Numo NArray compatible mode.

  Cumo returns 0-dimensional NArray instead of ruby numeric object
  for some methods such as `extract`, and `[]` not to synchronize
  between CPU and GPU for performance as default.

  Enabling the compatible mode makes Cumo behave as Numo. But, please
  note that it makes Cumo slow.

  @return [Boolean] Returns previous state (true if enabled)
 */
static VALUE
rb_enable_compatible_mode(VALUE self)
{
    VALUE ret = (cumo_compatible_mode_enabled ? Qtrue : Qfalse);
    cumo_compatible_mode_enabled = true;
    return ret;
}

/*
  Disable Numo NArray compatible mode.

  @return [Boolean] Returns previous state (true if enabled)
 */
static VALUE
rb_disable_compatible_mode(VALUE self)
{
    VALUE ret = (cumo_compatible_mode_enabled ? Qtrue : Qfalse);
    cumo_compatible_mode_enabled = false;
    return ret;
}

/*
  Returns whether Numo NArray compatible mode is enabled or not.

  @return [Boolean] Returns the state (true if enabled)
 */
static VALUE
rb_compatible_mode_enabled_p(VALUE self)
{
    return (cumo_compatible_mode_enabled ? Qtrue : Qfalse);
}

/* initialization of Cumo Module */
void
Init_cumo()
{
    VALUE mCumo = rb_define_module("Cumo");

    rb_define_const(mCumo, "VERSION", rb_str_new2(CUMO_VERSION));

    rb_define_singleton_method(mCumo, "enable_compatible_mode", RUBY_METHOD_FUNC(rb_enable_compatible_mode), 0);
    rb_define_singleton_method(mCumo, "disable_compatible_mode", RUBY_METHOD_FUNC(rb_disable_compatible_mode), 0);
    rb_define_singleton_method(mCumo, "compatible_mode_enabled?", RUBY_METHOD_FUNC(rb_compatible_mode_enabled_p), 0);

    // default is false
    char* env = getenv("CUMO_COMPATIBLE_MODE");
    cumo_compatible_mode_enabled = (env != NULL && strcmp(env, "OFF") != 0 && strcmp(env, "0") != 0 && strcmp(env, "NO") != 0);

    Init_cumo_narray();

    Init_cumo_nary_step();
    Init_cumo_nary_index();

    Init_cumo_nary_data();
    Init_cumo_nary_ndloop();

    Init_cumo_dcomplex();
    Init_cumo_dfloat();
    Init_cumo_scomplex();
    Init_cumo_sfloat();

    Init_cumo_int64();
    Init_cumo_uint64();
    Init_cumo_int32();
    Init_cumo_uint32();
    Init_cumo_int16();
    Init_cumo_uint16();
    Init_cumo_int8();
    Init_cumo_uint8();

    Init_cumo_bit();
    Init_cumo_robject();

    Init_cumo_nary_math();

    Init_cumo_nary_rand();
    Init_cumo_nary_array();
    Init_cumo_nary_struct();

    Init_cumo_cuda_driver();
    Init_cumo_cuda_memory_pool();
    Init_cumo_cuda_runtime();
    Init_cumo_cuda_nvrtc();
}
