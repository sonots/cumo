#define CUMO_C
#include <ruby.h>
#include <assert.h>
#include <stdlib.h>
#include "cumo.h"
#include "cumo/narray.h"

// Ruby's compare rather than strcasecmp, since it does not follow the locale.
int
cumo_env_truth(const char *name, int dflt)
{
    static const char* const yes[] = {"1", "on", "yes", "true"};
    static const char* const no[]  = {"0", "off", "no", "false"};
    const char *env = getenv(name);
    size_t i;

    if (env == NULL || *env == '\0') return dflt;
    for (i = 0; i < sizeof(yes) / sizeof(yes[0]); ++i) {
        if (st_locale_insensitive_strcasecmp(env, yes[i]) == 0) return 1;
        if (st_locale_insensitive_strcasecmp(env, no[i]) == 0)  return 0;
    }
    rb_warn("%s=%s is not a yes or a no, leaving it %s", name, env, dflt ? "on" : "off");
    return dflt;
}

void Init_cumo();
void Init_cumo_narray();
void Init_cumo_na_data();
void Init_cumo_na_ndloop();
void Init_cumo_na_step();
void Init_cumo_na_index();
void Init_cumo_bit();
void Init_cumo_int8();
void Init_cumo_int16();
void Init_cumo_int32();
void Init_cumo_int64();
void Init_cumo_uint8();
void Init_cumo_uint16();
void Init_cumo_uint32();
void Init_cumo_uint64();
void Init_cumo_hfloat();
void Init_cumo_bfloat();
void Init_cumo_sfloat();
void Init_cumo_scomplex();
void Init_cumo_dfloat();
void Init_cumo_dcomplex();
void Init_cumo_robject();
void Init_cumo_dcomplex_upcast();
void Init_cumo_dfloat_upcast();
void Init_cumo_scomplex_upcast();
void Init_cumo_sfloat_upcast();
void Init_cumo_hfloat_upcast();
void Init_cumo_bfloat_upcast();
void Init_cumo_int64_upcast();
void Init_cumo_uint64_upcast();
void Init_cumo_int32_upcast();
void Init_cumo_uint32_upcast();
void Init_cumo_int16_upcast();
void Init_cumo_uint16_upcast();
void Init_cumo_int8_upcast();
void Init_cumo_uint8_upcast();
void Init_cumo_bit_upcast();
void Init_cumo_robject_upcast();
void Init_cumo_na_math();
void Init_cumo_na_rand();
void Init_cumo_na_array();
void Init_cumo_na_struct();
void Init_cumo_cuda_driver();
void Init_cumo_cuda_memory_pool();
void Init_cumo_cuda_runtime();
void Init_cumo_cuda_nvrtc();
void Init_cumo_cuda_cublas();
void Init_cumo_cuda_cudnn();

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

static bool cumo_show_warning_enabled;

bool cumo_show_warning_enabled_p()
{
    return cumo_show_warning_enabled;
}

static bool cumo_show_warning_once_enabled;

bool cumo_show_warning_once_enabled_p()
{
    return cumo_show_warning_once_enabled;
}

static bool cumo_allow_tf32;

bool cumo_allow_tf32_p()
{
    return cumo_allow_tf32;
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

/*
  Returns whether single precision may run on tensor cores, set by
  CUMO_ALLOW_TF32. Off unless asked for: tensor cores round the operands to
  a 10 bit significand.

  @return [Boolean]
 */
static VALUE
rb_allow_tf32_p(VALUE self)
{
    return (cumo_allow_tf32 ? Qtrue : Qfalse);
}

/* initialization of Cumo Module */
void
Init_cumo()
{
    VALUE mCumo;

#ifdef HAVE_RB_EXT_RACTOR_SAFE
    rb_ext_ractor_safe(true);
#endif

    mCumo = rb_define_module("Cumo");

    rb_define_const(mCumo, "VERSION", rb_str_new2(CUMO_VERSION));

    rb_define_singleton_method(mCumo, "enable_compatible_mode", rb_enable_compatible_mode, 0);
    rb_define_singleton_method(mCumo, "disable_compatible_mode", rb_disable_compatible_mode, 0);
    rb_define_singleton_method(mCumo, "compatible_mode_enabled?", rb_compatible_mode_enabled_p, 0);
    rb_define_singleton_method(mCumo, "allow_tf32?", rb_allow_tf32_p, 0);

    cumo_compatible_mode_enabled = cumo_env_truth("CUMO_COMPATIBLE_MODE", 0);
    cumo_show_warning_enabled = cumo_env_truth("CUMO_SHOW_WARNING", 0);
    cumo_show_warning_once_enabled = cumo_env_truth("CUMO_SHOW_WARNING_ONCE", 1);
    cumo_allow_tf32 = cumo_env_truth("CUMO_ALLOW_TF32", 0);

    Init_cumo_narray();

    Init_cumo_na_step();
    Init_cumo_na_index();

    Init_cumo_na_data();
    Init_cumo_na_ndloop();

    Init_cumo_dcomplex();
    Init_cumo_dfloat();
    Init_cumo_scomplex();
    Init_cumo_sfloat();
    Init_cumo_hfloat();
    Init_cumo_bfloat();

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

    // Every class exists now, so the UPCAST tables can name all of them.
    Init_cumo_dcomplex_upcast();
    Init_cumo_dfloat_upcast();
    Init_cumo_scomplex_upcast();
    Init_cumo_sfloat_upcast();
    Init_cumo_hfloat_upcast();
    Init_cumo_bfloat_upcast();
    Init_cumo_int64_upcast();
    Init_cumo_uint64_upcast();
    Init_cumo_int32_upcast();
    Init_cumo_uint32_upcast();
    Init_cumo_int16_upcast();
    Init_cumo_uint16_upcast();
    Init_cumo_int8_upcast();
    Init_cumo_uint8_upcast();
    Init_cumo_bit_upcast();
    Init_cumo_robject_upcast();

    Init_cumo_na_math();

    Init_cumo_na_rand();
    Init_cumo_na_array();
    Init_cumo_na_struct();

    Init_cumo_cuda_driver();
    Init_cumo_cuda_memory_pool();
    Init_cumo_cuda_runtime();
    Init_cumo_cuda_nvrtc();

    Init_cumo_cuda_cublas();
    Init_cumo_cuda_cudnn();
}
