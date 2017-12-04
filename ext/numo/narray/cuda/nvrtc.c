#include <ruby.h>
#include <assert.h>
#include "numo/cuda/nvrtc.h"

VALUE cumo_cuda_eNVRTCError;
VALUE cumo_cuda_mNVRTC;
#define eNVRTCError cumo_cuda_eNVRTCError
#define mNVRTC cumo_cuda_mNVRTC

static void
check_status(nvrtcResult status)
{
    if (status != 0) {
        rb_raise(cumo_cuda_eNVRTCError, "NVRTC error: %d %s", status, nvrtcGetErrorString(status));
    }
}

static VALUE
create_program(
        VALUE self,
        VALUE src,
        VALUE name,
        VALUE headers,
        VALUE include_names)
{
    nvrtcResult status;
    nvrtcProgram _prog;
    const char* _src = StringValueCStr(src);
    const char* _name = StringValueCStr(name);
    int num_headers = RARRAY_LEN(headers);
    const char** ary_headers = (const char **)malloc(num_headers);
    const char** ary_include_names = (const char **)malloc(num_headers);
    int i;
    for (i = 0; i < num_headers; i++) {
        VALUE header = RARRAY_PTR(headers)[i];
        ary_headers[i] = StringValueCStr(header);
    }
    for (i = 0; i < num_headers; i++) {
        VALUE include_name = RARRAY_PTR(include_names)[i];
        ary_include_names[i] = StringValueCStr(include_name);
    }

    status = nvrtcCreateProgram(&_prog, _src, _name, num_headers, ary_headers, ary_include_names);

    free(ary_headers);
    free(ary_include_names);
    check_status(status);
    return SIZET2NUM((size_t)_prog);
}

static VALUE
compile_program(VALUE self, VALUE prog, VALUE options)
{
    nvrtcResult status;
    nvrtcProgram _prog = (nvrtcProgram)NUM2SIZET(prog);
    int num_options = RARRAY_LEN(options);
    const char** ary_options = (const char **)malloc(num_options);
    int i;
    for (i = 0; i < num_options; i++) {
        VALUE option = RARRAY_PTR(options)[i];
        ary_options[i] = StringValueCStr(option);
    }

    status = nvrtcCompileProgram(_prog, num_options, ary_options);

    free(ary_options);
    check_status(status);
    return Qnil;
}

void
Init_numo_cumo_narray_cuda_nvrtc()
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");
    mNVRTC = rb_define_module_under(mCUDA, "NVRTC");
    eNVRTCError = rb_define_class_under(mCUDA, "NVRTCError", rb_eStandardError);

    rb_define_singleton_method(mNVRTC, "create_program", create_program, 4);
    //rb_define_singleton_method(mNVRTC, "destroy_program", destroy_program, -1);
    rb_define_singleton_method(mNVRTC, "compile_program", compile_program, 2);
    //rb_define_singleton_method(mNVRTC, "get_ptx", get_ptx, -1);
    //rb_define_singleton_method(mNVRTC, "get_program_log", get_program_log, -1);
}
