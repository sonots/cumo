#include <ruby.h>
#include <ruby/thread.h>
#include <assert.h>
#include <nvrtc.h>
#include "numo/cuda/nvrtc.h"

VALUE numo_cuda_eNVRTCError;
VALUE numo_cuda_mNVRTC;
#define eNVRTCError numo_cuda_eNVRTCError
#define mNVRTC numo_cuda_mNVRTC

static void
check_status(nvrtcResult status)
{
    if (status != 0) {
        rb_raise(numo_cuda_eNVRTCError, "%s (error=%d)", nvrtcGetErrorString(status), status);
    }
}

static VALUE
version(VALUE self)
{
    int _major, _minor;
    int status = nvrtcVersion(&_major, &_minor);
    VALUE major = INT2NUM(_major);
    VALUE minor = INT2NUM(_minor);
    return rb_ary_new3(2, major, minor);
}

struct create_program_args {
    nvrtcProgram *prog;
    const char* src;
    const char *name;
    int num_headers;
    const char** headers;
    const char** include_names;
};

static void*
create_program_without_gvl(void *ptr)
{
    struct create_program_args *a = ptr;
    nvrtcResult status;
    status = nvrtcCreateProgram(a->prog, a->src, a->name, a->num_headers, a->headers, a->include_names);
    return (void *)status;
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
    const char** ary_headers = (const char **)malloc(num_headers * sizeof(char *));
    const char** ary_include_names = (const char **)malloc(num_headers * sizeof(char *));
    int i;
    for (i = 0; i < num_headers; i++) {
        VALUE header = RARRAY_PTR(headers)[i];
        ary_headers[i] = StringValueCStr(header);
    }
    for (i = 0; i < num_headers; i++) {
        VALUE include_name = RARRAY_PTR(include_names)[i];
        ary_include_names[i] = StringValueCStr(include_name);
    }

    struct create_program_args args = {&_prog, _src, _name, num_headers, ary_headers, ary_include_names};
    status = (nvrtcResult)rb_thread_call_without_gvl(create_program_without_gvl, &args, NULL, NULL);

    free(ary_headers);
    free(ary_include_names);
    check_status(status);
    return SIZET2NUM((size_t)_prog);
}

struct destroy_program_args {
    nvrtcProgram *prog;
};

static void*
destroy_program_without_gvl(void *ptr)
{
    struct destroy_program_args *a = ptr;
    nvrtcResult status;
    status = nvrtcDestroyProgram(a->prog);
    return (void *)status;
}

static VALUE
destroy_program(VALUE self, VALUE prog)
{
    nvrtcResult status;
    nvrtcProgram _prog = (nvrtcProgram)NUM2SIZET(prog);

    struct destroy_program_args args = {&_prog};
    status = (nvrtcResult)rb_thread_call_without_gvl(destroy_program_without_gvl, &args, NULL, NULL);

    check_status(status);
    return Qnil;
}

struct compile_program_args {
    nvrtcProgram prog;
    int num_options;
    const char** options;
};

static void*
compile_program_without_gvl(void *ptr)
{
    struct compile_program_args *a = ptr;
    nvrtcResult status;
    status = nvrtcCompileProgram(a->prog, a->num_options, a->options);
    return (void *)status;
}

static VALUE
compile_program(VALUE self, VALUE prog, VALUE options)
{
    nvrtcResult status;
    nvrtcProgram _prog = (nvrtcProgram)NUM2SIZET(prog);
    int num_options = RARRAY_LEN(options);
    const char** ary_options = (const char **)malloc(num_options * sizeof(char *));
    int i;
    for (i = 0; i < num_options; i++) {
        VALUE option = RARRAY_PTR(options)[i];
        ary_options[i] = StringValueCStr(option);
    }

    struct compile_program_args args = {_prog, num_options, ary_options};
    status = (nvrtcResult)rb_thread_call_without_gvl(compile_program_without_gvl, &args, NULL, NULL);

    free(ary_options);
    check_status(status);
    return Qnil;
}

static VALUE
get_ptx(VALUE self, VALUE prog)
{
    nvrtcResult status;
    nvrtcProgram _prog = (nvrtcProgram)NUM2SIZET(prog);
    size_t _ptxSizeRet;
    char *_ptx;
    VALUE ptx;

    status = nvrtcGetPTXSize(_prog, &_ptxSizeRet);
    check_status(status);

    ptx = rb_str_new(NULL, _ptxSizeRet);
    _ptx = RSTRING_PTR(ptx);
    status = nvrtcGetPTX(_prog, _ptx);
    check_status(status);

    return ptx;
}

static VALUE
get_program_log(VALUE self, VALUE prog)
{
    nvrtcResult status;
    nvrtcProgram _prog = (nvrtcProgram)NUM2SIZET(prog);
    size_t _logSizeRet;
    char *_log;
    VALUE log;

    status = nvrtcGetProgramLogSize(_prog, &_logSizeRet);
    check_status(status);

    log = rb_str_new(NULL, _logSizeRet);
    _log = RSTRING_PTR(log);
    status = nvrtcGetProgramLog(_prog, _log);
    check_status(status);

    return log;
}

void
Init_numo_cuda_nvrtc()
{
    VALUE mNumo = rb_define_module("Numo");
    VALUE mCUDA = rb_define_module_under(mNumo, "CUDA");
    mNVRTC = rb_define_module_under(mCUDA, "NVRTC");
    eNVRTCError = rb_define_class_under(mCUDA, "NVRTCError", rb_eStandardError);

    rb_define_singleton_method(mNVRTC, "version", version, 0);
    rb_define_singleton_method(mNVRTC, "create_program", create_program, 4);
    rb_define_singleton_method(mNVRTC, "destroy_program", destroy_program, 1);
    rb_define_singleton_method(mNVRTC, "compile_program", compile_program, 2);
    rb_define_singleton_method(mNVRTC, "get_ptx", get_ptx, 1);
    rb_define_singleton_method(mNVRTC, "get_program_log", get_program_log, 1);
}
