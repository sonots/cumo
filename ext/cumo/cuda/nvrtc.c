#include <ruby.h>
#include <ruby/thread.h>
#include <assert.h>
#include <nvrtc.h>
#include "cumo/cuda/nvrtc.h"

VALUE cumo_cuda_eNVRTCError;
VALUE cumo_cuda_mNVRTC;
#define eNVRTCError cumo_cuda_eNVRTCError
#define mNVRTC cumo_cuda_mNVRTC

static void
check_status(nvrtcResult status)
{
    if (status != 0) {
        rb_raise(cumo_cuda_eNVRTCError, "%s (error=%d)", nvrtcGetErrorString(status), status);
    }
}

static VALUE
rb_nvrtcVersion(VALUE self)
{
    int _major, _minor;
    nvrtcResult status;

    status = nvrtcVersion(&_major, &_minor);

    check_status(status);
    VALUE major = INT2NUM(_major);
    VALUE minor = INT2NUM(_minor);
    return rb_ary_new3(2, major, minor);
}

struct nvrtcCreateProgramParam {
    nvrtcProgram *prog;
    const char* src;
    const char *name;
    int numHeaders;
    const char** headers;
    const char** includeNames;
};

static void*
nvrtcCreateProgram_without_gvl_cb(void *param)
{
    struct nvrtcCreateProgramParam *p = param;
    nvrtcResult status;
    status = nvrtcCreateProgram(p->prog, p->src, p->name, p->numHeaders, p->headers, p->includeNames);
    return (void *)status;
}

static VALUE
rb_nvrtcCreateProgram(
        VALUE self,
        VALUE src,
        VALUE name,
        VALUE headers,
        VALUE includeNames)
{
    nvrtcResult status;
    nvrtcProgram _prog;
    const char* _src = StringValueCStr(src);
    const char* _name = StringValueCStr(name);
    int _numHeaders = RARRAY_LEN(headers);
    const char** _headers = (const char **)malloc(_numHeaders * sizeof(char *));
    const char** _includeNames = (const char **)malloc(_numHeaders * sizeof(char *));
    int i;
    for (i = 0; i < _numHeaders; i++) {
        VALUE header = RARRAY_PTR(headers)[i];
        _headers[i] = StringValueCStr(header);
    }
    for (i = 0; i < _numHeaders; i++) {
        VALUE include_name = RARRAY_PTR(includeNames)[i];
        _includeNames[i] = StringValueCStr(include_name);
    }

    struct nvrtcCreateProgramParam param = {&_prog, _src, _name, _numHeaders, _headers, _includeNames};
    status = (nvrtcResult)rb_thread_call_without_gvl(nvrtcCreateProgram_without_gvl_cb, &param, NULL, NULL);

    free(_headers);
    free(_includeNames);
    check_status(status);
    return SIZET2NUM((size_t)_prog);
}

struct nvrtcDestroyProgramParam {
    nvrtcProgram *prog;
};

static void*
nvrtcDestroyProgram_without_gvl_cb(void *param)
{
    struct nvrtcDestroyProgramParam *p = param;
    nvrtcResult status;
    status = nvrtcDestroyProgram(p->prog);
    return (void *)status;
}

static VALUE
rb_nvrtcDestroyProgram(VALUE self, VALUE prog)
{
    nvrtcResult status;
    nvrtcProgram _prog = (nvrtcProgram)NUM2SIZET(prog);

    struct nvrtcDestroyProgramParam param = {&_prog};
    status = (nvrtcResult)rb_thread_call_without_gvl(nvrtcDestroyProgram_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return Qnil;
}

struct nvrtcCompileProgramParam {
    nvrtcProgram prog;
    int numOptions;
    const char** options;
};

static void*
nvrtcCompileProgram_without_gvl_cb(void *param)
{
    struct nvrtcCompileProgramParam *p = param;
    nvrtcResult status;
    status = nvrtcCompileProgram(p->prog, p->numOptions, p->options);
    return (void *)status;
}

static VALUE
rb_nvrtcCompileProgram(VALUE self, VALUE prog, VALUE options)
{
    nvrtcResult status;
    nvrtcProgram _prog = (nvrtcProgram)NUM2SIZET(prog);
    int _numOptions = RARRAY_LEN(options);
    const char** _options = (const char **)malloc(_numOptions * sizeof(char *));
    int i;
    for (i = 0; i < _numOptions; i++) {
        VALUE option = RARRAY_PTR(options)[i];
        _options[i] = StringValueCStr(option);
    }

    struct nvrtcCompileProgramParam param = {_prog, _numOptions, _options};
    status = (nvrtcResult)rb_thread_call_without_gvl(nvrtcCompileProgram_without_gvl_cb, &param, NULL, NULL);

    free(_options);
    check_status(status);
    return Qnil;
}

static VALUE
rb_nvrtcGetPTX(VALUE self, VALUE prog)
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
rb_nvrtcGetProgramLog(VALUE self, VALUE prog)
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
Init_cumo_cuda_nvrtc()
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");
    mNVRTC = rb_define_module_under(mCUDA, "NVRTC");
    eNVRTCError = rb_define_class_under(mCUDA, "NVRTCError", rb_eStandardError);

    rb_define_singleton_method(mNVRTC, "nvrtcVersion", rb_nvrtcVersion, 0);
    rb_define_singleton_method(mNVRTC, "nvrtcCreateProgram", rb_nvrtcCreateProgram, 4);
    rb_define_singleton_method(mNVRTC, "nvrtcDestroyProgram", rb_nvrtcDestroyProgram, 1);
    rb_define_singleton_method(mNVRTC, "nvrtcCompileProgram", rb_nvrtcCompileProgram, 2);
    rb_define_singleton_method(mNVRTC, "nvrtcGetPTX", rb_nvrtcGetPTX, 1);
    rb_define_singleton_method(mNVRTC, "nvrtcGetProgramLog", rb_nvrtcGetProgramLog, 1);
}
