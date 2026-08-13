#include <ruby.h>
#include <ruby/thread.h>
#include <assert.h>
#include <limits.h>
#include <nvrtc.h>
#include "cumo/cuda/nvrtc.h"
#include "cumo/cuda/handle.h"

VALUE cumo_cuda_eNVRTCError;
VALUE cumo_cuda_mNVRTC;
#define eNVRTCError cumo_cuda_eNVRTCError
#define mNVRTC cumo_cuda_mNVRTC

static cumo_cuda_handle_set_t programs;

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
    VALUE major, minor;

    status = nvrtcVersion(&_major, &_minor);

    check_status(status);
    major = INT2NUM(_major);
    minor = INT2NUM(_minor);
    return rb_ary_new3(2, major, minor);
}

// NVRTC counts its string vectors with an int, so reject anything longer
// before the length is truncated into one.
static int
ary_len(VALUE ary, const char *name)
{
    long len;

    Check_Type(ary, T_ARRAY);
    len = RARRAY_LEN(ary);
    if (len > INT_MAX) {
        rb_raise(rb_eArgError, "%s is too long (%ld elements)", name, len);
    }
    return (int)len;
}

// Convert every element of ary to a String and stash the results in strings.
//
// Everything that can raise -- a non-String element, or a String holding an
// embedded NUL -- happens here, before the caller allocates its vector, so a
// bad argument cannot leak it. Holding the results in strings also keeps a
// String that came out of to_str alive while the caller uses its char*.
static void
push_strings(VALUE strings, VALUE ary, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        VALUE str = rb_str_to_str(RARRAY_AREF(ary, i));
        StringValueCStr(str);
        rb_ary_push(strings, str);
    }
}

// Fill buf from strings[from, n]. Cannot raise: push_strings converted and
// NUL-terminated every element already.
static void
fill_cstrs(const char **buf, VALUE strings, int from, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        buf[i] = RSTRING_PTR(RARRAY_AREF(strings, (long)from + i));
    }
}

static const char **
alloc_cstrs(long n)
{
    const char **buf = (const char **)malloc((size_t)n * sizeof(char *));
    if (buf == NULL) {
        rb_raise(rb_eNoMemError, "failed to allocate %ld string pointers", n);
    }
    return buf;
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
    int _numHeaders = ary_len(headers, "headers");
    const char** _headers = NULL;
    const char** _includeNames = NULL;
    VALUE strings;

    // nvrtcCreateProgram reads numHeaders entries out of includeNames as well,
    // so a shorter array would be read past its end.
    if (ary_len(includeNames, "include_names") != _numHeaders) {
        rb_raise(rb_eArgError,
                 "headers and include_names must have the same length (%d vs %ld)",
                 _numHeaders, RARRAY_LEN(includeNames));
    }

    strings = rb_ary_new_capa((long)_numHeaders * 2);
    push_strings(strings, headers, _numHeaders);
    push_strings(strings, includeNames, _numHeaders);

    if (_numHeaders > 0) {
        // Both vectors come out of one allocation, so there is one pointer to
        // free rather than two.
        _headers = alloc_cstrs((long)_numHeaders * 2);
        _includeNames = _headers + _numHeaders;
        fill_cstrs(_headers, strings, 0, _numHeaders);
        fill_cstrs(_includeNames, strings, _numHeaders, _numHeaders);
    }

    {
        struct nvrtcCreateProgramParam param = {&_prog, _src, _name, _numHeaders, _headers, _includeNames};
        status = (nvrtcResult)rb_thread_call_without_gvl(nvrtcCreateProgram_without_gvl_cb, &param, NULL, NULL);
    }

    free(_headers);
    RB_GC_GUARD(strings);
    check_status(status);
    cumo_cuda_handle_set_add(&programs, (size_t)_prog);
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
    nvrtcProgram _prog = (nvrtcProgram)cumo_cuda_handle_take(&programs, prog, "nvrtcProgram");

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
    nvrtcProgram _prog = (nvrtcProgram)cumo_cuda_handle_get(&programs, prog, "nvrtcProgram");
    int _numOptions = ary_len(options, "options");
    const char** _options = NULL;
    VALUE strings;

    strings = rb_ary_new_capa(_numOptions);
    push_strings(strings, options, _numOptions);

    if (_numOptions > 0) {
        _options = alloc_cstrs(_numOptions);
        fill_cstrs(_options, strings, 0, _numOptions);
    }

    {
        struct nvrtcCompileProgramParam param = {_prog, _numOptions, _options};
        status = (nvrtcResult)rb_thread_call_without_gvl(nvrtcCompileProgram_without_gvl_cb, &param, NULL, NULL);
    }

    free(_options);
    RB_GC_GUARD(strings);
    check_status(status);
    return Qnil;
}

static VALUE
rb_nvrtcGetPTX(VALUE self, VALUE prog)
{
    nvrtcResult status;
    nvrtcProgram _prog = (nvrtcProgram)cumo_cuda_handle_get(&programs, prog, "nvrtcProgram");
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
    nvrtcProgram _prog = (nvrtcProgram)cumo_cuda_handle_get(&programs, prog, "nvrtcProgram");
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

    cumo_cuda_handle_set_init(&programs);

    rb_define_singleton_method(mNVRTC, "nvrtcVersion", rb_nvrtcVersion, 0);
    rb_define_singleton_method(mNVRTC, "nvrtcCreateProgram", rb_nvrtcCreateProgram, 4);
    rb_define_singleton_method(mNVRTC, "nvrtcDestroyProgram", rb_nvrtcDestroyProgram, 1);
    rb_define_singleton_method(mNVRTC, "nvrtcCompileProgram", rb_nvrtcCompileProgram, 2);
    rb_define_singleton_method(mNVRTC, "nvrtcGetPTX", rb_nvrtcGetPTX, 1);
    rb_define_singleton_method(mNVRTC, "nvrtcGetProgramLog", rb_nvrtcGetProgramLog, 1);
}
