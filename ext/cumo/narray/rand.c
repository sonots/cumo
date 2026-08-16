#include "ruby.h"
#include "cumo/narray.h"
#include "SFMT.h"

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <time.h>
#ifdef HAVE_SYS_TIME_H
#include <sys/time.h>
#endif

static u_int64_t
random_seed()
{
    static int n = 0;
    struct timeval tv;

    gettimeofday(&tv, 0);
    return tv.tv_sec ^ tv.tv_usec ^ getpid() ^ n++;
}

// The kernels draw from Philox keyed by this seed, giving every element its own
// subsequence. The offset advances by the number of elements drawn so that two
// calls under one seed do not repeat the same stream.
static u_int64_t cumo_cuda_rand_seed_value;
static u_int64_t cumo_cuda_rand_offset_value;

u_int64_t
cumo_cuda_rand_seed(void)
{
    return cumo_cuda_rand_seed_value;
}

u_int64_t
cumo_cuda_rand_offset(void)
{
    return cumo_cuda_rand_offset_value;
}

void
cumo_cuda_rand_set_offset(u_int64_t offset)
{
    cumo_cuda_rand_offset_value = offset;
}

static void
cumo_na_seed(u_int64_t seed)
{
    init_gen_rand(seed);
    cumo_cuda_rand_seed_value = seed;
    cumo_cuda_rand_offset_value = 0;
}

static VALUE
cumo_na_s_srand(int argc, VALUE *argv, VALUE obj)
{
    VALUE vseed;
    u_int64_t seed;

    //rb_secure(4);
    if (rb_scan_args(argc, argv, "01", &vseed) == 0) {
        seed = random_seed();
    }
    else {
        seed = NUM2UINT64(vseed);
    }
    cumo_na_seed(seed);

    return Qnil;
}

void
Init_cumo_na_rand() {
    rb_define_singleton_method(cNArray, "srand", cumo_na_s_srand, -1);
    cumo_na_seed(0);
}
