/*
  step.c
  Numerical Array Extension for Ruby
    (C) Copyright 2007,2013 by Masahiro TANAKA
*/
#include <ruby.h>
#include <math.h>

#include "cumo/narray.h"

#if defined(__FreeBSD__) && __FreeBSD__ < 4
#include <floatingpoint.h>
#endif

#ifdef HAVE_FLOAT_H
#include <float.h>
#endif

#ifdef HAVE_IEEEFP_H
#include <ieeefp.h>
#endif

#ifndef DBL_EPSILON
#define DBL_EPSILON 2.2204460492503131e-16
#endif

void
cumo_na_step_array_index(VALUE obj, size_t ary_size,
                      size_t *plen, ssize_t *pbeg, ssize_t *pstep)
{
    size_t len;
    ssize_t beg=0, step=1;
    VALUE vbeg, vend, vstep;
    ssize_t end=ary_size;

    rb_arithmetic_sequence_components_t x;
    rb_arithmetic_sequence_extract(obj, &x);

    vstep = x.step;
    vbeg = x.begin;
    vend = x.end;

    if (RTEST(vbeg)) {
        beg = NUM2SSIZET(vbeg);
        if (beg<0) {
            beg += ary_size;
        }
    }
    if (RTEST(vend)) {
        end = NUM2SSIZET(vend);
        if (end<0) {
            end += ary_size;
        }
    }

    if (RTEST(vstep)) {
        step = NUM2SSIZET(vstep);
    } else {
        step = 1;
    }
    if (step>0) {
        if (!RTEST(vbeg)) {
            beg = 0;
        }
        if (!RTEST(vend)) {
            end = ary_size-1;
        }
        else if (RTEST(x.exclude_end)) {
            end--;
        }
        if (beg<=end) {
            len = (end-beg)/step+1;
        } else {
            len = 0;
        }
    } else if (step<0) {
        if (!RTEST(vbeg)) {
            beg = ary_size-1;
        }
        if (!RTEST(vend)) {
            end = 0;
        }
        else if (RTEST(x.exclude_end)) {
            end++;
        }
        if (beg>=end) {
            len = (beg-end)/(-step)+1;
        } else {
            len = 0;
        }
    } else {
        rb_raise( rb_eStandardError, "step must be non-zero" );
    }

    if (beg<0 || beg>=(ssize_t)ary_size ||
        end<0 || end>=(ssize_t)ary_size) {
        rb_raise( rb_eRangeError,
                  "beg=%"SZF"d,end=%"SZF"d is out of array size (%"SZF"u)",
                  beg, end, ary_size );
    }
    if (plen) *plen = len;
    if (pbeg) *pbeg = beg;
    if (pstep) *pstep = step;
}

void
cumo_na_step_sequence( VALUE obj, size_t *plen, double *pbeg, double *pstep )
{
    VALUE vend, vstep;
    double dbeg, dend, dstep=1, dsize, err;
    size_t size;

    rb_arithmetic_sequence_components_t x;
    rb_arithmetic_sequence_extract(obj, &x);

    vstep = x.step;
    dbeg = NUM2DBL(x.begin);
    vend = x.end;

    if (!RTEST(vstep)) {
        dstep = 1;
    } else {
        dstep = NUM2DBL(vstep);
    }
    if (RTEST(vend)) {
        dend = NUM2DBL(vend);
        err = (fabs(dbeg)+fabs(dend)+fabs(dend-dbeg))/fabs(dstep)*DBL_EPSILON;
        if (err>0.5) err=0.5;
        dsize = (dend-dbeg)/dstep;
        if (RTEST(x.exclude_end))
            dsize -= err;
        else
            dsize += err;
        dsize = floor(dsize) + 1;
        if (dsize<0) dsize=0;
        if (isinf(dsize) || isnan(dsize)) {
            rb_raise(rb_eArgError, "not finite size");
        }
        size = dsize;
    } else {
        rb_raise(rb_eArgError, "cannot determine length argument");
    }

    if (plen) *plen = size;
    if (pbeg) *pbeg = dbeg;
    if (pstep) *pstep = dstep;
}
