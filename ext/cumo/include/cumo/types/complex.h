static inline dtype c_new(rtype r, rtype i) {
    dtype z;
    CUMO_REAL(z) = r;
    CUMO_IMAG(z) = i;
    return z;
}

static inline dtype c_set_real(dtype x, rtype r) {
    CUMO_REAL(x)=r;
    return x;
}

static inline dtype c_set_imag(dtype x, rtype i) {
    CUMO_IMAG(x)=i;
    return x;
}

static inline VALUE COMP2NUM(dtype x) {
    VALUE v;
    v = rb_funcall(rb_intern("Kernel"), rb_intern("Complex"), 2,
                   rb_float_new(CUMO_REAL(x)), rb_float_new(CUMO_IMAG(x)));
    return v;
}

static inline dtype NUM2COMP(VALUE v) {
    dtype z;
    CUMO_REAL(z) = NUM2DBL(rb_funcall(v,cumo_id_real,0));
    CUMO_IMAG(z) = NUM2DBL(rb_funcall(v,cumo_id_imag,0));
    return z;
}

#define c_is_zero(x) (CUMO_REAL(x)==0 && CUMO_IMAG(x)==0)
#define c_eq(x,y) (CUMO_REAL(x)==CUMO_REAL(y) && CUMO_IMAG(x)==CUMO_IMAG(y))
#define c_ne(x,y) (CUMO_REAL(x)!=CUMO_REAL(y) || CUMO_IMAG(x)!=CUMO_IMAG(y))
#define c_isnan(x) (isnan(CUMO_REAL(x)) || isnan(CUMO_IMAG(x)))
#define c_isinf(x) (isinf(CUMO_REAL(x)) || isinf(CUMO_IMAG(x)))
#define c_isposinf(x) ((isinf(CUMO_REAL(x)) && signbit(CUMO_REAL(x))==0) || \
                       (isinf(CUMO_IMAG(x)) && signbit(CUMO_IMAG(x))==0))
#define c_isneginf(x) ((isinf(CUMO_REAL(x)) && signbit(CUMO_REAL(x))) || \
                       (isinf(CUMO_IMAG(x)) && signbit(CUMO_IMAG(x))))
#define c_isfinite(x) (isfinite(CUMO_REAL(x)) && isfinite(CUMO_IMAG(x)))

static inline dtype c_zero() {
    dtype z;
    CUMO_REAL(z) = 0;
    CUMO_IMAG(z) = 0;
    return z;
}

static inline dtype c_one() {
    dtype z;
    CUMO_REAL(z) = 1;
    CUMO_IMAG(z) = 0;
    return z;
}

static inline dtype c_minus(dtype x) {
    dtype z;
    CUMO_REAL(z) = -CUMO_REAL(x);
    CUMO_IMAG(z) = -CUMO_IMAG(x);
    return z;
}

static inline dtype c_im(dtype x) {
    dtype z;
    CUMO_REAL(z) = -CUMO_IMAG(x);
    CUMO_IMAG(z) = CUMO_REAL(x);
    return z;
}

static inline dtype c_add(dtype x, dtype y) {
    dtype z;
    CUMO_REAL(z) = CUMO_REAL(x)+CUMO_REAL(y);
    CUMO_IMAG(z) = CUMO_IMAG(x)+CUMO_IMAG(y);
    return z;
}

static inline dtype c_sub(dtype x, dtype y) {
    dtype z;
    CUMO_REAL(z) = CUMO_REAL(x)-CUMO_REAL(y);
    CUMO_IMAG(z) = CUMO_IMAG(x)-CUMO_IMAG(y);
    return z;
}


static inline dtype c_mul(dtype x, dtype y) {
    dtype z;
    CUMO_REAL(z) = CUMO_REAL(x)*CUMO_REAL(y)-CUMO_IMAG(x)*CUMO_IMAG(y);
    CUMO_IMAG(z) = CUMO_REAL(x)*CUMO_IMAG(y)+CUMO_IMAG(x)*CUMO_REAL(y);
    return z;
}

static inline dtype c_mul_r(dtype x, rtype y) {
    dtype z;
    CUMO_REAL(z) = CUMO_REAL(x)*y;
    CUMO_IMAG(z) = CUMO_IMAG(x)*y;
    return z;
}

static inline dtype c_div(dtype x, dtype y) {
    dtype z;
    rtype s,yr,yi;
    s  = r_hypot(CUMO_REAL(y),CUMO_IMAG(y));
    yr = CUMO_REAL(y)/s;
    yi = CUMO_IMAG(y)/s;
    CUMO_REAL(z) = (CUMO_REAL(x)*yr+CUMO_IMAG(x)*yi)/s;
    CUMO_IMAG(z) = (CUMO_IMAG(x)*yr-CUMO_REAL(x)*yi)/s;
    return z;
}

static inline dtype c_div_r(dtype x, rtype y) {
    dtype z;
    CUMO_REAL(z) = CUMO_REAL(x)/y;
    CUMO_IMAG(z) = CUMO_IMAG(x)/y;
    return z;
}

static inline dtype c_reciprocal(dtype x) {
    dtype z;
    if ( r_abs(CUMO_REAL(x)) > r_abs(CUMO_IMAG(x)) ) {
        CUMO_IMAG(z) = CUMO_IMAG(x)/CUMO_REAL(x);
        CUMO_REAL(z) = (1+CUMO_IMAG(z)*CUMO_IMAG(z))*CUMO_REAL(x);
        CUMO_IMAG(z) /= -CUMO_REAL(z);
        CUMO_REAL(z) = 1/CUMO_REAL(z);
    } else {
        CUMO_REAL(z) = CUMO_REAL(x)/CUMO_IMAG(x);
        CUMO_IMAG(z) = (1+CUMO_REAL(z)*CUMO_REAL(z))*CUMO_IMAG(x);
        CUMO_REAL(z) /= CUMO_IMAG(z);
        CUMO_IMAG(z) = -1/CUMO_IMAG(z);
    }
    return z;
}

static inline dtype c_square(dtype x) {
    dtype z;
    CUMO_REAL(z) = CUMO_REAL(x)*CUMO_REAL(x)-CUMO_IMAG(x)*CUMO_IMAG(x);
    CUMO_IMAG(z) = 2*CUMO_REAL(x)*CUMO_IMAG(x);
    return z;
}

static inline dtype c_sqrt(dtype x) {
    dtype z;
    rtype xr, xi, r;
    xr = CUMO_REAL(x)/2;
    xi = CUMO_IMAG(x)/2;
    r  = r_hypot(xr,xi);
    if (xr>0) {
        CUMO_REAL(z) = sqrt(r+xr);
        CUMO_IMAG(z) = xi/CUMO_REAL(z);
    } else if ( (r-=xr)!=0 ) {
        CUMO_IMAG(z) = (xi>=0) ? sqrt(r):-sqrt(r);
        CUMO_REAL(z) = xi/CUMO_IMAG(z);
    } else {
        CUMO_REAL(z) = CUMO_IMAG(z) = 0;
    }
    return z;
}

static inline dtype c_log(dtype x) {
    dtype z;
    CUMO_REAL(z) = r_log(r_hypot(CUMO_REAL(x),CUMO_IMAG(x)));
    CUMO_IMAG(z) = r_atan2(CUMO_IMAG(x),CUMO_REAL(x));
    return z;
}

static inline dtype c_log2(dtype x) {
    dtype z;
    z = c_log(x);
    z = c_mul_r(z,M_LOG2E);
    return z;
}

static inline dtype c_log10(dtype x) {
    dtype z;
    z = c_log(x);
    z = c_mul_r(z,M_LOG10E);
    return z;
}

static inline dtype c_exp(dtype x) {
    dtype z;
    rtype a = r_exp(CUMO_REAL(x));
    CUMO_REAL(z) = a*r_cos(CUMO_IMAG(x));
    CUMO_IMAG(z) = a*r_sin(CUMO_IMAG(x));
    return z;
}

static inline dtype c_exp2(dtype x) {
    dtype z;
    rtype a = r_exp(CUMO_REAL(x)*M_LN2);
    CUMO_REAL(z) = a*r_cos(CUMO_IMAG(x));
    CUMO_IMAG(z) = a*r_sin(CUMO_IMAG(x));
    return z;
}

static inline dtype c_exp10(dtype x) {
    dtype z;
    rtype a = r_exp(CUMO_REAL(x)*M_LN10);
    CUMO_REAL(z) = a*r_cos(CUMO_IMAG(x));
    CUMO_IMAG(z) = a*r_sin(CUMO_IMAG(x));
    return z;
}

static inline dtype c_sin(dtype x) {
    dtype z;
    CUMO_REAL(z) = r_sin(CUMO_REAL(x))*r_cosh(CUMO_IMAG(x));
    CUMO_IMAG(z) = r_cos(CUMO_REAL(x))*r_sinh(CUMO_IMAG(x));
    return z;
}

static inline dtype c_sinh(dtype x) {
    dtype z;
    CUMO_REAL(z) = r_sinh(CUMO_REAL(x))*r_cos(CUMO_IMAG(x));
    CUMO_IMAG(z) = r_cosh(CUMO_REAL(x))*r_sin(CUMO_IMAG(x));
    return z;
}

static inline dtype c_cos(dtype x) {
    dtype z;
    CUMO_REAL(z) = r_cos(CUMO_REAL(x))*r_cosh(CUMO_IMAG(x));
    CUMO_IMAG(z) = -r_sin(CUMO_REAL(x))*r_sinh(CUMO_IMAG(x));
    return z;
}

static inline dtype c_cosh(dtype x) {
    dtype z;
    CUMO_REAL(z) = r_cosh(CUMO_REAL(x))*r_cos(CUMO_IMAG(x));
    CUMO_IMAG(z) = r_sinh(CUMO_REAL(x))*r_sin(CUMO_IMAG(x));
    return z;
}

static inline dtype c_tan(dtype x) {
    dtype z;
    rtype c, d;
    if (r_abs(CUMO_IMAG(x))<1) {
        c = r_cos(CUMO_REAL(x));
        d = r_sinh(CUMO_IMAG(x));
        d = c*c + d*d;
        CUMO_REAL(z) = 0.5*r_sin(2*CUMO_REAL(x))/d;
        CUMO_IMAG(z) = 0.5*r_sinh(2*CUMO_IMAG(x))/d;
    } else {
        d = r_exp(-CUMO_IMAG(x));
        c = 2*d/(1-d*d);
        c = c*c;
        d = r_cos(CUMO_REAL(x));
        d = 1.0 + d*d*c;
        CUMO_REAL(z) = 0.5*r_sin(2*CUMO_REAL(x))*c/d;
        CUMO_IMAG(z) = 1/r_tanh(CUMO_IMAG(x))/d;
    }
    return z;
}

static inline dtype c_tanh(dtype x) {
    dtype z;
    rtype c, d, s;
    c = r_cos(CUMO_IMAG(x));
    s = r_sinh(CUMO_REAL(x));
    d = c*c + s*s;
    if (r_abs(CUMO_REAL(x))<1) {
        CUMO_REAL(z) = s*r_cosh(CUMO_REAL(x))/d;
        CUMO_IMAG(z) = 0.5*r_sin(2*CUMO_IMAG(x))/d;
    } else {
        c = c / s;
        c = 1 + c*c;
        CUMO_REAL(z) = 1/(r_tanh(CUMO_REAL(x))*c);
        CUMO_IMAG(z) = 0.5*r_sin(2*CUMO_IMAG(x))/d;
    }
    return z;
}

static inline dtype c_asin(dtype x) {
    dtype z, y;
    y = c_square(x);
    CUMO_REAL(y) = 1-CUMO_REAL(y);
    CUMO_IMAG(y) = -CUMO_IMAG(y);
    y = c_sqrt(y);
    CUMO_REAL(y) -= CUMO_IMAG(x);
    CUMO_IMAG(y) += CUMO_REAL(x);
    y = c_log(y);
    CUMO_REAL(z) = CUMO_IMAG(y);
    CUMO_IMAG(z) = -CUMO_REAL(y);
    return z;
}

static inline dtype c_asinh(dtype x) {
    dtype z, y;
    y = c_square(x);
    CUMO_REAL(y) += 1;
    y = c_sqrt(y);
    CUMO_REAL(y) += CUMO_REAL(x);
    CUMO_IMAG(y) += CUMO_IMAG(x);
    z = c_log(y);
    return z;
}

static inline dtype c_acos(dtype x) {
    dtype z, y;
    y = c_square(x);
    CUMO_REAL(y) = 1-CUMO_REAL(y);
    CUMO_IMAG(y) = -CUMO_IMAG(y);
    y = c_sqrt(y);
    CUMO_REAL(z) = CUMO_REAL(x)-CUMO_IMAG(y);
    CUMO_IMAG(z) = CUMO_IMAG(x)+CUMO_REAL(y);
    y = c_log(z);
    CUMO_REAL(z) = CUMO_IMAG(y);
    CUMO_IMAG(z) = -CUMO_REAL(y);
    return z;
}

static inline dtype c_acosh(dtype x) {
    dtype z, y;
    y = c_square(x);
    CUMO_REAL(y) -= 1;
    y = c_sqrt(y);
    CUMO_REAL(y) += CUMO_REAL(x);
    CUMO_IMAG(y) += CUMO_IMAG(x);
    z = c_log(y);
    return z;
}

static inline dtype c_atan(dtype x) {
    dtype z, y;
    CUMO_REAL(y) = -CUMO_REAL(x);
    CUMO_IMAG(y) = 1-CUMO_IMAG(x);
    CUMO_REAL(z) = CUMO_REAL(x);
    CUMO_IMAG(z) = 1+CUMO_IMAG(x);
    y = c_div(z,y);
    y = c_log(y);
    CUMO_REAL(z) = -CUMO_IMAG(y)/2;
    CUMO_IMAG(z) = CUMO_REAL(y)/2;
    return z;
}

static inline dtype c_atanh(dtype x) {
    dtype z, y;
    CUMO_REAL(y) = 1-CUMO_REAL(x);
    CUMO_IMAG(y) = -CUMO_IMAG(x);
    CUMO_REAL(z) = 1+CUMO_REAL(x);
    CUMO_IMAG(z) = CUMO_IMAG(x);
    y = c_div(z,y);
    y = c_log(y);
    CUMO_REAL(z) = CUMO_REAL(y)/2;
    CUMO_IMAG(z) = CUMO_IMAG(y)/2;
    return z;
}

static inline dtype c_pow(dtype x, dtype y)
{
    dtype z;
    if (c_is_zero(y)) {
        z = c_one();
    } else if (c_is_zero(x) && CUMO_REAL(y)>0 && CUMO_IMAG(y)==0) {
        z = c_zero();
    } else {
        z = c_log(x);
        z = c_mul(y,z);
        z = c_exp(z);
    }
    return z;
}

static inline dtype c_pow_int(dtype x, int p)
{
    dtype z = c_one();
    if (p<0) {
	x = c_pow_int(x,-p);
	return c_reciprocal(x);
    }
    if (p==2) {return c_square(x);}
    if (p&1) {z = x;}
    p >>= 1;
    while (p) {
	x = c_square(x);
	if (p&1) z = c_mul(z,x);
	p >>= 1;
    }
    return z;
}

static inline dtype c_cbrt(dtype x) {
    dtype z;
    z = c_log(x);
    z = c_div_r(z,3);
    z = c_exp(z);
    return z;
}

static inline rtype c_abs(dtype x) {
    return r_hypot(CUMO_REAL(x),CUMO_IMAG(x));
}

static inline rtype c_abs_square(dtype x) {
    return CUMO_REAL(x)*CUMO_REAL(x)+CUMO_IMAG(x)*CUMO_IMAG(x);
}



/*
static inline rtype c_hypot(dtype x, dtype y) {
    return r_hypot(c_abs(x),c_abs(y));
}
*/
