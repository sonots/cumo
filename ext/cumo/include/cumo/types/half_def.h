#ifndef CUMO_HALF_DEF_H
#define CUMO_HALF_DEF_H

// A struct rather than a bare uint16, so that C code which reaches for an
// arithmetic operator on an element fails to compile instead of computing on
// the bit pattern. Its layout matches __half, which the kernels use.
typedef struct { unsigned short x; } cumo_half;

// GCC gained _Float16 on x86-64 only in 12, and Ubuntu 22.04 ships 11, so the
// conversions are written out. Both round the way __float2half does, which
// test/hfloat_test.rb checks against the GPU over every 32-bit pattern.
static inline float cumo_half2float(cumo_half h)
{
    union { unsigned int u; float f; } o;
    unsigned int m = h.x & 0x3ffu;
    unsigned int s = (unsigned int)(h.x & 0x8000u) << 16;
    int e = (h.x >> 10) & 0x1f;

    if (e == 0) {
        if (m == 0) {
            o.u = s;
            return o.f;
        }
        e = 1;
        while (!(m & 0x400u)) {
            m <<= 1;
            e--;
        }
        m &= 0x3ffu;
    } else if (e == 31) {
        o.u = s | 0x7f800000u | (m << 13);
        return o.f;
    }
    o.u = s | ((unsigned int)(e + 112) << 23) | (m << 13);
    return o.f;
}

static inline cumo_half cumo_float2half(float f)
{
    union { unsigned int u; float f; } v;
    cumo_half h;
    unsigned int u, sign, mant, round_bits, shift, out;
    int exp;

    v.f = f;
    u = v.u;
    sign = (u >> 16) & 0x8000u;
    u &= 0x7fffffffu;

    if (u > 0x7f800000u) {
        h.x = (unsigned short)(sign | 0x7e00u);
        return h;
    }
    if (u >= 0x47800000u) {
        h.x = (unsigned short)(sign | 0x7c00u);
        return h;
    }
    if (u < 0x33000000u) {
        h.x = (unsigned short)sign;
        return h;
    }

    exp = (int)(u >> 23) - 127;
    mant = u & 0x7fffffu;
    if (exp >= -14) {
        out = (unsigned int)(exp + 15) << 10;
        shift = 13;
    } else {
        mant |= 0x800000u;
        shift = (unsigned int)(-exp - 1);
        out = 0;
    }
    round_bits = mant & ((1u << shift) - 1u);
    out += mant >> shift;
    if (round_bits > (1u << (shift - 1)) ||
        (round_bits == (1u << (shift - 1)) && ((mant >> shift) & 1u))) {
        out += 1;
    }
    h.x = (unsigned short)(sign | out);
    return h;
}

static inline cumo_half cumo_double2half(double d)
{
    union { unsigned long long u; double d; } v;
    cumo_half h;
    unsigned long long u, sign, mant, round_bits, tie;
    unsigned int shift, out;
    int exp;

    v.d = d;
    u = v.u;
    sign = (u >> 48) & 0x8000ull;
    u &= 0x7fffffffffffffffull;

    if (u > 0x7ff0000000000000ull) {
        h.x = (unsigned short)(sign | 0x7e00u);
        return h;
    }
    if (u >= 0x40f0000000000000ull) {
        h.x = (unsigned short)(sign | 0x7c00u);
        return h;
    }
    if (u < 0x3e60000000000000ull) {
        h.x = (unsigned short)sign;
        return h;
    }

    exp = (int)(u >> 52) - 1023;
    mant = u & 0xfffffffffffffull;
    if (exp >= -14) {
        out = (unsigned int)(exp + 15) << 10;
        shift = 42;
    } else {
        mant |= 0x10000000000000ull;
        shift = (unsigned int)(28 - exp);
        out = 0;
    }
    round_bits = mant & ((1ull << shift) - 1ull);
    tie = 1ull << (shift - 1);
    out += (unsigned int)(mant >> shift);
    if (round_bits > tie || (round_bits == tie && ((mant >> shift) & 1ull))) {
        out += 1;
    }
    h.x = (unsigned short)(sign | out);
    return h;
}

#endif // CUMO_HALF_DEF_H
