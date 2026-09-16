#ifndef CUMO_BF16_DEF_H
#define CUMO_BF16_DEF_H

// A struct rather than a bare uint16, so that C code which reaches for an
// arithmetic operator on an element fails to compile instead of computing on
// the bit pattern. Its layout matches __nv_bfloat16, which the kernels use.
typedef struct { unsigned short x; } cumo_bfloat;

static inline float cumo_bfloat2float(cumo_bfloat b)
{
    union { unsigned int u; float f; } o;
    o.u = (unsigned int)b.x << 16;
    return o.f;
}

static inline cumo_bfloat cumo_float2bfloat(float f)
{
    union { unsigned int u; float f; } v;
    cumo_bfloat b;

    v.f = f;
    // bfloat16 keeps the exponent of a float, so every finite value is the top
    // 16 bits rounded to nearest even. A NaN carries its payload in the low
    // bits, and truncating one leaves an all-ones exponent over a zero
    // mantissa, which reads as an infinity.
    if ((v.u & 0x7f800000u) == 0x7f800000u && (v.u & 0x007fffffu) != 0u) {
        b.x = (unsigned short)(((v.u >> 16) & 0x8000u) | 0x7fc0u);
        return b;
    }
    b.x = (unsigned short)((v.u + 0x7fffu + ((v.u >> 16) & 1u)) >> 16);
    return b;
}

static inline cumo_bfloat cumo_double2bfloat(double d)
{
    union { unsigned long long u; double d; } v;
    cumo_bfloat b;
    unsigned long long u, sign, mant, round_bits, tie;
    unsigned int shift, out;
    int exp;

    // Going through float would round twice, and the first rounding can land
    // on a tie the second one then moves the wrong way.
    v.d = d;
    u = v.u;
    sign = (u >> 48) & 0x8000ull;
    u &= 0x7fffffffffffffffull;

    if (u > 0x7ff0000000000000ull) {
        b.x = (unsigned short)(sign | 0x7fc0u);
        return b;
    }
    if (u >= 0x47f0000000000000ull) {
        b.x = (unsigned short)(sign | 0x7f80u);
        return b;
    }
    if (u < 0x3790000000000000ull) {
        b.x = (unsigned short)sign;
        return b;
    }

    exp = (int)(u >> 52) - 1023;
    mant = u & 0xfffffffffffffull;
    if (exp >= -126) {
        out = (unsigned int)(exp + 127) << 7;
        shift = 45;
    } else {
        mant |= 0x10000000000000ull;
        shift = (unsigned int)(-81 - exp);
        out = 0;
    }
    round_bits = mant & ((1ull << shift) - 1ull);
    tie = 1ull << (shift - 1);
    out += (unsigned int)(mant >> shift);
    if (round_bits > tie || (round_bits == tie && ((mant >> shift) & 1ull))) {
        out += 1;
    }
    b.x = (unsigned short)(sign | out);
    return b;
}

#endif // CUMO_BF16_DEF_H
