# frozen_string_literal: true

require_relative "test_helper"

class NArrayMathTest < CumoTestBase
  FLOAT_TYPES = [
    Cumo::SFloat,
    Cumo::SComplex,
    Cumo::DFloat,
    Cumo::DComplex,
  ]

  def test_sqrt
    # NArray
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-4 + 1i, -1 + 4i, 1 - 4i, 4 - 1i]
          else
            dtype[0, 1, 4, 9, 16]
          end
      b = Cumo::NMath.sqrt(a)
      expected = if complex_type?(dtype)
                   dtype[zsqrt(-4 + 1i), zsqrt(-1 + 4i), zsqrt(1 - 4i), zsqrt(4 - 1i)]
                 else
                   dtype[0, 1, 2, 3, 4]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end

    # Ruby Array
    a = Cumo::NMath.sqrt([4, 9])
    b = Cumo::NMath.sqrt([-2 + 3i, 4 - 5i])

    assert_kind_of(Cumo::DFloat, a)
    assert_in_delta(2.0, a[0].extract_cpu)
    assert_in_delta(3.0, a[1].extract_cpu)
    assert_equal(1, a.ndim)
    assert_equal(2, a.size)
    assert_kind_of(Cumo::DComplex, b)
    assert_equal(1, b.ndim)
    assert_equal(2, b.size)

    # Scalar
    a = Cumo::NMath.sqrt(4)
    b = Cumo::NMath.sqrt(-2 + 3i)

    assert_kind_of(Cumo::DFloat, a)
    assert_in_delta(2, a.extract_cpu)
    assert_equal(0, a.ndim)
    assert_equal(1, a.size)
    assert_kind_of(Cumo::DComplex, b)
    assert_operator((b - zsqrt(-2 + 3i)).abs.extract_cpu, :<, 1e-8)
    assert_equal(0, b.ndim)
    assert_equal(1, b.size)
  end

  def test_cbrt
    # NArray
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-8 + 1i, -1 + 27i, 1 - 8i, 8 - 1i]
          else
            dtype[-8, -1, 0, 1, 8]
          end
      b = Cumo::NMath.cbrt(a)
      expected = if complex_type?(dtype)
                   dtype[zcbrt(-8 + 1i), zcbrt(-1 + 27i), zcbrt(1 - 8i), zcbrt(8 - 1i)]
                 else
                   dtype[-2, -1, 0, 1, 2]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end

    # Ruby Array
    a = Cumo::NMath.cbrt([8, 27])
    b = Cumo::NMath.cbrt([-2 + 3i, 4 - 5i])

    assert_kind_of(Cumo::DFloat, a)
    assert_in_delta(2, a[0].extract_cpu)
    assert_in_delta(3, a[1].extract_cpu)
    assert_equal(1, a.ndim)
    assert_equal(2, a.size)
    assert_kind_of(Cumo::DComplex, b)
    assert_equal(1, b.ndim)
    assert_equal(2, b.size)

    # Scalar
    a = Cumo::NMath.cbrt(8)
    b = Cumo::NMath.cbrt(-2 + 3i)

    assert_kind_of(Cumo::DFloat, a)
    assert_in_delta(2, a.extract_cpu)
    assert_equal(0, a.ndim)
    assert_equal(1, a.size)
    assert_kind_of(Cumo::DComplex, b)
    assert_operator((b - zcbrt(-2 + 3i)).abs.extract_cpu, :<, 1e-8)
    assert_equal(0, b.ndim)
    assert_equal(1, b.size)
  end

  def test_log
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 1 - 2i, 2 - 1i]
          else
            dtype[1, Math::E, 0.1, 10]
          end
      b = Cumo::NMath.log(a)
      expected = if complex_type?(dtype)
                   dtype[zlog(-2 + 1i), zlog(-1 + 2i), zlog(1 - 2i), zlog(2 - 1i)]
                 else
                   dtype[0, 1, Math.log(0.1), Math.log(10)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_log2
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 1 - 2i, 2 - 1i]
          else
            dtype[1, 0.5, 2, 4]
          end
      b = Cumo::NMath.log2(a)
      expected = if complex_type?(dtype)
                   dtype[zlog2(-2 + 1i), zlog2(-1 + 2i), zlog2(1 - 2i), zlog2(2 - 1i)]
                 else
                   dtype[0, -1, 1, 2]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_log10
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 1 - 2i, 2 - 1i]
          else
            dtype[1, 0.1, 5, 10, 100]
          end
      b = Cumo::NMath.log10(a)
      expected = if complex_type?(dtype)
                   dtype[zlog10(-2 + 1i), zlog10(-1 + 2i), zlog10(1 - 2i), zlog10(2 - 1i)]
                 else
                   dtype[0, -1, Math.log10(5), 1, 2]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_exp
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.exp(a)
      expected = if complex_type?(dtype)
                   dtype[zexp(-2 + 1i), zexp(-1 + 2i), 1, zexp(1 - 2i), zexp(2 - 1i)]
                 else
                   dtype[Math.exp(-2), Math.exp(-1), 1, Math.exp(1), Math.exp(2)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_exp2
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.exp2(a)
      expected = if complex_type?(dtype)
                   dtype[zexp2(-2 + 1i), zexp2(-1 + 2i), 1, zexp2(1 - 2i), zexp2(2 - 1i)]
                 else
                   dtype[2**-2, 2**-1, 1, 2**1, 2**2]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_exp10
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.exp10(a)
      expected = if complex_type?(dtype)
                   dtype[zexp10(-2 + 1i), zexp10(-1 + 2i), 1, zexp10(1 - 2i), zexp10(2 - 1i)]
                 else
                   dtype[10**-2, 10**-1, 1, 10**1, 10**2]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_sin
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.sin(a)
      expected = if complex_type?(dtype)
                   dtype[zsin(-2 + 1i), zsin(-1 + 2i), 0, zsin(1 - 2i), zsin(2 - 1i)]
                 else
                   dtype[Math.sin(-2), Math.sin(-1), 0, Math.sin(1), Math.sin(2)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_cos
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.cos(a)
      expected = if complex_type?(dtype)
                   dtype[zcos(-2 + 1i), zcos(-1 + 2i), 1, zcos(1 - 2i), zcos(2 - 1i)]
                 else
                   dtype[Math.cos(-2), Math.cos(-1), 1, Math.cos(1), Math.cos(2)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_tan
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.tan(a)
      expected = if complex_type?(dtype)
                   dtype[ztan(-2 + 1i), ztan(-1 + 2i), 0, ztan(1 - 2i), ztan(2 - 1i)]
                 else
                   dtype[Math.tan(-2), Math.tan(-1), 0, Math.tan(1), Math.tan(2)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_asin
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-1, -0.5, 0, 0.5, 1]
          end
      b = Cumo::NMath.asin(a)
      expected = if complex_type?(dtype)
                   dtype[zasin(-2 + 1i), zasin(-1 + 2i), 0, zasin(1 - 2i), zasin(2 - 1i)]
                 else
                   dtype[-Math::PI / 2, Math.asin(-0.5), 0, Math.asin(0.5), Math::PI / 2]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_acos
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-1, -0.5, 0, 0.5, 1]
          end
      b = Cumo::NMath.acos(a)
      expected = if complex_type?(dtype)
                   dtype[zacos(-2 + 1i), zacos(-1 + 2i), Math::PI / 2,
                         zacos(1 - 2i), zacos(2 - 1i)]
                 else
                   dtype[Math::PI, Math.acos(-0.5), Math::PI / 2, Math.acos(0.5), 0]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_atan
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.atan(a)
      expected = if complex_type?(dtype)
                   dtype[zatan(-2 + 1i), zatan(-1 + 2i), 0, zatan(1 - 2i), zatan(2 - 1i)]
                 else
                   dtype[Math.atan(-2), Math.atan(-1), 0, Math.atan(1), Math.atan(2)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_sinh
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.sinh(a)
      expected = if complex_type?(dtype)
                   dtype[zsinh(-2 + 1i), zsinh(-1 + 2i), 0, zsinh(1 - 2i), zsinh(2 - 1i)]
                 else
                   dtype[Math.sinh(-2), Math.sinh(-1), 0, Math.sinh(1), Math.sinh(2)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_cosh
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.cosh(a)
      expected = if complex_type?(dtype)
                   dtype[zcosh(-2 + 1i), zcosh(-1 + 2i), 1, zcosh(1 - 2i), zcosh(2 - 1i)]
                 else
                   dtype[Math.cosh(-2), Math.cosh(-1), 1, Math.cosh(1), Math.cosh(2)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_tanh
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.tanh(a)
      expected = if complex_type?(dtype)
                   dtype[ztanh(-2 + 1i), ztanh(-1 + 2i), 0, ztanh(1 - 2i), ztanh(2 - 1i)]
                 else
                   dtype[Math.tanh(-2), Math.tanh(-1), 0, Math.tanh(1), Math.tanh(2)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_asinh
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.asinh(a)
      expected = if complex_type?(dtype)
                   dtype[zasinh(-2 + 1i), zasinh(-1 + 2i), 0, zasinh(1 - 2i), zasinh(2 - 1i)]
                 else
                   dtype[Math.asinh(-2), Math.asinh(-1), 0, Math.asinh(1), Math.asinh(2)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_acosh
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[1 + 2i, 3 + 4i, 2 - 1i, 4 - 3i]
          else
            dtype[1, 2, 3, 4]
          end
      b = Cumo::NMath.acosh(a)
      expected = if complex_type?(dtype)
                   dtype[zacosh(1 + 2i), zacosh(3 + 4i), zacosh(2 - 1i), zacosh(4 - 3i)]
                 else
                   dtype[Math.acosh(1), Math.acosh(2), Math.acosh(3), Math.acosh(4)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_atanh
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-0.2 + 0.1i, -0.1 + 0.2i, 0, 0.1 - 0.2i, 0.2 - 0.1i]
          else
            dtype[-0.2, -0.1, 0, 0.1, 0.2]
          end
      b = Cumo::NMath.atanh(a)
      expected = if complex_type?(dtype)
                   dtype[zatanh(-0.2 + 0.1i), zatanh(-0.1 + 0.2i), 0,
                         zatanh(0.1 - 0.2i), zatanh(0.2 - 0.1i)]
                 else
                   dtype[Math.atanh(-0.2), Math.atanh(-0.1), 0,
                         Math.atanh(0.1), Math.atanh(0.2)]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_sinc
    FLOAT_TYPES.each do |dtype|
      a = if complex_type?(dtype)
            dtype[-2 + 1i, -1 + 2i, 0, 1 - 2i, 2 - 1i]
          else
            dtype[-2, -1, 0, 1, 2]
          end
      b = Cumo::NMath.sinc(a)
      expected = if complex_type?(dtype)
                   dtype[zsin(-2 + 1i) / (-2 + 1i), zsin(-1 + 2i) / (-1 + 2i),
                         1, zsin(1 - 2i) / (1 - 2i), zsin(2 - 1i) / (2 - 1i)]
                 else
                   dtype[Math.sin(-2) / -2, Math.sin(-1) / -1,
                         1, Math.sin(1) / 1, Math.sin(2) / 2]
                 end
      assert_kind_of(dtype, b)
      assert_close(expected, b)
    end
  end

  def test_atan2
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      a = dtype[0, 1, 0, -1]
      b = dtype[1, 0, -1, 0]
      c = Cumo::NMath.atan2(a, b)

      assert_kind_of(dtype, c)
      assert_equal(dtype[0, Math::PI / 2, Math::PI, -Math::PI / 2], c)
    end
  end

  def test_hypot
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      a = dtype[3, 5, 8]
      b = dtype[4, 12, 15]
      c = Cumo::NMath.hypot(a, b)

      assert_kind_of(dtype, c)
      assert_equal(dtype[5, 13, 17], c)
    end
  end

  def test_erf
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      a = dtype[-2, -1, 0, 1, 2]
      b = Cumo::NMath.erf(a)
      assert_kind_of(dtype, b)
      assert_close(dtype[Math.erf(-2), Math.erf(-1), 0, Math.erf(1), Math.erf(2)], b)
    end
  end

  def test_erfc
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      a = dtype[-2, -1, 0, 1, 2]
      b = Cumo::NMath.erfc(a)
      assert_kind_of(dtype, b)
      assert_close(dtype[Math.erfc(-2), Math.erfc(-1), 1, Math.erfc(1), Math.erfc(2)], b)
    end
  end

  def test_log1p
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      a = dtype[-0.5, 0, 1, 2]
      b = Cumo::NMath.log1p(a)
      assert_kind_of(dtype, b)
      assert_close(dtype[Math.log(0.5), Math.log(1), Math.log(2), Math.log(3)], b)
    end
  end

  def test_expm1
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      a = dtype[-1.0, 0.0, 1.0, 2.0]
      b = Cumo::NMath.expm1(a)
      assert_kind_of(dtype, b)
      assert_close(dtype[Math.exp(-1) - 1, 0, Math.exp(1) - 1, Math.exp(2) - 1], b)
    end
  end

  def test_ldexp
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      a = dtype[0.0, 0.5, 0.5, 0.75, 0.5, 0.625]
      b = Cumo::Int32[0, 1, 2, 2, 3, 3]
      c = Cumo::NMath.ldexp(a, b)

      assert_kind_of(dtype, c)
      assert_equal(dtype[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], c)
    end
  end

  def test_frexp
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      a = dtype[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
      mant, expo = Cumo::NMath.frexp(a)

      assert_kind_of(dtype, mant)
      assert_kind_of(Cumo::Int32, expo)
      assert_equal(dtype[0.0, 0.5, 0.5, 0.75, 0.5, 0.625], mant)
      assert_equal(Cumo::Int32[0, 1, 2, 2, 3, 3], expo)
    end
  end

  private

  def assert_close(expected, actual, rtol = 1e-6)
    scale = [1.0, expected.abs.max.extract_cpu].max
    assert_operator((expected - actual).abs.max.extract_cpu, :<, rtol * scale)
  end

  def complex_type?(type)
    [Cumo::DComplex, Cumo::SComplex].include?(type)
  end

  def zsqrt(z)
    r = Math.sqrt(z.abs)
    theta = Math.atan2(z.imag, z.real) / 2
    r * (Math.cos(theta) + (1i * Math.sin(theta)))
  end

  def zcbrt(z)
    r = Math.cbrt(z.abs)
    theta = Math.atan2(z.imag, z.real) / 3
    r * (Math.cos(theta) + (1i * Math.sin(theta)))
  end

  def zlog(z)
    Math.log(z.abs) + (1i * Math.atan2(z.imag, z.real))
  end

  def zlog2(z)
    zlog(z) / zlog(2 + 0i)
  end

  def zlog10(z)
    zlog(z) / zlog(10 + 0i)
  end

  def zexp(z)
    exp_real = Math.exp(z.real)
    real_part = exp_real * Math.cos(z.imag)
    imag_part = exp_real * Math.sin(z.imag)
    real_part + (1i * imag_part)
  end

  def zexp2(z)
    exp_real = Math.exp(z.real * Math.log(2))
    real_part = exp_real * Math.cos(z.imag)
    imag_part = exp_real * Math.sin(z.imag)
    real_part + (1i * imag_part)
  end

  def zexp10(z)
    exp_real = Math.exp(z.real * Math.log(10))
    real_part = exp_real * Math.cos(z.imag)
    imag_part = exp_real * Math.sin(z.imag)
    real_part + (1i * imag_part)
  end

  def zsin(z)
    (Math.sin(z.real) * Math.cosh(z.imag)) + (1i * Math.cos(z.real) * Math.sinh(z.imag))
  end

  def zcos(z)
    (Math.cos(z.real) * Math.cosh(z.imag)) - (1i * Math.sin(z.real) * Math.sinh(z.imag))
  end

  def ztan(z)
    sin_z = zsin(z)
    cos_z = zcos(z)
    sin_z / cos_z
  end

  def zasin(z)
    -1i * zlog((1i * z) + zsqrt(1 - (z * z)))
  end

  def zacos(z)
    -1i * zlog(z + (1i * zsqrt(1 - (z * z))))
  end

  def zatan(z)
    (1i / 2) * zlog((1 - (1i * z)) / (1 + (1i * z)))
  end

  def zsinh(z)
    (Math.sinh(z.real) * Math.cos(z.imag)) + (1i * Math.cosh(z.real) * Math.sin(z.imag))
  end

  def zcosh(z)
    (Math.cosh(z.real) * Math.cos(z.imag)) + (1i * Math.sinh(z.real) * Math.sin(z.imag))
  end

  def ztanh(z)
    sinh_z = zsinh(z)
    cosh_z = zcosh(z)
    sinh_z / cosh_z
  end

  def zasinh(z)
    zlog(z + zsqrt((z * z) + 1))
  end

  def zacosh(z)
    zlog(z + zsqrt((z * z) - 1))
  end

  def zatanh(z)
    0.5 * zlog((1 + z) / (1 - z))
  end
end
