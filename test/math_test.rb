# frozen_string_literal: true

require_relative "test_helper"

class NArrayMathTest < CumoTestBase
  FLOAT_TYPES = [
    Cumo::HFloat,
    Cumo::BFloat,
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

  # Values from an independent source rather than the kernel's own spelling:
  # torch.nn.functional.gelu in double, which is the definition cumo follows.
  GELU_EXACT = {
    -6.0 => -5.9195258694799691e-09,
    -2.0 => -0.04550026389635842,
    -1.0 => -0.15865525393145707,
    0.0 => 0.0,
    1.0 => 0.8413447460685429,
    2.0 => 1.9544997361036416,
    6.0 => 5.999999994080474,
  }.freeze

  GELU_TANH = {
    -6.0 => -0.0,
    -2.0 => -0.04540230330138159,
    -1.0 => -0.15880800939172324,
    0.0 => 0.0,
    1.0 => 0.8411919906082768,
    2.0 => 1.9545976966986184,
    6.0 => 6.0,
  }.freeze

  def test_gelu
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      b = Cumo::NMath.gelu(dtype[*GELU_EXACT.keys])
      assert_kind_of(dtype, b)
      assert_close(dtype[*GELU_EXACT.values], b)
    end
  end

  def test_gelu_tanh
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      b = Cumo::NMath.gelu_tanh(dtype[*GELU_TANH.keys])
      assert_kind_of(dtype, b)
      assert_close(dtype[*GELU_TANH.values], b)
      # The two are different functions, not two roundings of one. Checked per
      # dtype, since wiring one to the other is a per-dtype macro.
      a = dtype[-2, -1, 1, 2]
      assert_operator((Cumo::NMath.gelu(a) - Cumo::NMath.gelu_tanh(a)).abs.max.extract_cpu,
                      :>, [Cumo::HFloat, Cumo::BFloat].include?(dtype) ? 1e-4 : 1e-5)
    end
  end

  # Both forms are x * (something that goes to zero), so a negative infinity
  # reaches infinity times zero. torch answers NaN here too, and cumo follows
  # the framework rather than the limit, which is zero.
  def test_gelu_at_infinity
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      inf = dtype[-Float::INFINITY, Float::INFINITY]
      %i[gelu gelu_tanh].each do |name|
        y = Cumo::NMath.send(name, inf).to_a
        assert(y[0].nan?, "#{name} #{dtype} at -Infinity answered #{y[0]}")
        assert(y[1].infinite? == 1, "#{name} #{dtype} at +Infinity answered #{y[1]}")
        assert(Cumo::NMath.send(name, dtype[Float::NAN]).to_a.first.nan?)
      end
    end
  end

  # 1 + erf(x/sqrt(2)) cancels for a large negative x, so the answer runs out of
  # significant digits and reaches exactly zero well before the true value does.
  # torch answers the same, and matching it is worth more here than the digits.
  def test_gelu_underflows_like_torch
    assert_equal(0.0, Cumo::NMath.gelu(Cumo::DFloat[-10.0]).to_a.first.abs)
    assert_in_delta(-4.884981308350689e-15, Cumo::NMath.gelu(Cumo::DFloat[-8.0]).to_a.first, 1e-29)
  end

  def test_gelu_reads_a_non_contiguous_view
    base = Cumo::DFloat.new(4, 3).seq(-6) / 2
    view = base.transpose
    assert_close(Cumo::NMath.gelu(view.dup), Cumo::NMath.gelu(view))
    assert_close(Cumo::NMath.gelu_tanh(view.dup), Cumo::NMath.gelu_tanh(view))
  end

  # The definition evaluated at 200 decimal places and rounded once, which is
  # the true value rather than any implementation of it. A division cannot
  # always reach it: this kernel is within one unit in the last place at every
  # point below, and exactly on five of the nine. torch answers the same except
  # at 2.5, where it is a unit high.
  SILU = {
    -6.0 => -0.014835738939808645,
    -2.5 => -0.18964545005310887,
    -2.0 => -0.23840584404423512,
    -1.0 => -0.2689414213699951,
    0.0 => 0.0,
    1.0 => 0.7310585786300049,
    2.0 => 1.7615941559557649,
    2.5 => 2.310354549946891,
    6.0 => 5.985164261060191,
  }.freeze

  # The definition evaluated at 200 decimal places and rounded once, so these
  # are neither this kernel's spelling nor another implementation's. Written
  # out, log(1 + exp(x)) is about an ulp off the true value for a negative x,
  # the sum losing what the logarithm then asks for.
  def test_silu
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      b = Cumo::NMath.silu(dtype[*SILU.keys])
      assert_kind_of(dtype, b)
      assert_close(dtype[*SILU.values], b)
    end
  end

  # x * sigmoid(x) reaches infinity times zero at a negative infinity, the same
  # way both gelu forms do. torch answers NaN here, and cumo follows the
  # framework rather than the limit, which is zero.
  def test_silu_at_infinity
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      y = Cumo::NMath.silu(dtype[-Float::INFINITY, Float::INFINITY]).to_a
      assert(y[0].nan?, "silu #{dtype} at -Infinity answered #{y[0]}")
      assert(y[1].infinite? == 1, "silu #{dtype} at +Infinity answered #{y[1]}")
      assert(Cumo::NMath.silu(dtype[Float::NAN]).to_a.first.nan?)
    end
  end

  # The quotient reaches exactly zero rather than the denormal the true value
  # is, and keeps its sign, which is what tells it from a plain zero. Ruby has
  # -0.0 == 0.0, so the reciprocal is what actually asks.
  def test_silu_underflows_like_torch
    [[Cumo::SFloat, -800.0], [Cumo::SFloat, -100.0], [Cumo::DFloat, -800.0]].each do |dtype, x|
      y = Cumo::NMath.silu(dtype[x]).to_a.first
      assert_equal(0.0, y.abs, "#{dtype} at #{x}")
      assert_equal(-Float::INFINITY, 1.0 / y, "#{dtype} at #{x} lost the sign of its zero")
    end
    # Double still has room at -100, where single does not.
    assert_in_delta(-3.720075976020836e-42, Cumo::NMath.silu(Cumo::DFloat[-100.0]).to_a.first, 1e-56)
  end

  # Half runs out four times earlier than the others, and not for the same
  # reason: expf is still finite at -21, and it is the round back to half that
  # answers zero. bfloat16 carries a float's exponent, so it goes with single.
  def test_silu_underflows_earlier_in_half
    first_zero = lambda do |dtype|
      (1..200).find { |n| Cumo::NMath.silu(dtype[-n.to_f]).to_a.first == 0.0 }
    end
    assert_equal(21, first_zero.call(Cumo::HFloat))
    assert_equal(89, first_zero.call(Cumo::BFloat))
    assert_equal(89, first_zero.call(Cumo::SFloat))
    assert_nil(first_zero.call(Cumo::DFloat))
  end

  # silu is not either gelu, and wiring one to the other is a per-dtype macro.
  def test_silu_is_not_gelu
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      a = dtype[-2, -1, 1, 2]
      tol = [Cumo::HFloat, Cumo::BFloat].include?(dtype) ? 1e-4 : 1e-5
      assert_operator((Cumo::NMath.silu(a) - Cumo::NMath.gelu(a)).abs.max.extract_cpu, :>, tol)
      assert_operator((Cumo::NMath.silu(a) - Cumo::NMath.gelu_tanh(a)).abs.max.extract_cpu, :>, tol)
    end
  end

  def test_silu_reads_a_non_contiguous_view
    view = (Cumo::DFloat.new(4, 3).seq(-6) / 2).transpose
    assert_close(Cumo::NMath.silu(view.dup), Cumo::NMath.silu(view))
  end

  # The definition evaluated at 200 decimal places and rounded once, so these
  # are neither this kernel's spelling nor another implementation's.
  SOFTPLUS = {
    -6.0 => 0.0024756851377304495,
    -2.5 => 0.07888973429254963,
    -2.0 => 0.1269280110429725,
    -1.0 => 0.3132616875182228,
    0.0 => 0.6931471805599453,
    1.0 => 1.3132616875182228,
    2.0 => 2.1269280110429727,
    2.5 => 2.5788897342925496,
    6.0 => 6.00247568513773,
  }.freeze

  def test_softplus
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      b = Cumo::NMath.softplus(dtype[*SOFTPLUS.keys])
      assert_kind_of(dtype, b)
      assert_close(dtype[*SOFTPLUS.values], b)
    end
  end

  # Writing the same thing out of the operators reaches an infinity as soon as
  # exp does. The three widths give out at the same x, the sixteen-bit ones
  # taking the single-precision exp, and what changes between them is how far
  # the type can carry the answer afterwards.
  def test_softplus_answers_x_where_exp_gives_out
    [[Cumo::HFloat, 60000.0], [Cumo::BFloat, 100.0], [Cumo::BFloat, 3.0e38],
     [Cumo::SFloat, 89.0], [Cumo::SFloat, 200.0], [Cumo::DFloat, 800.0]].each do |dtype, x|
      written_out = Cumo::NMath.log(1.0 + Cumo::NMath.exp(dtype[x])).to_a.first
      assert_equal(Float::INFINITY, written_out, "#{dtype} at #{x} was expected to overflow written out")
      assert_equal(dtype[x].to_a.first, Cumo::NMath.softplus(dtype[x]).to_a.first, "#{dtype} at #{x}")
    end
  end

  # An x that exp still reaches keeps the value the written-out spelling
  # answers, bit for bit. The range runs past where each type overflows, so
  # the two halves of the branch are both walked.
  def test_softplus_matches_the_written_out_form_where_it_is_finite
    [[Cumo::SFloat, 90.0], [Cumo::DFloat, 712.0]].each do |dtype, top|
      x = dtype.new(2001).seq * (top / 1000.0) - top
      written_out = Cumo::NMath.log(1.0 + Cumo::NMath.exp(x))
      finite = written_out.isfinite
      assert_operator(finite.count_false, :>, 0, "#{dtype}: the range never overflowed")
      assert_operator(finite.count_true, :>, 1900, "#{dtype}: too few finite points to compare")
      assert_equal(written_out[finite].to_a, Cumo::NMath.softplus(x)[finite].to_a, dtype.to_s)
    end
  end

  # The cost of the spelling, at the other end. Adding one drops what exp
  # answers once it falls under the type's epsilon, so softplus reaches zero
  # where the true value is still a number log1p would have kept. Single gives
  # out at -17 and double at -37; the error grows without bound before that,
  # 4.3% at DFloat -36 and 5.9% at SFloat -16.
  def test_softplus_reaches_zero_where_adding_one_loses_exp
    [[Cumo::SFloat, -17.0], [Cumo::HFloat, -17.0], [Cumo::BFloat, -17.0],
     [Cumo::DFloat, -37.0]].each do |dtype, x|
      assert_equal(0.0, Cumo::NMath.softplus(dtype[x]).to_a.first, "#{dtype} at #{x}")
    end
    assert_operator(Cumo::NMath.softplus(Cumo::SFloat[-16.0]).to_a.first, :>, 0.0)
    assert_operator(Cumo::NMath.softplus(Cumo::DFloat[-36.0]).to_a.first, :>, 0.0)
  end

  def test_softplus_at_infinity
    FLOAT_TYPES.reject { |t| complex_type?(t) }.each do |dtype|
      y = Cumo::NMath.softplus(dtype[-Float::INFINITY, Float::INFINITY]).to_a
      assert_equal(0.0, y[0], "softplus #{dtype} at -Infinity answered #{y[0]}")
      assert_equal(1, y[1].infinite?, "softplus #{dtype} at +Infinity answered #{y[1]}")
      assert(Cumo::NMath.softplus(dtype[Float::NAN]).to_a.first.nan?, dtype.to_s)
    end
  end

  def test_softplus_reads_a_non_contiguous_view
    view = (Cumo::DFloat.new(4, 3).seq(-6) / 2).transpose
    assert_close(Cumo::NMath.softplus(view.dup), Cumo::NMath.softplus(view))
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

  def test_respond_to_a_dispatched_method
    %i[sqrt exp log gelu gelu_tanh silu].each do |name|
      assert_true(Cumo::NMath.respond_to?(name), name.to_s)
      assert_kind_of(Method, Cumo::NMath.method(name))
    end
    assert_false(Cumo::NMath.respond_to?(:no_such_math_function))
    assert_raise(NameError) { Cumo::NMath.method(:no_such_math_function) }
  end

  def test_respond_to_a_method_no_dtype_can_reach
    %i[gamma lgamma].each do |name|
      assert_true(Math.respond_to?(name), name.to_s)
      assert_raise(NoMethodError) { Cumo::NMath.send(name, 3.0) }
      assert_raise(NoMethodError) { Cumo::NMath.send(name, Cumo::DFloat[3.0]) }
      assert_false(Cumo::NMath.respond_to?(name), name.to_s)
    end
  end

  def test_dfloat_math_answers_for_every_module_a_call_can_reach
    reachable = Cumo::NMath::DISPATCH.select { |type,| type < Cumo::NArray }.values.uniq
    assert_operator(reachable.size, :>, 1)
    assert_include(reachable, Cumo::DFloat::Math)
    reachable.each do |mod|
      assert_equal([], mod.singleton_methods - Cumo::DFloat::Math.singleton_methods, mod.to_s)
    end
  end

  private

  def assert_close(expected, actual, rtol = nil)
    # binary16 keeps 11 bits of significand and bfloat16 only 8, so neither
    # holds a result to 1e-6
    rtol ||= expected.is_a?(Cumo::BFloat) ? 1e-2 : expected.is_a?(Cumo::HFloat) ? 1e-3 : 1e-6
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
