# frozen_string_literal: true

require "numo/narray"
require_relative "test_helper"
require "cumo/fft"

class FFTTest < CumoTestBase
  def setup
    omit("Cumo is built without cuFFT") unless Cumo::CUDA::Cufft.available?
  end

  def dft_matrix(n, inverse)
    k = Numo::Int64.new(n).seq
    angle = ((k.reshape(n, 1) * k) % n) * ((inverse ? 2 : -2) * Math::PI / n)
    Numo::NMath.cos(angle) + (Complex::I * Numo::NMath.sin(angle))
  end

  def dft(x, axis: -1, inverse: false)
    x = Numo::DComplex.cast(x)
    axis += x.ndim if axis.negative?
    last = x.ndim - 1
    y = x.swapaxes(axis, last).dup
    y = y.dot(dft_matrix(y.shape[-1], inverse))
    y /= y.shape[-1] if inverse
    y.swapaxes(axis, last).dup
  end

  def hermitian(x, n)
    x = Numo::DComplex.cast(x)
    full = Numo::DComplex.zeros(*x.shape[0...-1], n)
    m = (n / 2) + 1
    full[false, 0...m] = x[false, 0...m]
    full[false, 0] = full[false, 0].real
    full[false, n / 2] = full[false, n / 2].real if n.even?
    (1...((n + 1) / 2)).each { |k| full[false, n - k] = x[false, k].conj }
    full
  end

  def numo_input(type, *shape)
    re = Numo::DFloat.new(*shape).rand(-1, 1)
    im = Numo::DFloat.new(*shape).rand(-1, 1)
    case type.name
    when /Complex/ then re + (Complex::I * im)
    when /Int/ then (re * 100).round
    else re
    end
  end

  def assert_close(expected, actual, tol, message = nil)
    actual = Cumo::NArray.to_numo(actual)
    expected = Numo::DComplex.cast(expected)
    assert_equal(expected.shape, actual.shape, message)
    return if expected.empty?

    scale = [expected.abs.max, 1.0].max
    err = (Numo::DComplex.cast(actual) - expected).abs.max / scale
    assert_operator(err, :<=, tol, message)
  end

  def tol(type)
    [Cumo::SFloat, Cumo::SComplex, Cumo::HFloat, Cumo::BFloat].include?(type) ? 1e-5 : 1e-12
  end

  test "fft and ifft match the DFT for every length along the last axis" do
    [Cumo::DComplex, Cumo::SComplex, Cumo::DFloat, Cumo::SFloat, Cumo::Int32].each do |type|
      [1, 2, 3, 7, 8, 30, 64, 97, 1000].each do |n|
        src = numo_input(type, 3, n)
        a = type.cast(Cumo::NArray.from_numo(src))
        ref = Cumo::NArray.to_numo(a)
        assert_close(dft(ref), Cumo::FFT.fft(a), tol(type), "fft #{type} #{n}")
        assert_close(dft(ref, inverse: true), Cumo::FFT.ifft(a), tol(type), "ifft #{type} #{n}")
      end
    end
  end

  test "single precision stays single and everything else becomes double" do
    {
      Cumo::SFloat => Cumo::SComplex,
      Cumo::SComplex => Cumo::SComplex,
      Cumo::HFloat => Cumo::SComplex,
      Cumo::BFloat => Cumo::SComplex,
      Cumo::DFloat => Cumo::DComplex,
      Cumo::DComplex => Cumo::DComplex,
      Cumo::Int32 => Cumo::DComplex,
      Cumo::UInt8 => Cumo::DComplex
    }.each do |type, out|
      a = type.cast([1, 2, 3, 4])
      assert_equal(out, Cumo::FFT.fft(a).class, type.name)
      assert_equal(out, Cumo::FFT.ifft(a).class, type.name)
      next if [Cumo::SComplex, Cumo::DComplex].include?(type)

      assert_equal(out, Cumo::FFT.rfft(a).class, type.name)
      assert_equal(out == Cumo::SComplex ? Cumo::SFloat : Cumo::DFloat, Cumo::FFT.irfft(a).class, type.name)
    end
  end

  test "every axis of a 3-dimensional array, and views as input" do
    src = numo_input(Cumo::DComplex, 4, 5, 6)
    a = Cumo::NArray.from_numo(src)
    [0, 1, 2, -1, -2, -3].each do |axis|
      assert_close(dft(src, axis: axis), Cumo::FFT.fft(a, axis: axis), 1e-12, "axis #{axis}")
      assert_close(dft(src, axis: axis, inverse: true), Cumo::FFT.ifft(a, axis: axis), 1e-12, "axis #{axis}")
    end
    [[true, (0..).step(2), 1..4], [1..2, true, true], [true, 3, true]].each do |index|
      view = a[*index]
      assert_close(dft(src[*index]), Cumo::FFT.fft(view), 1e-12, index.inspect)
    end
    assert_close(dft(src.transpose), Cumo::FFT.fft(a.transpose), 1e-12, "transpose")
  end

  test "n crops or zero-pads the transformed axis" do
    src = numo_input(Cumo::DComplex, 2, 10)
    a = Cumo::NArray.from_numo(src)
    assert_close(dft(src[true, 0...6]), Cumo::FFT.fft(a, n: 6), 1e-12)
    padded = Numo::DComplex.zeros(2, 16)
    padded[true, 0...10] = src
    assert_close(dft(padded), Cumo::FFT.fft(a, n: 16), 1e-12)
    assert_close(dft(src[0...1, true], axis: 0), Cumo::FFT.fft(a, n: 1, axis: 0), 1e-12)
  end

  test "norm scales the forward or the inverse transform" do
    src = numo_input(Cumo::DComplex, 12)
    a = Cumo::NArray.from_numo(src)
    n = 12.0
    assert_close(dft(src), Cumo::FFT.fft(a, norm: "backward"), 1e-12)
    assert_close(dft(src) / Math.sqrt(n), Cumo::FFT.fft(a, norm: :ortho), 1e-12)
    assert_close(dft(src) / n, Cumo::FFT.fft(a, norm: "forward"), 1e-12)
    assert_close(dft(src, inverse: true) * Math.sqrt(n), Cumo::FFT.ifft(a, norm: "ortho"), 1e-12)
    assert_close(dft(src, inverse: true) * n, Cumo::FFT.ifft(a, norm: :forward), 1e-12)
  end

  test "rfft answers the non-negative frequencies of the DFT" do
    [Cumo::DFloat, Cumo::SFloat].each do |type|
      [1, 2, 7, 8, 31, 32, 1000].each do |n|
        src = numo_input(Cumo::DFloat, 3, n)
        a = type.cast(Cumo::NArray.from_numo(src))
        ref = Cumo::NArray.to_numo(a)
        assert_close(dft(ref)[true, 0..(n / 2)], Cumo::FFT.rfft(a), tol(type), "rfft #{type} #{n}")
      end
    end
    src = numo_input(Cumo::DFloat, 5, 6)
    assert_close(dft(src, axis: 0)[0..2, true], Cumo::FFT.rfft(Cumo::NArray.from_numo(src), axis: 0), 1e-12)
  end

  test "rfft reads real views that start between two complex elements" do
    [Cumo::DFloat, Cumo::SFloat].each do |type|
      a = type.cast(Cumo::NArray.from_numo(numo_input(Cumo::DFloat, 3, 9)))
      ref = Cumo::NArray.to_numo(a)
      [[1, true], [2, 3..8], [1, 1..-1], [true, 1..7]].each do |index|
        view = ref[*index]
        expected = dft(view)[false, 0..(view.shape[-1] / 2)]
        assert_close(expected, Cumo::FFT.rfft(a[*index]), tol(type), "#{type} #{index}")
      end
    end
  end

  test "irfft inverts rfft and drops the imaginary part of the DC and Nyquist terms" do
    [Cumo::DComplex, Cumo::SComplex].each do |type|
      [2, 7, 8, 31, 32].each do |n|
        m = (n / 2) + 1
        src = numo_input(Cumo::DComplex, 3, m)
        a = type.cast(Cumo::NArray.from_numo(src))
        ref = Cumo::NArray.to_numo(a)
        expected = dft(hermitian(ref, n), inverse: true).real
        assert_close(expected, Cumo::FFT.irfft(a, n: n), tol(type), "irfft #{type} #{n}")
      end
    end
    real = Cumo::DFloat.new(4, 10).rand
    assert_close(Cumo::NArray.to_numo(real), Cumo::FFT.irfft(Cumo::FFT.rfft(real)), 1e-12)
    assert_close(Cumo::NArray.to_numo(real), Cumo::FFT.irfft(Cumo::FFT.rfft(real, axis: 0), n: 4, axis: 0), 1e-12)
  end

  test "irfft leaves its input as it was" do
    a = Cumo::DComplex.new(2, 5).rand + (Complex::I * Cumo::DComplex.new(2, 5).rand)
    before = a.to_a
    Cumo::FFT.irfft(a)
    assert_equal(before, a.to_a)
  end

  test "a multi-dimensional C2R plan, which overwrites its input, leaves the array as it was" do
    a = Cumo::DComplex.new(4, 3).rand + (Complex::I * Cumo::DComplex.new(4, 3).rand)
    before = a.to_a
    Cumo::CUDA::Cufft.__send__(:c2r, a, 2, 4, false)
    assert_equal(before, a.to_a)
  end

  test "an empty batch answers an empty result" do
    a = Cumo::DFloat.new(0, 4)
    assert_equal([0, 4], Cumo::FFT.fft(a).shape)
    assert_equal([0, 3], Cumo::FFT.rfft(a).shape)
    assert_equal([0, 6], Cumo::FFT.irfft(Cumo::DComplex.new(0, 4)).shape)
  end

  test "errors follow numpy" do
    a = Cumo::DFloat[1, 2, 3, 4]
    assert_raise(TypeError) { Cumo::FFT.fft([1, 2, 3]) }
    assert_raise(TypeError) { Cumo::FFT.rfft(Cumo::DComplex[1, 2]) }
    assert_raise(ArgumentError) { Cumo::FFT.fft(a, norm: "bad") }
    assert_raise(ArgumentError) { Cumo::FFT.fft(a, n: 0) }
    assert_raise(ArgumentError) { Cumo::FFT.fft(a, n: -1) }
    assert_raise(ArgumentError) { Cumo::FFT.fft(Cumo::DFloat.new(4, 0)) }
    assert_raise(ArgumentError) { Cumo::FFT.irfft(Cumo::DComplex[1]) }
    assert_raise(Cumo::NArray::DimensionError) { Cumo::FFT.fft(a, axis: 1) }
    assert_raise(Cumo::NArray::DimensionError) { Cumo::FFT.fft(a, axis: -2) }
    assert_raise(Cumo::NArray::DimensionError) { Cumo::FFT.fft(Cumo::DFloat.cast(1.5)) }
  end

  def dftn(x, axes, inverse: false)
    axes.reduce(Numo::DComplex.cast(x)) { |acc, elem| dft(acc, axis: elem, inverse: inverse) }
  end

  def irfft_along(x, axis, n)
    last = x.ndim - 1
    axis += x.ndim if axis.negative?
    y = dft(hermitian(x.swapaxes(axis, last).dup, n), inverse: true).real
    y.swapaxes(axis, last).dup
  end

  def irfftn_ref(x, axes, n)
    irfft_along(dftn(x, axes[0...-1], inverse: true), axes[-1], n)
  end

  def half(x, axis)
    index = Array.new(x.ndim, true)
    index[axis] = 0..(x.shape[axis] / 2)
    x[*index]
  end

  test "fft2 and ifft2 transform the last two axes" do
    [Cumo::DComplex, Cumo::SComplex, Cumo::DFloat, Cumo::SFloat].each do |type|
      [[6, 10], [3, 5, 7]].each do |shape|
        a = type.cast(Cumo::NArray.from_numo(numo_input(type, *shape)))
        ref = Cumo::NArray.to_numo(a)
        assert_close(dftn(ref, [-2, -1]), Cumo::FFT.fft2(a), tol(type), "fft2 #{type} #{shape}")
        assert_close(dftn(ref, [-2, -1], inverse: true), Cumo::FFT.ifft2(a), tol(type), "ifft2 #{type} #{shape}")
      end
    end
  end

  test "fftn and ifftn transform every axis or any set of them" do
    {
      [7] => [nil, [0]],
      [4, 6] => [nil, [0], [1, 0]],
      [3, 4, 5] => [nil, [0], [1], [2], [0, 1], [1, 2], [0, 2], [2, 0], [-1, 0, 1]],
      [2, 3, 4, 5] => [nil, [0, 2, 3], [1, 3], [3, 0, 1, 2]],
      [2, 2, 3, 2, 3] => [nil, [0, 1, 2, 4]]
    }.each do |shape, axes_list|
      src = numo_input(Cumo::DComplex, *shape)
      a = Cumo::NArray.from_numo(src)
      axes_list.each do |axes|
        all = axes || (0...shape.size).to_a
        assert_close(dftn(src, all), Cumo::FFT.fftn(a, axes: axes), 1e-12, "fftn #{shape} #{axes}")
        assert_close(dftn(src, all, inverse: true), Cumo::FFT.ifftn(a, axes: axes), 1e-12, "ifftn #{shape} #{axes}")
      end
    end
  end

  test "s crops, pads or keeps each axis, and alone picks the last axes" do
    src = numo_input(Cumo::DComplex, 3, 5, 6)
    a = Cumo::NArray.from_numo(src)
    padded = Numo::DComplex.zeros(2, 5, 8)
    padded[true, true, 0...6] = src[0...2, true, true]
    assert_close(dftn(padded, [0, 2]), Cumo::FFT.fftn(a, s: [2, 8], axes: [0, 2]), 1e-12)
    assert_close(dftn(src[true, true, 0...4], [0, 2]), Cumo::FFT.fftn(a, s: [-1, 4], axes: [0, 2]), 1e-12)
    assert_close(dftn(src[true, 0...4, true], [1, 2]), Cumo::FFT.fftn(a, s: [4, 6]), 1e-12)
    assert_close(dftn(src[true, 0...4, true], [1, 2]), Cumo::FFT.fft2(a, s: [4, 6]), 1e-12)
  end

  test "a repeated axis is transformed once for each time it is named" do
    src = numo_input(Cumo::DComplex, 3, 4)
    a = Cumo::NArray.from_numo(src)
    assert_close(dftn(src, [1, 1]), Cumo::FFT.fftn(a, axes: [1, 1]), 1e-12)
    assert_close(dftn(src, [1, 1], inverse: true), Cumo::FFT.ifftn(a, axes: [1, 1]), 1e-12)
  end

  test "rfftn halves the last of the axes" do
    [Cumo::DFloat, Cumo::SFloat].each do |type|
      {
        [6, 9] => [nil, [1, 0], [0]],
        [3, 4, 7] => [nil, [0, 1], [2, 0], [1, 2]],
        [2, 3, 4, 5] => [nil, [3, 1, 0, 2]]
      }.each do |shape, axes_list|
        a = type.cast(Cumo::NArray.from_numo(numo_input(Cumo::DFloat, *shape)))
        ref = Cumo::NArray.to_numo(a)
        axes_list.each do |axes|
          all = axes || (0...shape.size).to_a
          expected = half(dftn(ref, all), all[-1] % shape.size)
          assert_close(expected, Cumo::FFT.rfftn(a, axes: axes), tol(type), "rfftn #{type} #{shape} #{axes}")
        end
      end
    end
    src = numo_input(Cumo::DFloat, 4, 6)
    assert_close(half(dftn(src, [0, 1]), 1), Cumo::FFT.rfft2(Cumo::NArray.from_numo(src)), 1e-12)
  end

  test "irfftn matches numpy for any input, Hermitian or not" do
    [Cumo::DComplex, Cumo::SComplex].each do |type|
      {
        [6, 5] => [nil, [1, 0]],
        [3, 4, 5] => [nil, [0, 1], [2, 0], [1, 2]],
        [2, 3, 4, 5] => [nil, [3, 1, 0, 2]]
      }.each do |shape, axes_list|
        a = type.cast(Cumo::NArray.from_numo(numo_input(Cumo::DComplex, *shape)))
        ref = Cumo::NArray.to_numo(a)
        axes_list.each do |axes|
          all = axes || (0...shape.size).to_a
          n = 2 * (shape[all[-1]] - 1)
          assert_close(irfftn_ref(ref, all, n), Cumo::FFT.irfftn(a, axes: axes), tol(type), "irfftn #{type} #{shape} #{axes}")
        end
      end
    end
    src = numo_input(Cumo::DComplex, 3, 4, 5)
    a = Cumo::NArray.from_numo(src)
    assert_close(irfftn_ref(src, [1, 2], 9), Cumo::FFT.irfftn(a, s: [4, 9], axes: [1, 2]), 1e-12)
    assert_close(irfftn_ref(src, [1, 2], 9), Cumo::FFT.irfft2(a, s: [4, 9]), 1e-12)
    assert_close(irfftn_ref(src, [1, 2], 5), Cumo::FFT.irfftn(a, s: [-1, -1], axes: [1, 2]), 1e-12)
    real = Cumo::DFloat.new(4, 6, 7).rand
    assert_close(Cumo::NArray.to_numo(real), Cumo::FFT.irfftn(Cumo::FFT.rfftn(real), s: [4, 6, 7]), 1e-12)
  end

  test "irfftn to one point along the last axis, which a multi-dimensional plan reads wrong" do
    src = numo_input(Cumo::DComplex, 3, 1)
    assert_close(irfftn_ref(src, [0, 1], 1), Cumo::FFT.irfftn(Cumo::NArray.from_numo(src), s: [3, 1]), 1e-12)
    src = numo_input(Cumo::DComplex, 5, 2, 3)
    expected = irfftn_ref(src[true, true, 0...1], [0, 2], 1)
    assert_close(expected, Cumo::FFT.irfftn(Cumo::NArray.from_numo(src), s: [5, 1], axes: [0, 2]), 1e-12)
    src = numo_input(Cumo::DComplex, 2, 3, 2, 2, 1)
    assert_close(irfftn_ref(src, [0, 1, 2, 3, 4], 1), Cumo::FFT.irfftn(Cumo::NArray.from_numo(src), s: [2, 3, 2, 2, 1]), 1e-12)
  end

  test "irfftn leaves its input as it was, which a multi-dimensional plan would overwrite" do
    a = Cumo::DComplex.new(4, 3).rand + (Complex::I * Cumo::DComplex.new(4, 3).rand)
    before = a.to_a
    Cumo::FFT.irfftn(a)
    Cumo::FFT.irfft2(a)
    assert_equal(before, a.to_a)
  end

  test "norm scales the N-dimensional transforms by every transformed length" do
    src = numo_input(Cumo::DComplex, 4, 6)
    a = Cumo::NArray.from_numo(src)
    assert_close(dftn(src, [0, 1]) / Math.sqrt(24), Cumo::FFT.fftn(a, norm: "ortho"), 1e-12)
    assert_close(dftn(src, [0, 1]) / 24, Cumo::FFT.fft2(a, norm: :forward), 1e-12)
    assert_close(dftn(src, [0, 1], inverse: true) * 24, Cumo::FFT.ifftn(a, norm: "forward"), 1e-12)
    assert_close(irfftn_ref(src, [0, 1], 10) * Math.sqrt(40), Cumo::FFT.irfftn(a, norm: "ortho"), 1e-12)
  end

  test "the N-dimensional transforms keep single precision single" do
    assert_equal(Cumo::SComplex, Cumo::FFT.fft2(Cumo::SFloat.new(2, 2).seq).class)
    assert_equal(Cumo::SComplex, Cumo::FFT.rfftn(Cumo::HFloat.new(2, 2).seq).class)
    assert_equal(Cumo::SFloat, Cumo::FFT.irfftn(Cumo::SComplex.new(2, 2).seq).class)
    assert_equal(Cumo::DComplex, Cumo::FFT.fftn(Cumo::Int32.new(2, 2).seq).class)
  end

  test "N-dimensional errors follow numpy" do
    a = Cumo::DFloat.new(3, 4).seq
    assert_raise(Cumo::NArray::DimensionError) { Cumo::FFT.fft2(Cumo::DFloat[1, 2, 3]) }
    assert_raise(Cumo::NArray::DimensionError) { Cumo::FFT.fftn(a, axes: [0, 2]) }
    assert_raise(ArgumentError) { Cumo::FFT.fftn(a, s: [3], axes: [0, 1]) }
    assert_raise(ArgumentError) { Cumo::FFT.fftn(a, s: [0, 4]) }
    assert_raise(ArgumentError) { Cumo::FFT.rfftn(a, axes: []) }
    assert_raise(ArgumentError) { Cumo::FFT.irfftn(a, axes: []) }
    assert_raise(TypeError) { Cumo::FFT.rfftn(Cumo::DComplex.new(2, 2).seq) }
    assert_raise(TypeError) { Cumo::FFT.fftn(a, s: 4) }
    assert_raise(TypeError) { Cumo::FFT.fftn(a, axes: 1) }
    assert_same(a, Cumo::FFT.fftn(a, axes: []))
    assert_equal([0, 3, 4], Cumo::FFT.fftn(Cumo::DFloat.new(0, 3, 4), axes: [1, 2]).shape)
  end

  def rolled(x, shifts)
    out = x.class.zeros(*x.shape)
    x.shape.map { |n| (0...n).to_a }.reduce { |acc, elem| acc.product(elem).map(&:flatten) }.each do |idx|
      idx = [idx].flatten
      src = idx.each_with_index.map { |v, ax| (v - shifts.fetch(ax, 0)) % x.shape[ax] }
      out[*idx] = x[*src]
    end
    out
  end

  test "hfft and ihfft are irfft and rfft of the conjugate with the norm turned around" do
    [Cumo::DComplex, Cumo::SComplex].each do |type|
      [5, 8].each do |m|
        a = type.cast(Cumo::NArray.from_numo(numo_input(Cumo::DComplex, 3, m)))
        ref = Cumo::NArray.to_numo(a)
        n = 2 * (m - 1)
        expected = irfft_along(ref.conj, 1, n)
        assert_close(expected * n, Cumo::FFT.hfft(a), tol(type), "hfft #{type} #{m}")
        assert_close(expected * Math.sqrt(n), Cumo::FFT.hfft(a, norm: "ortho"), tol(type), "hfft ortho #{type} #{m}")
        assert_close(expected, Cumo::FFT.hfft(a, norm: :forward), tol(type), "hfft forward #{type} #{m}")
      end
    end
    [Cumo::DFloat, Cumo::SFloat].each do |type|
      a = type.cast(Cumo::NArray.from_numo(numo_input(Cumo::DFloat, 3, 9)))
      expected = dft(Cumo::NArray.to_numo(a))[true, 0..4].conj
      assert_close(expected / 9, Cumo::FFT.ihfft(a), tol(type), "ihfft #{type}")
      assert_close(expected, Cumo::FFT.ihfft(a, norm: "forward"), tol(type), "ihfft forward #{type}")
    end
    src = numo_input(Cumo::DComplex, 6, 4)
    a = Cumo::NArray.from_numo(src)
    assert_close(irfft_along(src.conj, 0, 7) * 7, Cumo::FFT.hfft(a, n: 7, axis: 0), 1e-12)
    real = Cumo::NArray.from_numo(numo_input(Cumo::DFloat, 3, 9))
    assert_close(dft(Cumo::NArray.to_numo(real))[true, 0..4].conj / 3, Cumo::FFT.ihfft(real, norm: "ortho"), 1e-12)
    real = Cumo::DFloat.new(10).rand
    assert_close(Cumo::NArray.to_numo(real), Cumo::FFT.hfft(Cumo::FFT.ihfft(real), n: 10), 1e-12)
    flagged = Cumo::DComplex[1, Complex(2, 1), Complex(3, -1), 4]
    before = flagged.to_a
    Cumo::FFT.hfft(flagged.inplace)
    Cumo::FFT.hfft(flagged)
    assert_equal(before, flagged.to_a)
    assert_equal(Cumo::SFloat, Cumo::FFT.hfft(Cumo::SComplex[1, 2, 3]).class)
    assert_equal(Cumo::SComplex, Cumo::FFT.ihfft(Cumo::SFloat[1, 2, 3]).class)
    assert_raise(TypeError) { Cumo::FFT.ihfft(Cumo::DComplex[1, 2]) }
  end

  test "fftfreq and rfftfreq answer numpy's frequencies" do
    (1..9).each do |n|
      [1.0, 0.1, 3].each do |d|
        scale = 1.0 / (n * d)
        ints = (0...n).map { |k| k < ((n - 1) / 2) + 1 ? k : k - n }
        assert_equal(ints.map { |k| k * scale }, Cumo::FFT.fftfreq(n, d: d).to_a, "fftfreq #{n} #{d}")
        assert_equal((0..(n / 2)).map { |k| k * scale }, Cumo::FFT.rfftfreq(n, d: d).to_a, "rfftfreq #{n} #{d}")
      end
    end
    assert_equal(Cumo::DFloat, Cumo::FFT.fftfreq(4).class)
    assert_equal(Cumo::FFT.fftfreq(4, d: 0.5).to_a, Cumo::FFT.fftfreq(4, d: Cumo::DFloat.cast(0.5)).to_a)
    assert_equal(Cumo::FFT.rfftfreq(6, d: 0.25).to_a, Cumo::FFT.rfftfreq(6, d: Rational(1, 4)).to_a)
    assert_raise(ArgumentError) { Cumo::FFT.fftfreq(4.0) }
    assert_raise(ArgumentError) { Cumo::FFT.rfftfreq(-1) }
    assert_raise(ZeroDivisionError) { Cumo::FFT.fftfreq(0) }
    assert_raise(ZeroDivisionError) { Cumo::FFT.rfftfreq(4, d: 0) }
  end

  test "fftshift and ifftshift roll each axis by half its length" do
    x = Numo::Int32.new(3, 4, 5).seq
    a = Cumo::NArray.from_numo(x)
    half = { 0 => 1, 1 => 2, 2 => 2 }
    assert_equal(rolled(x, half).to_a, Cumo::FFT.fftshift(a).to_a)
    assert_equal(rolled(x, half.transform_values(&:-@)).to_a, Cumo::FFT.ifftshift(a).to_a)
    assert_equal(rolled(x, { 1 => 2 }).to_a, Cumo::FFT.fftshift(a, axes: 1).to_a)
    assert_equal(rolled(x, { 0 => 1, 2 => 2 }).to_a, Cumo::FFT.fftshift(a, axes: [0, -1]).to_a)
    assert_equal(rolled(x, { 2 => -2 }).to_a, Cumo::FFT.ifftshift(a, axes: [2]).to_a)
    assert_equal(rolled(x, { 2 => 4 }).to_a, Cumo::FFT.fftshift(a, axes: [2, 2]).to_a)
    assert_equal(x.to_a, Cumo::FFT.ifftshift(Cumo::FFT.fftshift(a)).to_a)
    bits = Cumo::Bit[[1, 0, 0], [0, 1, 1]]
    assert_equal([[0, 1, 1], [1, 0, 0]], Cumo::FFT.fftshift(bits, axes: 0).to_a)
    c = Cumo::DComplex[1, Complex(2, 3), 4, 5]
    shifted = Cumo::FFT.fftshift(c)
    assert_equal(Cumo::DComplex, shifted.class)
    shifted[0] = 0
    assert_equal([1, Complex(2, 3), 4, 5], c.to_a)
    scalar = Cumo::FFT.fftshift(Cumo::Int32.cast(7))
    assert_equal([[], 7], [scalar.shape, Integer(scalar)])
    [[0, 4], [4, 0], [3, 0, 4]].each do |shape|
      assert_equal(shape, Cumo::FFT.fftshift(Cumo::DFloat.new(*shape)).shape, shape.inspect)
      assert_equal(shape, Cumo::FFT.ifftshift(Cumo::DFloat.new(*shape), axes: -1).shape, shape.inspect)
    end
    flagged = Cumo::DFloat[1, 2, 3, 4, 5]
    shifted = Cumo::FFT.fftshift(flagged.inplace)
    assert_equal([[1, 2, 3, 4, 5], [4, 5, 1, 2, 3]], [flagged.to_a, shifted.to_a])
    assert_not_same(shifted, shifted * 2)
    assert_raise(Cumo::NArray::DimensionError) { Cumo::FFT.fftshift(a, axes: 3) }
    assert_raise(TypeError) { Cumo::FFT.fftshift(a, axes: 1.0) }
    assert_raise(TypeError) { Cumo::FFT.fftshift([1, 2, 3]) }
  end

  def convolve_ref(a, b)
    a = Numo::DComplex.cast(a)
    b = Numo::DComplex.cast(b)
    out = Numo::DComplex.zeros(*a.shape.zip(b.shape).map { |m, n| m + n - 1 })
    b.shape.map { |n| (0...n).to_a }.reduce { |acc, elem| acc.product(elem).map(&:flatten) }.each do |j|
      j = [j].flatten
      out[*a.shape.each_with_index.map { |m, i| j[i]...(j[i] + m) }] += a * b[*j]
    end
    out
  end

  test "fftconvolve answers the full linear convolution" do
    [
      [[Cumo::DFloat, [3]], [Cumo::DFloat, [2]]],
      [[Cumo::DFloat, [5]], [Cumo::DFloat, [4]]],
      [[Cumo::SFloat, [9]], [Cumo::SFloat, [6]]],
      [[Cumo::DComplex, [6]], [Cumo::DFloat, [5]]],
      [[Cumo::SComplex, [3, 4]], [Cumo::SComplex, [2, 5]]],
      [[Cumo::DFloat, [3, 4, 5]], [Cumo::DFloat, [2, 3, 2]]],
      [[Cumo::Int32, [7]], [Cumo::Int32, [1]]]
    ].each do |(ta, sa), (tb, sb)|
      a = ta.cast(Cumo::NArray.from_numo(numo_input(ta, *sa)))
      b = tb.cast(Cumo::NArray.from_numo(numo_input(tb, *sb)))
      expected = convolve_ref(Cumo::NArray.to_numo(a), Cumo::NArray.to_numo(b))
      expected = expected.real unless [ta, tb].include?(Cumo::DComplex) || [ta, tb].include?(Cumo::SComplex)
      tol = [ta, tb].all? { |t| [Cumo::SFloat, Cumo::SComplex].include?(t) } ? 1e-5 : 1e-12
      assert_close(expected, Cumo::FFT.fftconvolve(a, b), tol, "#{ta} #{sa} #{tb} #{sb}")
    end
  end

  test "fftconvolve keeps single precision single and real real" do
    {
      [Cumo::SFloat, Cumo::SFloat] => Cumo::SFloat,
      [Cumo::SFloat, Cumo::SComplex] => Cumo::SComplex,
      [Cumo::HFloat, Cumo::SFloat] => Cumo::SFloat,
      [Cumo::SFloat, Cumo::DFloat] => Cumo::DFloat,
      [Cumo::Int32, Cumo::Int32] => Cumo::DFloat,
      [Cumo::DComplex, Cumo::Int32] => Cumo::DComplex
    }.each do |(ta, tb), out|
      assert_equal(out, Cumo::FFT.fftconvolve(ta.new(4).seq, tb.new(3).seq).class, "#{ta} #{tb}")
    end
    assert_equal(6, Integer(Cumo::FFT.fftconvolve(Cumo::Int32.cast(2), Cumo::Int32.cast(3))))
    assert_equal([0], Cumo::FFT.fftconvolve(Cumo::DFloat.new(0), Cumo::DFloat[1, 2]).shape)
    assert_raise(ArgumentError) { Cumo::FFT.fftconvolve(Cumo::DFloat[1, 2], Cumo::DFloat[[1]]) }
    assert_raise(TypeError) { Cumo::FFT.fftconvolve([1, 2], Cumo::DFloat[1]) }
  end

  test "plan_cache_info counts the plans built and reused on this thread" do
    a = Cumo::DComplex.new(37).rand
    Cumo::CUDA::Cufft.clear_plan_cache
    assert_equal({ size: 0, capacity: 16, hits: 0, misses: 0 }, Cumo::CUDA::Cufft.plan_cache_info)
    Cumo::FFT.fft(a)
    Cumo::FFT.fft(a)
    Cumo::FFT.ifft(a)
    assert_equal({ size: 1, capacity: 16, hits: 2, misses: 1 }, Cumo::CUDA::Cufft.plan_cache_info)
    thread = Thread.new do
      Cumo::CUDA::Cufft.clear_plan_cache
      Cumo::FFT.fft(a)
      Cumo::CUDA::Cufft.plan_cache_info
    end
    other = thread.value
    assert_equal({ size: 1, capacity: 16, hits: 0, misses: 1 }, other)
    assert_equal({ size: 1, capacity: 16, hits: 2, misses: 1 }, Cumo::CUDA::Cufft.plan_cache_info)
    Cumo::CUDA::Cufft.clear_plan_cache
    assert_equal({ size: 0, capacity: 16, hits: 0, misses: 0 }, Cumo::CUDA::Cufft.plan_cache_info)
  end

  test "the result is a fresh array that later operations do not write into" do
    a = Cumo::DComplex[1, 2, 3, 4]
    b = Cumo::FFT.ifft(a)
    c = b * 2
    assert_not_same(b, c)
    assert_close(dft(Cumo::NArray.to_numo(a), inverse: true), b, 1e-12)
  end

  test "more lengths than the plan cache holds come back right" do
    a = Cumo::DComplex.new(64).seq
    ref = Cumo::NArray.to_numo(a)
    3.times do
      (1..20).each { |n| assert_close(dft(ref[0...n]), Cumo::FFT.fft(a, n: n), 1e-12, n.to_s) }
    end
  end

  test "a transform on a stream of its own and on other threads" do
    a = Cumo::DComplex.new(8, 128).rand
    expected = dft(Cumo::NArray.to_numo(a))
    stream = Cumo::CUDA::Stream.new
    begin
      result = stream.with { Cumo::FFT.fft(a) }
      assert_close(expected, result, 1e-12)
    ensure
      stream.destroy
    end
    results = Array.new(4) { Thread.new { Cumo::FFT.fft(a) } }.map(&:value)
    results.each { |r| assert_close(expected, r, 1e-12) }
  end
end
