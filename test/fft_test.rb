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
    assert_raise(ArgumentError) { Cumo::FFT.fft(Cumo::DFloat.new(4, 0)) }
    assert_raise(ArgumentError) { Cumo::FFT.irfft(Cumo::DComplex[1]) }
    assert_raise(Cumo::NArray::DimensionError) { Cumo::FFT.fft(a, axis: 1) }
    assert_raise(Cumo::NArray::DimensionError) { Cumo::FFT.fft(a, axis: -2) }
    assert_raise(Cumo::NArray::DimensionError) { Cumo::FFT.fft(Cumo::DFloat.cast(1.5)) }
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
