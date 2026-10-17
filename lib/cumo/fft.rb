# frozen_string_literal: true

require 'cumo'

module Cumo
  # Discrete Fourier transforms on the GPU through cuFFT, with the functions
  # and arguments of numpy.fft.
  #
  # Single precision stays single: an SFloat, SComplex, HFloat or BFloat input
  # gives an SComplex, and every other type a DComplex.
  module FFT
    NORMS = %w[backward ortho forward].freeze
    private_constant :NORMS

    module_function

    # Computes the one-dimensional discrete Fourier transform.
    # @param a [Cumo::NArray] input array
    # @param n [Integer, nil] length of the transformed axis; the input is cropped or zero-padded to it
    # @param axis [Integer] axis to transform
    # @param norm [String, Symbol, nil] "backward" (the default), "ortho" or "forward"
    # @return [Cumo::SComplex, Cumo::DComplex]
    def fft(a, n: nil, axis: -1, norm: nil)
      c2cn(a, one_length(n), [axis], norm, false)
    end

    # Computes the one-dimensional inverse discrete Fourier transform.
    # @param (see .fft)
    # @return [Cumo::SComplex, Cumo::DComplex]
    def ifft(a, n: nil, axis: -1, norm: nil)
      c2cn(a, one_length(n), [axis], norm, true)
    end

    # Computes the one-dimensional discrete Fourier transform of a real input,
    # answering the n / 2 + 1 non-negative frequencies.
    # @param (see .fft)
    # @return [Cumo::SComplex, Cumo::DComplex]
    # @raise [TypeError] if the input is complex
    def rfft(a, n: nil, axis: -1, norm: nil)
      r2cn(a, one_length(n), [axis], norm)
    end

    # Computes the inverse of rfft.
    # @param a [Cumo::NArray] input array, the non-negative frequencies
    # @param n [Integer, nil] length of the output along the axis; 2 * (m - 1) for an input of m points by default
    # @param axis [Integer] axis to transform
    # @param norm [String, Symbol, nil] "backward" (the default), "ortho" or "forward"
    # @return [Cumo::SFloat, Cumo::DFloat]
    def irfft(a, n: nil, axis: -1, norm: nil)
      c2rn(a, one_length(n), [axis], norm, false)
    end

    # Computes the two-dimensional discrete Fourier transform.
    # @param a [Cumo::NArray] input array
    # @param s [Array<Integer>, nil] lengths of the transformed axes, each cropped or zero-padded to it; -1 keeps one
    # @param axes [Array<Integer>] axes to transform, the last two by default
    # @param norm [String, Symbol, nil] "backward" (the default), "ortho" or "forward"
    # @return [Cumo::SComplex, Cumo::DComplex, Cumo::NArray] the input itself if axes is empty
    def fft2(a, s: nil, axes: [-2, -1], norm: nil)
      c2cn(a, s, axes, norm, false)
    end

    # Computes the two-dimensional inverse discrete Fourier transform.
    # @param (see .fft2)
    # @return (see .fft2)
    def ifft2(a, s: nil, axes: [-2, -1], norm: nil)
      c2cn(a, s, axes, norm, true)
    end

    # Computes the N-dimensional discrete Fourier transform.
    # @param a [Cumo::NArray] input array
    # @param s [Array<Integer>, nil] lengths of the transformed axes, each cropped or zero-padded to it; -1 keeps one
    # @param axes [Array<Integer>, nil] axes to transform; every axis by default, or the last s.size when only s is given
    # @param norm [String, Symbol, nil] "backward" (the default), "ortho" or "forward"
    # @return (see .fft2)
    def fftn(a, s: nil, axes: nil, norm: nil)
      c2cn(a, s, axes, norm, false)
    end

    # Computes the N-dimensional inverse discrete Fourier transform.
    # @param (see .fftn)
    # @return (see .fft2)
    def ifftn(a, s: nil, axes: nil, norm: nil)
      c2cn(a, s, axes, norm, true)
    end

    # Computes the two-dimensional discrete Fourier transform of a real input.
    # The last of the axes answers its s / 2 + 1 non-negative frequencies.
    # @param (see .fft2)
    # @return [Cumo::SComplex, Cumo::DComplex]
    # @raise [TypeError] if the input is complex
    def rfft2(a, s: nil, axes: [-2, -1], norm: nil)
      r2cn(a, s, axes, norm)
    end

    # Computes the inverse of rfft2.
    # @param a [Cumo::NArray] input array, the non-negative frequencies along the last of the axes
    # @param s [Array<Integer>, nil] lengths of the output along the axes; the last is 2 * (m - 1) by default
    # @param axes [Array<Integer>] axes to transform, the last two by default
    # @param norm [String, Symbol, nil] "backward" (the default), "ortho" or "forward"
    # @return [Cumo::SFloat, Cumo::DFloat]
    def irfft2(a, s: nil, axes: [-2, -1], norm: nil)
      c2rn(a, s, axes, norm, false)
    end

    # Computes the N-dimensional discrete Fourier transform of a real input.
    # The last of the axes answers its s / 2 + 1 non-negative frequencies.
    # @param (see .fftn)
    # @return [Cumo::SComplex, Cumo::DComplex]
    # @raise [TypeError] if the input is complex
    def rfftn(a, s: nil, axes: nil, norm: nil)
      r2cn(a, s, axes, norm)
    end

    # Computes the inverse of rfftn.
    # @param a [Cumo::NArray] input array, the non-negative frequencies along the last of the axes
    # @param s [Array<Integer>, nil] lengths of the output along the axes; the last is 2 * (m - 1) by default
    # @param axes [Array<Integer>, nil] axes to transform; every axis by default, or the last s.size when only s is given
    # @param norm [String, Symbol, nil] "backward" (the default), "ortho" or "forward"
    # @return [Cumo::SFloat, Cumo::DFloat]
    def irfftn(a, s: nil, axes: nil, norm: nil)
      c2rn(a, s, axes, norm, false)
    end

    # Computes the discrete Fourier transform of a signal with Hermitian
    # symmetry, from the half of it, answering its spectrum, which is real.
    # @param a [Cumo::NArray] input array, the half of a Hermitian signal
    # @param n [Integer, nil] length of the output along the axis; 2 * (m - 1) for an input of m points by default
    # @param axis [Integer] axis to transform
    # @param norm [String, Symbol, nil] "backward" (the default), "ortho" or "forward"
    # @return [Cumo::SFloat, Cumo::DFloat]
    def hfft(a, n: nil, axis: -1, norm: nil)
      check_array(a)
      norm = swap_norm(norm)
      x = complex_type(a).cast(a)
      x = x.dup if x.equal?(a)
      x.inplace.conj
      c2rn(x, one_length(n), [axis], norm, true)
    end

    # Computes the inverse of hfft from a real input.
    # @param (see .fft)
    # @return [Cumo::SComplex, Cumo::DComplex]
    # @raise [TypeError] if the input is complex
    def ihfft(a, n: nil, axis: -1, norm: nil)
      out = r2cn(a, one_length(n), [axis], swap_norm(norm))
      out.inplace.conj
      out
    end

    # Answers the sample frequencies of an n-point discrete Fourier
    # transform, in cycles per unit of the sample spacing d.
    # @param n [Integer] window length
    # @param d [Numeric] sample spacing
    # @return [Cumo::DFloat]
    def fftfreq(n, d: 1.0)
      scale = freq_scale(n, d)
      k = DFloat.new(n).seq
      half = ((n - 1) / 2) + 1
      k[half..] -= n if n > half
      k * scale
    end

    # Answers the sample frequencies of an n-point rfft.
    # @param (see .fftfreq)
    # @return [Cumo::DFloat]
    def rfftfreq(n, d: 1.0)
      scale = freq_scale(n, d)
      DFloat.new((n / 2) + 1).seq * scale
    end

    # Moves the zero-frequency term to the middle of the spectrum.
    # @param x [Cumo::NArray] input array
    # @param axes [Integer, Array<Integer>, nil] axes to shift, every axis by default
    # @return [Cumo::NArray] a new array of the type of the input
    def fftshift(x, axes: nil)
      roll_half(x, axes, 1)
    end

    # Undoes fftshift.
    # @param (see .fftshift)
    # @return (see .fftshift)
    def ifftshift(x, axes: nil)
      roll_half(x, axes, -1)
    end

    # Convolves two N-dimensional arrays through the discrete Fourier
    # transform, answering their full discrete linear convolution, as
    # scipy.signal.fftconvolve does in its "full" mode.
    # @param a [Cumo::NArray] first input
    # @param b [Cumo::NArray] second input, with as many dimensions as a
    # @return [Cumo::NArray] real if both inputs are real, and single precision if both are
    def fftconvolve(a, b)
      check_array(a)
      check_array(b)
      return a * b if a.ndim.zero? && b.ndim.zero?
      raise ArgumentError, 'in1 and in2 should have the same dimensionality' if a.ndim != b.ndim
      return DFloat.new(0) if a.empty? || b.empty?

      s = a.shape.zip(b.shape).map! { |m, n| m + n - 1 }
      axes = (0...a.ndim).to_a
      return irfftn(rfftn(a, s: s, axes: axes) * rfftn(b, s: s, axes: axes), s: s, axes: axes) unless complex?(a) || complex?(b)

      ifftn(fftn(a, s: s, axes: axes) * fftn(b, s: s, axes: axes), axes: axes)
    end

    def c2cn(a, s, axes, norm, inverse)
      check_array(a)
      norm = check_norm(norm)
      axes, s = check_axes(a, s, axes, false)
      return a if axes.empty?
      if axes.uniq.size < axes.size
        return (axes.size - 1).downto(0).reduce(a) { |acc, elem| c2cn(acc, [s[elem]], [axes[elem]], norm, inverse) }
      end

      x, = fit(a, complex_type(a), axes, s)
      sorted = axes.sort
      scale(across(x, sorted) { |b| c2c_axes(b, trailing(b, sorted.size), inverse) }, s.reduce(:*), norm, inverse)
    end

    def r2cn(a, s, axes, norm)
      check_array(a)
      raise TypeError, "expected a real array, got a #{a.class}" if a.is_a?(SComplex) || a.is_a?(DComplex)

      norm = check_norm(norm)
      axes, s = check_axes(a, s, axes, false)
      raise ArgumentError, 'a real transform needs an axis' if axes.empty?

      if axes.uniq.size < axes.size
        x = r2cn(a, [s[-1]], [axes[-1]], norm)
        return (axes.size - 2).downto(0).reduce(x) { |acc, elem| c2cn(acc, [s[elem]], [axes[elem]], norm, false) }
      end

      x, = fit(a, real_type(a), axes, s)
      order = axes[0...-1].sort << axes[-1]
      scale(across(x, order) { |b| r2c_trailing(b, order.size) }, s.reduce(:*), norm, false)
    end

    def c2rn(a, s, axes, norm, own)
      check_array(a)
      norm = check_norm(norm)
      axes, s = check_axes(a, s, axes, true)
      raise ArgumentError, 'a real transform needs an axis' if axes.empty?

      if axes.uniq.size < axes.size
        x = (0...(axes.size - 1)).reduce(a) { |acc, elem| c2cn(acc, [s[elem]], [axes[elem]], norm, true) }
        return c2rn(x, [s[-1]], [axes[-1]], norm, !x.equal?(a))
      end

      half = s.dup
      half[-1] = (s[-1] / 2) + 1
      x, fresh = fit(a, complex_type(a), axes, half)
      order = axes[0...-1].sort << axes[-1]
      out = across(x, order) { |b| c2r_trailing(b, order.size, s[-1], own || fresh || !b.equal?(x)) }
      scale(out, s.reduce(:*), norm, true)
    end

    def check_array(a)
      raise NotImplementedError, 'Cumo is built without cuFFT' unless CUDA::Cufft.available?

      check_narray(a)
    end

    def check_narray(a)
      raise TypeError, "expected a Cumo::NArray, got a #{a.class}" unless a.is_a?(NArray)
    end

    def check_norm(norm)
      return 'backward' if norm.nil?

      name = norm.to_s if norm.is_a?(String) || norm.is_a?(Symbol)
      unless NORMS.include?(name)
        raise ArgumentError, %(Invalid norm value #{norm.inspect}; should be "backward", "ortho" or "forward".)
      end

      name
    end

    def swap_norm(norm)
      case check_norm(norm)
      when 'backward' then 'forward'
      when 'forward' then 'backward'
      else 'ortho'
      end
    end

    def freq_scale(n, d)
      raise ArgumentError, 'n should be an integer' unless n.is_a?(Integer)
      raise ArgumentError, 'n should not be negative' if n.negative?

      spacing = n * Float(d)
      raise ZeroDivisionError, 'float division by zero' if spacing.zero?

      1.0 / spacing
    end

    def roll_half(x, axes, sign)
      check_narray(x)

      axes = (0...x.ndim).to_a if axes.nil?
      axes = [axes] if axes.is_a?(Integer)
      raise TypeError, "axes must be an Integer or an Array, got a #{axes.class}" unless axes.is_a?(Array)

      shifts = Hash.new(0)
      axes.each do |ax|
        i = check_axis(x, ax)
        shifts[i] += sign * (x.shape[i] / 2)
      end
      return x.dup if x.empty?

      blocks = x.shape.each_with_index.map do |n, i|
        k = shifts[i] % n
        k.zero? ? [[true, true]] : [[(n - k)...n, 0...k], [0...(n - k), k...n]]
      end
      return x.dup if blocks.all? { |b| b.size == 1 }

      out = x.class.new(*x.shape).allocate
      blocks.reduce([[]]) { |acc, elem| acc.product(elem).map! { |prefix, pair| prefix + [pair] } }.each do |pairs|
        out[*pairs.map(&:last)] = x[*pairs.map(&:first)]
      end
      out
    end

    def check_axis(a, axis)
      raise TypeError, "axis must be an Integer, got a #{axis.class}" unless axis.is_a?(Integer)

      i = axis.negative? ? axis + a.ndim : axis
      raise NArray::DimensionError, "invalid axis (#{axis} for #{a.ndim}-dimension)" if i.negative? || i >= a.ndim

      i
    end

    def check_axes(a, s, axes, inverse_real)
      raise TypeError, "s must be an Array, got a #{s.class}" unless s.nil? || s.is_a?(Array)
      raise TypeError, "axes must be an Array, got a #{axes.class}" unless axes.nil? || axes.is_a?(Array)

      axes = (s.nil? ? (0...a.ndim) : (-s.size...0)).to_a if axes.nil?
      axes = axes.map { |ax| check_axis(a, ax) }
      lengths = s.nil? ? axes.map { |ax| a.shape[ax] } : s.dup
      raise ArgumentError, 'Shape and axes have different lengths.' unless lengths.size == axes.size

      lengths[-1] = 2 * (a.shape[axes[-1]] - 1) if inverse_real && s.nil? && !axes.empty?
      lengths = lengths.each_with_index.map { |n, i| check_length(n == -1 ? a.shape[axes[i]] : n) }
      [axes, lengths]
    end

    def check_length(n)
      raise TypeError, "n must be an Integer, got a #{n.class}" unless n.is_a?(Integer)
      raise ArgumentError, "Invalid number of FFT data points (#{n}) specified." if n < 1

      n
    end

    def one_length(n)
      n.nil? ? nil : [check_length(n)]
    end

    def complex?(a)
      a.is_a?(SComplex) || a.is_a?(DComplex)
    end

    def complex_type(a)
      case a
      when SComplex, SFloat, HFloat, BFloat then SComplex
      else DComplex
      end
    end

    def real_type(a)
      case a
      when SFloat, HFloat, BFloat then SFloat
      else DFloat
      end
    end

    def fit(a, type, axes, lengths)
      index = Array.new(a.ndim, true)
      shape = a.shape
      pad = false
      axes.each_with_index do |ax, i|
        index[ax] = 0...lengths[i] if lengths[i] < a.shape[ax]
        pad ||= lengths[i] > a.shape[ax]
        shape[ax] = lengths[i]
      end
      cropped = index.all?(true) ? a : a[*index]
      x = type.cast(cropped)
      return [x, !x.equal?(cropped)] unless pad

      z = type.zeros(*shape)
      z[*x.shape.map { |m| 0...m }] = x unless x.empty?
      [z, true]
    end

    def across(x, axes)
      perm = ((0...x.ndim).to_a - axes) + axes
      return yield(x.contiguous? ? x : x.dup) if perm == (0...x.ndim).to_a

      back = Array.new(perm.size)
      perm.each_with_index { |ax, i| back[ax] = i }
      yield(x.transpose(*perm).dup).transpose(*back).dup
    end

    def trailing(x, k)
      ((x.ndim - k)...x.ndim).to_a
    end

    def c2c_axes(x, axes, inverse)
      axes.reverse.each_slice(3).reduce(x) do |acc, chunk|
        across(acc, chunk.reverse) { |b| CUDA::Cufft.__send__(:c2c, b, chunk.size, inverse) }
      end
    end

    def r2c_trailing(x, k)
      j = [k, 3].min
      c2c_axes(CUDA::Cufft.__send__(:r2c, x, j), trailing(x, k)[0...(k - j)], false)
    end

    def c2r_trailing(x, k, n, overwrite)
      j = n == 1 ? 1 : [k, 3].min
      return CUDA::Cufft.__send__(:c2r, x, j, n, overwrite) if j == k

      CUDA::Cufft.__send__(:c2r, c2c_axes(x, trailing(x, k)[0...(k - j)], true), j, n, true)
    end

    def scale(out, n, norm, inverse)
      divisor = case norm
                when 'backward' then n if inverse
                when 'forward' then n unless inverse
                else Math.sqrt(n)
                end
      out.inplace / divisor if divisor
      out
    end

    private_class_method :c2cn, :r2cn, :c2rn, :check_array, :check_narray, :check_norm, :swap_norm, :freq_scale, :roll_half,
                         :check_axis, :check_axes,
                         :check_length, :one_length, :complex?, :complex_type, :real_type, :fit, :across, :trailing,
                         :c2c_axes, :r2c_trailing, :c2r_trailing, :scale
  end
end
