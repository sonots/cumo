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
      c2c(a, n, axis, norm, false)
    end

    # Computes the one-dimensional inverse discrete Fourier transform.
    # @param (see .fft)
    # @return [Cumo::SComplex, Cumo::DComplex]
    def ifft(a, n: nil, axis: -1, norm: nil)
      c2c(a, n, axis, norm, true)
    end

    # Computes the one-dimensional discrete Fourier transform of a real input,
    # answering the n / 2 + 1 non-negative frequencies.
    # @param (see .fft)
    # @return [Cumo::SComplex, Cumo::DComplex]
    # @raise [TypeError] if the input is complex
    def rfft(a, n: nil, axis: -1, norm: nil)
      check_array(a)
      raise TypeError, "rfft takes a real array, got a #{a.class}" if a.is_a?(SComplex) || a.is_a?(DComplex)

      norm = check_norm(norm)
      axis = check_axis(a, axis)
      n = check_length(n || a.shape[axis])
      x = resize(real_type(a).cast(a), axis, n)
      scale(across(x, [axis]) { |b| CUDA::Cufft.__send__(:r2c, b, 1) }, n, norm, false)
    end

    # Computes the inverse of rfft.
    # @param a [Cumo::NArray] input array, the non-negative frequencies
    # @param n [Integer, nil] length of the output along the axis; 2 * (m - 1) for an input of m points by default
    # @param axis [Integer] axis to transform
    # @param norm [String, Symbol, nil] "backward" (the default), "ortho" or "forward"
    # @return [Cumo::SFloat, Cumo::DFloat]
    def irfft(a, n: nil, axis: -1, norm: nil)
      check_array(a)
      norm = check_norm(norm)
      axis = check_axis(a, axis)
      n = check_length(n || (2 * (a.shape[axis] - 1)))
      m = (n / 2) + 1
      x = complex_type(a).cast(a)
      own = !x.equal?(a) || x.shape[axis] < m
      x = resize(x, axis, m)
      scale(across(x, [axis]) { |b| CUDA::Cufft.__send__(:c2r, b, 1, n, own || !b.equal?(x)) }, n, norm, true)
    end

    # Computes the two-dimensional discrete Fourier transform.
    # @param a [Cumo::NArray] input array
    # @param s [Array<Integer>, nil] lengths of the transformed axes, each cropped or zero-padded to it; -1 keeps one
    # @param axes [Array<Integer>] axes to transform
    # @param norm [String, Symbol, nil] "backward" (the default), "ortho" or "forward"
    # @return [Cumo::SComplex, Cumo::DComplex]
    def fft2(a, s: nil, axes: [-2, -1], norm: nil)
      c2cn(a, s, axes, norm, false)
    end

    # Computes the two-dimensional inverse discrete Fourier transform.
    # @param (see .fft2)
    # @return [Cumo::SComplex, Cumo::DComplex]
    def ifft2(a, s: nil, axes: [-2, -1], norm: nil)
      c2cn(a, s, axes, norm, true)
    end

    # Computes the N-dimensional discrete Fourier transform, over every axis
    # by default, or over the last s.size axes when only s is given.
    # @param (see .fft2)
    # @return [Cumo::SComplex, Cumo::DComplex]
    def fftn(a, s: nil, axes: nil, norm: nil)
      c2cn(a, s, axes, norm, false)
    end

    # Computes the N-dimensional inverse discrete Fourier transform.
    # @param (see .fftn)
    # @return [Cumo::SComplex, Cumo::DComplex]
    def ifftn(a, s: nil, axes: nil, norm: nil)
      c2cn(a, s, axes, norm, true)
    end

    # Computes the two-dimensional discrete Fourier transform of a real input.
    # @param (see .fft2)
    # @return [Cumo::SComplex, Cumo::DComplex]
    def rfft2(a, s: nil, axes: [-2, -1], norm: nil)
      rfftn(a, s: s, axes: axes, norm: norm)
    end

    # Computes the inverse of rfft2.
    # @param (see .irfftn)
    # @return [Cumo::SFloat, Cumo::DFloat]
    def irfft2(a, s: nil, axes: [-2, -1], norm: nil)
      irfftn(a, s: s, axes: axes, norm: norm)
    end

    # Computes the N-dimensional discrete Fourier transform of a real input.
    # The last of the axes answers its s / 2 + 1 non-negative frequencies.
    # @param (see .fftn)
    # @return [Cumo::SComplex, Cumo::DComplex]
    # @raise [TypeError] if the input is complex
    def rfftn(a, s: nil, axes: nil, norm: nil)
      check_array(a)
      raise TypeError, "rfftn takes a real array, got a #{a.class}" if a.is_a?(SComplex) || a.is_a?(DComplex)

      norm = check_norm(norm)
      axes, s = check_axes(a, s, axes, false)
      raise ArgumentError, 'rfftn needs an axis to transform' if axes.empty?

      if axes.uniq.size < axes.size
        x = rfft(a, n: s[-1], axis: axes[-1], norm: norm)
        return (axes.size - 2).downto(0).reduce(x) { |acc, elem| c2c(acc, s[elem], axes[elem], norm, false) }
      end

      x = real_type(a).cast(a)
      axes.each_with_index { |ax, i| x = resize(x, ax, s[i]) }
      scale(across(x, axes) { |b| r2c_trailing(b, axes.size) }, s.reduce(:*), norm, false)
    end

    # Computes the inverse of rfftn.
    # @param a [Cumo::NArray] input array, the non-negative frequencies along the last of the axes
    # @param s [Array<Integer>, nil] lengths of the output along the axes; the last is 2 * (m - 1) by default
    # @param axes [Array<Integer>, nil] axes to transform, every axis by default
    # @param norm [String, Symbol, nil] "backward" (the default), "ortho" or "forward"
    # @return [Cumo::SFloat, Cumo::DFloat]
    def irfftn(a, s: nil, axes: nil, norm: nil)
      check_array(a)
      norm = check_norm(norm)
      axes, s = check_axes(a, s, axes, true)
      raise ArgumentError, 'irfftn needs an axis to transform' if axes.empty?

      if axes.uniq.size < axes.size
        x = (0...(axes.size - 1)).reduce(a) { |acc, elem| c2c(acc, s[elem], axes[elem], norm, true) }
        return irfft(x, n: s[-1], axis: axes[-1], norm: norm)
      end

      half = s.dup
      half[-1] = (s[-1] / 2) + 1
      x = complex_type(a).cast(a)
      own = !x.equal?(a)
      axes.each_with_index do |ax, i|
        own ||= x.shape[ax] < half[i]
        x = resize(x, ax, half[i])
      end
      out = across(x, axes) { |b| c2r_trailing(b, axes.size, s[-1], own || !b.equal?(x)) }
      scale(out, s.reduce(:*), norm, true)
    end

    def c2c(a, n, axis, norm, inverse)
      check_array(a)
      norm = check_norm(norm)
      axis = check_axis(a, axis)
      n = check_length(n || a.shape[axis])
      x = resize(complex_type(a).cast(a), axis, n)
      scale(across(x, [axis]) { |b| CUDA::Cufft.__send__(:c2c, b, 1, inverse) }, n, norm, inverse)
    end

    def c2cn(a, s, axes, norm, inverse)
      check_array(a)
      norm = check_norm(norm)
      axes, s = check_axes(a, s, axes, false)
      return a if axes.empty?
      return (axes.size - 1).downto(0).reduce(a) { |acc, elem| c2c(acc, s[elem], axes[elem], norm, inverse) } if axes.uniq.size < axes.size

      x = complex_type(a).cast(a)
      axes.each_with_index { |ax, i| x = resize(x, ax, s[i]) }
      sorted = axes.sort
      scale(across(x, sorted) { |b| c2c_trailing(b, sorted.size, inverse) }, s.reduce(:*), norm, inverse)
    end

    def check_array(a)
      raise NotImplementedError, 'Cumo is built without cuFFT' unless CUDA::Cufft.available?
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

    def check_axis(a, axis)
      raise TypeError, "axis must be an Integer, got a #{axis.class}" unless axis.is_a?(Integer)

      i = axis.negative? ? axis + a.ndim : axis
      raise NArray::DimensionError, "invalid axis (#{axis} for #{a.ndim}-dimension)" if i.negative? || i >= a.ndim

      i
    end

    def check_axes(a, s, axes, inverse_real)
      shapeless = s.nil?
      raise TypeError, "s must be an Array, got a #{s.class}" unless shapeless || s.is_a?(Array)

      axes = (shapeless ? (0...a.ndim) : (-s.size...0)).to_a if axes.nil?
      axes = Array(axes).map { |ax| check_axis(a, ax) }
      s = shapeless ? axes.map { |ax| a.shape[ax] } : s.dup
      raise ArgumentError, 'Shape and axes have different lengths.' unless s.size == axes.size

      s[-1] = 2 * (a.shape[axes[-1]] - 1) if inverse_real && shapeless && !axes.empty?
      s = s.each_with_index.map { |n, i| check_length(n == -1 ? a.shape[axes[i]] : n) }
      [axes, s]
    end

    def check_length(n)
      raise TypeError, "n must be an Integer, got a #{n.class}" unless n.is_a?(Integer)
      raise ArgumentError, "Invalid number of FFT data points (#{n}) specified." if n < 1

      n
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

    def resize(x, axis, n)
      m = x.shape[axis]
      return x if m == n

      index = Array.new(x.ndim, true)
      if m > n
        index[axis] = 0...n
        return x[*index]
      end

      shape = x.shape
      shape[axis] = n
      z = x.class.zeros(*shape)
      index[axis] = 0...m
      z[*index] = x if m.positive?
      z
    end

    def across(x, axes)
      perm = ((0...x.ndim).to_a - axes) + axes
      return yield(x.contiguous? ? x : x.dup) if perm == (0...x.ndim).to_a

      back = Array.new(perm.size)
      perm.each_with_index { |ax, i| back[ax] = i }
      yield(x.transpose(*perm).dup).transpose(*back).dup
    end

    def c2c_trailing(x, k, inverse)
      return CUDA::Cufft.__send__(:c2c, x, k, inverse) if k <= 3

      c2c_before_last3(CUDA::Cufft.__send__(:c2c, x, 3, inverse), k - 3, inverse)
    end

    def c2c_before_last3(x, k, inverse)
      across(x, ((x.ndim - 3 - k)...(x.ndim - 3)).to_a) { |b| c2c_trailing(b, k, inverse) }
    end

    def r2c_trailing(x, k)
      return CUDA::Cufft.__send__(:r2c, x, k) if k <= 3

      c2c_before_last3(CUDA::Cufft.__send__(:r2c, x, 3), k - 3, false)
    end

    def c2r_trailing(x, k, n, overwrite)
      return CUDA::Cufft.__send__(:c2r, x, k, n, overwrite) if k <= 3

      CUDA::Cufft.__send__(:c2r, c2c_before_last3(x, k - 3, true), 3, n, true)
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

    private_class_method :c2c, :c2cn, :check_array, :check_norm, :check_axis, :check_axes, :check_length,
                         :complex_type, :real_type, :resize, :across, :c2c_trailing, :c2c_before_last3,
                         :r2c_trailing, :c2r_trailing, :scale
  end
end
