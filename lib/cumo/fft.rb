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
      scale(along(x, axis) { |b| CUDA::Cufft.__send__(:r2c, b, 1) }, n, norm, false)
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
      scale(along(x, axis) { |b| CUDA::Cufft.__send__(:c2r, b, 1, n, own || !b.equal?(x)) }, n, norm, true)
    end

    def c2c(a, n, axis, norm, inverse)
      check_array(a)
      norm = check_norm(norm)
      axis = check_axis(a, axis)
      n = check_length(n || a.shape[axis])
      x = resize(complex_type(a).cast(a), axis, n)
      scale(along(x, axis) { |b| CUDA::Cufft.__send__(:c2c, b, 1, inverse) }, n, norm, inverse)
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

    def along(x, axis)
      last = x.ndim - 1
      if axis == last
        yield(x.contiguous? ? x : x.dup)
      else
        yield(x.swapaxes(axis, last).dup).swapaxes(axis, last).dup
      end
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

    private_class_method :c2c, :check_array, :check_norm, :check_axis, :check_length,
                         :complex_type, :real_type, :resize, :along, :scale
  end
end
