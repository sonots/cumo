# frozen_string_literal: true

require 'cumo/fft'

module Cumo
  # The functions of Numo::Pocketfft from numo-pocketfft, computed on the GPU
  # by Cumo::FFT. As there, they take one array and nothing else, the 1- and
  # 2-dimensional ones take only 1- and 2-dimensional arrays, the input is
  # cast to double precision, and what numo-pocketfft refuses raises the same
  # ArgumentError.
  module Pocketfft
    DIMENSIONS = { 1 => 'one', 2 => 'two' }.freeze
    private_constant :DIMENSIONS

    module_function

    # Computes the 1-dimensional discrete Fourier transform.
    # @param a [Cumo::NArray] real or complex 1-dimensional input array
    # @return [Cumo::DComplex]
    def fft(a)
      FFT.fft(DComplex.cast(check(a, 1)))
    end

    # Computes the 1-dimensional inverse discrete Fourier transform.
    # @param a [Cumo::NArray] complex 1-dimensional input array
    # @return [Cumo::DComplex]
    def ifft(a)
      FFT.ifft(DComplex.cast(check(a, 1)))
    end

    # Computes the 2-dimensional discrete Fourier transform.
    # @param a [Cumo::NArray] real or complex 2-dimensional input array
    # @return [Cumo::DComplex]
    def fft2(a)
      FFT.fftn(DComplex.cast(check(a, 2)))
    end

    # Computes the 2-dimensional inverse discrete Fourier transform.
    # @param a [Cumo::NArray] complex 2-dimensional input array
    # @return [Cumo::DComplex]
    def ifft2(a)
      FFT.ifftn(DComplex.cast(check(a, 2)))
    end

    # Computes the N-dimensional discrete Fourier transform.
    # @param a [Cumo::NArray] real or complex input array
    # @return [Cumo::DComplex]
    def fftn(a)
      FFT.fftn(DComplex.cast(check(a)))
    end

    # Computes the N-dimensional inverse discrete Fourier transform.
    # @param a [Cumo::NArray] complex input array
    # @return [Cumo::DComplex]
    def ifftn(a)
      FFT.ifftn(DComplex.cast(check(a)))
    end

    # Computes the 1-dimensional discrete Fourier transform of a real input.
    # @param a [Cumo::NArray] real 1-dimensional input array
    # @return [Cumo::DComplex]
    def rfft(a)
      FFT.rfft(DFloat.cast(check(a, 1)))
    end

    # Computes the inverse of rfft.
    # @param a [Cumo::NArray] complex 1-dimensional input array
    # @return [Cumo::DFloat]
    def irfft(a)
      FFT.irfft(DComplex.cast(check_half(check(a, 1))))
    end

    # Computes the 2-dimensional discrete Fourier transform of a real input.
    # @param a [Cumo::NArray] real 2-dimensional input array
    # @return [Cumo::DComplex]
    def rfft2(a)
      FFT.rfftn(DFloat.cast(check(a, 2)))
    end

    # Computes the inverse of rfft2.
    # @param a [Cumo::NArray] complex 2-dimensional input array
    # @return [Cumo::DFloat]
    def irfft2(a)
      FFT.irfftn(DComplex.cast(check_half(check(a, 2))))
    end

    # Computes the N-dimensional discrete Fourier transform of a real input.
    # @param a [Cumo::NArray] real input array
    # @return [Cumo::DComplex]
    def rfftn(a)
      FFT.rfftn(DFloat.cast(check(a)))
    end

    # Computes the inverse of rfftn.
    # @param a [Cumo::NArray] complex input array
    # @return [Cumo::DFloat]
    def irfftn(a)
      FFT.irfftn(DComplex.cast(check_half(check(a))))
    end

    # Convolves two N-dimensional arrays through the discrete Fourier transform.
    # @param a [Cumo::NArray] first input array
    # @param b [Cumo::NArray] second input array, with as many dimensions as a
    # @return [Cumo::DFloat, Cumo::DComplex] the full discrete linear convolution,
    #   or a * b when every axis of both has one element
    def fftconvolve(a, b)
      raise ArgumentError, 'Expect class of input array to be Cumo::NArray.' unless a.is_a?(NArray) && b.is_a?(NArray)
      raise ArgumentError, 'Expect input array to be non-empty.' if a.empty? || b.empty?
      raise ArgumentError, 'Input arrays should have the same dimensionarity' if a.ndim != b.ndim
      return a * b if (a.shape + b.shape).all?(1)

      type = [a, b].any? { |x| x.is_a?(SComplex) || x.is_a?(DComplex) } ? DComplex : DFloat
      FFT.fftconvolve(type.cast(a), type.cast(b))
    end

    def check(a, ndim = nil)
      raise ArgumentError, 'Expect class of input array to be Cumo::NArray.' unless a.is_a?(NArray)
      raise ArgumentError, 'Expect input array to be non-empty.' if a.empty?
      raise ArgumentError, "Expect input array to be #{DIMENSIONS[ndim]}-dimensional." if ndim && a.ndim != ndim

      a
    end

    def check_half(a)
      raise NArray::ShapeError, 'cannot get element of empty array' if a.shape[-1] == 1

      a
    end

    private_class_method :check, :check_half
  end
end
