# frozen_string_literal: true

require 'cumo'

# Provide compatibility layers with numo/linalg
module Cumo
  module Blas
    class << self
      def gemm(a, *args, **kwargs)
        a.gemm(*args, **kwargs)
      end
    end
  end

  # Linear algebra on the GPU, with the functions of Numo::Linalg from
  # numo-linalg-alt under the same names, arguments and errors.
  module Linalg
    # Raised where Numo::Linalg raises Numo::Linalg::LapackError.
    class LapackError < StandardError; end

    BLAS_CLASSES = { s: SFloat, d: DFloat, c: SComplex, z: DComplex }.freeze
    INTEGER_CLASSES = [Bit, Int64, Int32, Int16, Int8, UInt64, UInt32, UInt16, UInt8].freeze
    private_constant :BLAS_CLASSES, :INTEGER_CLASSES

    module_function

    # Returns the BLAS and LAPACK type the arrays are computed in: 's' for
    # SFloat, 'd' for DFloat, 'c' for SComplex and 'z' for DComplex. Integers
    # and Bit are computed in DFloat.
    #
    # @param args [Array<Cumo::NArray, Array>]
    # @return [String]
    # @raise [TypeError] if none of the arrays has such a type
    def blas_char(*args)
      type = 'n'
      args.each do |arg|
        arg = NArray.asarray(arg) if arg.is_a?(Array)
        klass = arg.class
        if INTEGER_CLASSES.include?(klass)
          type = 'd' if type == 'n'
        elsif klass == DFloat
          type = %w[c z].include?(type) ? 'z' : 'd'
        elsif klass == SFloat
          type = 's' if type == 'n'
        elsif klass == DComplex
          type = 'z'
        elsif klass == SComplex
          if %w[n s].include?(type)
            type = 'c'
          elsif type == 'd'
            type = 'z'
          end
        end
      end
      raise TypeError, 'invalid data type for BLAS/LAPACK' if type == 'n'

      type
    end

    # Returns the dot product of two vectors, a matrix and a vector, or two
    # matrices. The dot product of two vectors is a zero-dimensional array, or
    # a Float or Complex in compatible mode.
    #
    # @param a [Cumo::NArray, Array] 1- or 2-dimensional
    # @param b [Cumo::NArray, Array] 1- or 2-dimensional
    # @return [Cumo::NArray]
    def dot(a, b)
      a = NArray.asarray(a) unless a.is_a?(NArray)
      b = NArray.asarray(b) unless b.is_a?(NArray)
      klass = BLAS_CLASSES[blas_char(a, b).to_sym]
      a = klass.cast(a)
      b = klass.cast(b)

      if a.ndim == 1
        b.ndim == 1 ? blas_dot(a, b) : blas_gemv(b, a, trans: true)
      elsif b.ndim == 1
        blas_gemv(a, b)
      else
        blas_gemm(a, b)
      end
    end

    # Returns the product of two matrices.
    #
    # @param a [Cumo::NArray, Array] 2-dimensional
    # @param b [Cumo::NArray, Array] 2-dimensional
    # @return [Cumo::NArray]
    def matmul(a, b)
      klass = BLAS_CLASSES[blas_char(a, b).to_sym]
      blas_gemm(klass.cast(a), klass.cast(b))
    end

    def blas_dot(x, y)
      raise ArgumentError, 'x must not be empty' if x.empty? || y.empty?
      raise ArgumentError, 'x and y must have same size' if x.size != y.size

      x.mulsum(y)
    end

    def blas_gemv(a, x, trans: false)
      raise ArgumentError, 'a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'x must be 1-dimensional' if x.ndim != 1
      raise ArgumentError, 'a must not be empty' if a.empty?
      raise ArgumentError, 'x must not be empty' if x.empty?

      n = trans ? a.shape[0] : a.shape[1]
      raise NArray::ShapeError, "shape1[1](=#{n}) != shape2[0](=#{x.size})" if n != x.size

      trans ? x[:new, true].gemm(a).flatten : a.gemm(x[true, :new]).flatten
    end

    def blas_gemm(a, b)
      raise ArgumentError, 'a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'b must be 2-dimensional' if b.ndim != 2
      raise ArgumentError, 'a must not be empty' if a.empty?
      raise ArgumentError, 'b must not be empty' if b.empty?
      raise NArray::ShapeError, "shape1[1](=#{a.shape[1]}) != shape2[0](=#{b.shape[0]})" if a.shape[1] != b.shape[0]

      a.gemm(b)
    end

    private_class_method :blas_dot, :blas_gemv, :blas_gemm
  end
end
