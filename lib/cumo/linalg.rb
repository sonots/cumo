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
    # @raise [TypeError] if none of the arrays has such a type, or one is
    #   HFloat or BFloat
    def blas_char(*args)
      type = 'n'
      args.each do |arg|
        arg = NArray.asarray(arg) if arg.is_a?(Array)
        klass = arg.class
        raise TypeError, 'invalid data type for BLAS/LAPACK' if [HFloat, BFloat].include?(klass)

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
    # matrices. The dot product of two vectors is a Float or a Complex, which
    # waits for the GPU.
    #
    # @param a [Cumo::NArray, Array] 1- or 2-dimensional
    # @param b [Cumo::NArray, Array] 1- or 2-dimensional
    # @return [Cumo::NArray, Float, Complex]
    def dot(a, b)
      a, b = cast_to_blas_class(a, b)
      if a.ndim == 1
        if b.ndim == 1
          check_dot(a, b)
          c = a.mulsum(b)
          return c.is_a?(NArray) ? c.extract_cpu : c
        end
        check_gemv(b, a, trans: true)
      elsif b.ndim == 1
        check_gemv(a, b)
      else
        check_gemm(a, b)
      end
      a.dot(b)
    end

    # Returns the product of two matrices.
    #
    # @param a [Cumo::NArray, Array] 2-dimensional
    # @param b [Cumo::NArray, Array] 2-dimensional
    # @return [Cumo::NArray]
    def matmul(a, b)
      a, b = cast_to_blas_class(a, b)
      check_gemm(a, b)
      a.dot(b)
    end

    def cast_to_blas_class(a, b)
      a = NArray.asarray(a) unless a.is_a?(NArray)
      b = NArray.asarray(b) unless b.is_a?(NArray)
      klass = BLAS_CLASSES[blas_char(a, b).to_sym]
      [klass.cast(a), klass.cast(b)]
    end

    def check_dot(x, y)
      raise ArgumentError, 'x must not be empty' if x.empty? || y.empty?
      raise ArgumentError, 'x and y must have same size' if x.size != y.size
    end

    def check_gemv(a, x, trans: false)
      raise ArgumentError, 'a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'x must be 1-dimensional' if x.ndim != 1
      raise ArgumentError, 'a must not be empty' if a.empty?
      raise ArgumentError, 'x must not be empty' if x.empty?

      n = trans ? a.shape[0] : a.shape[1]
      raise NArray::ShapeError, "shape1[1](=#{n}) != shape2[0](=#{x.size})" if n != x.size
    end

    def check_gemm(a, b)
      raise ArgumentError, 'a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'b must be 2-dimensional' if b.ndim != 2
      raise ArgumentError, 'a must not be empty' if a.empty?
      raise ArgumentError, 'b must not be empty' if b.empty?
      raise NArray::ShapeError, "shape1[1](=#{a.shape[1]}) != shape2[0](=#{b.shape[0]})" if a.shape[1] != b.shape[0]
    end

    private_class_method :cast_to_blas_class, :check_dot, :check_gemv, :check_gemm
  end
end
