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
          return to_ruby(a.mulsum(b))
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

    # Computes the LU factorization of a matrix with partial pivoting,
    # A = P L U, as LAPACK's getrf does.
    #
    # @param a [Cumo::NArray] 2-dimensional
    # @return [Array<Cumo::NArray, Cumo::Int32>] the factors L and U in one
    #   matrix, the unit diagonal of L left out, and the 1-based pivot indices
    def lu_fact(a)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      klass = BLAS_CLASSES[blas_char(a).to_sym]
      lu = klass.new(*a.shape.reverse).store(a.transpose)
      ipiv, info = cusolver(:getrf, lu)
      raise LapackError, "the #{info.abs}-th argument of getrf had illegal value" if info.negative?

      if info.positive?
        warn("the factorization has been completed, but the factor U[#{info - 1}, #{info - 1}] is " \
             'exactly zero, indicating that the matrix is singular.')
      end

      [lu.transpose.dup, Int32.cast(ipiv)]
    end

    # Solves A X = B, A^T X = B or A^H X = B from the LU factorization of A
    # that lu_fact answers.
    #
    # @param lu [Cumo::NArray] the square matrix of factors
    # @param ipiv [Cumo::NArray] the 1-based pivot indices
    # @param b [Cumo::NArray] 1- or 2-dimensional
    # @param trans [String] 'N', 'T' or 'C'
    # @return [Cumo::NArray] X, of the shape of b
    def lu_solve(lu, ipiv, b, trans: 'N')
      raise NArray::ShapeError, 'input array lu must be 2-dimensional' if lu.ndim != 2
      raise NArray::ShapeError, 'input array lu must be square' if lu.shape[0] != lu.shape[1]
      raise NArray::ShapeError, "incompatible dimensions: lu.shape[0] = #{lu.shape[0]} != b.shape[0] = #{b.shape[0]}" if lu.shape[0] != b.shape[0]
      raise ArgumentError, 'trans must be "N", "T", or "C"' unless %w[N T C].include?(trans)

      klass = BLAS_CLASSES[blas_char(lu).to_sym]
      n = lu.shape[0]
      ipiv = NArray.asarray(ipiv) unless ipiv.is_a?(NArray)
      raise ArgumentError, 'input array ipiv must be 1-dimensional' if ipiv.ndim != 1
      raise ArgumentError, "input array ipiv must have #{n} elements" if ipiv.size != n
      raise ArgumentError, 'input array b must be 1 or 2-dimensional' unless [1, 2].include?(b.ndim)

      x = b.ndim == 1 ? klass.new(n).store(b) : klass.new(*b.shape.reverse).store(b.transpose)
      lu = klass.new(n, n).store(lu.transpose)
      info = cusolver(:getrs, lu, Int64.new(n).store(ipiv), x, trans)
      raise LapackError, "the #{info.abs}-th argument of getrs had illegal value" if info.negative?

      b.ndim == 1 ? x : x.transpose.dup
    end

    def cusolver(routine, *args)
      raise NotImplementedError, 'Cumo is built without cuSOLVER' unless CUDA::Cusolver.available?

      CUDA::Cusolver.__send__(routine, *args)
    end

    def to_ruby(x)
      x.is_a?(NArray) ? x.extract_cpu : x
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

    private_class_method :cusolver, :to_ruby, :cast_to_blas_class, :check_dot, :check_gemv, :check_gemm
  end
end
