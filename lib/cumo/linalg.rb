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

      lu, ipiv, info = getrf(BLAS_CLASSES[blas_char(a).to_sym], a)
      raise LapackError, "the #{info.abs}-th argument of getrf had illegal value" if info.negative?

      warn_singular_factor(info) if info.positive?

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
      ipiv = pivots(ipiv, n)
      raise ArgumentError, 'input array b must be 1 or 2-dimensional' unless [1, 2].include?(b.ndim)

      x, info = getrs(klass, to_column_major(klass, lu), ipiv, b, trans)
      raise LapackError, "the #{info.abs}-th argument of getrs had illegal value" if info.negative?

      x
    end

    # Computes the LU factorization of a matrix and returns its separate
    # factors, as scipy.linalg.lu does.
    #
    # @param a [Cumo::NArray] 2-dimensional, of shape [m, n]
    # @param permute_l [Boolean] whether to answer P L in place of P and L
    # @return [Array<Cumo::NArray>] P, L and U, or P L and U
    def lu(a, permute_l: false)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      m, n = a.shape
      k = [m, n].min
      lu, piv = lu_fact(a)
      l = lu.tril.tap { |x| x[x.diag_indices] = 1 }[true, 0...k].dup
      u = lu.triu[0...k, 0...n].dup
      columns = (0...m).to_a
      piv.to_a.each_with_index { |i, j| columns[i - 1], columns[j] = columns[j], columns[i - 1] }
      perm = a.class.eye(m)[true, columns].dup

      permute_l ? [perm.dot(l), u] : [perm, l, u]
    end

    # Computes the inverse of a matrix from the LU factorization lu_fact
    # answers.
    #
    # @param lu [Cumo::NArray] the square matrix of factors
    # @param ipiv [Cumo::NArray] the 1-based pivot indices
    # @return [Cumo::NArray]
    # @raise [LapackError] if the matrix is singular
    def lu_inv(lu, ipiv)
      lu = NArray.asarray(lu) unless lu.is_a?(NArray)
      bchr = blas_char(lu)
      klass = BLAS_CLASSES[bchr.to_sym]
      raise ArgumentError, 'input array a must be 2-dimensional' if lu.ndim != 2
      raise ArgumentError, 'input array a must be square' if lu.shape[0] != lu.shape[1]

      n = lu.shape[0]
      ipiv = pivots(ipiv, n)
      return klass.new(0, 0) if n.zero?

      lu = to_column_major(klass, lu)
      raise LapackError, 'the matrix is singular and its inverse could not be computed' if singular?(lu)

      invert(klass, lu, ipiv, "#{bchr}getri")
    end

    # Solves A X = B for a square matrix A.
    #
    # @param a [Cumo::NArray] the square matrix
    # @param b [Cumo::NArray] 1- or 2-dimensional
    # @param driver [String] accepted and ignored, as numo-linalg-alt does
    # @param uplo [String] accepted and ignored, as numo-linalg-alt does
    # @return [Cumo::NArray] X, of the shape of b
    def solve(a, b, driver: 'gen', uplo: 'U')
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      klass = BLAS_CLASSES[blas_char(a, b).to_sym]
      b = NArray.asarray(b) unless b.is_a?(NArray)
      raise ArgumentError, 'input array b must be 1- or 2-dimensional' unless [1, 2].include?(b.ndim)

      n = a.shape[0]
      raise NArray::ShapeError, "shape1[1](=#{n}) != shape2[0](=#{b.shape[0]})" if b.shape[0] != n

      lu, ipiv, info = getrf(klass, a)
      raise LapackError, "the #{-info}-th argument of getrf had illegal value" if info.negative?

      if info.positive?
        warn('the factorization has been completed, but the factor is singular, ' \
             'so the solution could not be computed.')
        return klass.new(*b.shape).store(b)
      end

      x, info = getrs(klass, lu, ipiv, b, 'N')
      raise LapackError, "the #{-info}-th argument of getrf had illegal value" if info.negative?

      x
    end

    # Computes the inverse of a square matrix.
    #
    # @param a [Cumo::NArray] the square matrix
    # @param driver [String] accepted and ignored, as numo-linalg-alt does
    # @param uplo [String] accepted and ignored, as numo-linalg-alt does
    # @return [Cumo::NArray]
    # @raise [LapackError] if the matrix is singular
    def inv(a, driver: 'getrf', uplo: 'U')
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      klass = BLAS_CLASSES[blas_char(a).to_sym]
      lu, ipiv, info = getrf(klass, a)
      raise LapackError, "the #{-info}-th argument of getrf had illegal value" if info.negative?
      raise LapackError, 'The matrix is singular, and the inverse matrix could not be computed.' if info.positive?

      invert(klass, lu, ipiv, 'getrf')
    end

    # Computes the Cholesky factorization of a Hermitian positive definite
    # matrix, A = U^H U or A = L L^H.
    #
    # @param a [Cumo::NArray] the square matrix
    # @param uplo [String] 'U' for the upper factor, 'L' for the lower
    # @return [Cumo::NArray] the factor, with zeros in the other triangle
    def cholesky(a, uplo: 'U')
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      klass = BLAS_CLASSES[blas_char(a).to_sym]
      fill = lapack_uplo(uplo)
      raise ArgumentError, "invalid uplo: #{uplo}" unless %w[U L].include?(uplo)

      c, = potrf(klass, a, fill)
      fill == 'U' ? c.transpose.triu : c.transpose.tril
    end

    # Computes the Cholesky factorization of a Hermitian positive definite
    # matrix as LAPACK's potrf does, leaving the other triangle as it was.
    #
    # @param a [Cumo::NArray] the square matrix
    # @param uplo [String] 'U' for the upper factor, 'L' for the lower
    # @return [Cumo::NArray]
    # @raise [LapackError] if the matrix is not positive definite
    def cho_fact(a, uplo: 'U')
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]
      raise ArgumentError, 'uplo must be "U" or "L"' unless %w[U L].include?(uplo)

      bchr = blas_char(a)
      c, info = potrf(BLAS_CLASSES[bchr.to_sym], a, uplo)
      raise LapackError, "the #{-info}-th argument of #{bchr}potrf had illegal value" if info.negative?

      if info.positive?
        raise LapackError,
              "the leading principal minor of order #{info} is not positive, " \
              'and the factorization could not be completed.'
      end

      c.transpose.dup
    end

    # Computes the inverse of a Hermitian positive definite matrix from the
    # factor cho_fact answers, as LAPACK's potri does: only the uplo triangle
    # holds the inverse.
    #
    # @param a [Cumo::NArray] the factor
    # @param uplo [String] the triangle the factor is in
    # @return [Cumo::NArray]
    # @raise [LapackError] if the factor has a zero on its diagonal
    def cho_inv(a, uplo: 'U')
      a = NArray.asarray(a) unless a.is_a?(NArray)
      bchr = blas_char(a)
      klass = BLAS_CLASSES[bchr.to_sym]
      fill = lapack_uplo(uplo)
      raise ArgumentError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      return klass.new(0, 0) if a.empty?

      inv = to_column_major(klass, a)
      zero = inv.diagonal.eq(0).where.to_a.first
      unless zero.nil?
        raise LapackError,
              "the (#{zero}, #{zero})-th element of the factor U or L is zero, " \
              'and the inverse could not be computed.'
      end

      info = cusolver(:potri, inv, fill)
      raise LapackError, "the #{info.abs}-th argument of #{bchr}potri had illegal value" if info.negative?

      inv.transpose.dup
    end

    # Solves A X = B from the Cholesky factor of A that cho_fact answers.
    #
    # @param a [Cumo::NArray] the factor
    # @param b [Cumo::NArray] 1- or 2-dimensional
    # @param uplo [String] the triangle the factor is in
    # @return [Cumo::NArray] X, of the shape of b
    def cho_solve(a, b, uplo: 'U')
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]
      raise NArray::ShapeError, "incompatible dimensions: a.shape[0] = #{a.shape[0]} != b.shape[0] = #{b.shape[0]}" if a.shape[0] != b.shape[0]

      klass = BLAS_CLASSES[blas_char(a, b).to_sym]
      fill = lapack_uplo(uplo)
      raise ArgumentError, 'input array b must be 1- or 2-dimensional' unless [1, 2].include?(b.ndim)

      x = to_column_major(klass, b)
      info = cusolver(:potrs, to_column_major(klass, a), x, fill)
      raise LapackError, "the #{-info}-th argument of potrs had illegal value" if info.negative?

      from_column_major(x)
    end

    # Computes the eigenvalues and eigenvectors of a Hermitian matrix, or of
    # the generalized problem A x = lambda B x with B positive definite. The
    # upper triangles are read, and uplo: and turbo: are accepted and ignored,
    # as numo-linalg-alt does.
    #
    # @param a [Cumo::NArray] the square matrix
    # @param b [Cumo::NArray, nil] the square matrix B, or nil
    # @param vals_only [Boolean] whether to leave the eigenvectors out
    # @param vals_range [Range, Array, nil] the indices of the eigenvalues to
    #   answer, 0-based, in ascending order of the eigenvalues
    # @return [Array] the eigenvalues in ascending order, and the eigenvectors
    #   in the columns of a matrix, or nil
    def eigh(a, b = nil, vals_only: false, vals_range: nil, uplo: 'U', turbo: false)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      b_given = !b.nil?
      raise NArray::ShapeError, 'input array b must be 2-dimensional' if b_given && b.ndim != 2
      raise NArray::ShapeError, 'input array b must be square' if b_given && b.shape[0] != b.shape[1]

      blas_char(b) if b_given
      klass = BLAS_CLASSES[blas_char(a).to_sym]
      n = a.shape[0]
      raise NArray::ShapeError, 'input array b must have the shape of a' if b_given && b.shape != a.shape

      il, iu = eigen_range(vals_range, n)
      w = ([SFloat, SComplex].include?(klass) ? SFloat : DFloat).new(n).fill(Float::NAN)
      v = to_column_major(klass, a)
      meig, info = if b_given
                     cusolver(:sygvd, v, to_column_major(klass, b), w, !vals_only, il, iu)
                   else
                     cusolver(:syevd, v, w, !vals_only, il, iu)
                   end
      raise LapackError, "the #{-info}-th argument of #{b_given ? 'sygvd' : 'syevd'} had illegal value" if info.negative?

      vals = il ? w[0...meig].dup : w
      vecs = if vals_only
               nil
             elsif il
               v.transpose[true, 0...meig].dup
             else
               v.transpose.dup
             end
      [vals, vecs]
    end

    # Computes the eigenvalues of a Hermitian matrix, or of the generalized
    # problem A x = lambda B x.
    #
    # @return [Cumo::NArray] the eigenvalues in ascending order
    def eigvalsh(a, b = nil, vals_range: nil, uplo: 'U', turbo: false)
      eigh(a, b, vals_only: true, vals_range: vals_range, uplo: uplo, turbo: turbo)[0]
    end

    # Computes the determinant of a square matrix from its LU factorization.
    #
    # @param a [Cumo::NArray] the square matrix
    # @return [Float, Complex]
    def det(a)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      return one(bchr) if a.shape[0].zero?

      dg, sign, info = lu_diagonal(BLAS_CLASSES[bchr.to_sym], a)
      raise LapackError, "the #{-info}-th argument of getrf had illegal value" if info.negative?

      to_ruby(dg.prod * sign)
    end

    # Computes the sign and the natural logarithm of the absolute value of the
    # determinant, which does not overflow where det does.
    #
    # @param a [Cumo::NArray] the square matrix
    # @return [Array] the sign, a Float or a Complex of absolute value 1, or 0
    #   for a singular matrix, and the logarithm, -Infinity for a singular
    #   matrix
    def slogdet(a)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      bchr = blas_char(a)
      return [one(bchr), 0.0] if a.shape == [0, 0]

      dg, sign, info = lu_diagonal(BLAS_CLASSES[bchr.to_sym], a)
      raise LapackError, "the #{info.abs}-th argument of getrf had illegal value" if info.negative?

      warn_singular_factor(info) if info.positive?
      return 0, -Float::INFINITY if to_ruby(dg.eq(0).count_true).positive?

      abs = dg.abs
      [to_ruby((dg / abs).prod * sign), to_ruby(NMath.log(abs).sum)]
    end

    # Computes a square matrix raised to an integer power.
    #
    # @param a [Cumo::NArray] the square matrix
    # @param n [Integer] the exponent, which may be negative
    # @return [Cumo::NArray]
    def matrix_power(a, n)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]
      raise ArgumentError, "exponent n must be an integer: #{n}" unless n.is_a?(Integer)

      return a.class.eye(a.shape[0]) if n.zero?
      return a.dup if n == 1

      power(n.positive? ? BLAS_CLASSES[blas_char(a).to_sym].cast(a) : inv(a), n.abs)
    end

    def power(a, n)
      result = nil
      square = a
      while n.positive?
        result = result ? matmul(square, result) : square if n.odd?
        n >>= 1
        square = matmul(square, square) if n.positive?
      end
      result
    end

    def getrf(klass, a)
      lu = to_column_major(klass, a)
      ipiv, info = cusolver(:getrf, lu)
      [lu, ipiv, info]
    end

    def getrs(klass, lu, ipiv, b, trans)
      x = to_column_major(klass, b)
      info = cusolver(:getrs, lu, ipiv, x, trans)
      [from_column_major(x), info]
    end

    def potrf(klass, a, uplo)
      c = to_column_major(klass, a)
      [c, cusolver(:potrf, c, uplo)]
    end

    def to_column_major(klass, x)
      x.ndim == 1 ? klass.new(*x.shape).store(x) : klass.new(*x.shape.reverse).store(x.transpose)
    end

    def from_column_major(x)
      x.ndim == 1 ? x : x.transpose.dup
    end

    def eigen_range(vals_range, n)
      return [nil, nil] if vals_range.nil?

      il = vals_range.first(1)[0] + 1
      iu = vals_range.last(1)[0] + 1
      raise ArgumentError, 'il must satisfy 1 <= il <= n' if il < 1 || il > n
      raise ArgumentError, 'iu must satisfy 1 <= iu <= n' if iu < 1 || iu > n
      raise ArgumentError, 'iu must be greater than or equal to il' if iu < il

      [il, iu]
    end

    def lapack_uplo(uplo)
      cusolver(:uplo, uplo)
    end

    def invert(klass, lu, ipiv, routine)
      n = lu.shape[0]
      return klass.new(0, 0) if n.zero?

      x = klass.eye(n)
      info = cusolver(:getrs, lu, ipiv, x, 'N')
      raise LapackError, "the #{info.abs}-th argument of #{routine} had illegal value" if info.negative?

      x.transpose.dup
    end

    def warn_singular_factor(info)
      warn("the factorization has been completed, but the factor U[#{info - 1}, #{info - 1}] is " \
           'exactly zero, indicating that the matrix is singular.')
    end

    def one(bchr)
      %w[c z].include?(bchr) ? Complex(1.0, 0.0) : 1.0
    end

    def lu_diagonal(klass, a)
      lu, ipiv, info = getrf(klass, a)
      swapped = ipiv.ne(ipiv.class.new(ipiv.size).seq(1)).count_true % 2
      [lu.transpose.diagonal, (swapped * -2.0) + 1.0, info]
    end

    def pivots(ipiv, n)
      ipiv = NArray.asarray(ipiv) unless ipiv.is_a?(NArray)
      raise ArgumentError, 'input array ipiv must be 1-dimensional' if ipiv.ndim != 1
      raise ArgumentError, "input array ipiv must have #{n} elements" if ipiv.size != n

      Int64.new(n).store(ipiv)
    end

    def singular?(lu)
      diagonal = lu.diagonal
      to_ruby((diagonal.eq(0) | diagonal.isnan).count_true).positive?
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

    private_class_method :eigen_range, :potrf, :to_column_major, :from_column_major, :lapack_uplo, :warn_singular_factor, :one, :lu_diagonal, :power, :getrf, :getrs, :invert, :pivots, :singular?, :cusolver, :to_ruby, :cast_to_blas_class, :check_dot, :check_gemv, :check_gemm
  end
end
