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

    # Computes the eigenvalues and the right and left eigenvectors of a
    # general square matrix with cuSOLVER's geev. cuSOLVER has no left
    # eigenvectors, so they are the right eigenvectors of the conjugate
    # transpose, whose eigenvalues are matched to the eigenvalues of the
    # matrix. For an eigenvalue that is ill-conditioned, as in a defective
    # matrix, the left eigenvector then belongs to an eigenvalue that
    # differs from the one answered by about its rounding error.
    #
    # @param a [Cumo::NArray] the square matrix
    # @param left [Boolean] whether to compute the left eigenvectors
    # @param right [Boolean] whether to compute the right eigenvectors
    # @return [Array] the eigenvalues, which are complex, and the left and
    #   the right eigenvectors, each nil when not computed and complex only
    #   when an eigenvalue is
    # @raise [NotImplementedError] if the cuSOLVER Cumo is built with has no
    #   cusolverDnXgeev, which CUDA 11 does not have
    def eig(a, left: false, right: true)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, 'input array a must be square' if a.shape[0] != a.shape[1]

      w, vr = geev(a, right)
      [w, left ? left_eigenvectors(a, w) : nil, vr]
    end

    # Computes the Bunch-Kaufman decomposition of a symmetric or Hermitian
    # matrix, A = U D U^T or A = L D L^T, with U^H and L^H for a Hermitian
    # one, where U or L is a permuted unit triangular matrix and D is block
    # diagonal with blocks of order 1 and 2.
    #
    # @param a [Cumo::NArray] the square matrix, of which only the uplo
    #   triangle is read
    # @param uplo [String] 'U' or 'L'
    # @param hermitian [Boolean] whether a complex matrix is Hermitian rather
    #   than symmetric
    # @return [Array] the permuted triangular factor, D, and the
    #   permutation, a Cumo::Int32, that makes the factor triangular
    # @raise [NotImplementedError] for a Hermitian complex matrix if the
    #   cuSOLVER Cumo is built with has no cusolverDnXhetrf, which CUDA 11
    #   does not have
    def ldl(a, uplo: 'U', hermitian: true)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      bchr = blas_char(a)
      klass = BLAS_CLASSES[bchr.to_sym]
      uplo = lapack_uplo(uplo)
      return [klass.new(0, 0), klass.new(0, 0), Int32.new(0)] if a.shape[0].zero?

      lud = to_column_major(klass, a)
      ipiv, info = cusolver(:sytrf, lud, uplo, hermitian)
      fnc = %w[c z].include?(bchr) && hermitian ? "#{bchr}hetrf" : "#{bchr}sytrf"
      raise LapackError, "the #{info.abs}-th argument of #{fnc} had illegal value" if info.negative?

      if info.positive?
        warn("the factorization has been completed, but the D[#{info - 1}, #{info - 1}] is " \
             'exactly zero, indicating that the block diagonal matrix is singular.')
      end
      ldl_factors(from_column_major(lud), ipiv.to_a, uplo, hermitian)
    end

    # Computes the eigenvalues of a general square matrix.
    #
    # @param a [Cumo::NArray] the square matrix
    # @return [Cumo::SComplex, Cumo::DComplex]
    # @raise [NotImplementedError] if the cuSOLVER Cumo is built with has no
    #   cusolverDnXgeev, which CUDA 11 does not have
    def eigvals(a)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, 'input array a must be square' if a.shape[0] != a.shape[1]

      geev(a, false)[0]
    end

    # Computes the singular value decomposition A = U S V^H.
    #
    # @param a [Cumo::NArray] 2-dimensional, of shape [m, n]
    # @param driver [String] 'svd' or 'sdd', which both run cuSOLVER's gesvd
    # @param job [String] 'A' for U of shape [m, m] and V^H of [n, n], 'S' for
    #   [m, k] and [k, n] with k = min(m, n), 'N' for neither
    # @return [Array] the singular values in descending order, U and V^H, or
    #   nil in place of U and V^H for job 'N'
    def svd(a, driver: 'svd', job: 'A')
      raise ArgumentError, "invalid job: #{job}" unless /^[ASN]/i.match?(job.to_s)

      svd_call(a, driver, job, 'the did not converge')
    end

    # Computes the singular values of a matrix.
    #
    # @param a [Cumo::NArray] 2-dimensional
    # @param driver [String] 'sdd' or 'svd', which both run cuSOLVER's gesvd
    # @return [Cumo::NArray] the singular values in descending order
    def svdvals(a, driver: 'sdd')
      svd_call(a, driver, 'N', 'the decomposition did not converge')[0]
    end

    # Computes the rank of a matrix, the number of its singular values above
    # tol.
    #
    # @param a [Cumo::NArray]
    # @param tol [Float, nil] by default the largest singular value times the
    #   larger dimension times the machine epsilon
    # @param driver [String] passed to svdvals
    # @return [Integer]
    def matrix_rank(a, tol: nil, driver: 'svd')
      return to_ruby(a.ne(0).count_true).positive? ? 1 : 0 if a.ndim < 2

      s = svdvals(a, driver: driver)
      tol ||= s.max(axis: -1, keepdims: true) * (a.shape[-2..].max * s.class::EPSILON)
      to_ruby(s.gt(tol).count_true(axis: -1))
    end

    # Computes the Moore-Penrose pseudoinverse of a matrix from its singular
    # value decomposition.
    #
    # @param a [Cumo::NArray] 2-dimensional, of shape [m, n]
    # @param driver [String] passed to svd
    # @param rcond [Float, nil] singular values below rcond times the largest
    #   count as zero
    # @return [Cumo::NArray] of shape [n, m]
    def pinv(a, driver: 'svd', rcond: nil)
      s, u, vh = svd(a, driver: driver, job: 'S')
      rcond = a.shape.max * s.class::EPSILON if rcond.nil?
      rank = count_above(s, rcond)
      return u.class.zeros(*a.shape.reverse) if rank.zero?

      u = u[true, 0...rank] / s[0...rank]
      u.dot(vh[0...rank, true]).conj.transpose.dup
    end

    # Computes an orthonormal basis of the range of a matrix.
    #
    # @param a [Cumo::NArray] 2-dimensional, of shape [m, n]
    # @param rcond [Float, nil] singular values below rcond times the largest
    #   count as zero
    # @return [Cumo::NArray] of shape [m, rank]
    def orth(a, rcond: nil)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      s, u, = svd(a, driver: 'sdd', job: 'S')
      rank = numerical_rank(s, a, rcond)
      rank.zero? ? u.class.new(a.shape[0], 0) : u[true, 0...rank].dup
    end

    # Computes an orthonormal basis of the null space of a matrix.
    #
    # @param a [Cumo::NArray] 2-dimensional, of shape [m, n]
    # @param rcond [Float, nil] singular values below rcond times the largest
    #   count as zero
    # @return [Cumo::NArray] of shape [n, n - rank]
    def null_space(a, rcond: nil)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2

      s, _u, vt = svd(a, driver: 'sdd', job: a.shape[0] >= a.shape[1] ? 'S' : 'A')
      rank = numerical_rank(s, a, rcond)
      n = vt.shape[0]
      rank == n ? vt.class.new(n, 0) : vt[rank...n, true].conj.transpose.dup
    end

    # Computes the norm of a vector or a matrix, or the norms of the vectors or
    # the matrices along the given axes.
    #
    #   |  ord  |  matrix norm           | vector norm                 |
    #   | ----- | ---------------------- | --------------------------- |
    #   |  nil  | Frobenius norm         | 2-norm                      |
    #   | 'fro' | Frobenius norm         |  -                          |
    #   | 'nuc' | nuclear norm           |  -                          |
    #   | 'inf' | x.abs.sum(axis:-1).max | x.abs.max                   |
    #   |    0  |  -                     | (x.ne 0).sum                |
    #   |    1  | x.abs.sum(axis:-2).max | same as below               |
    #   |    2  | 2-norm (max sing_vals) | same as below               |
    #   | other |  -                     | (x.abs**ord).sum**(1.0/ord) |
    #
    # @param a [Cumo::NArray, Array]
    # @param ord [String, Numeric, nil] the order of the norm
    # @param axis [Integer, Array, nil] one axis for vector norms or two for
    #   matrix norms; by default the whole array, which must be 1- or
    #   2-dimensional unless ord is nil
    # @param keepdims [Boolean] whether the normed axes are left with size one
    # @return [Float, Integer, Complex, Cumo::NArray] a scalar, which waits for
    #   the GPU, where numo-linalg-alt answers one
    def norm(a, ord = nil, axis: nil, keepdims: false)
      a = NArray.asarray(a) unless a.is_a?(NArray)
      return 0.0 if a.empty?

      blas_char(a)
      a = a.reshape(1) if a.ndim.zero?
      ord = Float::INFINITY if ord == 'inf'
      ord = -Float::INFINITY if ord == '-inf'
      if axis.nil?
        norm = whole_norm(a, ord)
        return keepdims ? NArray.asarray(norm).reshape(*([1] * a.ndim)) : norm unless norm.nil?

        axis = Array.new(a.ndim) { |d| d }
      else
        axis = norm_axes(axis)
      end
      raise ArgumentError, "the number of dimensions of axis is inappropriate for the norm: #{axis.size}" unless [1, 2].include?(axis.size)
      raise ArgumentError, "axis is out of range: #{axis}" unless axis.all? { |ax| (-a.ndim...a.ndim).cover?(ax) }

      a = DFloat.cast(a) if a.is_a?(Bit)
      return vector_norm(a, ord || 2, axis[0], keepdims) if axis.size == 1

      axes = axis.map { |ax| ax.negative? ? ax + a.ndim : ax }
      raise ArgumentError, "invalid axis: #{axis}" if axes.uniq.size == 1

      norm = matrix_norm(a, ord || 'fro', *axes)
      return norm unless keepdims

      shape = a.shape.dup
      axes.each { |ax| shape[ax] = 1 }
      NArray.asarray(norm).reshape(*shape)
    end

    # Computes the condition number of a matrix, the norm of the matrix times
    # the norm of its inverse. For nil and 2, the ratio of its largest
    # singular value to its smallest.
    #
    # @param a [Cumo::NArray] 2-dimensional
    # @param ord [String, Integer, nil] nil or 2, or -2 for the inverse ratio,
    #   or an ord of the matrix norm: 'fro', 'nuc', 1, -1, 'inf' or '-inf'
    # @return [Cumo::NArray, Float] zero-dimensional for nil, 2, -2 and 'fro',
    #   as numo-linalg-alt answers, and a Float for the others
    def cond(a, ord = nil)
      if ord.nil? || ord == 2 || ord == -2
        svals = svdvals(a)
        return ord == -2 ? svals[false, -1] / svals[false, 0] : svals[false, 0] / svals[false, -1]
      end

      inv_a = inv(a)
      norm(a, ord, axis: [-2, -1]) * norm(inv_a, ord, axis: [-2, -1])
    end

    # Computes the least-squares solution to A x = b, the one of least norm
    # when A has less than full rank, from the singular value decomposition of
    # A, as LAPACK's gelsd does.
    #
    # @param a [Cumo::NArray] 2-dimensional, of shape [m, n]
    # @param b [Cumo::NArray] 1- or 2-dimensional, with m rows
    # @param driver [String] accepted and ignored, as numo-linalg-alt does
    # @param rcond [Float, nil] singular values below rcond times the largest
    #   count as zero; the machine epsilon when nil or negative
    # @return [Array] x, the squared residuals when m > n and A has rank n,
    #   the rank, and the singular values
    def lstsq(a, b, driver: 'lsd', rcond: nil)
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise NArray::ShapeError, "incompatible dimensions: a.shape[0] = #{a.shape[0]} != b.shape[0] = #{b.shape[0]}" if a.shape[0] != b.shape[0]

      bchr = blas_char(a)
      klass = BLAS_CLASSES[bchr.to_sym]
      raise ArgumentError, 'input array b must be 1 or 2-dimensional' unless [1, 2].include?(b.ndim)

      m, n = a.shape
      resid_class = m < n ? DFloat : b.class
      b = klass.cast(b)
      s, u, vt, info = gesvd(klass, a, 'S')
      raise LapackError, "the #{info.abs}-th argument of #{bchr}gelsd had illegal value" if info.negative?
      raise LapackError, 'the algorithm for computing the SVD failed to converge' if info.positive?

      rcond = s.class::EPSILON / 2 if rcond.nil? || rcond <= 0 || rcond >= 1
      rank = count_above(s, rcond)
      x = if rank.zero?
            klass.zeros(*([n] + b.shape[1..]))
          else
            c = u[true, 0...rank].conj.transpose.dot(b)
            c /= b.ndim == 1 ? s[0...rank] : s[0...rank][true, :new]
            vt[0...rank, true].conj.transpose.dot(c)
          end

      resids = if m > n && rank == n
                 r = klass.cast(a).dot(x) - b
                 to_ruby((r.abs**2).sum(axis: 0))
               else
                 resid_class[]
               end
      [x, resids, rank, s]
    end

    # Computes the QR factorization A = Q R.
    #
    # @param a [Cumo::NArray] 2-dimensional, of shape [m, n]
    # @param mode [String] 'reduce' for Q of shape [m, m] and R of [m, n],
    #   'r' for R alone, 'economic' for [m, k] and [k, n] with k = min(m, n),
    #   'raw' for the Householder vectors and their factors as LAPACK's geqrf
    #   answers them
    # @return [Array<Cumo::NArray>, Cumo::NArray] Q and R, R, or the vectors
    #   and the factors
    def qr(a, mode: 'reduce')
      raise NArray::ShapeError, 'input array a must be 2-dimensional' if a.ndim != 2
      raise ArgumentError, "invalid mode: #{mode}" unless %w[reduce r economic raw].include?(mode)

      bchr = blas_char(a)
      klass = BLAS_CLASSES[bchr.to_sym]
      m, n = a.shape
      k = [m, n].min
      return empty_qr(klass, m, n, mode) if k.zero?

      buf = to_column_major(klass, a)
      tau = klass.new(k)
      info = cusolver(:geqrf, buf, tau)
      raise LapackError, "the #{-info}-th argument of #{bchr}geqrf had illegal value" if info.negative?

      qr = buf.transpose.dup
      return [qr, tau] if mode == 'raw'

      r = m > n && mode == 'economic' ? qr[0...n, true].triu : qr.triu!
      return r if mode == 'r'

      q = if m < n
            buf[0...m, true].dup
          elsif mode == 'economic'
            buf
          else
            klass.zeros(m, m).tap { |x| x[0...n, true] = buf }
          end
      info = cusolver(:orgqr, q, tau)
      raise LapackError, "the #{-info}-th argument of #{bchr}orgqr had illegal value" if info.negative?

      [q.transpose.dup, r]
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

    def empty_qr(klass, m, n, mode)
      return [klass.new(m, n), klass.new(0)] if mode == 'raw'

      r = mode == 'economic' ? klass.new(0, n) : klass.new(m, n)
      return r if mode == 'r'

      q = mode == 'economic' || m.zero? ? klass.new(m, 0) : klass.eye(m)
      [q, r]
    end

    def svd_call(a, driver, job, not_converged)
      a = NArray.asarray(a) unless a.is_a?(NArray)
      bchr = blas_char(a)
      raise ArgumentError, "invalid driver: #{driver}" unless %w[svd sdd].include?(driver.to_s)

      fill = svd_job(job, driver.to_s)
      raise ArgumentError, 'input array must be 2-dimensional' if a.ndim != 2

      s, u, vt, info = gesvd(BLAS_CLASSES[bchr.to_sym], a, fill)
      raise LapackError, "the #{info.abs}-th argument had illegal value" if info.negative?
      raise LapackError, not_converged if info.positive?

      [s, u, vt]
    end

    def count_above(s, factor)
      s.empty? ? 0 : to_ruby(s.gt(s.max * factor).count_true)
    end

    def svd_job(job, driver)
      raise TypeError, "no implicit conversion of #{job.class} into String" unless job.is_a?(String)

      fill = job[0]
      return fill if %w[A S N].include?(fill)

      raise ArgumentError, driver == 'sdd' ? "jobz must be one of 'A', 'S', 'O', or 'N'" : "jobu must be 'A', 'S', 'O', or 'N'"
    end

    def gesvd(klass, a, job)
      m, n = a.shape
      tall = m >= n
      rows, cols = tall ? [m, n] : [n, m]
      complex = [SComplex, DComplex].include?(klass)
      buf = tall ? to_column_major(klass, a) : klass.new(m, n).store(a)
      buf = buf.conj if !tall && complex && job != 'N'
      s = ([SFloat, SComplex].include?(klass) ? SFloat : DFloat).new(cols)
      u = vt = nil
      unless job == 'N'
        u_rows = job == 'A' ? rows : cols
        u = cols.zero? && u_rows.positive? ? klass.eye(rows) : klass.new(u_rows, rows)
        vt = klass.new(cols, cols)
      end
      info = cusolver(:gesvd, buf, s, u, vt, job)
      return [s, nil, nil, info] if job == 'N'

      if tall
        [s, u.transpose.dup, vt.transpose.dup, info]
      else
        [s, complex ? vt.conj : vt, complex ? u.conj : u, info]
      end
    end

    def numerical_rank(s, a, rcond)
      count_above(s, rcond.nil? || rcond.negative? ? a.shape.max * s.class::EPSILON : rcond)
    end

    def whole_norm(a, ord)
      case a.ndim
      when 1
        frobenius(a) if ord.nil? || ord == 2
      when 2
        if ord.nil? || ord == 'fro'
          frobenius(a)
        elsif [1, Float::INFINITY].include?(ord)
          matrix_norm(to_float(a), ord, 0, 1)
        end
      else
        frobenius(a) if ord.nil?
      end
    end

    def frobenius(a)
      x = magnitudes(to_float(a))
      squares = to_ruby(x.mulsum(x))
      return squares if squares.nan?
      return Math.sqrt(squares) if squares.finite? && squares >= x.size * x.class::MIN / x.class::EPSILON

      x = x.abs
      scale = x.max
      largest = to_ruby(scale)
      return largest unless largest.positive? && largest.finite?

      scaled = x / scale
      to_ruby(NMath.sqrt(scaled.mulsum(scaled)) * scale)
    end

    def norm_axes(axis)
      case axis
      when Integer
        [axis]
      when Array, NArray
        axis.flatten.to_a
      else
        raise ArgumentError, "invalid axis: #{axis}"
      end
    end

    def vector_norm(a, ord, axis, keepdims)
      raise ArgumentError, "invalid ord: #{ord}" unless ord.is_a?(Numeric)

      norm = if ord.infinite?
               ord.positive? ? a.abs.max(axis: axis, keepdims: keepdims) : a.abs.min(axis: axis, keepdims: keepdims)
             elsif ord.zero?
               a.class.cast(a.ne(0)).sum(axis: axis, keepdims: keepdims)
             elsif ord == 1
               a.abs.sum(axis: axis, keepdims: keepdims)
             elsif ord == 2
               x = magnitudes(to_float(a))
               NMath.sqrt(x.mulsum(x, axis: axis, keepdims: keepdims))
             else
               (to_float(a).abs**ord).sum(axis: axis, keepdims: keepdims)**1.fdiv(ord)
             end
      to_scalar(norm)
    end

    def matrix_norm(a, ord, r_axis, c_axis)
      raise ArgumentError, "invalid ord: #{ord}" unless ord.is_a?(String) || ord.is_a?(Numeric)

      case ord
      when 'fro'
        x = magnitudes(to_float(a))
        sum = x.mulsum(x, axis: [r_axis, c_axis])
        NMath.sqrt(sum.ndim.zero? ? DFloat.cast(sum) : sum)
      when 'nuc'
        to_scalar(stacked_svdvals(a, r_axis, c_axis).sum(axis: -1))
      when String
        raise ArgumentError, "invalid ord: #{ord}"
      when 2, -2
        s = stacked_svdvals(a, r_axis, c_axis)
        to_scalar(ord == 2 ? s.max(axis: -1) : s.min(axis: -1))
      when 1, -1
        sums = a.abs.sum(axis: r_axis)
        c_axis -= 1 if c_axis > r_axis
        to_scalar(ord == 1 ? sums.max(axis: c_axis) : sums.min(axis: c_axis))
      else
        raise ArgumentError, "invalid ord: #{ord}" unless ord.infinite?

        sums = a.abs.sum(axis: c_axis)
        r_axis -= 1 if r_axis > c_axis
        to_scalar(ord.positive? ? sums.max(axis: r_axis) : sums.min(axis: r_axis))
      end
    end

    def stacked_svdvals(a, r_axis, c_axis)
      return svdvals(a) if a.ndim == 2

      b = a.transpose(*((0...a.ndim).to_a - [r_axis, c_axis]), r_axis, c_axis)
      batch = b.shape[0...-2]
      matrices = b.dup.reshape(batch.reduce(:*), *b.shape[-2..])
      s = nil
      matrices.shape[0].times do |i|
        vals = svdvals(matrices[i, true, true])
        s ||= vals.class.zeros(matrices.shape[0], vals.size)
        s[i, true] = vals
      end
      s.reshape(*batch, s.shape[1])
    end

    def magnitudes(x)
      x.is_a?(SComplex) || x.is_a?(DComplex) ? x.abs : x
    end

    def to_float(a)
      INTEGER_CLASSES.include?(a.class) ? DFloat.cast(a) : a
    end

    def to_scalar(x)
      x.ndim.zero? ? to_ruby(x) : x
    end

    def geev(a, vectors)
      bchr = blas_char(a)
      raise NotImplementedError, 'Cumo is built with a cuSOLVER that has no cusolverDnXgeev' if CUDA::Cusolver.available? && !CUDA::Cusolver.respond_to?(:geev, true)

      klass = BLAS_CLASSES[bchr.to_sym]
      n = a.shape[0]
      w = (%w[s c].include?(bchr) ? SComplex : DComplex).new(n)
      vr = vectors ? klass.new(n, n) : nil
      info = cusolver(:geev, to_column_major(klass, a), w, vr)
      raise LapackError, "the #{info.abs}-th argument of #{bchr}geev had illegal value" if info.negative?
      raise LapackError, 'the QR algorithm failed to compute all the eigenvalues.' if info.positive?
      return [w, nil] unless vectors

      vr = from_column_major(vr)
      [w, %w[s d].include?(bchr) ? unpack_eigenvectors(w, vr) : vr]
    end

    def unpack_eigenvectors(w, v)
      ids = w.imag.gt(0).where
      return v if ids.empty?

      packed = w.class.cast(v)
      packed[true, ids].imag = v[true, ids + 1]
      packed[true, ids + 1] = packed[true, ids].conj
      packed
    end

    def left_eigenvectors(a, w)
      mu, y = geev(BLAS_CLASSES[blas_char(a).to_sym].cast(a).conj.transpose, true)
      y.empty? ? y : y[true, match_eigenvalues(w, mu.conj)].dup
    end

    def match_eigenvalues(w, target)
      n = w.size
      dist = (w.reshape(n, 1) - target.reshape(1, n)).abs
      flat = dist.min_index(axis: 1)
      gaps = dist[flat].to_a
      nearest = (flat % n).to_a
      order = (0...n).sort_by { |i| gaps[i] }
      perm = Array.new(n)
      free = Array.new(n, true)
      order.each do |i|
        next unless free[nearest[i]]

        perm[i] = nearest[i]
        free[nearest[i]] = false
      end
      rows = order.select { |i| perm[i].nil? }
      return Int32.cast(perm) if rows.empty?

      cols = (0...n).select { |j| free[j] }
      sub = dist[rows, cols].to_a
      open = cols.each_index.to_a
      rows.each_with_index do |i, r|
        c = open.min_by { |k| sub[r][k] }
        perm[i] = cols[c]
        open.delete(c)
      end
      Int32.cast(perm)
    end

    def ldl_factors(lud, piv, uplo, hermitian)
      n = lud.shape[0]
      perm = Array.new(n) { |k| k }
      blocks = []
      swaps = []
      upper = uplo == 'U'
      pending = false
      (upper ? 0.upto(n - 1) : (n - 1).downto(0)).each do |k|
        other = upper ? k - 1 : k + 1
        if piv[k].positive?
          swaps << [piv[k] - 1, k, upper ? 0..k : k...n]
        elsif piv[k].negative? && other.between?(0, n - 1) && piv[k] == piv[other] && !pending
          blocks << [other, k]
          swaps << [-piv[k] - 1, other, upper ? 0..k : k...n]
          pending = true
          next
        end
        pending = false
      end

      factor = upper ? lud.triu : lud.tril
      diagonal = lud.diag_indices
      factor[diagonal] = 1
      d = lud.class.zeros(n, n)
      d[diagonal] = lud[diagonal]
      unless blocks.empty?
        inner = Int32.cast(blocks.map { |r, c| (r * n) + c })
        mirror = Int32.cast(blocks.map { |r, c| (c * n) + r })
        off = lud[inner]
        d[inner] = off
        d[mirror] = hermitian ? off.conj : off
        factor[inner] = 0
      end
      swaps.each do |i, k, cols|
        perm[i], perm[k] = perm[k], perm[i]
        factor[[i, k], cols] = factor[[k, i], cols] if i != k
      end
      inverse = Array.new(n)
      perm.each_with_index { |p, j| inverse[p] = j }
      [factor, d, Int32.cast(inverse)]
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

    private_class_method :ldl_factors, :geev, :unpack_eigenvectors, :left_eigenvectors, :match_eigenvalues, :whole_norm, :frobenius, :norm_axes, :vector_norm, :matrix_norm, :stacked_svdvals, :magnitudes, :to_float, :to_scalar, :empty_qr, :svd_call, :count_above, :svd_job, :gesvd, :numerical_rank, :eigen_range, :potrf, :to_column_major, :from_column_major, :lapack_uplo, :warn_singular_factor, :one, :lu_diagonal, :power, :getrf, :getrs, :invert, :pivots, :singular?, :cusolver, :to_ruby, :cast_to_blas_class, :check_dot, :check_gemv, :check_gemm
  end
end
