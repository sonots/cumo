# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgCholeskyTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-12, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-12 }

  types.each do |type, tol|
    %w[U L].each do |uplo|
      sub_test_case "#{type} #{uplo}" do
        setup do
          rng = Random.new(7)
          m = Array.new(4) { Array.new(4) { (rng.rand * 6).round - 3 } }
          m = linalg_values(type, m)
          @host = Array.new(4) do |i|
            Array.new(4) { |j| (0...4).sum { |k| conj(m[k][i]) * m[k][j] } + (i == j ? 4 : 0) }
          end
          @a = type.cast(@host)
          @uplo = uplo
        end

        test "cholesky answers the factor with zeros in the other triangle" do
          c = Cumo::Linalg.cholesky(@a, uplo: @uplo)
          assert_kind_of(type, c)
          host = c.to_a
          assert_true(other_triangle(host).all?(&:zero?))
          assert_close(@host, @uplo == 'U' ? host_dot(conj_t(host), host) : host_dot(host, conj_t(host)), tol * 100)
        end

        test "cho_fact leaves the other triangle as it was" do
          c = Cumo::Linalg.cho_fact(@a, uplo: @uplo)
          assert_equal(other_triangle(@host), other_triangle(c.to_a))
          assert_equal(own_triangle(Cumo::Linalg.cholesky(@a, uplo: @uplo).to_a), own_triangle(c.to_a))
        end

        test "cho_inv answers the inverse in the uplo triangle" do
          c = Cumo::Linalg.cho_fact(@a, uplo: @uplo)
          inv = Cumo::Linalg.cho_inv(c, uplo: @uplo).to_a
          assert_equal(other_triangle(c.to_a), other_triangle(inv))
          full = Array.new(4) { |i| Array.new(4) { |j| in_triangle?(i, j) ? inv[i][j] : conj(inv[j][i]) } }
          assert_close(Array.new(4) { |i| Array.new(4) { |j| i == j ? 1 : 0 } }, host_dot(@host, full), tol * 100)
        end

        test "cho_solve" do
          c = Cumo::Linalg.cho_fact(@a, uplo: @uplo)
          b = type.new(4, 2).rand
          x = Cumo::Linalg.cho_solve(c, b, uplo: @uplo)
          assert_kind_of(type, x)
          assert_close(b.to_a, host_dot(@host, x.to_a), tol * 100)
          v = type.new(4).rand
          assert_close(v.to_a, host_dot(@host, Cumo::Linalg.cho_solve(c, v, uplo: @uplo).to_a), tol * 100)
        end

        private

        def in_triangle?(i, j)
          @uplo == 'U' ? i <= j : i >= j
        end

        def other_triangle(m)
          m.each_with_index.flat_map do |row, i|
            row.each_with_index.reject { |_, j| in_triangle?(i, j) }
               .map(&:first)
          end
        end

        def own_triangle(m)
          m.each_with_index.flat_map do |row, i|
            row.each_with_index.select { |_, j| in_triangle?(i, j) }
               .map(&:first)
          end
        end
      end
    end
  end

  { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-12, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-12 }.each do |type, tol|
    [40, 840].each do |n|
      %w[U L].each do |uplo|
        test "#{type} #{n}x#{n} #{uplo} reads only the uplo triangle" do
          b = type.new(n, n).rand(-1, 1)
          a = (b.dot(b.transpose.conj) / n) + type.eye(n)
          noise = type.new(n, n).rand(-9, 9)
          g = uplo == 'U' ? a.triu + noise.tril(-1) : a.tril + noise.triu(1)
          c = Cumo::Linalg.cholesky(g, uplo: uplo)
          assert_equal(0.0, (uplo == 'U' ? c.tril(-1) : c.triu(1)).abs.max.to_f)
          product = uplo == 'U' ? c.transpose.conj.dot(c) : c.dot(c.transpose.conj)
          assert_operator((product - a).abs.max.to_f, :<, tol * n)
          f = Cumo::Linalg.cho_fact(g, uplo: uplo)
          other = uplo == 'U' ? [f.tril(-1), g.tril(-1)] : [f.triu(1), g.triu(1)]
          assert_equal(other[1].to_a, other[0].to_a)
          assert_equal((uplo == 'U' ? c.triu : c.tril).to_a, (uplo == 'U' ? f.triu : f.tril).to_a)
        end
      end
    end
  end

  test "an integer matrix" do
    c = Cumo::Linalg.cholesky(Cumo::Int32[[4, 2], [2, 5]])
    assert_kind_of(Cumo::DFloat, c)
    assert_equal([[2.0, 1.0], [0.0, 2.0]], c.to_a)
  end

  test "uplo is read by its first character, as LAPACK reads it" do
    a = Cumo::DFloat[[4, 2], [2, 5]]
    c = Cumo::Linalg.cho_fact(a)
    assert_equal(Cumo::Linalg.cho_inv(c).to_a, Cumo::Linalg.cho_inv(c, uplo: 'Upper').to_a)
    error = assert_raise(ArgumentError) { Cumo::Linalg.cholesky(a, uplo: 'Upper') }
    assert_equal('invalid uplo: Upper', error.message)
    error = assert_raise(TypeError) { Cumo::Linalg.cholesky(a, uplo: :U) }
    assert_equal('no implicit conversion of Symbol into Integer', error.message)
    assert_equal(Cumo::Linalg.cho_inv(c).to_a, Cumo::Linalg.cho_inv(c, uplo: 85).to_a)
    error = assert_raise(TypeError) { Cumo::Linalg.cho_inv(c, uplo: '') }
    assert_equal('no implicit conversion of String into Integer', error.message)
  end

  test "a matrix that is not positive definite" do
    error = assert_raise(Cumo::Linalg::LapackError) { Cumo::Linalg.cho_fact(Cumo::DFloat[[1, 2], [2, 1]]) }
    assert_equal('the leading principal minor of order 2 is not positive, and the factorization could not be completed.',
                 error.message)
    assert_equal([1.0, 2.0], Cumo::Linalg.cholesky(Cumo::DFloat[[1, 2], [2, 1]]).to_a.first)
  end

  test "a zero on the diagonal of the factor" do
    error = assert_raise(Cumo::Linalg::LapackError) { Cumo::Linalg.cho_inv(Cumo::DFloat[[2, 1, 0], [0, 3, 1], [0, 0, 0]]) }
    assert_equal('the (2, 2)-th element of the factor U or L is zero, and the inverse could not be computed.', error.message)
  end

  test "empty" do
    assert_equal([0, 0], Cumo::Linalg.cho_fact(Cumo::DFloat.new(0, 0)).shape)
    assert_equal([0, 0], Cumo::Linalg.cho_inv(Cumo::DFloat.new(0, 0)).shape)
    assert_equal([0], Cumo::Linalg.cho_solve(Cumo::DFloat.new(0, 0), Cumo::DFloat[]).shape)
  end

  test "errors" do
    s = Cumo::DFloat[[4, 2], [2, 5]]
    [
      [:cholesky, [Cumo::DFloat[1, 2]], {}, Cumo::NArray::ShapeError, 'input array a must be 2-dimensional'],
      [:cholesky, [Cumo::DFloat[[1, 2, 3], [4, 5, 6]]], {}, Cumo::NArray::ShapeError, 'input array a must be square'],
      [:cholesky, [s], { uplo: 'X' }, ArgumentError, "uplo must be 'U' or 'L'"],
      [:cholesky, [Cumo::RObject[[1]]], {}, TypeError, 'invalid data type for BLAS/LAPACK'],
      [:cho_fact, [s], { uplo: 'X' }, ArgumentError, 'uplo must be "U" or "L"'],
      [:cho_inv, [s], { uplo: 'X' }, ArgumentError, "uplo must be 'U' or 'L'"],
      [:cho_inv, [Cumo::DFloat[1, 2]], {}, ArgumentError, 'input array a must be 2-dimensional'],
      [:cho_inv, [Cumo::DFloat[[1, 2, 3], [4, 5, 6]]], {}, ArgumentError, 'input array a must be square'],
      [:cho_solve, [s, Cumo::DFloat[1, 2]], { uplo: 'X' }, ArgumentError, "uplo must be 'U' or 'L'"],
      [:cho_solve, [s, Cumo::DFloat[1, 2, 3]], {}, Cumo::NArray::ShapeError, 'incompatible dimensions: a.shape[0] = 2 != b.shape[0] = 3'],
      [:cho_solve, [s, Cumo::DFloat.new(2, 1, 1).seq], {}, ArgumentError, 'input array b must be 1- or 2-dimensional']
    ].each do |method, args, kwargs, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.public_send(method, *args, **kwargs) }
      assert_equal(message, error.message)
    end
  end

  def conj(v)
    v.respond_to?(:conj) ? v.conj : v
  end

  def conj_t(m)
    m.transpose.map { |row| row.map { |v| conj(v) } }
  end
end
