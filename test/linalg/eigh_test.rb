# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgEighTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-11, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-11 }

  types.each do |type, tol|
    sub_test_case type.to_s do
      setup do
        rng = Random.new(11)
        m = linalg_values(type, Array.new(4) { Array.new(4) { (rng.rand * 8).round - 4 } })
        q = linalg_values(type, Array.new(4) { Array.new(4) { (rng.rand * 6).round - 3 } })
        @a = Array.new(4) { |i| Array.new(4) { |j| m[i][j] + conj(m[j][i]) } }
        @b = Array.new(4) { |i| Array.new(4) { |j| (0...4).sum { |k| conj(q[k][i]) * q[k][j] } + (i == j ? 4 : 0) } }
        @real = if COMPLEX_TYPES.include?(type)
                  type == Cumo::SComplex ? Cumo::SFloat : Cumo::DFloat
                else
                  type
                end
      end

      test "eigh" do
        vals, vecs = Cumo::Linalg.eigh(type.cast(@a))
        assert_kind_of(@real, vals)
        assert_kind_of(type, vecs)
        assert_equal(vals.to_a.sort, vals.to_a)
        v = vecs.to_a
        assert_close(host_dot(v, diag(vals.to_a)), host_dot(@a, v), tol * 100)
        assert_close(identity(4), host_dot(conj_t(v), v), tol * 10)
      end

      test "the generalized problem" do
        vals, vecs = Cumo::Linalg.eigh(type.cast(@a), type.cast(@b))
        v = vecs.to_a
        assert_close(host_dot(host_dot(@b, v), diag(vals.to_a)), host_dot(@a, v), tol * 1000)
        assert_close(identity(4), host_dot(conj_t(v), host_dot(@b, v)), tol * 100)
      end

      test "vals_only and eigvalsh" do
        vals, = Cumo::Linalg.eigh(type.cast(@a))
        only, vecs = Cumo::Linalg.eigh(type.cast(@a), vals_only: true)
        assert_nil(vecs)
        assert_close(vals.to_a, only, tol * 10)
        assert_close(vals.to_a, Cumo::Linalg.eigvalsh(type.cast(@a)), tol * 10)
        assert_close(Cumo::Linalg.eigh(type.cast(@a), type.cast(@b))[0].to_a,
                     Cumo::Linalg.eigvalsh(type.cast(@a), type.cast(@b)),
                     tol * 100)
      end

      test "vals_range" do
        vals, vecs = Cumo::Linalg.eigh(type.cast(@a))
        part, part_vecs = Cumo::Linalg.eigh(type.cast(@a), vals_range: 1..2)
        assert_close(vals.to_a[1..2], part, tol * 10)
        assert_equal([4, 2], part_vecs.shape)
        p = part_vecs.to_a
        assert_close(host_dot(p, diag(part.to_a)), host_dot(@a, p), tol * 100)
        assert_equal([4, 4], vecs.shape)

        gvals = Cumo::Linalg.eigvalsh(type.cast(@a), type.cast(@b))
        assert_close(gvals.to_a[0...2], Cumo::Linalg.eigvalsh(type.cast(@a), type.cast(@b), vals_range: 0...2), tol * 100)
      end
    end
  end

  test "the upper triangle is read and uplo: is ignored, as numo-linalg-alt does" do
    a = Cumo::DFloat[[1, 0], [5, 1]]
    assert_equal([1.0, 1.0], Cumo::Linalg.eigh(a)[0].to_a)
    assert_equal([1.0, 1.0], Cumo::Linalg.eigh(a, uplo: 'L', turbo: true)[0].to_a)
  end

  test "vals_only with vals_range" do
    a = Cumo::DFloat[[4, 1, 2], [1, 3, 0], [2, 0, 5]]
    vals, vecs = Cumo::Linalg.eigh(a, vals_only: true, vals_range: 1..2)
    assert_nil(vecs)
    assert_close(Cumo::Linalg.eigvalsh(a).to_a[1..2], vals, 1e-12)
  end

  test "a B that is not positive definite answers NaN rather than stale memory" do
    10.times { Cumo::DFloat.new(2).fill(12_345.0) }
    GC.start
    vals, vecs = Cumo::Linalg.eigh(Cumo::DFloat[[2, 1], [1, 2]], Cumo::DFloat[[1, 0], [0, -1]])
    assert_true(vals.to_a.all?(&:nan?))
    assert_equal([2, 2], vecs.shape)
  end

  test "an integer matrix" do
    vals, vecs = Cumo::Linalg.eigh(Cumo::Int32[[2, 1], [1, 2]])
    assert_kind_of(Cumo::DFloat, vals)
    assert_kind_of(Cumo::DFloat, vecs)
    assert_close([1.0, 3.0], vals, 1e-14)
  end

  test "empty" do
    vals, vecs = Cumo::Linalg.eigh(Cumo::DFloat.new(0, 0))
    assert_equal([[0], [0, 0]], [vals.shape, vecs.shape])
  end

  test "errors" do
    a = Cumo::DFloat[[4, 1, 2], [1, 3, 0], [2, 0, 5]]
    [
      [[Cumo::DFloat[1, 2]], {}, Cumo::NArray::ShapeError, 'input array a must be 2-dimensional'],
      [[Cumo::DFloat[[1, 2, 3], [4, 5, 6]]], {}, Cumo::NArray::ShapeError, 'input array a must be square'],
      [[a, Cumo::DFloat[1, 2]], {}, Cumo::NArray::ShapeError, 'input array b must be 2-dimensional'],
      [[a, Cumo::DFloat[[1, 2, 3], [4, 5, 6]]], {}, Cumo::NArray::ShapeError, 'input array b must be square'],
      [[a, Cumo::DFloat.eye(2)], {}, Cumo::NArray::ShapeError, 'input array b must have the shape of a'],
      [[a, Cumo::RObject[[1]]], {}, TypeError, 'invalid data type for BLAS/LAPACK'],
      [[a], { vals_range: 5..6 }, ArgumentError, 'il must satisfy 1 <= il <= n'],
      [[a], { vals_range: 2..5 }, ArgumentError, 'iu must satisfy 1 <= iu <= n'],
      [[a], { vals_range: [2, 1] }, ArgumentError, 'iu must be greater than or equal to il']
    ].each do |args, kwargs, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.eigh(*args, **kwargs) }
      assert_equal(message, error.message)
    end
  end

  private

  def conj(v)
    v.respond_to?(:conj) ? v.conj : v
  end

  def conj_t(m)
    m.transpose.map { |row| row.map { |v| conj(v) } }
  end

  def diag(values)
    Array.new(values.size) { |i| Array.new(values.size) { |j| i == j ? values[i] : 0 } }
  end

  def identity(n)
    Array.new(n) { |i| Array.new(n) { |j| i == j ? 1 : 0 } }
  end
end
