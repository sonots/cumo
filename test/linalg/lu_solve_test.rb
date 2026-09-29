# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgLuSolveTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-10, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-10 }

  types.each do |type, tol|
    sub_test_case type.to_s do
      setup do
        @a = type.new(5, 5).rand - 0.5
        @lu, @piv = Cumo::Linalg.lu_fact(@a)
        @host_a = @a.to_a
      end

      test "a vector" do
        b = type.new(5).rand
        x = Cumo::Linalg.lu_solve(@lu, @piv, b)
        assert_kind_of(type, x)
        assert_equal([5], x.shape)
        assert_close(b.to_a, host_dot(@host_a, x.to_a), tol)
      end

      test "a matrix" do
        b = type.new(5, 3).rand
        x = Cumo::Linalg.lu_solve(@lu, @piv, b)
        assert_equal([5, 3], x.shape)
        assert_close(b.to_a, host_dot(@host_a, x.to_a), tol)
      end

      test "the transpose" do
        b = type.new(5, 2).rand
        x = Cumo::Linalg.lu_solve(@lu, @piv, b, trans: 'T')
        assert_close(b.to_a, host_dot(@host_a.transpose, x.to_a), tol)
      end

      test "the conjugate transpose" do
        b = type.new(5).rand
        x = Cumo::Linalg.lu_solve(@lu, @piv, b, trans: 'C')
        a_h = @host_a.transpose.map { |row| row.map { |v| v.respond_to?(:conj) ? v.conj : v } }
        assert_close(b.to_a, host_dot(a_h, x.to_a), tol)
      end

      test "b is left as it was" do
        b = type.new(5, 2).rand
        orig = b.to_a
        Cumo::Linalg.lu_solve(@lu, @piv, b)
        assert_equal(orig, b.to_a)
      end
    end
  end

  test "b is cast to the type of lu" do
    lu, piv = Cumo::Linalg.lu_fact(Cumo::SFloat[[1, 2], [3, 4]])
    x = Cumo::Linalg.lu_solve(lu, piv, Cumo::DFloat[1, 2])
    assert_kind_of(Cumo::SFloat, x)
    assert_close([0.0, 0.5], x, 1e-6)
    assert_close([0.0, 0.5], Cumo::Linalg.lu_solve(lu, piv, Cumo::Int32[1, 2]), 1e-6)
  end

  test "pivots as an Array" do
    lu, piv = Cumo::Linalg.lu_fact(Cumo::DFloat[[1, 2], [3, 4]])
    assert_close([0.0, 0.5], Cumo::Linalg.lu_solve(lu, piv.to_a, Cumo::DFloat[1, 2]), 1e-15)
  end

  test "empty" do
    x = Cumo::Linalg.lu_solve(Cumo::DFloat.new(0, 0), Cumo::Int32[], Cumo::DFloat[])
    assert_equal([0], x.shape)
  end

  test "errors" do
    lu, piv = Cumo::Linalg.lu_fact(Cumo::DFloat[[1, 2], [3, 4]])
    b = Cumo::DFloat[1, 2]
    [
      [[Cumo::DFloat[1, 2], piv, b], Cumo::NArray::ShapeError, 'input array lu must be 2-dimensional'],
      [[Cumo::DFloat[[1, 2, 3], [4, 5, 6]], piv, b], Cumo::NArray::ShapeError, 'input array lu must be square'],
      [[lu, piv, Cumo::DFloat[1, 2, 3]], Cumo::NArray::ShapeError, 'incompatible dimensions: lu.shape[0] = 2 != b.shape[0] = 3'],
      [[lu, piv, b, { trans: 'X' }], ArgumentError, 'trans must be "N", "T", or "C"'],
      [[lu, piv.reshape(2, 1), b], ArgumentError, 'input array ipiv must be 1-dimensional'],
      [[lu, Cumo::Int32[1], b], ArgumentError, 'input array ipiv must have 2 elements'],
      [[lu, Cumo::Int32[0, 2], b], ArgumentError, 'input array ipiv must be in 1..2'],
      [[lu, Cumo::Int32[1, 3], b], ArgumentError, 'input array ipiv must be in 1..2'],
      [[lu, piv, Cumo::DFloat.new(2, 1, 1).seq], ArgumentError, 'input array b must be 1 or 2-dimensional']
    ].each do |args, klass, message|
      kwargs = args.last.is_a?(Hash) ? args.pop : {}
      error = assert_raise(klass) { Cumo::Linalg.lu_solve(*args, **kwargs) }
      assert_equal(message, error.message)
    end
  end
end
