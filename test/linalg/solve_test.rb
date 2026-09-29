# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgSolveTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-10, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-10 }

  types.each do |type, tol|
    sub_test_case type.to_s do
      setup do
        @a = type.new(5, 5).rand - 0.5
      end

      test "a vector" do
        b = type.new(5).rand
        x = Cumo::Linalg.solve(@a, b)
        assert_kind_of(type, x)
        assert_equal([5], x.shape)
        assert_close(b.to_a, host_dot(@a.to_a, x.to_a), tol)
      end

      test "a matrix" do
        b = type.new(5, 3).rand
        x = Cumo::Linalg.solve(@a, b)
        assert_equal([5, 3], x.shape)
        assert_close(b.to_a, host_dot(@a.to_a, x.to_a), tol)
      end

      test "a transposed view" do
        b = type.new(5).rand
        x = Cumo::Linalg.solve(@a.transpose, b)
        assert_close(b.to_a, host_dot(@a.to_a.transpose, x.to_a), tol)
      end

      test "the operands are left as they were" do
        b = type.new(5, 2).rand
        a_orig = @a.to_a
        b_orig = b.to_a
        Cumo::Linalg.solve(@a, b)
        assert_equal([a_orig, b_orig], [@a.to_a, b.to_a])
      end
    end
  end

  test "the type of the answer" do
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.solve(Cumo::Int32[[1, 2], [3, 4]], Cumo::Int32[1, 2]))
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.solve(Cumo::SFloat[[1, 2], [3, 4]], Cumo::DFloat[1, 2]))
    assert_close([0.0, 0.5], Cumo::Linalg.solve(Cumo::DFloat[[1, 2], [3, 4]], [1, 2]), 1e-15)
  end

  test "driver: and uplo: are accepted and ignored" do
    assert_close([0.0, 0.5], Cumo::Linalg.solve(Cumo::DFloat[[1, 2], [3, 4]], Cumo::DFloat[1, 2], driver: 'sym', uplo: 'L'), 1e-15)
  end

  test "a singular matrix warns and answers b as LAPACK's gesv does" do
    x = nil
    _, stderr = capture_output { x = Cumo::Linalg.solve(Cumo::DFloat[[1, 2], [2, 4]], Cumo::Int32[1, 3]) }
    assert_equal("the factorization has been completed, but the factor is singular, " \
                 "so the solution could not be computed.\n",
                 stderr)
    assert_kind_of(Cumo::DFloat, x)
    assert_equal([1.0, 3.0], x.to_a)
  end

  test "empty" do
    assert_equal([0], Cumo::Linalg.solve(Cumo::DFloat.new(0, 0), Cumo::DFloat[]).shape)
  end

  test "errors" do
    m = Cumo::DFloat[[1, 2], [3, 4]]
    [
      [[Cumo::DFloat[1, 2], Cumo::DFloat[1, 2]], Cumo::NArray::ShapeError, 'input array a must be 2-dimensional'],
      [[Cumo::DFloat[[1, 2, 3], [4, 5, 6]], Cumo::DFloat[1, 2]], Cumo::NArray::ShapeError, 'input array a must be square'],
      [[m, Cumo::DFloat[1, 2, 3]], Cumo::NArray::ShapeError, 'shape1[1](=2) != shape2[0](=3)'],
      [[m, Cumo::DFloat.new(2, 1, 1).seq], ArgumentError, 'input array b must be 1- or 2-dimensional'],
      [[Cumo::RObject[[1]], Cumo::RObject[1]], TypeError, 'invalid data type for BLAS/LAPACK']
    ].each do |args, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.solve(*args) }
      assert_equal(message, error.message)
    end
  end
end
