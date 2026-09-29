# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgInvTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-10, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-10 }

  types.each do |type, tol|
    sub_test_case type.to_s do
      setup do
        @a = type.new(5, 5).rand - 0.5
        @identity = Array.new(5) { |i| Array.new(5) { |j| i == j ? 1 : 0 } }
      end

      test "inv" do
        x = Cumo::Linalg.inv(@a)
        assert_kind_of(type, x)
        assert_close(@identity, host_dot(@a.to_a, x.to_a), tol)
        assert_close(@identity, host_dot(x.to_a, @a.to_a), tol)
      end

      test "lu_inv" do
        lu, piv = Cumo::Linalg.lu_fact(@a)
        x = Cumo::Linalg.lu_inv(lu, piv)
        assert_kind_of(type, x)
        assert_close(@identity, host_dot(@a.to_a, x.to_a), tol)
      end

      test "inv of a transposed view" do
        x = Cumo::Linalg.inv(@a.transpose)
        assert_close(@identity, host_dot(@a.to_a.transpose, x.to_a), tol)
      end
    end
  end

  test "inv of an integer matrix" do
    x = Cumo::Linalg.inv(Cumo::Int32[[1, 2], [3, 4]])
    assert_kind_of(Cumo::DFloat, x)
    assert_close([[-2.0, 1.0], [1.5, -0.5]], x, 1e-15)
  end

  test "driver: and uplo: are accepted and ignored" do
    assert_close([[-2.0, 1.0], [1.5, -0.5]], Cumo::Linalg.inv(Cumo::DFloat[[1, 2], [3, 4]], driver: 'gesv', uplo: 'L'), 1e-15)
  end

  test "a singular matrix raises" do
    error = assert_raise(Cumo::Linalg::LapackError) { Cumo::Linalg.inv(Cumo::DFloat[[1, 2], [2, 4]]) }
    assert_equal('The matrix is singular, and the inverse matrix could not be computed.', error.message)

    lu = piv = nil
    capture_output { lu, piv = Cumo::Linalg.lu_fact(Cumo::DFloat[[1, 2], [2, 4]]) }
    error = assert_raise(Cumo::Linalg::LapackError) { Cumo::Linalg.lu_inv(lu, piv) }
    assert_equal('the matrix is singular and its inverse could not be computed', error.message)
  end

  test "an empty matrix answers an empty matrix" do
    assert_equal([0, 0], Cumo::Linalg.inv(Cumo::DFloat.new(0, 0)).shape)
    assert_equal([0, 0], Cumo::Linalg.lu_inv(Cumo::DFloat.new(0, 0), Cumo::Int32[]).shape)
  end

  test "errors" do
    [
      [:inv, [Cumo::DFloat[1, 2]], Cumo::NArray::ShapeError, 'input array a must be 2-dimensional'],
      [:inv, [Cumo::DFloat[[1, 2, 3], [4, 5, 6]]], Cumo::NArray::ShapeError, 'input array a must be square'],
      [:inv, [Cumo::RObject[[1]]], TypeError, 'invalid data type for BLAS/LAPACK'],
      [:lu_inv, [Cumo::DFloat[1, 2], Cumo::Int32[1, 2]], ArgumentError, 'input array a must be 2-dimensional'],
      [:lu_inv, [Cumo::DFloat[[1, 2, 3], [4, 5, 6]], Cumo::Int32[1, 2]], ArgumentError, 'input array a must be square'],
      [:lu_inv, [Cumo::DFloat[[1, 2], [3, 4]], Cumo::Int32[[1], [2]]], ArgumentError, 'input array ipiv must be 1-dimensional'],
      [:lu_inv, [Cumo::DFloat[[1, 2], [3, 4]], Cumo::Int32[1]], ArgumentError, 'input array ipiv must have 2 elements'],
      [:lu_inv, [Cumo::DFloat[[1, 2], [3, 4]], Cumo::Int32[1, 5]], ArgumentError, 'input array ipiv must be in 1..2']
    ].each do |method, args, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.public_send(method, *args) }
      assert_equal(message, error.message)
    end
  end
end
