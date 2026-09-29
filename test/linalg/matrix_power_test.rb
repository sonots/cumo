# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgMatrixPowerTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  A = [[1, 2], [3, 4]].freeze
  private_constant :A

  test "positive powers" do
    assert_equal([[37.0, 54.0], [81.0, 118.0]], Cumo::Linalg.matrix_power(Cumo::DFloat.cast(A), 3).to_a)
    assert_equal(host_dot(A, A), Cumo::Linalg.matrix_power(Cumo::DFloat.cast(A), 2).to_a)
  end

  test "the power of one and zero keep the class of the input" do
    one = Cumo::Linalg.matrix_power(Cumo::Int32.cast(A), 1)
    assert_kind_of(Cumo::Int32, one)
    assert_equal(A, one.to_a)
    zero = Cumo::Linalg.matrix_power(Cumo::Int32.cast(A), 0)
    assert_kind_of(Cumo::Int32, zero)
    assert_equal([[1, 0], [0, 1]], zero.to_a)
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.matrix_power(Cumo::Int32.cast(A), 2))
  end

  test "negative powers" do
    assert_close([[-2.0, 1.0], [1.5, -0.5]], Cumo::Linalg.matrix_power(Cumo::DFloat.cast(A), -1), 1e-14)
    assert_close([[5.5, -2.5], [-3.75, 1.75]], Cumo::Linalg.matrix_power(Cumo::DFloat.cast(A), -2), 1e-13)
  end

  test "errors" do
    [
      [[Cumo::DFloat[1, 2], 2], Cumo::NArray::ShapeError, 'input array a must be 2-dimensional'],
      [[Cumo::DFloat[[1, 2, 3], [4, 5, 6]], 2], Cumo::NArray::ShapeError, 'input array a must be square'],
      [[Cumo::DFloat.cast(A), 1.5], ArgumentError, 'exponent n must be an integer: 1.5'],
      [[Cumo::DFloat[[1, 2], [2, 4]], -1], Cumo::Linalg::LapackError, 'The matrix is singular, and the inverse matrix could not be computed.']
    ].each do |args, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.matrix_power(*args) }
      assert_equal(message, error.message)
    end
  end
end
