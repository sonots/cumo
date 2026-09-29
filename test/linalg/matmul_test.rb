# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgMatmulTest < Test::Unit::TestCase
  include LinalgTestHelper

  A = [[1, -2], [3, 0], [2, 1]].freeze
  B = [[2, 0, -1], [1, 3, 1]].freeze
  private_constant :A, :B

  types = [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex]

  types.each do |type|
    test type.to_s do
      a = linalg_values(type, A)
      b = linalg_values(type, B)
      c = Cumo::Linalg.matmul(type.cast(a), type.cast(b))
      assert_kind_of(type, c)
      assert_equal(host_dot(a, b), c.to_a)
      assert_equal(host_dot(b.transpose, a.transpose), Cumo::Linalg.matmul(type.cast(b).transpose, type.cast(a).transpose).to_a)
      assert_equal(host_dot(a.values_at(2, 0), b), Cumo::Linalg.matmul(type.cast(a)[[2, 0], true], type.cast(b)).to_a)
    end
  end

  test "as numo-linalg-alt does" do
    a = Cumo::DFloat[[1, 0], [0, 1]]
    b = Cumo::DFloat[[4, 1], [2, 2]]
    assert_close([[4, 1], [2, 2]], Cumo::Linalg.matmul(a, b), 1e-7)
  end

  test "the type of the answer" do
    i = Cumo::Int32[[1, 2], [3, 4]]
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.matmul(i, i))
    assert_equal([[7.0, 10.0], [15.0, 22.0]], Cumo::Linalg.matmul(i, i).to_a)
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.matmul(Cumo::SFloat[[1, 2]], Cumo::DFloat[[3], [4]]))
    assert_kind_of(Cumo::SComplex, Cumo::Linalg.matmul(Cumo::SFloat[[1, 2]], Cumo::SComplex[[1], [2]]))
    assert_equal([[0.0, 1.0], [1.0, 2.0]], Cumo::Linalg.matmul(Cumo::Bit[[1, 0], [1, 1]], Cumo::Bit[[0, 1], [1, 1]]).to_a)
  end

  test "Arrays" do
    c = Cumo::Linalg.matmul([[1, 2], [3, 4]], [[5], [6]])
    assert_kind_of(Cumo::DFloat, c)
    assert_equal([[17.0], [39.0]], c.to_a)
  end

  test "errors" do
    m = Cumo::DFloat[[1, 2], [3, 4]]
    c3 = Cumo::DFloat.new(2, 2, 2).seq
    [
      [[Cumo::DFloat[1, 2], Cumo::DFloat[3, 4]], ArgumentError, 'a must be 2-dimensional'],
      [[m, Cumo::DFloat[3, 4]], ArgumentError, 'b must be 2-dimensional'],
      [[c3, m], ArgumentError, 'a must be 2-dimensional'],
      [[Cumo::DFloat.new(0, 2), m], ArgumentError, 'a must not be empty'],
      [[m, Cumo::DFloat.new(2, 0)], ArgumentError, 'b must not be empty'],
      [[Cumo::DFloat.ones(2, 3), Cumo::DFloat.ones(2, 3)], Cumo::NArray::ShapeError, 'shape1[1](=3) != shape2[0](=2)'],
      [[Cumo::RObject[[1]], Cumo::RObject[[1]]], TypeError, 'invalid data type for BLAS/LAPACK'],
      [[Cumo::SFloat[[1]], Cumo::HFloat[[1]]], TypeError, 'invalid data type for BLAS/LAPACK']
    ].each do |args, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.matmul(*args) }
      assert_equal(message, error.message)
    end
  end
end
