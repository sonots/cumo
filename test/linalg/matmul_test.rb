# frozen_string_literal: true

require_relative "../test_helper"
require "cumo/linalg"

class LinalgMatmulTest < Test::Unit::TestCase
  types = [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex]

  types.each do |type|
    test type.to_s do
      a = type.new(3, 2).seq(1)
      b = type.new(2, 3).seq(-2)
      c = Cumo::Linalg.matmul(a, b)
      assert_kind_of(type, c)
      assert_equal(a.dot(b).to_a, c.to_a)
      assert_equal(b.transpose.dot(a.transpose).to_a, Cumo::Linalg.matmul(b.transpose, a.transpose).to_a)
    end
  end

  test "as numo-linalg-alt does" do
    a = Cumo::DFloat[[1, 0], [0, 1]]
    b = Cumo::DFloat[[4, 1], [2, 2]]
    assert_operator((Cumo::DFloat[[4, 1], [2, 2]] - Cumo::Linalg.matmul(a, b)).abs.max, :<, 1e-7)
  end

  test "the type of the answer" do
    i = Cumo::Int32[[1, 2], [3, 4]]
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.matmul(i, i))
    assert_equal([[7.0, 10.0], [15.0, 22.0]], Cumo::Linalg.matmul(i, i).to_a)
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.matmul(Cumo::SFloat[[1, 2]], Cumo::DFloat[[3], [4]]))
    assert_kind_of(Cumo::SComplex, Cumo::Linalg.matmul(Cumo::SFloat[[1, 2]], Cumo::SComplex[[1], [2]]))
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
      [[Cumo::RObject[[1]], Cumo::RObject[[1]]], TypeError, 'invalid data type for BLAS/LAPACK']
    ].each do |args, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.matmul(*args) }
      assert_equal(message, error.message)
    end
  end
end
