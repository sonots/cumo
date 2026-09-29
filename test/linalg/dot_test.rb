# frozen_string_literal: true

require_relative "../test_helper"
require "cumo/linalg"

class LinalgDotTest < Test::Unit::TestCase
  types = [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex]

  types.each do |type|
    sub_test_case type.to_s do
      setup do
        @v = type.new(3).seq(1)
        @w = type.new(3).seq(-1)
        @a = type.new(3, 2).seq(1)
        @b = type.new(2, 3).seq(-2)
      end

      test "vector and vector" do
        c = Cumo::Linalg.dot(@v, @w)
        assert_kind_of(type, c)
        assert_equal(0, c.ndim)
        assert_equal(@v.to_a.zip(@w.to_a).sum { |x, y| x * y }, c.to_a.flatten.first)
      end

      test "vector and matrix" do
        c = Cumo::Linalg.dot(@v, @a)
        assert_kind_of(type, c)
        assert_equal(@v.dot(@a).to_a, c.to_a)
      end

      test "matrix and vector" do
        c = Cumo::Linalg.dot(@b, @v)
        assert_kind_of(type, c)
        assert_equal(@b.dot(@v).to_a, c.to_a)
      end

      test "matrix and matrix" do
        c = Cumo::Linalg.dot(@a, @b)
        assert_kind_of(type, c)
        assert_equal(@a.dot(@b).to_a, c.to_a)
      end

      test "transposed views" do
        assert_equal(@a.transpose.dup.dot(@v).to_a, Cumo::Linalg.dot(@a.transpose, @v).to_a)
        assert_equal(@v[0...2].dot(@a.transpose.dup).to_a, Cumo::Linalg.dot(@v[0...2], @a.transpose).to_a)
        assert_equal(@a.dot(@a.transpose.dup).to_a, Cumo::Linalg.dot(@a, @a.transpose).to_a)
        assert_equal(@b.transpose.dup.dot(@a.transpose.dup).to_a, Cumo::Linalg.dot(@b.transpose, @a.transpose).to_a)
      end
    end
  end

  test "mixed float types as numo-linalg-alt does" do
    a = Cumo::DFloat.new(3).rand
    b = Cumo::SFloat.new(3).rand
    c = Cumo::DFloat.new(3, 2).rand
    d = Cumo::SFloat.new(2, 3).rand
    assert_operator((a.dot(b) - Cumo::Linalg.dot(a, b)).abs, :<, 1e-7)
    assert_operator((a.dot(c) - Cumo::Linalg.dot(a, c)).abs.max, :<, 1e-7)
    assert_operator((c.transpose.dot(b) - Cumo::Linalg.dot(c.transpose, b)).abs.max, :<, 1e-7)
    assert_operator((c.dot(d) - Cumo::Linalg.dot(c, d)).abs.max, :<, 1e-7)
  end

  test "the type of the answer" do
    i = Cumo::Int32[[1, 2], [3, 4]]
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.dot(i, i))
    assert_equal([[7.0, 10.0], [15.0, 22.0]], Cumo::Linalg.dot(i, i).to_a)
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.dot(Cumo::SFloat[1, 2], Cumo::DFloat[3, 4]))
    assert_kind_of(Cumo::SComplex, Cumo::Linalg.dot(Cumo::SFloat[[1, 2]], Cumo::SComplex[[1], [2]]))
    assert_kind_of(Cumo::DComplex, Cumo::Linalg.dot(Cumo::DFloat[[1, 2]], Cumo::SComplex[[1], [2]]))
  end

  test "Arrays" do
    c = Cumo::Linalg.dot([[1, 2], [3, 4]], [5, 6])
    assert_kind_of(Cumo::DFloat, c)
    assert_equal([17.0, 39.0], c.to_a)
  end

  test "complex vectors are not conjugated" do
    c = Cumo::Linalg.dot(Cumo::DComplex[Complex(1, 1), 2], Cumo::DComplex[Complex(1, 1), 1])
    assert_equal(Complex(2, 2), c.to_a.flatten.first)
  end

  sub_test_case "compatible mode" do
    setup do
      @orig_compatible_mode = Cumo.compatible_mode_enabled?
      Cumo.enable_compatible_mode
    end

    teardown do
      @orig_compatible_mode ? Cumo.enable_compatible_mode : Cumo.disable_compatible_mode
    end

    test "vector and vector answer a Ruby number" do
      assert_equal(11.0, Cumo::Linalg.dot(Cumo::DFloat[1, 2], Cumo::DFloat[3, 4]))
      assert_equal(Complex(2, 2), Cumo::Linalg.dot(Cumo::DComplex[Complex(1, 1), 2], Cumo::DComplex[Complex(1, 1), 1]))
    end
  end

  test "errors" do
    v = Cumo::DFloat[1, 2]
    m = Cumo::DFloat[[1, 2], [3, 4]]
    c3 = Cumo::DFloat.new(2, 2, 2).seq
    [
      [[v, Cumo::DFloat[1, 2, 3]], ArgumentError, 'x and y must have same size'],
      [[Cumo::DFloat[], Cumo::DFloat[]], ArgumentError, 'x must not be empty'],
      [[Cumo::DFloat[1], Cumo::DFloat[]], ArgumentError, 'x must not be empty'],
      [[Cumo::DFloat[1, 2, 3], m], Cumo::NArray::ShapeError, 'shape1[1](=2) != shape2[0](=3)'],
      [[m, Cumo::DFloat[1, 2, 3]], Cumo::NArray::ShapeError, 'shape1[1](=2) != shape2[0](=3)'],
      [[Cumo::DFloat.ones(2, 3), Cumo::DFloat.ones(2, 3)], Cumo::NArray::ShapeError, 'shape1[1](=3) != shape2[0](=2)'],
      [[Cumo::DFloat.new(0, 2), v], ArgumentError, 'a must not be empty'],
      [[v, Cumo::DFloat.new(2, 0)], ArgumentError, 'a must not be empty'],
      [[Cumo::DFloat.new(0, 0), Cumo::DFloat.new(0, 0)], ArgumentError, 'a must not be empty'],
      [[c3, m], ArgumentError, 'a must be 2-dimensional'],
      [[m, c3], ArgumentError, 'b must be 2-dimensional'],
      [[v, c3], ArgumentError, 'a must be 2-dimensional'],
      [[Cumo::RObject[[1]], Cumo::RObject[[1]]], TypeError, 'invalid data type for BLAS/LAPACK'],
      [[Cumo::HFloat[[1]], Cumo::HFloat[[1]]], TypeError, 'invalid data type for BLAS/LAPACK']
    ].each do |args, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.dot(*args) }
      assert_equal(message, error.message)
    end
  end
end
