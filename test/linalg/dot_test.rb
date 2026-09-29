# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgDotTest < Test::Unit::TestCase
  include LinalgTestHelper

  V = [1, 2, 3].freeze
  W = [-1, 0, 2].freeze
  A = [[1, -2], [3, 0], [2, 1]].freeze
  B = [[2, 0, -1], [1, 3, 1]].freeze
  private_constant :V, :W, :A, :B

  types = [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex]

  types.each do |type|
    sub_test_case type.to_s do
      setup do
        @v, @w, @a, @b = [V, W, A, B].map { |x| linalg_values(type, x) }
        @cv, @cw, @ca, @cb = [@v, @w, @a, @b].map { |x| type.cast(x) }
      end

      test "vector and vector answer a Ruby number" do
        c = Cumo::Linalg.dot(@cv, @cw)
        assert_kind_of(COMPLEX_TYPES.include?(type) ? Complex : Float, c)
        assert_equal(host_dot(@v, @w), c)
      end

      test "vector and matrix" do
        c = Cumo::Linalg.dot(@cv, @ca)
        assert_kind_of(type, c)
        assert_equal(host_dot(@v, @a), c.to_a)
      end

      test "matrix and vector" do
        c = Cumo::Linalg.dot(@cb, @cv)
        assert_kind_of(type, c)
        assert_equal(host_dot(@b, @v), c.to_a)
      end

      test "matrix and matrix" do
        c = Cumo::Linalg.dot(@ca, @cb)
        assert_kind_of(type, c)
        assert_equal(host_dot(@a, @b), c.to_a)
      end

      test "transposed views" do
        assert_equal(host_dot(@a.transpose, @v), Cumo::Linalg.dot(@ca.transpose, @cv).to_a)
        assert_equal(host_dot(@v[0...2], @a.transpose), Cumo::Linalg.dot(@cv[0...2], @ca.transpose).to_a)
        assert_equal(host_dot(@a, @a.transpose), Cumo::Linalg.dot(@ca, @ca.transpose).to_a)
        assert_equal(host_dot(@b.transpose, @a.transpose), Cumo::Linalg.dot(@cb.transpose, @ca.transpose).to_a)
      end

      test "index views" do
        rows = [2, 0]
        assert_equal(host_dot(@a.values_at(*rows), @b), Cumo::Linalg.dot(@ca[rows, true], @cb).to_a)
        assert_equal(host_dot(@v.values_at(*rows), @v.values_at(0, 1)), Cumo::Linalg.dot(@cv[rows], @cv[[0, 1]]))
        assert_equal(host_dot(@b, @v.values_at(2, 1, 0)), Cumo::Linalg.dot(@cb, @cv[[2, 1, 0]]).to_a)
      end
    end
  end

  test "mixed float types as numo-linalg-alt does" do
    a = Cumo::DFloat.new(3).rand
    b = Cumo::SFloat.new(3).rand
    c = Cumo::DFloat.new(3, 2).rand
    d = Cumo::SFloat.new(2, 3).rand
    assert_close(host_dot(a.to_a, b.to_a), Cumo::Linalg.dot(a, b), 1e-7)
    assert_close(host_dot(a.to_a, c.to_a), Cumo::Linalg.dot(a, c), 1e-7)
    assert_close(host_dot(c.transpose.to_a, b.to_a), Cumo::Linalg.dot(c.transpose, b), 1e-7)
    assert_close(host_dot(c.to_a, d.to_a), Cumo::Linalg.dot(c, d), 1e-7)
  end

  test "the type of the answer" do
    i = Cumo::Int32[[1, 2], [3, 4]]
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.dot(i, i))
    assert_equal([[7.0, 10.0], [15.0, 22.0]], Cumo::Linalg.dot(i, i).to_a)
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.dot(Cumo::SFloat[[1, 2]], Cumo::DFloat[3, 4]))
    assert_kind_of(Cumo::SComplex, Cumo::Linalg.dot(Cumo::SFloat[[1, 2]], Cumo::SComplex[[1], [2]]))
    assert_kind_of(Cumo::DComplex, Cumo::Linalg.dot(Cumo::DFloat[[1, 2]], Cumo::SComplex[[1], [2]]))
  end

  test "Bit" do
    assert_equal(2.0, Cumo::Linalg.dot(Cumo::Bit[1, 0, 1], Cumo::Bit[1, 1, 1]))
    c = Cumo::Linalg.dot(Cumo::Bit[[1, 0], [1, 1]], Cumo::Bit[[0, 1], [1, 1]])
    assert_kind_of(Cumo::DFloat, c)
    assert_equal([[0.0, 1.0], [1.0, 2.0]], c.to_a)
  end

  test "Arrays" do
    c = Cumo::Linalg.dot([[1, 2], [3, 4]], [5, 6])
    assert_kind_of(Cumo::DFloat, c)
    assert_equal([17.0, 39.0], c.to_a)
    assert_equal(11.0, Cumo::Linalg.dot([1, 2], [3, 4]))
  end

  test "complex vectors are not conjugated" do
    assert_equal(Complex(2, 2), Cumo::Linalg.dot(Cumo::DComplex[Complex(1, 1), 2], Cumo::DComplex[Complex(1, 1), 1]))
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
      [[Cumo::HFloat[[1]], Cumo::HFloat[[1]]], TypeError, 'invalid data type for BLAS/LAPACK'],
      [[Cumo::HFloat[[1]], Cumo::DFloat[[1]]], TypeError, 'invalid data type for BLAS/LAPACK'],
      [[Cumo::DFloat[[1]], Cumo::BFloat[[1]]], TypeError, 'invalid data type for BLAS/LAPACK']
    ].each do |args, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.dot(*args) }
      assert_equal(message, error.message)
    end
  end
end
