# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgDetTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-12, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-12 }

  types.each do |type, tol|
    sub_test_case type.to_s do
      setup do
        @a = type.new(5, 5).rand - 0.5
        @expected = host_det(@a.to_a)
      end

      test "det" do
        d = Cumo::Linalg.det(@a)
        assert_kind_of(COMPLEX.include?(type) ? Complex : Float, d)
        assert_operator((d - @expected).abs, :<=, tol * [1, @expected.abs].max)
      end

      test "slogdet" do
        sign, logdet = Cumo::Linalg.slogdet(@a)
        assert_kind_of(COMPLEX.include?(type) ? Complex : Float, sign)
        assert_kind_of(Float, logdet)
        assert_in_delta(1.0, sign.abs, tol)
        assert_operator((sign * Math.exp(logdet) - @expected).abs, :<=, tol * [1, @expected.abs].max)
      end

      test "a transposed view" do
        assert_operator((Cumo::Linalg.det(@a.transpose) - @expected).abs, :<=, tol * [1, @expected.abs].max)
      end
    end
  end

  test "the sign of a permutation" do
    assert_equal(-1.0, Cumo::Linalg.det(Cumo::DFloat[[0, 1], [1, 0]]))
    assert_equal(1.0, Cumo::Linalg.det(Cumo::DFloat[[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert_equal([-1.0, 0.0], Cumo::Linalg.slogdet(Cumo::DFloat[[0, 1], [1, 0]]))
  end

  test "an integer matrix" do
    assert_equal(-2.0, Cumo::Linalg.det(Cumo::Int32[[1, 2], [3, 4]]))
    sign, logdet = Cumo::Linalg.slogdet(Cumo::Int32[[1, 2], [3, 4]])
    assert_equal(-1.0, sign)
    assert_in_delta(Math.log(2), logdet, 1e-15)
  end

  test "a singular matrix" do
    assert_equal(0.0, Cumo::Linalg.det(Cumo::DFloat[[1, 2], [2, 4]]))
    result = nil
    _, stderr = capture_output { result = Cumo::Linalg.slogdet(Cumo::DFloat[[1, 2], [2, 4]]) }
    assert_equal([0, -Float::INFINITY], result)
    assert_match(/singular/, stderr)
  end

  test "slogdet does not overflow where det does" do
    a = Cumo::DFloat.eye(3) * 1e200
    assert_equal(Float::INFINITY, Cumo::Linalg.det(a))
    sign, logdet = Cumo::Linalg.slogdet(a)
    assert_equal(1.0, sign)
    assert_in_delta(Math.log(10) * 600, logdet, 1e-10)
  end

  test "an empty matrix" do
    assert_equal(1.0, Cumo::Linalg.det(Cumo::DFloat.new(0, 0)))
    assert_equal(Complex(1.0, 0.0), Cumo::Linalg.det(Cumo::DComplex.new(0, 0)))
    assert_equal([1.0, 0.0], Cumo::Linalg.slogdet(Cumo::DFloat.new(0, 0)))
  end

  test "errors" do
    [
      [:det, [Cumo::DFloat[1, 2]], Cumo::NArray::ShapeError, 'input array a must be 2-dimensional'],
      [:det, [Cumo::DFloat[[1, 2, 3], [4, 5, 6]]], Cumo::NArray::ShapeError, 'input array a must be square'],
      [:det, [Cumo::RObject[[1]]], TypeError, 'invalid data type for BLAS/LAPACK'],
      [:slogdet, [Cumo::DFloat[1, 2]], Cumo::NArray::ShapeError, 'input array a must be 2-dimensional']
    ].each do |method, args, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.public_send(method, *args) }
      assert_equal(message, error.message)
    end
  end

  private

  COMPLEX = [Cumo::SComplex, Cumo::DComplex].freeze
  private_constant :COMPLEX

  def host_det(m)
    m = m.map(&:dup)
    n = m.size
    d = 1
    n.times do |k|
      p = (k...n).max_by { |i| m[i][k].abs }
      if p != k
        m[k], m[p] = m[p], m[k]
        d = -d
      end
      d *= m[k][k]
      ((k + 1)...n).each do |i|
        f = m[i][k] / m[k][k]
        (k...n).each { |j| m[i][j] -= f * m[k][j] }
      end
    end
    d
  end
end
