# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgExpmTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-5, Cumo::DFloat => 1e-13, Cumo::SComplex => 1e-5, Cumo::DComplex => 1e-13 }

  types.each do |type, tol|
    sub_test_case type.to_s do
      test "a diagonal matrix" do
        e = Cumo::Linalg.expm(type.cast([[1, 0], [0, -2]]))
        assert_kind_of(type, e)
        assert_close([[Math.exp(1), 0], [0, Math.exp(-2)]], e, Math.exp(1) * tol)
      end

      test "a rotation" do
        t = 0.7
        expected = [[Math.cos(t), -Math.sin(t)], [Math.sin(t), Math.cos(t)]]
        assert_close(expected, Cumo::Linalg.expm(type.cast([[0, -t], [t, 0]])), tol)
      end

      test "a nilpotent matrix" do
        assert_close([[1, 1, 3.5], [0, 1, 3], [0, 0, 1]], Cumo::Linalg.expm(type.cast([[0, 1, 2], [0, 0, 3], [0, 0, 0]])), tol * 10)
      end

      test "a matrix of large norm is scaled and squared" do
        e = Cumo::Linalg.expm(type.cast([[20, 0], [0, 1]]))
        assert_in_delta(1.0, e[0, 0].to_a.flatten.first.real / Math.exp(20), tol * 100)
        assert_in_delta(Math.exp(1), e[1, 1].to_a.flatten.first.real, tol * 100)
      end

      test "expm(a) times expm(-a) is the identity" do
        rng = Random.new(3)
        host = linalg_values(type, Array.new(5) { Array.new(5) { (rng.rand - 0.5) * 2 } })
        a = type.cast(host)
        product = host_dot(Cumo::Linalg.expm(a).to_a, Cumo::Linalg.expm(-a).to_a)
        assert_close(Array.new(5) { |i| Array.new(5) { |j| i == j ? 1 : 0 } }, product, tol * 100)
      end
    end
  end

  test "a complex matrix" do
    e = Cumo::Linalg.expm(Cumo::DComplex[[0, Complex(0, 1)], [Complex(0, 1), 0]])
    assert_close([[Math.cos(1), Complex(0, Math.sin(1))], [Complex(0, Math.sin(1)), Math.cos(1)]], e, 1e-15)
  end

  test "the values numo-linalg-alt answers" do
    a = Cumo::DFloat[[1, 2], [3, 4]]
    assert_close([[51.96895619870497, 74.73656456700311], [112.1048468505047, 164.07380304920954]], Cumo::Linalg.expm(a), 1e-12)
    assert_close([[51.96921208507579, 74.73693750273655], [112.10540625410484, 164.07461833918066]], Cumo::Linalg.expm(a, 3), 1e-12)
    assert_equal([[1.0, 2.0], [3.0, 4.0]], a.to_a)
  end

  test "integers and Bit are exponentiated in DFloat" do
    e = Cumo::Linalg.expm(Cumo::Int32[[1, 2], [3, 4]])
    assert_kind_of(Cumo::DFloat, e)
    assert_close([[51.96895619870497, 74.73656456700311], [112.1048468505047, 164.07380304920954]], e, 1e-12)
    assert_close([[Math::E, 0], [Math::E, Math::E]], Cumo::Linalg.expm(Cumo::Bit[[1, 0], [1, 1]]), 1e-14)
  end

  test "an empty matrix answers an empty matrix" do
    e = Cumo::Linalg.expm(Cumo::Int32.new(0, 0))
    assert_equal([Cumo::DFloat, [0, 0]], [e.class, e.shape])
  end

  test "errors" do
    error = assert_raise(Cumo::NArray::ShapeError) { Cumo::Linalg.expm(Cumo::DFloat[[1, 2]]) }
    assert_equal('input array a must be square', error.message)
    error = assert_raise(Cumo::NArray::ShapeError) { Cumo::Linalg.expm(Cumo::DFloat[1]) }
    assert_equal('input array a must be 2-dimensional', error.message)
    assert_raise(TypeError) { Cumo::Linalg.expm(Cumo::HFloat[[1]]) }
    assert_raise(FloatDomainError) { Cumo::Linalg.expm(Cumo::DFloat[[Float::INFINITY, 0], [0, 1]]) }
  end
end
