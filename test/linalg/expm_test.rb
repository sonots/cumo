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

      test "a matrix whose 1-norm far exceeds its largest element" do
        n = 32
        expected = Array.new(n) { |i| Array.new(n) { |j| (i == j ? 1 : 0) + ((Math.exp(n) - 1) / n) } }
        assert_close(expected, Cumo::Linalg.expm(type.new(n, n).fill(1)), Math.exp(n) / n * tol * 100)
      end

      test "expm(a) times expm(-a) is the identity" do
        rng = Random.new(3)
        host = linalg_values(type, Array.new(5) { Array.new(5) { (rng.rand - 0.5) * 2 } })
        a = type.cast(host)
        product = host_dot(Cumo::Linalg.expm(a).to_a, Cumo::Linalg.expm(-a).to_a)
        assert_close(Array.new(5) { |i| Array.new(5) { |j| i == j ? 1 : 0 } }, product, tol * 100)
      end

      test "a triangular matrix with an off-diagonal element far larger than its diagonal" do
        b = [Cumo::SFloat, Cumo::SComplex].include?(type) ? 1e30 : 1e300
        e = Math.exp(1)
        upper = [[e, b * ((e * e) - e)], [0, e * e]]
        assert_relatively_close(upper, Cumo::Linalg.expm(type.cast([[1, b], [0, 2]])), tol)
        assert_relatively_close(upper.transpose, Cumo::Linalg.expm(type.cast([[1, 0], [b, 2]])), tol)
      end
    end
  end

  test "a 3x3 upper triangular matrix with elements far larger than its diagonal" do
    a = Cumo::DFloat[[1, 1e100, 3e99], [0, -0.5, -2e100], [0, 0, 0.25]]
    expected = [
      [2.7182818284590452, 1.4078341124976079e+100, -1.3453540529710146e+200],
      [0, 0.60653065971263342, -1.8066526852669549e+100],
      [0, 0, 1.2840254166877415]
    ]
    assert_relatively_close(expected, Cumo::Linalg.expm(a), 1e-13)
    assert_relatively_close(expected.transpose, Cumo::Linalg.expm(a.transpose.dup), 1e-13)
  end

  test "a triangular matrix with nearly equal diagonal elements" do
    e = Cumo::Linalg.expm(Cumo::DFloat[[1, 1e50], [0, 1 + 1e-9]])
    assert_relatively_close([[Math.exp(1), 2.7182818298181862e+50], [0, 2.7182818311773271]], e, 1e-13)
  end

  def assert_relatively_close(expected, actual, tol)
    actual = actual.to_a.flatten
    expected.flatten.each_with_index do |x, i|
      assert_operator((actual[i] - x).abs, :<=, x.abs * tol, "element #{i}: #{actual[i]} for #{x}")
    end
  end

  [1e-3, 0.02, 0.2, 0.5, 1.5, 3.0].each do |scale|
    test "a matrix scaled by #{scale} agrees with its Taylor series" do
      rng = Random.new(5)
      a = Array.new(6) { Array.new(6) { (rng.rand - 0.5) * 2 * scale } }
      term = Array.new(6) { |i| Array.new(6) { |j| i == j ? 1.0 : 0.0 } }
      expected = term
      (1..60).each do |k|
        term = host_dot(term, a).map { |row| row.map { |x| x / k } }
        expected = expected.zip(term).map { |row, t| row.zip(t).map(&:sum) }
      end
      assert_close(expected, Cumo::Linalg.expm(Cumo::DFloat.cast(a)), Math.exp(scale * 6) * 1e-14)
    end
  end

  test "the logarithm in experiment 1 of Al-Mohy and Higham (2012) exponentiates back" do
    expected = [[3.2346e-1, 3e4, 3e4, 3e4], [0, 3.0089e-1, 3e4, 3e4], [0, 0, 3.221e-1, 3e4], [0, 0, 0, 3.0744e-1]]
    logm = Cumo::DFloat[
      [-1.12867982029050462e+00, 9.61418377142025565e+04, -4.52485573953179264e+09, 2.92496941103871812e+14],
      [0, -1.20101052953082288e+00, 9.63469687211303099e+04, -4.68104828911105442e+09],
      [0, 0, -1.13289322264498393e+00, 9.53249183094775653e+04],
      [0, 0, 0, -1.17947533272554850e+00]
    ]
    expected.flatten.zip(Cumo::Linalg.expm(logm).to_a.flatten) do |e, x|
      assert_in_delta(e, x, e.abs * 1e-4)
    end
  end

  { Cumo::DFloat => [1e31, 1e-12], Cumo::SFloat => [1e4, 1e-6] }.each do |type, (b, tol)|
    test "a #{type} triangular matrix of off-diagonal #{b} keeps its diagonal" do
      e = Math::E
      expected = [e, b * ((e * e) - e), 0, e * e]
      expected.zip(Cumo::Linalg.expm(type[[1, b], [0, 2]]).to_a.flatten) do |x, actual|
        assert_in_delta(x, actual, x.abs * tol)
      end
    end
  end

  test "the backward error bound ell agrees with scipy" do
    [[1e4, 3, 0, 10], [1e4, 3, 3, 7], [1e4, 5, 3, 2], [1e4, 7, 0, 2], [1e4, 9, 0, 1], [1e8, 3, 10, 2], [1e20, 13, 0, 2]].each do |b, m, s, ell|
      a = Cumo::DFloat[[1, b, b], [0, 2, b], [0, 0, 0.5]]
      norm = Cumo::Linalg.__send__(:onenorm, a)
      assert_equal(ell, Cumo::Linalg.__send__(:pade_ell, a.abs, norm, m, s), [b, m, s].inspect)
    end
  end

  test "the tenth power is formed only when it can change the number of squarings" do
    squarings = ->(a4, a6, d6, d8) { Cumo::Linalg.__send__(:pade13_squarings, a4, a6, d6, d8) }
    assert_equal([3, 3], [squarings.call(nil, nil, 10.0, 30.0), squarings.call(nil, nil, 32.0, 20.0)])
    eye = Cumo::DFloat.eye(2)
    assert_equal(3, squarings.call(eye * (30.0**4), eye * (30.0**6), 60.0, 10.0))
    assert_equal(2, squarings.call(eye * (5.0**4), eye * (5.0**6), 60.0, 10.0))
    assert_equal(4, squarings.call(eye * (100.0**4), eye * (100.0**6), 60.0, 10.0))
    assert_nil(squarings.call(eye * 1e200, eye * 1e200, 60.0, 10.0))
  end

  test "a nilpotent matrix whose eighth power vanishes takes the degree 13 approximant" do
    a = Cumo::DFloat.zeros(8, 8)
    7.times { |i| a[i, i + 1] = 10 }
    expected = Array.new(8) { |i| Array.new(8) { |j| j >= i ? (10.0**(j - i)) / (1..(j - i)).reduce(1, :*) : 0 } }
    assert_close(expected, Cumo::Linalg.expm(a), (10.0**7) / 5040 * 1e-12)
  end

  sub_test_case "compatible mode" do
    setup do
      @orig_compatible_mode = Cumo.compatible_mode_enabled?
      Cumo.enable_compatible_mode
    end

    teardown do
      @orig_compatible_mode ? Cumo.enable_compatible_mode : Cumo.disable_compatible_mode
    end

    test "expm reads its norms back" do
      a = Cumo::DFloat[[1, 2], [3, 4]]
      assert_close([[51.96895619870497, 74.73656456700311], [112.1048468505047, 164.07380304920954]], Cumo::Linalg.expm(a), 1e-12)
      e = Cumo::Linalg.expm(Cumo::SComplex[[0, Complex(0, 1)], [Complex(0, 1), 0]])
      assert_close([[Math.cos(1), Complex(0, Math.sin(1))], [Complex(0, Math.sin(1)), Math.cos(1)]], e, 1e-6)
      n = Cumo::DFloat.zeros(8, 8)
      7.times { |i| n[i, i + 1] = 10 }
      assert_in_delta(1e7 / 5040, Cumo::Linalg.expm(n)[0, 7], 1e-6)
    end
  end

  test "a matrix with a NaN answers NaN" do
    assert(Cumo::Linalg.expm(Cumo::DFloat[[Float::NAN, 0], [0, 1]]).to_a.flatten.all?(&:nan?))
  end

  test "a zero matrix answers the identity" do
    assert_equal([[1.0, 0.0], [0.0, 1.0]], Cumo::Linalg.expm(Cumo::DFloat.zeros(2, 2)).to_a)
  end

  test "a double precision matrix of norm near the largest float is scaled without overflow" do
    assert_equal([[0.0, 0.0], [0.0, 0.0]], Cumo::Linalg.expm(Cumo::DFloat[[-1e308, 0], [0, -2e307]]).to_a)
  end

  test "a single precision matrix of norm near the largest float is scaled without overflow" do
    assert_equal([[0.0, 0.0], [0.0, 0.0]], Cumo::Linalg.expm(Cumo::SFloat[[-3e38, 0], [0, -2e38]]).to_a)
  end

  test "a complex matrix" do
    e = Cumo::Linalg.expm(Cumo::DComplex[[0, Complex(0, 1)], [Complex(0, 1), 0]])
    assert_close([[Math.cos(1), Complex(0, Math.sin(1))], [Complex(0, Math.sin(1)), Math.cos(1)]], e, 1e-13)
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
    error = assert_raise(FloatDomainError) { Cumo::Linalg.expm(Cumo::DFloat[[Float::INFINITY, 0], [0, 1]]) }
    assert_equal('Infinity', error.message)
    { Cumo::SFloat => -3e38, Cumo::DFloat => -1e308, Cumo::SComplex => -3e38, Cumo::DComplex => -1e308 }.each do |type, v|
      error = assert_raise(FloatDomainError) { Cumo::Linalg.expm(type[[v, 0], [v, 0]]) }
      assert_equal("the 1-norm of the matrix overflows #{type}", error.message)
    end
  end
end
