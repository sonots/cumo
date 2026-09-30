# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgEigTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
    omit("cuSOLVER has no cusolverDnXgeev") unless Cumo::CUDA::Cusolver.respond_to?(:geev, true)
  end

  types = {
    Cumo::SFloat => [1e-4, Cumo::SComplex],
    Cumo::DFloat => [1e-12, Cumo::DComplex],
    Cumo::SComplex => [1e-4, Cumo::SComplex],
    Cumo::DComplex => [1e-12, Cumo::DComplex]
  }

  types.each do |type, (tol, complex)|
    sub_test_case type.to_s do
      setup do
        rng = Random.new(5)
        @host = linalg_values(type, Array.new(5) { Array.new(5) { (rng.rand * 8).round - 4 } })
        @a = type.cast(@host)
      end

      test "eigenvalues and right and left eigenvectors" do
        w, vl, vr = Cumo::Linalg.eig(@a, left: true)
        assert_kind_of(complex, w)
        assert_equal([[5], [5, 5], [5, 5]], [w.shape, vl.shape, vr.shape])
        w = w.to_a
        av = host_dot(@host, vr.to_a)
        assert_close(vr.to_a.map { |row| row.each_with_index.map { |x, j| x * w[j] } }, av, tol * 1000)
        vlh = conj_t(vl.to_a)
        assert_close(vlh.each_with_index.map { |row, i| row.map { |x| x * w[i] } }, host_dot(vlh, @host), tol * 1000)
        [vl, vr].each do |v|
          assert_close([1.0] * 5, v.to_a.transpose.map { |col| Math.sqrt(col.sum { |x| x.abs**2 }) }, tol * 100)
          assert_largest_real(v, tol * 100)
        end
        assert_close(w, Cumo::Linalg.eigvals(@a), tol * 100)
      end

      test "the eigenvectors are real when every eigenvalue is" do
        a = type.cast([[2, 1, 0], [1, 3, 1], [0, 1, 4]])
        w, vl, vr = Cumo::Linalg.eig(a, left: true)
        assert_kind_of(complex, w)
        assert_kind_of(type, vl)
        assert_kind_of(type, vr)
      end

      test "a rotation has a conjugate pair" do
        w, vl, vr = Cumo::Linalg.eig(type.cast([[0, -1, 0], [1, 0, 0], [0, 0, 2]]), left: true)
        assert_close([Complex(0, 1), Complex(0, -1), 2], w, tol)
        assert_kind_of(complex, vr)
        assert_kind_of(complex, vl)
        s = Math.sqrt(0.5)
        expected = [[s, s, 0], [Complex(0, -s), Complex(0, s), 0], [0, 0, 1]]
        assert_same_columns(expected, vr, tol)
        assert_same_columns(expected, vl, tol)
      end
    end
  end

  test "the left eigenvectors of a defective matrix" do
    _, vl, vr = Cumo::Linalg.eig(Cumo::DFloat[[1, 1], [0, 1]], left: true)
    vl.to_a.transpose.each { |col| assert_in_delta(0.0, col[0], 1e-8) }
    vr.to_a.transpose.each { |col| assert_in_delta(0.0, col[1], 1e-8) }
  end

  test "left and right choose the eigenvectors" do
    a = Cumo::DFloat[[1, 2], [3, 4]]
    w, vl, vr = Cumo::Linalg.eig(a, left: true, right: false)
    assert_equal([[2], [2, 2]], [w.shape, vl.shape])
    assert_nil(vr)
    _, vl, vr = Cumo::Linalg.eig(a, right: false)
    assert_nil(vl)
    assert_nil(vr)
    assert_equal([[1.0, 2.0], [3.0, 4.0]], a.to_a)
  end

  test "an integer matrix is decomposed in DFloat" do
    w, _, vr = Cumo::Linalg.eig(Cumo::Int32[[2, 0], [0, 3]])
    assert_kind_of(Cumo::DComplex, w)
    assert_kind_of(Cumo::DFloat, vr)
    assert_equal([Complex(2, 0), Complex(3, 0)], w.to_a)
  end

  test "an empty matrix has no eigenvalues" do
    w, vl, vr = Cumo::Linalg.eig(Cumo::DFloat.new(0, 0), left: true)
    assert_equal([[0], [0, 0], [0, 0]], [w.shape, vl.shape, vr.shape])
    assert_equal([0], Cumo::Linalg.eigvals(Cumo::SFloat.new(0, 0)).shape)
  end

  test "errors" do
    error = assert_raise(ArgumentError) { Cumo::Linalg.eig(Cumo::DFloat[[1, 2]]) }
    assert_equal('input array a must be square', error.message)
    error = assert_raise(Cumo::NArray::ShapeError) { Cumo::Linalg.eigvals(Cumo::DFloat[[1, 2]]) }
    assert_equal('input array a must be square', error.message)
    %i[eig eigvals].each do |method|
      error = assert_raise(Cumo::NArray::ShapeError) { Cumo::Linalg.public_send(method, Cumo::DFloat[1]) }
      assert_equal('input array a must be 2-dimensional', error.message)
      assert_raise(TypeError) { Cumo::Linalg.public_send(method, Cumo::HFloat[[1]]) }
    end
  end

  test "geev checks what it is handed" do
    cusolver = Cumo::CUDA::Cusolver
    assert_raise(Cumo::NArray::ShapeError) { cusolver.__send__(:geev, Cumo::DFloat.new(2, 3), Cumo::DComplex.new(2), nil) }
    assert_raise(TypeError) { cusolver.__send__(:geev, Cumo::DFloat.new(2, 2), Cumo::SComplex.new(2), nil) }
    assert_raise(Cumo::NArray::ShapeError) { cusolver.__send__(:geev, Cumo::DFloat.new(2, 2), Cumo::DComplex.new(3), nil) }
    assert_raise(Cumo::NArray::ShapeError) { cusolver.__send__(:geev, Cumo::DFloat.new(2, 2), Cumo::DComplex.new(2), Cumo::DFloat.new(3, 2)) }
    assert_raise(TypeError) { cusolver.__send__(:geev, Cumo::DFloat.new(2, 2), Cumo::DComplex.new(2), Cumo::SFloat.new(2, 2)) }
  end

  private

  def assert_same_columns(expected, actual, tol)
    expected.transpose.zip(actual.to_a.transpose) do |e, a|
      k = e.each_index.max_by { |i| e[i].abs }
      phase = a[k] / e[k]
      assert_in_delta(1.0, phase.abs, tol)
      assert_close(e.map { |x| x * phase }, a, tol)
    end
  end

  def assert_largest_real(v, tol)
    v.to_a.transpose.each do |col|
      largest = col.max_by(&:abs)
      assert_in_delta(0.0, largest.imag, tol) if largest.is_a?(Complex)
    end
  end

  def conj_t(m)
    m.transpose.map { |row| row.map { |v| v.respond_to?(:conj) ? v.conj : v } }
  end
end

class LinalgEigWithoutGeevTest < Test::Unit::TestCase
  test "eig and eigvals raise NotImplementedError without cusolverDnXgeev" do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
    omit("cuSOLVER has cusolverDnXgeev") if Cumo::CUDA::Cusolver.respond_to?(:geev, true)
    assert_raise(NotImplementedError) { Cumo::Linalg.eig(Cumo::DFloat[[1]]) }
    assert_raise(NotImplementedError) { Cumo::Linalg.eigvals(Cumo::DFloat[[1]]) }
  end
end
