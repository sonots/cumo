# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgLdlTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-12, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-12 }

  types.each do |type, tol|
    complex = [Cumo::SComplex, Cumo::DComplex].include?(type)

    (complex ? [true, false] : [true]).each do |hermitian|
      %w[U L].each do |uplo|
        sub_test_case "#{type} uplo=#{uplo} hermitian=#{hermitian}" do
          setup do
            rng = Random.new(uplo.ord)
            @host = Array.new(6) { Array.new(6) { (rng.rand * 8).round - 4 + (complex ? Complex(0, (rng.rand * 4).round - 2) : 0) } }
            @host.each_with_index { |row, i| row[i] = 0 if i.even? }
            @a = type.cast(@host)
          end

          test "the factors rebuild the matrix from its uplo triangle" do
            u, d, perm = ldl(@a, uplo, hermitian)
            assert_equal([type, type, Cumo::Int32], [u.class, d.class, perm.class])
            conj = complex && hermitian
            full = Array.new(6) do |i|
              Array.new(6) do |j|
                next (conj ? @host[i][i].real : @host[i][i]) if i == j

                inside = uplo == 'U' ? i < j : i > j
                v = inside ? @host[i][j] : @host[j][i]
                conj && !inside ? v.conj : v
              end
            end
            ut = u.to_a.transpose.map { |row| row.map { |v| conj ? v.conj : v } }
            assert_close(full, host_dot(host_dot(u.to_a, d.to_a), ut), tol * 100)
            rows = perm.to_a.map { |p| u.to_a[p] }
            rows.each_with_index do |row, i|
              row.each_with_index { |v, j| assert_equal(0, v) unless uplo == 'U' ? j >= i : j <= i }
            end
          end
        end
      end
    end
  end

  test "the factors numo-linalg-alt answers" do
    u, d, perm = Cumo::Linalg.ldl(Cumo::DFloat[[4, 2, 1], [2, -3, 5], [1, 5, 0]])
    assert_close([[1.0, 0.2, 0.52], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], u, 1e-15)
    assert_close([[3.08, 0.0, 0.0], [0.0, -3.0, 5.0], [0.0, 5.0, 0.0]], d, 1e-14)
    assert_equal([0, 1, 2], perm.to_a)
    l, d, perm = Cumo::Linalg.ldl(Cumo::DFloat[[0, 1, 2], [1, 0, 3], [2, 3, 0]], uplo: 'L')
    assert_close([[1.0, 0.0, 0.0], [1.5, 0.5, 1.0], [0.0, 1.0, 0.0]], l, 1e-15)
    assert_close([[0.0, 2.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, -3.0]], d, 1e-15)
    assert_equal([0, 2, 1], perm.to_a)
  end

  test "a complex matrix is Hermitian or symmetric" do
    z = Cumo::DComplex[[2, Complex(1, 1)], [Complex(1, -1), 3]]
    u, d, = ldl(z, 'U', true)
    assert_close([[1, Complex(1.0 / 3, 1.0 / 3)], [0, 1]], u, 1e-15)
    assert_close([[4.0 / 3, 0], [0, 3]], d, 1e-15)
    _, d, = Cumo::Linalg.ldl(z, hermitian: false)
    assert_close([[Complex(2, -2.0 / 3), 0], [0, 3]], d, 1e-15)
  end

  test "the imaginary part of the diagonal of a Hermitian matrix is not read" do
    real = ldl(Cumo::DComplex[[-4, -1], [-1, 0]], 'U', true)
    imaginary = ldl(Cumo::DComplex[[Complex(-4, 1), -1], [0, Complex(0, -2)]], 'U', true)
    real.zip(imaginary) { |r, i| assert_equal(r.to_a, i.to_a) }
    _, d, = ldl(Cumo::DComplex[[Complex(0, -2), Complex(0, -1)], [Complex(0, 1), -2]], 'L', true)
    assert_false(d.to_a.flatten.any? { |v| v.abs.nan? || v.abs.infinite? })
  end

  test "sytrf zeroes the imaginary part of the diagonal itself" do
    a = Cumo::DComplex[[Complex(-4, 1), 0], [-1, Complex(0, -2)]]
    begin
      _, info = Cumo::CUDA::Cusolver.__send__(:sytrf, a, 'U', true)
    rescue NotImplementedError => e
      omit(e.message)
    end
    assert_equal(0, info)
    assert_false(a.to_a.flatten.any? { |v| v.abs.nan? || v.abs.infinite? })
  end

  test "an integer matrix is factored in DFloat" do
    u, d, = Cumo::Linalg.ldl(Cumo::Int32[[4, 2], [2, 3]])
    assert_kind_of(Cumo::DFloat, u)
    assert_close([[1.0, 2.0 / 3], [0.0, 1.0]], u, 1e-15)
    assert_close([[8.0 / 3, 0.0], [0.0, 3.0]], d, 1e-15)
  end

  test "a singular block diagonal matrix warns" do
    _, err = capture_output { Cumo::Linalg.ldl(Cumo::DFloat[[0, 0], [0, 0]]) }
    assert_match(/D\[1, 1\] is exactly zero/, err)
  end

  test "the input is left as it was" do
    a = Cumo::DComplex[[Complex(2, 5), 1], [1, 3]]
    ldl(a, 'U', true)
    assert_equal([[Complex(2, 5), 1], [1, 3]], a.to_a)
  end

  test "an empty matrix answers empty factors" do
    u, d, perm = Cumo::Linalg.ldl(Cumo::DFloat.new(0, 0))
    assert_equal([[0, 0], [0, 0], [0]], [u.shape, d.shape, perm.shape])
    assert_kind_of(Cumo::Int32, perm)
  end

  test "errors" do
    error = assert_raise(Cumo::NArray::ShapeError) { Cumo::Linalg.ldl(Cumo::DFloat[[1, 2]]) }
    assert_equal('input array a must be square', error.message)
    error = assert_raise(Cumo::NArray::ShapeError) { Cumo::Linalg.ldl(Cumo::DFloat[1]) }
    assert_equal('input array a must be 2-dimensional', error.message)
    a = Cumo::DFloat[[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    assert_equal(Cumo::Linalg.ldl(a, uplo: 'U').map(&:to_a), Cumo::Linalg.ldl(a, uplo: 'U'.ord).map(&:to_a))
    error = assert_raise(ArgumentError) { Cumo::Linalg.ldl(Cumo::DFloat[[1]], uplo: 'u') }
    assert_equal("uplo must be 'U' or 'L'", error.message)
    assert_raise(TypeError) { Cumo::Linalg.ldl(Cumo::HFloat[[1]]) }
  end

  test "sytrf checks what it is handed" do
    cusolver = Cumo::CUDA::Cusolver
    assert_raise(Cumo::NArray::ShapeError) { cusolver.__send__(:sytrf, Cumo::DFloat.new(2, 3), 'U', true) }
    assert_raise(ArgumentError) { cusolver.__send__(:sytrf, Cumo::DFloat.new(2, 2), 'X', true) }
    assert_raise(TypeError) { cusolver.__send__(:sytrf, Cumo::Int32.new(2, 2), 'U', true) }
  end

  private

  def ldl(a, uplo, hermitian)
    Cumo::Linalg.ldl(a, uplo: uplo, hermitian: hermitian)
  rescue NotImplementedError => e
    omit(e.message)
  end
end
