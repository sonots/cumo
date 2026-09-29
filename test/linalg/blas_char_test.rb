# frozen_string_literal: true

require_relative "../test_helper"
require "cumo/linalg"

class LinalgBlasCharTest < Test::Unit::TestCase
  test "one array" do
    assert_equal('d', Cumo::Linalg.blas_char([true, false]))
    assert_equal('d', Cumo::Linalg.blas_char([1, 2]))
    assert_equal('d', Cumo::Linalg.blas_char([1.1, 2.2]))
    assert_equal('z', Cumo::Linalg.blas_char([Complex(1, 2), 3]))
    assert_equal('d', Cumo::Linalg.blas_char(Cumo::NArray[1, 2]))
    assert_equal('d', Cumo::Linalg.blas_char(Cumo::UInt8[1, 2]))
    assert_equal('s', Cumo::Linalg.blas_char(Cumo::SFloat[1.1, 2.2]))
    assert_equal('d', Cumo::Linalg.blas_char(Cumo::DFloat[1.1, 2.2]))
    assert_equal('c', Cumo::Linalg.blas_char(Cumo::SComplex[1.1, 2.2]))
    assert_equal('z', Cumo::Linalg.blas_char(Cumo::DComplex[1.1, 2.2]))
  end

  test "two arrays" do
    assert_equal('d', Cumo::Linalg.blas_char(Cumo::SFloat[1, 2], Cumo::DFloat[1, 2]))
    assert_equal('c', Cumo::Linalg.blas_char(Cumo::SFloat[1, 2], Cumo::SComplex[1, 2]))
    assert_equal('z', Cumo::Linalg.blas_char(Cumo::SFloat[1, 2], Cumo::DComplex[1, 2]))
    assert_equal('d', Cumo::Linalg.blas_char(Cumo::DFloat[1, 2], Cumo::SFloat[1, 2]))
    assert_equal('z', Cumo::Linalg.blas_char(Cumo::DFloat[1, 2], Cumo::SComplex[1, 2]))
    assert_equal('z', Cumo::Linalg.blas_char(Cumo::DFloat[1, 2], Cumo::DComplex[1, 2]))
    assert_equal('c', Cumo::Linalg.blas_char(Cumo::SComplex[1, 2], Cumo::SFloat[1, 2]))
    assert_equal('z', Cumo::Linalg.blas_char(Cumo::SComplex[1, 2], Cumo::DFloat[1, 2]))
    assert_equal('z', Cumo::Linalg.blas_char(Cumo::SComplex[1, 2], Cumo::DComplex[1, 2]))
    assert_equal('z', Cumo::Linalg.blas_char(Cumo::DComplex[1, 2], Cumo::SFloat[1, 2]))
    assert_equal('z', Cumo::Linalg.blas_char(Cumo::DComplex[1, 2], Cumo::DFloat[1, 2]))
    assert_equal('z', Cumo::Linalg.blas_char(Cumo::DComplex[1, 2], Cumo::SComplex[1, 2]))
  end

  test "an integer array decides the type ahead of SFloat and not behind it" do
    assert_equal('d', Cumo::Linalg.blas_char(Cumo::Int32[1, 2], Cumo::SFloat[1, 2]))
    assert_equal('s', Cumo::Linalg.blas_char(Cumo::SFloat[1, 2], Cumo::Int32[1, 2]))
  end

  test "no type BLAS takes" do
    [
      [['1', 2, 3]],
      [Cumo::RObject[1, 2]],
      [Cumo::HFloat[1, 2]],
      [Cumo::BFloat[1, 2]],
      [1.0],
      []
    ].each do |args|
      error = assert_raise(TypeError) { Cumo::Linalg.blas_char(*args) }
      assert_equal('invalid data type for BLAS/LAPACK', error.message)
    end
  end
end
