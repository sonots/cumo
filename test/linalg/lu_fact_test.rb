# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgLuFactTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-5, Cumo::DFloat => 1e-12, Cumo::SComplex => 1e-5, Cumo::DComplex => 1e-12 }

  types.each do |type, tol|
    [[4, 4], [5, 3], [3, 5]].each do |m, n|
      test "#{type} #{m}x#{n} is P L U" do
        a = type.new(m, n).rand - 0.5
        lu, piv = Cumo::Linalg.lu_fact(a)
        assert_kind_of(type, lu)
        assert_equal([m, n], lu.shape)
        assert_kind_of(Cumo::Int32, piv)
        assert_equal([[m, n].min], piv.shape)
        assert_close(a.to_a, host_plu(lu.to_a, piv.to_a, m, n), tol)
      end
    end
  end

  test "the factors and pivots numo-linalg-alt answers" do
    lu, piv = Cumo::Linalg.lu_fact(Cumo::DFloat[[2, 1, 1], [4, 3, 3], [8, 7, 9]])
    assert_close([[8.0, 7.0, 9.0], [0.25, -0.75, -1.25], [0.5, 2.0 / 3.0, -2.0 / 3.0]], lu, 1e-15)
    assert_equal([3, 3, 3], piv.to_a)
  end

  test "an integer matrix is factorized in DFloat" do
    lu, = Cumo::Linalg.lu_fact(Cumo::Int32[[1, 2], [3, 4]])
    assert_kind_of(Cumo::DFloat, lu)
    assert_close([[3.0, 4.0], [1.0 / 3.0, 2.0 / 3.0]], lu, 1e-15)
  end

  test "the receiver is left as it was" do
    a = Cumo::DFloat[[1, 2], [3, 4]]
    Cumo::Linalg.lu_fact(a)
    assert_equal([[1.0, 2.0], [3.0, 4.0]], a.to_a)
  end

  test "a transposed view" do
    a = Cumo::DFloat.new(4, 3).rand
    lu, piv = Cumo::Linalg.lu_fact(a.transpose)
    assert_close(a.transpose.to_a, host_plu(lu.to_a, piv.to_a, 3, 4), 1e-12)
  end

  test "a singular matrix warns and answers the factors" do
    lu = piv = nil
    _, stderr = capture_output { lu, piv = Cumo::Linalg.lu_fact(Cumo::DFloat[[1, 2], [2, 4]]) }
    assert_equal("the factorization has been completed, but the factor U[1, 1] is exactly zero, " \
                 "indicating that the matrix is singular.\n",
                 stderr)
    assert_equal([[2.0, 4.0], [0.5, 0.0]], lu.to_a)
    assert_equal([2, 2], piv.to_a)
  end

  test "empty matrices" do
    [[0, 0], [0, 2], [2, 0]].each do |shape|
      lu, piv = Cumo::Linalg.lu_fact(Cumo::DFloat.new(*shape))
      assert_equal(shape, lu.shape)
      assert_equal([0], piv.shape)
    end
  end

  test "errors" do
    error = assert_raise(Cumo::NArray::ShapeError) { Cumo::Linalg.lu_fact(Cumo::DFloat[1, 2]) }
    assert_equal('input array a must be 2-dimensional', error.message)
    error = assert_raise(TypeError) { Cumo::Linalg.lu_fact(Cumo::RObject[[1, 2], [3, 4]]) }
    assert_equal('invalid data type for BLAS/LAPACK', error.message)
  end

  private

  def host_plu(lu, piv, m, n)
    k = [m, n].min
    l = Array.new(m) do |i|
      Array.new(k) do |j|
        if i == j
          1
        elsif i > j
          lu[i][j]
        else
          0
        end
      end
    end
    u = Array.new(k) { |i| Array.new(n) { |j| i <= j ? lu[i][j] : 0 } }
    a = host_dot(l, u)
    piv.each_with_index.reverse_each { |p, i| a[i], a[p - 1] = a[p - 1], a[i] }
    a
  end
end
