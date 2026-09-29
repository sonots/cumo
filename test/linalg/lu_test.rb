# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgLuTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-5, Cumo::DFloat => 1e-12, Cumo::SComplex => 1e-5, Cumo::DComplex => 1e-12 }

  types.each do |type, tol|
    [[4, 4], [5, 3], [3, 5]].each do |m, n|
      test "#{type} #{m}x#{n} is P L U" do
        a = type.new(m, n).rand - 0.5
        k = [m, n].min
        perm, l, u = Cumo::Linalg.lu(a)
        assert_equal([[m, m], [m, k], [k, n]], [perm.shape, l.shape, u.shape])
        assert_close(a.to_a, host_dot(host_dot(perm.to_a, l.to_a), u.to_a), tol)
        assert_equal(Array.new(k, 1.0), host_diagonal(l.to_a))
        assert_true(l.to_a.each_with_index.all? { |row, i| row.each_with_index.all? { |v, j| j <= i || v.zero? } })
        assert_true(u.to_a.each_with_index.all? { |row, i| row.each_with_index.all? { |v, j| j >= i || v.zero? } })

        pl, u2 = Cumo::Linalg.lu(a, permute_l: true)
        assert_close(host_dot(perm.to_a, l.to_a), pl, tol)
        assert_equal(u.to_a, u2.to_a)
      end
    end
  end

  test "P is a permutation of the class of the input" do
    perm, l, = Cumo::Linalg.lu(Cumo::Int32[[1, 2], [3, 4]])
    assert_kind_of(Cumo::Int32, perm)
    assert_equal([[0, 1], [1, 0]], perm.to_a)
    assert_kind_of(Cumo::DFloat, l)
  end

  test "the factors of numo-linalg-alt" do
    perm, l, u = Cumo::Linalg.lu(Cumo::DFloat[[2, 1, 1], [4, 3, 3], [8, 7, 9]])
    assert_equal([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], perm.to_a)
    assert_close([[1.0, 0.0, 0.0], [0.25, 1.0, 0.0], [0.5, 2.0 / 3.0, 1.0]], l, 1e-15)
    assert_close([[8.0, 7.0, 9.0], [0.0, -0.75, -1.25], [0.0, 0.0, -2.0 / 3.0]], u, 1e-15)
  end

  test "errors" do
    error = assert_raise(Cumo::NArray::ShapeError) { Cumo::Linalg.lu(Cumo::DFloat[1, 2]) }
    assert_equal('input array a must be 2-dimensional', error.message)
  end

  private

  def host_diagonal(m)
    m.each_with_index.filter_map { |row, i| row[i] }
  end
end
