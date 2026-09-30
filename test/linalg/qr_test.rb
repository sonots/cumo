# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgQrTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-12, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-12 }

  types.each do |type, tol|
    [[4, 4], [5, 3], [3, 5]].each do |m, n|
      sub_test_case "#{type} #{m}x#{n}" do
        setup do
          rng = Random.new(m * 7 + n)
          @host = linalg_values(type, Array.new(m) { Array.new(n) { (rng.rand * 8).round - 4 } })
          @a = type.cast(@host)
          @k = [m, n].min
        end

        test "reduce and economic" do
          [['reduce', [m, m], [m, n]], ['economic', [m, @k], [@k, n]]].each do |mode, q_shape, r_shape|
            q, r = Cumo::Linalg.qr(@a, mode: mode)
            assert_kind_of(type, q)
            assert_equal([q_shape, r_shape], [q.shape, r.shape], mode)
            assert_close(@host, host_dot(q.to_a, r.to_a), tol * 100)
            assert_close(identity(q_shape[1]), host_dot(conj_t(q.to_a), q.to_a), tol * 10)
            assert_true(r.to_a.each_with_index.all? { |row, i| row.each_with_index.all? { |v, j| j >= i || v.zero? } })
          end
        end

        test "r and raw agree with reduce" do
          _, r = Cumo::Linalg.qr(@a)
          assert_equal(r.to_a, Cumo::Linalg.qr(@a, mode: 'r').to_a)
          packed, tau = Cumo::Linalg.qr(@a, mode: 'raw')
          assert_equal([[m, n], [@k]], [packed.shape, tau.shape])
          assert_kind_of(type, tau)
          assert_equal(r.to_a, packed.triu.to_a)
        end
      end
    end
  end

  test "the factors numo-linalg-alt answers" do
    q, r = Cumo::Linalg.qr(Cumo::DFloat[[2, 1], [1, 3]])
    assert_close([[-0.8944271909999157, -0.4472135954999579], [-0.4472135954999579, 0.8944271909999159]], q, 1e-15)
    assert_close([[-2.23606797749979, -2.2360679774997894], [0.0, 2.23606797749979]], r, 1e-15)
    packed, tau = Cumo::Linalg.qr(Cumo::DFloat[[1, 2], [3, 4], [5, 6]], mode: 'raw')
    assert_close([
                   [-5.916079783099616, -7.437357441610946],
                   [0.4337717455676132, 0.828078671210825],
                   [0.7229529092793553, 0.8926238280228976]
                 ],
                 packed,
                 1e-14)
    assert_close([1.1690308509457032, 1.1131040011646904], tau, 1e-14)
  end

  test "an integer matrix is factored in DFloat" do
    q, r = Cumo::Linalg.qr(Cumo::Int32[[1, 2], [3, 4]])
    assert_kind_of(Cumo::DFloat, q)
    assert_close([[1.0, 2.0], [3.0, 4.0]], host_dot(q.to_a, r.to_a), 1e-14)
  end

  test "the input is left as it was" do
    a = Cumo::DFloat[[1, 2], [3, 4], [5, 6]]
    Cumo::Linalg.qr(a)
    assert_equal([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], a.to_a)
  end

  test "errors" do
    error = assert_raise(Cumo::NArray::ShapeError) { Cumo::Linalg.qr(Cumo::DFloat[1, 2]) }
    assert_equal('input array a must be 2-dimensional', error.message)
    error = assert_raise(ArgumentError) { Cumo::Linalg.qr(Cumo::DFloat[[1, 2], [3, 4]], mode: 'X') }
    assert_equal('invalid mode: X', error.message)
    error = assert_raise(TypeError) { Cumo::Linalg.qr(Cumo::RObject[[1]]) }
    assert_equal('invalid data type for BLAS/LAPACK', error.message)
  end

  private

  def conj(v)
    v.respond_to?(:conj) ? v.conj : v
  end

  def conj_t(m)
    m.transpose.map { |row| row.map { |v| conj(v) } }
  end

  def identity(n)
    Array.new(n) { |i| Array.new(n) { |j| i == j ? 1 : 0 } }
  end
end
