# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgSvdTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  types = { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-11, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-11 }

  types.each do |type, tol|
    real = [Cumo::SFloat, Cumo::SComplex].include?(type) ? Cumo::SFloat : Cumo::DFloat

    [[4, 4], [5, 3], [3, 5]].each do |m, n|
      sub_test_case "#{type} #{m}x#{n}" do
        setup do
          rng = Random.new(m * 10 + n)
          @host = linalg_values(type, Array.new(m) { Array.new(n) { (rng.rand * 8).round - 4 } })
          @a = type.cast(@host)
          @k = [m, n].min
        end

        test "svd with job A and S" do
          [['A', [m, m], [n, n]], ['S', [m, @k], [@k, n]]].each do |job, u_shape, vt_shape|
            s, u, vt = Cumo::Linalg.svd(@a, job: job)
            assert_kind_of(real, s)
            assert_equal([[@k], u_shape, vt_shape], [s.shape, u.shape, vt.shape])
            assert_equal(s.to_a.sort.reverse, s.to_a)
            us = u.to_a.map { |row| row.each_with_index.map { |v, j| j < @k ? v * s.to_a[j] : 0 } }
            assert_close(@host, host_dot(us.map { |row| row[0...@k] }, vt.to_a[0...@k]), tol * 100)
            assert_close(identity(u_shape[1]), host_dot(conj_t(u.to_a), u.to_a), tol * 10)
            assert_close(identity(vt_shape[0]), host_dot(vt.to_a, conj_t(vt.to_a)), tol * 10)
          end
        end

        test "svd with job N and svdvals" do
          s, u, vt = Cumo::Linalg.svd(@a, job: 'N')
          assert_nil(u)
          assert_nil(vt)
          assert_close(Cumo::Linalg.svd(@a)[0].to_a, s, tol * 10)
          assert_close(s.to_a, Cumo::Linalg.svdvals(@a), tol * 10)
          assert_close(s.to_a, Cumo::Linalg.svdvals(@a, driver: 'svd'), tol * 10)
        end

        test "pinv satisfies the Moore-Penrose conditions" do
          x = Cumo::Linalg.pinv(@a).to_a
          assert_equal([n, m], [x.size, x[0].size])
          assert_close(@host, host_dot(host_dot(@host, x), @host), tol * 1000)
          assert_close(x, host_dot(host_dot(x, @host), x), tol * 1000)
        end

        test "lstsq answers the least-norm least-squares solution" do
          b = linalg_values(type, Array.new(m) { |i| [i - 1, 2 - i] })
          x, resids, rank, s = Cumo::Linalg.lstsq(@a, type.cast(b))
          assert_equal(@k, rank)
          assert_close(Cumo::Linalg.svdvals(@a).to_a, s, tol * 10)
          assert_close(host_dot(Cumo::Linalg.pinv(@a).to_a, b), x, tol * 1000)
          if m > n
            r = host_dot(@host, x.to_a).zip(b).map { |ar, br| ar.zip(br).map { |p, q| (p - q).abs**2 } }
            assert_close(r.transpose.map(&:sum), resids, tol * 1000)
          else
            assert_equal([0], resids.shape)
          end
        end
      end
    end
  end

  test "lstsq reads rcond as LAPACK's gelsd does" do
    a = Cumo::DFloat[[1, 0], [0, 1e-20], [0, 0]]
    b = Cumo::DFloat[1, 1, 0]
    [0.0, 1.0, -1.0, nil].each do |rcond|
      x, _, rank, = Cumo::Linalg.lstsq(a, b, rcond: rcond)
      assert_equal(1, rank, "rcond #{rcond.inspect}")
      assert_equal([1.0, 0.0], x.to_a)
    end
    assert_equal(2, Cumo::Linalg.lstsq(Cumo::DFloat[[1, 0], [0, 1.5e-16]], Cumo::DFloat[1, 1])[2])
  end

  test "lstsq answers empty residuals of the class numo-linalg-alt answers" do
    assert_kind_of(Cumo::Int32, Cumo::Linalg.lstsq(Cumo::DFloat[[1, 0], [0, 1]], Cumo::Int32[1, 2])[1])
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.lstsq(Cumo::SFloat[[1, 0, 1], [0, 1, 1]], Cumo::SFloat[1, 2])[1])
  end

  test "matrix_rank" do
    assert_equal(2, Cumo::Linalg.matrix_rank(Cumo::DFloat[[1, 2, 3], [4, 5, 6]]))
    assert_equal(1, Cumo::Linalg.matrix_rank(Cumo::DFloat[[1, 2], [2, 4]]))
    assert_equal(1, Cumo::Linalg.matrix_rank(Cumo::DFloat[[1, 2, 3], [4, 5, 6]], tol: 5.0))
    assert_equal(1, Cumo::Linalg.matrix_rank(Cumo::DFloat[0, 1]))
    assert_equal(0, Cumo::Linalg.matrix_rank(Cumo::DFloat[0, 0]))
  end

  test "orth and null_space" do
    a = Cumo::DFloat[[1, 2, 3], [4, 5, 6]]
    q = Cumo::Linalg.orth(a)
    assert_equal([2, 2], q.shape)
    assert_close(identity(2), host_dot(q.to_a.transpose, q.to_a), 1e-12)
    ns = Cumo::Linalg.null_space(a)
    assert_equal([3, 1], ns.shape)
    assert_close([[0.0], [0.0]], host_dot(a.to_a, ns.to_a), 1e-12)
    assert_equal([2, 1], Cumo::Linalg.orth(Cumo::DFloat[[1, 2], [2, 4]]).shape)
  end

  test "a full rank matrix has an empty null space and a zero matrix an empty range" do
    assert_equal([2, 0], Cumo::Linalg.null_space(Cumo::DFloat[[1, 2], [3, 4]]).shape)
    assert_equal([2, 0], Cumo::Linalg.orth(Cumo::DFloat.zeros(2, 3)).shape)
  end

  test "pinv of a zero matrix is a zero matrix" do
    assert_equal([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], Cumo::Linalg.pinv(Cumo::DFloat.zeros(2, 3)).to_a)
  end

  test "cond answers a zero-dimensional array, as numo-linalg-alt does" do
    c = Cumo::Linalg.cond(Cumo::DFloat[[1, 2], [3, 4]])
    assert_equal(0, c.ndim)
    s = Cumo::Linalg.svdvals(Cumo::DFloat[[1, 2], [3, 4]]).to_a
    assert_in_delta(s[0] / s[1], c.to_a.flatten.first, 1e-12)
    assert_in_delta(s[1] / s[0], Cumo::Linalg.cond(Cumo::DFloat[[1, 2], [3, 4]], -2).to_a.flatten.first, 1e-12)
  end

  test "an integer matrix is decomposed in DFloat" do
    s, u, = Cumo::Linalg.svd(Cumo::Int32[[3, 0], [0, 4]])
    assert_kind_of(Cumo::DFloat, s)
    assert_kind_of(Cumo::DFloat, u)
    assert_equal([4.0, 3.0], s.to_a)
  end

  test "empty" do
    s, u, vt = Cumo::Linalg.svd(Cumo::DFloat.new(0, 0))
    assert_equal([[0], [0, 0], [0, 0]], [s.shape, u.shape, vt.shape])
  end

  test "errors" do
    w = Cumo::DFloat[[1, 2, 3], [4, 5, 6]]
    [
      [:svd, [w], { job: 'X' }, ArgumentError, 'invalid job: X'],
      [:svd, [w], { job: 's' }, ArgumentError, "jobu must be 'A', 'S', 'O', or 'N'"],
      [:svd, [w], { driver: 'sdd', job: 's' }, ArgumentError, "jobz must be one of 'A', 'S', 'O', or 'N'"],
      [:svd, [w], { driver: 'X' }, ArgumentError, 'invalid driver: X'],
      [:svd, [Cumo::DFloat[1, 2]], {}, ArgumentError, 'input array must be 2-dimensional'],
      [:svdvals, [w], { driver: 'X' }, ArgumentError, 'invalid driver: X'],
      [:orth, [Cumo::DFloat[1, 2]], {}, Cumo::NArray::ShapeError, 'input array a must be 2-dimensional'],
      [:lstsq, [w.transpose.dup, Cumo::DFloat.ones(3, 2, 2)], {}, ArgumentError, 'input array b must be 1 or 2-dimensional'],
      [
        :lstsq,
        [w.transpose.dup, Cumo::DFloat[1, 2]],
        {},
        Cumo::NArray::ShapeError,
        'incompatible dimensions: a.shape[0] = 3 != b.shape[0] = 2'
      ]
    ].each do |method, args, kwargs, klass, message|
      error = assert_raise(klass) { Cumo::Linalg.public_send(method, *args, **kwargs) }
      assert_equal(message, error.message)
    end
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
