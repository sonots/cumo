# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgNormTest < Test::Unit::TestCase
  include LinalgTestHelper

  types = { Cumo::SFloat => 1e-4, Cumo::DFloat => 1e-12, Cumo::SComplex => 1e-4, Cumo::DComplex => 1e-12 }

  types.each do |type, tol|
    sub_test_case type.to_s do
      setup do
        @vector_host = linalg_values(type, [3, -4, 0, 1, -2])
        @vector = type.cast(@vector_host)
        @matrix_host = linalg_values(type, [[1, 2, -3, 1], [-4, 1, 8, 2], [0, -1, 2, 5]])
        @matrix = type.cast(@matrix_host)
      end

      test "vector norms" do
        abs = @vector_host.map(&:abs)
        [
          [nil, Math.sqrt(abs.sum { |x| x**2 })],
          [2, Math.sqrt(abs.sum { |x| x**2 })],
          [1, abs.sum],
          [3, abs.sum { |x| x**3 }**(1.0 / 3)],
          [0.5, abs.sum { |x| x**0.5 }**2],
          ['inf', abs.max],
          ['-inf', abs.min],
          [Float::INFINITY, abs.max]
        ].each do |ord, expected|
          norm = Cumo::Linalg.norm(@vector, ord)
          assert_kind_of(Float, norm, ord.inspect)
          assert_in_delta(expected, norm, expected * tol, ord.inspect)
        end
        assert_equal(abs.count(&:nonzero?), Cumo::Linalg.norm(@vector, 0).real)
      end

      test "matrix norms" do
        omit_unless_cusolver
        abs = @matrix_host.map { |row| row.map(&:abs) }
        [
          [nil, Math.sqrt(abs.flatten.sum { |x| x**2 })],
          ['fro', Math.sqrt(abs.flatten.sum { |x| x**2 })],
          [1, abs.transpose.map(&:sum).max],
          [-1, abs.transpose.map(&:sum).min],
          ['inf', abs.map(&:sum).max],
          ['-inf', abs.map(&:sum).min]
        ].each do |ord, expected|
          norm = Cumo::Linalg.norm(@matrix, ord)
          assert_kind_of(Float, norm, ord.inspect)
          assert_in_delta(expected, norm, expected * tol, ord.inspect)
        end
        s = Cumo::Linalg.svdvals(@matrix).to_a
        [[2, s.max], [-2, s.min], ['nuc', s.sum]].each do |ord, expected|
          assert_in_delta(expected, Cumo::Linalg.norm(@matrix, ord), expected * tol, ord.inspect)
        end
      end

      test "norms along axes" do
        rows = @matrix_host.map { |row| Math.sqrt(row.sum { |x| x.abs**2 }) }
        assert_close(rows, Cumo::Linalg.norm(@matrix, axis: 1), rows.max * tol)
        assert_close(rows, Cumo::Linalg.norm(@matrix, 2, axis: -1), rows.max * tol)
        cols = @matrix_host.transpose.map { |col| col.sum(&:abs) }
        assert_close(cols, Cumo::Linalg.norm(@matrix, 1, axis: 0), cols.max * tol)
      end
    end
  end

  test "the classes numo-linalg-alt answers" do
    omit_unless_cusolver
    m = Cumo::DFloat[[1, 2, -3, 1], [-4, 1, 8, 2]]
    assert_kind_of(Float, Cumo::Linalg.norm(m, 1, axis: [1, 0]))
    assert_kind_of(Float, Cumo::Linalg.norm(m, 'nuc', axis: [0, 1]))
    assert_kind_of(Float, Cumo::Linalg.norm(Cumo::DFloat[3, -4], axis: 0))
    fro = Cumo::Linalg.norm(m, 'fro', axis: [0, 1])
    assert_kind_of(Cumo::DFloat, fro)
    assert_equal([[], [10.0]], [fro.shape, fro.to_a.flatten])
    assert_kind_of(Cumo::DFloat, Cumo::Linalg.norm(Cumo::SFloat[[3, 4]], 'fro', axis: [0, 1]))
    assert_kind_of(Cumo::SFloat, Cumo::Linalg.norm(Cumo::SFloat[[3, 4]], axis: 1))
    assert_equal(7, Cumo::Linalg.norm(Cumo::Int32[3, -4], 1))
    assert_equal(4, Cumo::Linalg.norm(Cumo::Int32[3, -4], 'inf'))
    assert_equal(Complex(1.0, 0.0), Cumo::Linalg.norm(Cumo::DComplex[3i, 0], 0))
    assert_equal(5.0, Cumo::Linalg.norm([3, 4]))
  end

  test "the values numo-linalg-alt answers" do
    omit_unless_cusolver
    m = Cumo::DFloat[[1, 2, -3, 1], [-4, 1, 8, 2]]
    assert_in_delta(9.614478163054194, Cumo::Linalg.norm(m, 2), 1e-13)
    assert_in_delta(2.749874479345213, Cumo::Linalg.norm(m, -2), 1e-13)
    assert_in_delta(12.364352642399407, Cumo::Linalg.norm(m, 'nuc'), 1e-13)
    assert_close([0.35294117647058826, 0.5333333333333333], Cumo::Linalg.norm(m, -1, axis: 1), 1e-15)
  end

  test "keepdims" do
    omit_unless_cusolver
    m = Cumo::DFloat[[1, 2, -3, 1], [-4, 1, 8, 2]]
    assert_equal([[[10.0]]], [Cumo::Linalg.norm(m, keepdims: true).to_a])
    assert_equal([[7.0], [15.0]], Cumo::Linalg.norm(m, 1, axis: 1, keepdims: true).to_a)
    assert_equal([1, 1], Cumo::Linalg.norm(m, 'nuc', keepdims: true).shape)
    assert_equal([1], Cumo::Linalg.norm(Cumo::DFloat[3, 4], keepdims: true).shape)
    t = Cumo::DFloat.new(2, 3, 4).seq - 10
    assert_equal([1, 3, 1], Cumo::Linalg.norm(t, 'fro', axis: [2, 0], keepdims: true).shape)
    assert_equal([1, 1, 1], Cumo::Linalg.norm(t, keepdims: true).shape)
  end

  test "matrix norms of the matrices along two axes" do
    t = Cumo::DFloat.new(2, 3, 4).seq - 10
    host = t.to_a
    fro = Array.new(3) { |j| Math.sqrt(host.sum { |plane| plane[j].sum { |x| x**2 } }) }
    assert_close(fro, Cumo::Linalg.norm(t, axis: [0, 2]), 1e-12)
    ones = Array.new(3) do |j|
      sums = Array.new(4) { |k| host.sum { |plane| plane[j][k].abs } }
      sums.max
    end
    assert_close(ones, Cumo::Linalg.norm(t, 1, axis: [0, 2]), 1e-12)
    infs = Array.new(3) do |j|
      sums = host.map { |plane| plane[j].sum(&:abs) }
      sums.max
    end
    assert_close(infs, Cumo::Linalg.norm(t, 'inf', axis: [0, 2]), 1e-12)
    assert_close(ones, Cumo::Linalg.norm(t, 'inf', axis: [2, 0]), 1e-12)
    assert_close(infs, Cumo::Linalg.norm(t, 1, axis: [2, 0]), 1e-12)
  end

  test "matrix norms from the singular values of the matrices along two axes" do
    omit_unless_cusolver
    t = Cumo::DFloat.new(2, 3, 4).seq - 10
    t[0, 1, 2] = 7
    slices = Array.new(3) { |j| Cumo::Linalg.svdvals(t[true, j, true].dup).to_a }
    [[2, :max], [-2, :min], ['nuc', :sum]].each do |ord, reduce|
      expected = slices.map(&reduce)
      assert_close(expected, Cumo::Linalg.norm(t, ord, axis: [0, 2]), 1e-12)
      assert_close(expected, Cumo::Linalg.norm(t, ord, axis: [2, 0]), 1e-12)
      assert_equal([1, 3, 1], Cumo::Linalg.norm(t, ord, axis: [2, 0], keepdims: true).shape)
    end
    s = Cumo::SFloat.new(2, 2, 3).seq
    assert_kind_of(Cumo::SFloat, Cumo::Linalg.norm(s, 2, axis: [1, 2]))
    assert_close(Array.new(2) { |i| Cumo::Linalg.svdvals(s[i, true, true].dup).to_a.sum }, Cumo::Linalg.norm(s, 'nuc', axis: [1, 2]), 1e-4)
  end

  test "stacked matrices up to 32 rows and columns and past them answer the singular values of each" do
    omit_unless_cusolver
    {
      Cumo::DFloat => [[3, 32, 7], [2, 33, 5]],
      Cumo::DComplex => [[3, 4, 5], [2, 5, 40]],
      Cumo::SFloat => [[3, 6, 4]],
      Cumo::SComplex => [[3, 4, 6]]
    }.each do |type, shapes|
      tol = [Cumo::SFloat, Cumo::SComplex].include?(type) ? 1e-4 : 1e-12
      shapes.each do |shape|
        t = type.new(*shape).rand_norm
        expected = Array.new(shape[0]) { |i| Cumo::Linalg.svdvals(t[i, true, true].dup).to_a }
        assert_close(expected.map(&:max), Cumo::Linalg.norm(t, 2, axis: [1, 2]), tol)
        assert_close(expected.map(&:min), Cumo::Linalg.norm(t, -2, axis: [1, 2]), tol)
        assert_close(expected.map(&:sum), Cumo::Linalg.norm(t, 'nuc', axis: [1, 2]), tol)
      end
    end
  end

  test "the reductions of magnitudes norm uses stay private" do
    [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex].each do |type|
      %i[abs_sum abs_max abs_min].each do |name|
        assert_raise(NoMethodError) { type[[1, -2]].public_send(name) }
      end
    end
  end

  test "a stacked matrix holding NaN or Infinity answers NaN whatever its size" do
    omit_unless_cusolver
    [[2, 3, 3], [2, 33, 3]].each do |shape|
      [Float::NAN, Float::INFINITY].each do |v|
        [Cumo::DFloat, Cumo::DComplex].each do |type|
          t = type.new(*shape).seq
          t[0, 0, 0] = v
          [2, -2, 'nuc'].each do |ord|
            norms = Cumo::Linalg.norm(t, ord, axis: [1, 2]).to_a
            assert_true(norms[0].nan?, "#{type} #{shape} #{v} #{ord}")
            assert_true(norms[1].finite?, "#{type} #{shape} #{v} #{ord}")
          end
        end
      end
    end
  end

  test "the whole array is scaled so its squares do not overflow or underflow" do
    assert_in_delta(Math.sqrt(2) * 1e200, Cumo::Linalg.norm(Cumo::DFloat[1e200, 1e200]), 1e188)
    assert_in_delta(Math.sqrt(10) * 1e-200, Cumo::Linalg.norm(Cumo::DFloat[[1e-200, 3e-200]]), 1e-212)
    assert_in_delta(Math.sqrt(10) * 1e-160, Cumo::Linalg.norm(Cumo::DFloat[1e-160, 3e-160]), 1e-172)
    assert_in_delta(Math.sqrt(10) * 1e30, Cumo::Linalg.norm(Cumo::SFloat[1e30, 3e30]), 1e25)
    tiny = Cumo::SFloat.new(1001).fill(1e-20)
    tiny[0] = 1.2e-19
    expected = Math.sqrt((1.2e-19**2) + ((1e-20**2) * 1000))
    assert_in_delta(expected, Cumo::Linalg.norm(tiny), expected * 1e-6)
    assert_in_delta(Math.sqrt(2) * 1e200, Cumo::Linalg.norm(Cumo::DComplex[[[1e200, 1e200i]]]), 1e188)
  end

  test "zeros, infinities and NaN" do
    assert_equal(0.0, Cumo::Linalg.norm(Cumo::DFloat[0, 0]))
    assert_equal(Float::INFINITY, Cumo::Linalg.norm(Cumo::DFloat[Float::INFINITY, 1]))
    assert_equal(Float::INFINITY, Cumo::Linalg.norm(Cumo::DFloat[-Float::INFINITY, 1]))
    [[Float::NAN, 1], [Float::NAN, 0], [Float::INFINITY, Float::NAN], [[Float::NAN, 0], [0, 0]]].each do |values|
      assert_true(Cumo::Linalg.norm(Cumo::DFloat[*values]).nan?, values.inspect)
    end
  end

  test "empty arrays answer zero, as numo-linalg-alt does" do
    assert_equal(0.0, Cumo::Linalg.norm(Cumo::DFloat[]))
    assert_equal(0.0, Cumo::Linalg.norm(Cumo::DFloat.new(0, 3), axis: 1))
  end

  test "integers and Bit are normed in DFloat where a power or singular values are taken" do
    omit_unless_cusolver
    assert_equal(5.0, Cumo::Linalg.norm(Cumo::Int32[3, -4]))
    assert_in_delta(12.0 / 7, Cumo::Linalg.norm(Cumo::Int32[3, -4], -1), 1e-15)
    assert_in_delta(Math.sqrt(2) * 1e5, Cumo::Linalg.norm(Cumo::Int32[100_000, 100_000], 2, axis: 0), 1e-9)
    assert_in_delta(Math.sqrt(2) * 1e5, Cumo::Linalg.norm(Cumo::Int32[[100_000, 100_000]], 'fro', axis: [0, 1]).to_a.flatten.first, 1e-9)
    assert_in_delta(5.0, Cumo::Linalg.norm(Cumo::Int32[[3, 0], [0, 5]], 2), 1e-14)
    assert_in_delta(Math.sqrt(2), Cumo::Linalg.norm(Cumo::Bit[1, 0, 1]), 1e-15)
    assert_equal(2.0, Cumo::Linalg.norm(Cumo::Bit[1, 0, 1], 1))
    assert_equal(2.0, Cumo::Linalg.norm(Cumo::Bit[[1, 0], [1, 1]], 1))
  end

  test "a zero-dimensional array is normed as a vector of one" do
    x = Cumo::DFloat[-3, 0].sum
    assert_equal(3.0, Cumo::Linalg.norm(x))
    assert_equal(3.0, Cumo::Linalg.norm(x, 1))
    assert_equal(3.0, Cumo::Linalg.norm(x, 'inf'))
    assert_equal(3.0, Cumo::Linalg.norm(x, axis: 0))
    assert_equal([[3.0], [3.0]], [Cumo::Linalg.norm(x, keepdims: true).to_a, Cumo::Linalg.norm(x, 1, axis: 0, keepdims: true).to_a])
  end

  test "HFloat and BFloat are not normed, as in the other functions" do
    [Cumo::HFloat, Cumo::BFloat].each do |type|
      [[nil, {}], [1, {}], [nil, { axis: 0 }], [3, {}]].each do |ord, kwargs|
        error = assert_raise(TypeError) { Cumo::Linalg.norm(type[3, 4], ord, **kwargs) }
        assert_equal('invalid data type for BLAS/LAPACK', error.message)
      end
    end
  end

  test "errors" do
    m = Cumo::DFloat[[1, 2], [3, 4]]
    [
      [[Cumo::DFloat[1, 2], 'fro'], {}, 'invalid ord: fro'],
      [[m, 0], {}, 'invalid ord: 0'],
      [[m, 3], {}, 'invalid ord: 3'],
      [[m, 'x'], {}, 'invalid ord: x'],
      [[m, :fro], {}, 'invalid ord: fro'],
      [[m, 'fro'], { axis: 1 }, 'invalid ord: fro'],
      [[m], { axis: [0, 1, 1] }, 'the number of dimensions of axis is inappropriate for the norm: 3'],
      [[Cumo::DFloat.new(2, 3, 4), 2], {}, 'the number of dimensions of axis is inappropriate for the norm: 3'],
      [[m], { axis: 2 }, 'axis is out of range: [2]'],
      [[m], { axis: -3 }, 'axis is out of range: [-3]'],
      [[m], { axis: [1, 1] }, 'invalid axis: [1, 1]'],
      [[m], { axis: [1, -1] }, 'invalid axis: [1, -1]'],
      [[m], { axis: 'a' }, 'invalid axis: a']
    ].each do |args, kwargs, message|
      error = assert_raise(ArgumentError) { Cumo::Linalg.norm(*args, **kwargs) }
      assert_equal(message, error.message)
    end
  end

  sub_test_case "cond" do
    setup do
      omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
    end

    test "cond in the matrix norms" do
      a = Cumo::DFloat[[1, 2], [3, 4]]
      [[1, 21.0], [-1, 6.0], ['inf', 21.0], ['-inf', 6.0], ['nuc', 17.0]].each do |ord, expected|
        c = Cumo::Linalg.cond(a, ord)
        assert_kind_of(Float, c, ord.inspect)
        assert_in_delta(expected, c, 1e-13, ord.inspect)
      end
      c = Cumo::Linalg.cond(a, 'fro')
      assert_equal(0, c.ndim)
      assert_in_delta(15.0, c.to_a.flatten.first, 1e-13)
      assert_in_delta(5.824352060364905, Cumo::Linalg.cond(Cumo::DComplex[[1, 2i], [3, 4]], 1), 1e-13)
    end

    test "cond errors" do
      error = assert_raise(Cumo::Linalg::LapackError) { Cumo::Linalg.cond(Cumo::DFloat[[1, 2], [2, 4]], 1) }
      assert_equal('The matrix is singular, and the inverse matrix could not be computed.', error.message)
    end

    test "cond rejects an invalid ord before it inverts the matrix" do
      singular = Cumo::DFloat[[1, 2], [2, 4]]
      [[3, 'invalid ord: 3'], ['x', 'invalid ord: x'], [:fro, 'invalid ord: fro'], [1.5, 'invalid ord: 1.5']].each do |ord, message|
        error = assert_raise(ArgumentError) { Cumo::Linalg.cond(singular, ord) }
        assert_equal(message, error.message)
      end
      error = assert_raise(ArgumentError) { Cumo::Linalg.cond(Cumo::DFloat[[1, 2, 3]], 'x') }
      assert_equal('invalid ord: x', error.message)
      assert_in_delta(21.0, Cumo::Linalg.cond(Cumo::DFloat[[1, 2], [3, 4]], Float::INFINITY), 1e-13)
      assert_in_delta(6.0, Cumo::Linalg.cond(Cumo::DFloat[[1, 2], [3, 4]], -1.0), 1e-13)
    end

    test "cond takes the infinite Numeric ords norm takes" do
      klass = Class.new(Numeric)
      klass.define_method(:infinite?) { 1 }
      klass.define_method(:positive?) { true }
      klass.define_method(:to_s) { 'Infinity' }
      infinity = klass.new
      a = Cumo::DFloat[[1, 2], [3, 4]]
      assert_in_delta(7.0, Cumo::Linalg.norm(a, infinity, axis: [-2, -1]), 1e-13)
      assert_in_delta(21.0, Cumo::Linalg.cond(a, infinity), 1e-13)
    end
  end

  private

  def omit_unless_cusolver
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end
end
