# frozen_string_literal: true

require_relative "test_helper"

class NArrayExtraTest < CumoTestBase
  # instance methods

  def test_new_narray
    TYPES.each do |dtype|
      a = dtype[[0, 1, 2], [3, 4, 5]]
      na = a.new_narray
      assert_equal(2, na.ndim)
      assert_equal(2, na.shape[0])
      assert_equal(3, na.shape[1])
    end
  end

  def test_new_zeros
    TYPES.each do |dtype|
      a = dtype[[1], [2]]
      na = a.new_zeros
      assert_equal(2, na.ndim)
      assert_equal(2, na.shape[0])
      assert_equal(1, na.shape[1])
      assert_equal(0, na[0, 0])
      assert_equal(0, na[1, 0])
    end
  end

  def test_new_ones
    TYPES.each do |dtype|
      a = dtype[[2, 3], [4, 5]]
      na = a.new_ones
      assert_equal(2, na.ndim)
      assert_equal(2, na.shape[0])
      assert_equal(2, na.shape[1])
      assert_equal(1, na[0, 0])
      assert_equal(1, na[0, 1])
      assert_equal(1, na[1, 0])
      assert_equal(1, na[1, 1])
    end
  end

  def test_new_fill
    TYPES.each do |dtype|
      a = dtype[[1], [2]]
      na = a.new_fill(5)
      assert_equal(2, na.ndim)
      assert_equal(2, na.shape[0])
      assert_equal(1, na.shape[1])
      assert_equal(5, na[0, 0])
      assert_equal(5, na[1, 0])
    end
  end

  def test_fliplr
    TYPES.each do |dtype|
      a = dtype[[2, 3], [4, 5]]
      na = a.fliplr
      assert_equal(3, na[0, 0])
      assert_equal(2, na[0, 1])
      assert_equal(5, na[1, 0])
      assert_equal(4, na[1, 1])
    end
  end

  def test_flipud
    TYPES.each do |dtype|
      a = dtype[[2, 3], [4, 5]]
      na = a.flipud
      assert_equal(4, na[0, 0])
      assert_equal(5, na[0, 1])
      assert_equal(2, na[1, 0])
      assert_equal(3, na[1, 1])
    end
  end

  def test_rot90
    TYPES.each do |dtype|
      a = dtype[[0, 1], [2, 3]]
      na = a.rot90(0)
      assert_equal(a[0, 0], na[0, 0])
      assert_equal(a[0, 1], na[0, 1])
      assert_equal(a[1, 0], na[1, 0])
      assert_equal(a[1, 1], na[1, 1])

      na = a.rot90
      assert_equal(a[0, 1], na[0, 0])
      assert_equal(a[1, 1], na[0, 1])
      assert_equal(a[0, 0], na[1, 0])
      assert_equal(a[1, 0], na[1, 1])

      na = a.rot90(2)
      assert_equal(a[1, 1], na[0, 0])
      assert_equal(a[1, 0], na[0, 1])
      assert_equal(a[0, 1], na[1, 0])
      assert_equal(a[0, 0], na[1, 1])

      na = a.rot90(3)
      assert_equal(a[1, 0], na[0, 0])
      assert_equal(a[0, 0], na[0, 1])
      assert_equal(a[1, 1], na[1, 0])
      assert_equal(a[0, 1], na[1, 1])
    end
  end

  def test_to_i
    TYPES.each do |dtype|
      a = dtype[1, 2, 3]
      assert_raise(TypeError) { a.to_i }
      a = dtype[5]
      if [Cumo::DComplex, Cumo::SComplex].include?(dtype)
        assert_raise(RangeError) { a.to_i }
      else
        ai = a.to_i
        assert_equal(5, ai)
        assert_kind_of(Integer, ai)
      end
    end
  end

  def test_to_f
    TYPES.each do |dtype|
      a = dtype[1, 2, 3]
      assert_raise(TypeError) { a.to_f }

      a = dtype[5]
      if [Cumo::DComplex, Cumo::SComplex].include?(dtype)
        assert_raise(RangeError) { a.to_f }
      else
        af = a.to_f
        assert_equal(5, af)
        assert_kind_of(Float, af)
      end
    end
  end

  def test_to_c
    TYPES.each do |dtype|
      a = dtype[1, 2, 3]
      assert_raise(TypeError) { a.to_c }

      a = dtype[5]
      ac = a.to_c
      assert_equal(5 + 0i, ac)
      assert_kind_of(Complex, ac)
    end
  end

  def test_each_over_axis
    TYPES.each do |dtype|
      a = dtype[[0, 1], [2, 3]]
      enm = a.each_over_axis
      assert_instance_of(Enumerator, enm)
      assert_equal(2, enm.to_a.size)
      assert_equal(dtype[0, 1], enm.next)
      assert_equal(dtype[2, 3], enm.next)

      enm = a.each_over_axis(1)
      assert_instance_of(Enumerator, enm)
      assert_equal(2, enm.to_a.size)
      assert_equal(dtype[0, 2], enm.next)
      assert_equal(dtype[1, 3], enm.next)

      a = dtype.new
      assert_raise(ArgumentError) { a.each_over_axis(1) { |_v| nil } }
    end
  end

  def test_each_over_axis_on_an_unallocated_array
    omit("Cumo yields the array instead of raising on an unallocated one")
    assert_raise(RuntimeError) { Cumo::DFloat.new.each_over_axis { |_v| nil } }
  end

  def test_append
    TYPES.each do |dtype|
      a = dtype[1, 2]
      na = a.append([[3, 4], [5, 6]])
      assert_equal(1, na.ndim)
      assert_equal(6, na.size)

      a = dtype[[1, 2]]
      na = a.append([[3, 4], [5, 6]], axis: 0)
      assert_equal(2, na.ndim)
      assert_equal(3, na.shape[0])
      assert_equal(2, na.shape[1])
      assert_raise(Cumo::NArray::DimensionError) { a.append([3, 4], axis: 0) }

      a = dtype[[1], [2]]
      na = a.append([[3, 4], [5, 6]], axis: 1)
      assert_equal(2, na.ndim)
      assert_equal(2, na.shape[0])
      assert_equal(3, na.shape[1])
    end
  end

  def test_delete
    TYPES.each do |dtype|
      a = dtype[[1, 2], [3, 4], [5, 6]]
      na = a.delete(1)
      assert_equal(1, na.ndim)
      assert_equal(5, na.size)

      na = a.delete(1, 0)
      assert_equal(2, na.ndim)
      assert_equal(2, na.shape[0])
      assert_equal(2, na.shape[1])

      na = a.delete(1, 1)
      assert_equal(2, na.ndim)
      assert_equal(3, na.shape[0])
      assert_equal(1, na.shape[1])

      na = a.delete([0, 2], 0)
      assert_equal(2, na.ndim)
      assert_equal(1, na.shape[0])
      assert_equal(2, na.shape[1])
    end
  end

  def test_insert
    TYPES.each do |dtype|
      a = dtype[[1, 2], [3, 4], [5, 6]]
      na = a.insert(1, 9)
      assert_equal(1, na.ndim)
      assert_equal(7, na.size)
      assert_equal(9, na[1])

      na = a.insert(1, 9, axis: 1)
      assert_equal(2, na.ndim)
      assert_equal(3, na.shape[0])
      assert_equal(3, na.shape[1])
      assert_equal(dtype[9, 9, 9], na[true, 1])

      na = a.insert(1, [7, 8], axis: 0)
      assert_equal(2, na.ndim)
      assert_equal(4, na.shape[0])
      assert_equal(2, na.shape[1])
      assert_equal(dtype[7, 8], na[1, true])

      na = a.insert([0, 2], [8, 9], axis: 1)
      assert_equal(2, na.ndim)
      assert_equal(3, na.shape[0])
      assert_equal(4, na.shape[1])
      assert_equal(dtype[8, 8, 8], na[true, 0])
      assert_equal(dtype[9, 9, 9], na[true, 3])
    end
  end

  def test_instance_method_concatenate
    TYPES.each do |dtype|
      a = dtype[[1, 2], [3, 4]]
      b = dtype[[5, 6]]
      na = a.concatenate(b, axis: 0)
      assert_equal(2, na.ndim)
      assert_equal(3, na.shape[0])
      assert_equal(2, na.shape[1])
      assert_equal(na[2, true], b[0, true])

      na = a.concatenate(b.transpose, axis: 1)
      assert_equal(2, na.ndim)
      assert_equal(2, na.shape[0])
      assert_equal(3, na.shape[1])
      assert_equal(na[true, 2], b[0, true])

      a = dtype[1, 2]
      na = a.concatenate([3, 4])
      assert_equal(1, na.ndim)
      assert_equal(4, na.size)
      assert_equal(3, na[2])
      assert_equal(4, na[3])

      na = a.concatenate(3)
      assert_equal(1, na.ndim)
      assert_equal(3, na.size)
      assert_equal(3, na[2])
      assert_raise(TypeError) { a.concatenate('3') }
    end
  end

  def test_split
    TYPES.each do |dtype|
      a = dtype.new(6).seq
      na = a.split(2)
      assert_equal(2, na.size)
      assert_equal(dtype[0, 1, 2], na[0])
      assert_equal(dtype[3, 4, 5], na[1])

      na = a.split([2, 5])
      assert_equal(3, na.size)
      assert_equal(dtype[0, 1], na[0])
      assert_equal(dtype[2, 3, 4], na[1])
      assert_equal(dtype[5], na[2])

      na = a.split(Cumo::UInt8[2, 5])
      assert_equal(3, na.size)
      assert_equal(dtype[0, 1], na[0])
      assert_equal(dtype[2, 3, 4], na[1])
      assert_equal(dtype[5], na[2])

      a = dtype.new(2, 3).seq
      na = a.split(2, axis: 1)
      assert_equal(2, na.size)
      assert_equal(dtype[[0, 1], [3, 4]], na[0])
      assert_equal(dtype[[2], [5]], na[1])
      assert_raise(TypeError) { a.split('2') }
    end
  end

  def test_vsplit
    TYPES.each do |dtype|
      a = dtype.new(3, 2).seq
      na = a.vsplit(2)
      assert_equal(2, na.size)
      assert_equal(dtype[[0, 1], [2, 3]], na[0])
      assert_equal(dtype[[4, 5]], na[1])
    end
  end

  def test_hsplit
    TYPES.each do |dtype|
      a = dtype.new(3, 2).seq
      na = a.hsplit(2)
      assert_equal(2, na.size)
      assert_equal(dtype[[0], [2], [4]], na[0])
      assert_equal(dtype[[1], [3], [5]], na[1])
    end
  end

  def test_dsplit
    TYPES.each do |dtype|
      a = dtype.new(3, 2, 2).seq
      na = a.dsplit(2)
      assert_equal(2, na.size)
      assert_equal(dtype[[[0], [2]], [[4], [6]], [[8], [10]]], na[0])
      assert_equal(dtype[[[1], [3]], [[5], [7]], [[9], [11]]], na[1])
    end
  end

  def test_tile
    TYPES.each do |dtype|
      a = dtype.new(3).seq
      na = a.tile(2)
      assert_equal(dtype[0, 1, 2, 0, 1, 2], na)

      na = a.tile(2, 2)
      assert_equal(dtype[[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]], na)

      na = a.tile(2, 1, 2)
      assert_equal(dtype[[[0, 1, 2, 0, 1, 2]], [[0, 1, 2, 0, 1, 2]]], na)

      a = dtype.new(2, 2).seq
      na = a.tile(2)
      assert_equal(dtype[[0, 1, 0, 1], [2, 3, 2, 3]], na)

      na = a.tile(2, 2)
      assert_equal(dtype[[0, 1, 0, 1], [2, 3, 2, 3], [0, 1, 0, 1], [2, 3, 2, 3]], na)
    end
  end

  def test_repeat
    TYPES.each do |dtype|
      a = dtype.new(2, 3).seq
      na = a.repeat(2)
      assert_equal(dtype[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], na)

      na = a.repeat(2, axis: 0)
      assert_equal(dtype[[0, 1, 2], [0, 1, 2], [3, 4, 5], [3, 4, 5]], na)

      na = a.repeat(2, axis: 1)
      assert_equal(dtype[[0, 0, 1, 1, 2, 2], [3, 3, 4, 4, 5, 5]], na)
      assert_raise(ArgumentError) { a.repeat(2, axis: '0') }

      na = a.repeat([1, 2], axis: 0)
      assert_equal(dtype[[0, 1, 2], [3, 4, 5], [3, 4, 5]], na)
      assert_raise(ArgumentError) { a.repeat([1, 2], axis: 1) }
    end
  end

  def test_diff
    TYPES.each do |dtype|
      a = dtype[1, 3, 5, 7, 9]
      na = a.diff
      assert_equal(dtype[2, 2, 2, 2], na)

      a = dtype[1, 3, 5, 9]
      na = a.diff(2)
      assert_equal(dtype[0, 2], na)

      a = dtype[[0, 3, 6, 8], [1, 5, 6, 10]]
      na = a.diff(axis: 0)
      assert_equal(dtype[[1, 2, 0, 2]], na)

      na = a.diff(axis: 1)
      assert_equal(dtype[[3, 3, 2], [4, 1, 4]], na)
    end
  end

  def test_triu
    TYPES.each do |dtype|
      a = dtype.new(3, 3).seq + 1
      na = a.triu
      assert_equal(dtype[[1, 2, 3], [0, 5, 6], [0, 0, 9]], na)

      na = a.triu(1)
      assert_equal(dtype[[0, 2, 3], [0, 0, 6], [0, 0, 0]], na)

      na = a.triu(-1)
      assert_equal(dtype[[1, 2, 3], [4, 5, 6], [0, 8, 9]], na)

      av = a[0...2, 0...2]
      refute_predicate(av, :contiguous?)

      av.triu!
      assert_equal(dtype[[1, 2, 3], [0, 5, 6], [7, 8, 9]], a)
    end
  end

  def test_tril
    TYPES.each do |dtype|
      a = dtype.new(3, 3).seq + 1
      na = a.tril
      assert_equal(dtype[[1, 0, 0], [4, 5, 0], [7, 8, 9]], na)

      na = a.tril(1)
      assert_equal(dtype[[1, 2, 0], [4, 5, 6], [7, 8, 9]], na)

      na = a.tril(-1)
      assert_equal(dtype[[0, 0, 0], [4, 0, 0], [7, 8, 0]], na)

      av = a[1...3, 1...3]
      refute_predicate(av, :contiguous?)

      av.tril!
      assert_equal(dtype[[1, 2, 3], [4, 5, 0], [7, 8, 9]], a)
    end
  end

  def test_diag_indices
    TYPES.each do |dtype|
      a = dtype[1, 2, 3]
      assert_raise(Cumo::NArray::ShapeError) { a.diag_indices }

      a = dtype[[1, 2], [3, 4]]
      assert_equal(Cumo::Int32[0, 3], a.diag_indices)

      a = dtype[[1, 2, 3], [4, 5, 6]]
      assert_equal(Cumo::Int32[0, 4], a.diag_indices)

      a = dtype[[1, 2], [3, 4], [5, 6]]
      assert_equal(Cumo::Int32[0, 3], a.diag_indices)
    end
  end

  def test_diag
    TYPES.each do |dtype|
      a = dtype[1, 2]
      na = a.diag
      assert_equal(dtype[[1, 0], [0, 2]], na)

      na = a.diag(1)
      assert_equal(dtype[[0, 1, 0], [0, 0, 2], [0, 0, 0]], na)

      na = a.diag(-1)
      assert_equal(dtype[[0, 0, 0], [1, 0, 0], [0, 2, 0]], na)
    end
  end

  def test_argsort
    omit("Cumo does not implement argsort")
    types = [Cumo::SFloat, Cumo::DFloat, Cumo::Int64, Cumo::Int32, Cumo::Int16, Cumo::Int8,
             Cumo::UInt64, Cumo::UInt32, Cumo::UInt16, Cumo::UInt8]
    types.each do |dtype|
      a = dtype.new(24).seq.reshape(2, 3, 4)
      s = a.argsort
      s0 = a.argsort(axis: 0)
      s1 = a.argsort(axis: 1)
      s2 = a.argsort(axis: 2)
      sn = a.argsort(axis: nil)

      assert_kind_of(Cumo::Int32, s)
      assert_kind_of(Cumo::Int32, s0)
      assert_kind_of(Cumo::Int32, s1)
      assert_kind_of(Cumo::Int32, s2)
      assert_kind_of(Cumo::Int32, sn)
      assert_equal(3, s.ndim)
      assert_equal(3, s0.ndim)
      assert_equal(3, s1.ndim)
      assert_equal(3, s2.ndim)
      assert_equal([2, 3, 4], s.shape)
      assert_equal([2, 3, 4], s0.shape)
      assert_equal([2, 3, 4], s1.shape)
      assert_equal([2, 3, 4], s2.shape)
      assert_equal(Cumo::Int32[[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                               [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]], s)
      assert_equal(Cumo::Int32[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                               [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]], s0)
      assert_equal(Cumo::Int32[[[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]],
                               [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]], s1)
      assert_equal(Cumo::Int32[[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                               [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]], s2)
      assert_equal(Cumo::Int32.new(24).seq, sn)
    end

    types = [Cumo::SFloat, Cumo::DFloat]
    types.each do |dtype|
      a = dtype.new(5).rand - 0.5
      s = a.argsort

      assert_kind_of(Cumo::Int32, s)
      assert_equal(1, s.ndim)
      assert_equal([5], s.shape)
      assert_equal(a.sort_index, s)

      shapes = [[3, 2], [2, 3], [3, 3]]
      shapes.each do |shape|
        a = dtype.new(*shape).rand - 0.5
        s0 = a.argsort(0)
        s1 = a.argsort(1)
        sn = a.argsort(nil)

        ex0 = Cumo::Int32.cast(Array.new(shape[1]) { |j| a[true, j].sort_index }).transpose.dup
        ex1 = Cumo::Int32.cast(Array.new(shape[0]) { |i| a[i, true].sort_index })
        exn = a.flatten.sort_index

        assert_kind_of(Cumo::Int32, s0)
        assert_kind_of(Cumo::Int32, s1)
        assert_kind_of(Cumo::Int32, sn)
        assert_equal(2, s0.ndim)
        assert_equal(2, s1.ndim)
        assert_equal(1, sn.ndim)
        assert_equal(shape, s0.shape)
        assert_equal(shape, s1.shape)
        assert_equal([shape.reduce(:*)], sn.shape)
        assert_equal(ex0, s0)
        assert_equal(ex1, s1)
        assert_equal(exn, sn)
        assert_equal(ex0, a.argsort(-2))
        assert_equal(ex1, a.argsort(-1))
        assert_equal(ex1, a.argsort)
      end
    end
  end

  def test_trace
    TYPES.each do |dtype|
      a = dtype.new(3, 3).seq + 1
      tr = a.trace
      assert_equal(15, tr)
    end
  end

  def test_dot
    TYPES.each do |dtype|
      # vector.dot(vector)
      a = dtype[1..3]
      b = dtype[2..4]
      assert_equal((1 * 2) + (2 * 3) + (3 * 4), a.dot(b))
      # matrix.dot(vector)
      a = dtype[1..6].reshape(3, 2)
      b = dtype[1..2]
      assert_equal(dtype[5, 11, 17], a.dot(b))
      # vector.dot(matrix)
      a = dtype[1..2]
      b = dtype[1..6].reshape(2, 3)
      assert_equal(dtype[9, 12, 15], a.dot(b))
      # matrix.dot(matrix)
      a = dtype[1..6].reshape(3, 2)
      b = dtype[1..6].reshape(2, 3)
      assert_equal(dtype[[9, 12, 15], [19, 26, 33], [29, 40, 51]], a.dot(b))
      assert_equal(dtype[[22, 28], [49, 64]], b.dot(a))
      # matrix.dot(matrix) with incorrect shape
      a = dtype[1..6].reshape(3, 2)
      b = dtype[1..9].reshape(3, 3)
      assert_raise(Cumo::NArray::ShapeError) { a.dot(b) }
    end

    # view
    a = Cumo::DFloat[[1, 2], [3, 4]]
    b = a.transpose.dot(a)
    assert_equal(Cumo::DFloat[[10, 14], [14, 20]], b)
  end

  def test_dot_between_types
    a = Cumo::Int32[1, 2, 3]
    b = Cumo::DFloat[[4], [5], [6]]
    assert_equal(Cumo::DFloat[32], a.dot(b))
    assert_equal(Cumo::DFloat[32], Cumo::DFloat[1, 2, 3].dot(Cumo::Int32[[4], [5], [6]]))
  end

  def test_inner
    TYPES.each do |dtype|
      a = dtype[1, 2, 3]
      b = dtype[4, 5, 6]
      assert_equal(32, a.inner(b))
    end
  end

  def test_outer
    TYPES.each do |dtype|
      a = dtype[1, 2]
      b = dtype[3, 4]
      assert_equal(dtype[[3, 4], [6, 8]], a.outer(b))
      assert_equal(dtype[[3, 6], [4, 8]], b.outer(a, axis: 0))
      assert_raise(ArgumentError) { a.outer(b, axis: 1) }
    end
  end

  def test_percentile
    TYPES.each do |dtype|
      next if [Cumo::DComplex, Cumo::SComplex, Cumo::RObject].include?(dtype)

      a = dtype[[1, 2, 3], [5, 7, 11]]
      assert_in_delta(a.percentile(0), 1.0, 1e-14)
      assert_in_delta(a.percentile(50), 4.0, 1e-14)
      assert_in_delta(a.percentile(90), 9.0, 1e-14)
      assert_in_delta(a.percentile(100), 11.0, 1e-14)
      assert_equal(dtype[1, 2, 3],        a.percentile(0, axis: 0))
      assert_equal(dtype[5, 7, 11],       a.percentile(100, axis: 0))
      assert_equal(dtype[1, 5],           a.percentile(0, axis: 1))
      assert_equal(dtype[2, 7],           a.percentile(50, axis: 1))
      assert_equal(dtype[3, 11],          a.percentile(100, axis: 1))
      if [Cumo::DFloat, Cumo::SFloat].include?(dtype)
        assert_equal(dtype[3, 4.5, 7],      a.percentile(50, axis: 0))
        assert_equal(dtype[4.6, 6.5, 10.2], a.percentile(90, axis: 0))
        assert_equal(dtype[2.8, 10.2],      a.percentile(90, axis: 1))
      else
        assert_equal(Cumo::DFloat[3, 4.5, 7],      a.percentile(50, axis: 0))
        assert_equal(Cumo::DFloat[4.6, 6.5, 10.2], a.percentile(90, axis: 0))
        assert_equal(Cumo::DFloat[2.8, 10.2],      a.percentile(90, axis: 1))
      end
    end
  end

  def test_kron
    TYPES.each do |dtype|
      a = dtype[1, 10, 100]
      assert_equal(dtype[5, 6, 7, 50, 60, 70, 500, 600, 700], a.kron(dtype[5, 6, 7]))
      assert_equal(dtype[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],
                   dtype.eye(2).kron(dtype.ones(2, 2)))
    end
  end

  def test_rad2deg
    FLOAT_TYPES.each do |dtype|
      a = dtype[Math::PI / 2, Math::PI / 4]
      na = a.rad2deg
      if [Cumo::DComplex, Cumo::SComplex].include?(dtype)
        assert_in_delta(90, na[0].real, 1e-6)
        assert_in_delta(0, na[0].imag, 1e-6)
        assert_in_delta(45, na[1].real, 1e-6)
        assert_in_delta(0, na[1].imag, 1e-6)
      else
        assert_in_delta(90, na[0], 1e-6)
        assert_in_delta(45, na[1], 1e-6)
      end
    end
  end

  def test_deg2rad
    FLOAT_TYPES.each do |dtype|
      a = dtype[90, 45]
      na = a.deg2rad
      if [Cumo::DComplex, Cumo::SComplex].include?(dtype)
        assert_in_delta(Math::PI / 2, na[0].real, 1e-6)
        assert_in_delta(0, na[0].imag, 1e-6)
        assert_in_delta(Math::PI / 4, na[1].real, 1e-6)
        assert_in_delta(0, na[1].imag, 1e-6)
      else
        assert_in_delta(Math::PI / 2, na[0], 1e-6)
        assert_in_delta(Math::PI / 4, na[1], 1e-6)
      end
    end
  end

  def test_cov
    [Cumo::SFloat, Cumo::SComplex, Cumo::DFloat, Cumo::DComplex].each do |dtype|
      a = dtype[1, 2, 3]
      if [Cumo::DComplex, Cumo::SComplex].include?(dtype)
        assert_equal(1.0 + 0.0i, a.cov)
      else
        assert_in_delta(1.0, a.cov.extract_cpu)
      end

      a = dtype[[1, 2], [3, 4]]
      assert_equal(dtype[[0.5, 0.5], [0.5, 0.5]], a.cov)
    end

    # xy
    x = Cumo::DFloat[[3, 2, 7], [4, 2, 3]]
    y = Cumo::DFloat[6, 5, 4]
    z = Cumo::NArray.vstack([x, y])
    assert_equal(z.cov, x.cov(y))
    x = Cumo::DFloat[[1, 2, 3]]
    y = Cumo::DComplex[[1i, 2i, 3i]]
    assert_equal(Cumo::DComplex[[1, -1i], [1i, 1]], x.cov(y))
    # ddof
    x = Cumo::DFloat[[3, 1, 2], [2, 5, 8]]
    assert_equal(Cumo::DFloat[[1, -1.5], [-1.5, 9]], x.cov(ddof: 1))
    assert_equal(Cumo::DFloat[[2.fdiv(3), -1], [-1, 6]], x.cov(ddof: 0))
    # fweights
    fw = Cumo::Int32[1, 4, 2]
    c = x.cov(fweights: fw)
    error = (Cumo::DFloat[[0.6190476, -0.2857143], [-0.2857143, 4.2857143]] - c).abs.max
    assert_operator(error, :<, 1e-6)
    # aweights
    aw = Cumo::DFloat[2, 0.5, 0.25]
    c = x.cov(aweights: aw)
    error = (Cumo::DFloat[[1.4230769, -2.6538461], [-2.6538461, 8.6538461]] - c).abs.max
    assert_operator(error, :<, 1e-6)
    # fweights and aweights
    c = x.cov(fweights: fw, aweights: aw)
    error = (Cumo::DFloat[[1.1900826, -1.7851240], [-1.7851240, 5.3553719]] - c).abs.max
    assert_operator(error, :<, 1e-6)
  end

  # class methods

  def test_cast
    TYPES.each do |dtype|
      a = dtype.cast(dtype[1, 2, 3])
      assert_kind_of(dtype, a)
      assert_equal(1, a.ndim)
      assert_equal(3, a.size)

      a = dtype.cast([1, 2, 3])
      assert_kind_of(dtype, a)
      assert_equal(1, a.ndim)
      assert_equal(3, a.size)

      a = dtype.cast(1)
      assert_kind_of(dtype, a)
      assert_equal(0, a.ndim)
      assert_equal(1, a.size)
      if [Cumo::RObject].include?(dtype)
        a = dtype.cast('a')
        assert_kind_of(dtype, a)
        assert_equal(0, a.ndim)
        assert_equal(1, a.size)
      else
        assert_raise(Cumo::NArray::CastError) { dtype.cast('a') }
      end
    end
  end

  def test_asarray
    TYPES.each do |dtype|
      a = dtype.asarray(dtype[1, 2, 3])
      assert_equal(dtype[1, 2, 3], a)

      a = dtype.asarray([1, 2, 3])
      assert_equal(dtype[1, 2, 3], a)

      a = dtype.asarray(dtype.cast(1))
      assert_equal(dtype[1], a)

      a = dtype.asarray(1)
      assert_equal(dtype[1], a)

      a = dtype.asarray(1..3)
      assert_equal(dtype[1, 2, 3], a)

      next unless [Cumo::RObject].include?(dtype)

      a = dtype.asarray('a')
      assert_equal(dtype.cast('a'), a)
    end
  end

  def test_parse
    TYPES.each do |dtype|
      a = dtype.parse('1 2 3')
      assert_equal(dtype[[1, 2, 3]], a)

      a = dtype.parse('1 2 3; 4 5 6')
      assert_equal(dtype[[1, 2, 3], [4, 5, 6]], a)

      a = dtype.parse('1 2; 3 4 & 5 6; 7 8', split3d: '&')
      assert_equal(dtype[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], a)
    end
  end

  def test_parse_rejects_non_literal_token
    ['system("id")', '`id`', 'Kernel.exit', '$stdout', 'self', 'x', 'TRUE', 'True', '[1, 2]'].each do |src|
      assert_raise(ArgumentError) { Cumo::DFloat.parse(src) }
    end
  end

  def test_parse_rejects_expression
    ['1+1', '2*3', '3/4', '3r', '1/3r'].each do |src|
      assert_raise(ArgumentError) { Cumo::DFloat.parse(src) }
    end
  end

  def test_parse_error_names_the_token
    error = assert_raise(ArgumentError) { Cumo::DFloat.parse('1 2 oops 4') }
    assert_include(error.message, 'oops')
  end

  def test_parse_integer_literals
    actual = Cumo::Int64.parse('1 -3 +4 1_000')
    assert_kind_of(Cumo::Int64, actual)
    assert_equal(Cumo::Int64[[1, -3, 4, 1000]], actual)

    actual = Cumo::Int64.parse('0x1f -0x1f 0b101 0o17 017')
    assert_kind_of(Cumo::Int64, actual)
    assert_equal(Cumo::Int64[[31, -31, 5, 15, 15]], actual)
  end

  def test_parse_float_literals
    actual = Cumo::DFloat.parse('1.5 -2.5 1e5 1.2e-3')
    assert_kind_of(Cumo::DFloat, actual)
    assert_equal(Cumo::DFloat[[1.5, -2.5, 100_000.0, 0.0012]], actual)
  end

  def test_parse_complex_literals
    actual = Cumo::DComplex.parse('2i 2+3i -2i 1.5i')
    assert_kind_of(Cumo::DComplex, actual)
    assert_equal(Cumo::DComplex[[Complex(0, 2), Complex(2, 3), Complex(0, -2), Complex(0, 1.5)]], actual)
  end

  def test_parse_boolean_literals
    actual = Cumo::NArray.parse('true false nil')
    assert_kind_of(Cumo::Bit, actual)
    assert_equal(Cumo::Bit[[1, 0, 0]], actual)

    actual = Cumo::NArray.parse("true false\nfalse true")
    assert_kind_of(Cumo::Bit, actual)
    assert_equal(Cumo::Bit[[1, 0], [0, 1]], actual)
  end

  def test_class_method_concatenate
    TYPES.each do |dtype|
      a = dtype[[1, 2], [3, 4]]
      b = dtype[[5, 6]]
      na = dtype.concatenate([a, b], axis: 0)
      assert_equal(dtype[[1, 2], [3, 4], [5, 6]], na)

      na = dtype.concatenate([a, b.transpose], axis: 1)
      assert_equal(dtype[[1, 2, 5], [3, 4, 6]], na)

      a = dtype[1, 2]
      na = dtype.concatenate([a, [3, 4]])
      assert_equal(dtype[1, 2, 3, 4], na)

      na = dtype.concatenate([a, 3])
      assert_equal(dtype[1, 2, 3], na)
      assert_raise(TypeError) { dtype.concatenate([a, '3']) }
    end
  end

  def test_vstack
    TYPES.each do |dtype|
      a = dtype[1, 2]
      b = dtype[3, 4]
      na = dtype.vstack([a, b])
      assert_equal(dtype[[1, 2], [3, 4]], na)
    end
  end

  def test_hstack
    TYPES.each do |dtype|
      a = dtype[1, 2]
      b = dtype[3, 4]
      na = dtype.hstack([a, b])
      assert_equal(dtype[1, 2, 3, 4], na)
    end
  end

  def test_dstack
    TYPES.each do |dtype|
      a = dtype[1, 2]
      b = dtype[3, 4]
      na = dtype.dstack([a, b])
      assert_equal(dtype[[[1, 3], [2, 4]]], na)
    end
  end

  def test_column_stack
    TYPES.each do |dtype|
      a = dtype.column_stack([dtype[1, 2], dtype[3, 4]])
      assert_equal(dtype[[1, 3], [2, 4]], a)

      a = dtype.column_stack([dtype.cast(1), dtype.cast(2)])
      assert_equal(dtype[[1, 2]], a)

      a = dtype.column_stack([dtype[[1, 2], [3, 4]], dtype[[5, 6], [7, 8]]])
      assert_equal(dtype[[1, 2, 5, 6], [3, 4, 7, 8]], a)
    end
  end

  def test_narray_cast
    a = Cumo::NArray.cast([1, 2, 3])
    assert_kind_of(Cumo::Int32, a)
    assert_equal(1, a.ndim)
    assert_equal(3, a.size)

    a = Cumo::NArray.cast(1)
    assert_kind_of(Cumo::Int32, a)
    assert_equal(0, a.ndim)
    assert_equal(1, a.size)
    assert_raise(TypeError) { Cumo::NArray.cast('1') }

    a = Cumo::NArray.cast(Set[1, 2, 3])
    assert_equal(Cumo::Int32[1, 2, 3], a)
  end

  def test_triu_indices
    TYPES.each do |dtype|
      a = dtype.new(3, 3).seq + 1
      idx = a.triu_indices
      assert_equal(Cumo::Int32[0, 1, 2, 4, 5, 8], idx)

      idx = a.triu_indices(1)
      assert_equal(Cumo::Int32[1, 2, 5], idx)

      idx = a.triu_indices(-1)
      assert_equal(Cumo::Int32[0, 1, 2, 3, 4, 5, 7, 8], idx)
    end
  end

  def test_triu_indices_class_method
    idx = Cumo::NArray.triu_indices(3, 3)
    assert_equal(Cumo::Int32[0, 1, 2, 4, 5, 8], idx)

    idx = Cumo::NArray.triu_indices(3, 3, 1)
    assert_equal(Cumo::Int32[1, 2, 5], idx)

    idx = Cumo::NArray.triu_indices(2, 4)
    assert_equal(Cumo::Int32[0, 1, 2, 3, 5, 6, 7], idx)

    idx = Cumo::NArray.triu_indices(4, 2)
    assert_equal(Cumo::Int32[0, 1, 3], idx)
  end

  def test_tril_indices
    TYPES.each do |dtype|
      a = dtype.new(3, 3).seq + 1
      idx = a.tril_indices
      assert_equal(Cumo::Int32[0, 3, 4, 6, 7, 8], idx)

      idx = a.tril_indices(1)
      assert_equal(Cumo::Int32[0, 1, 3, 4, 5, 6, 7, 8], idx)

      idx = a.tril_indices(-1)
      assert_equal(Cumo::Int32[3, 6, 7], idx)
    end
  end

  def test_tril_indices_class_method
    idx = Cumo::NArray.tril_indices(3, 3)
    assert_equal(Cumo::Int32[0, 3, 4, 6, 7, 8], idx)

    idx = Cumo::NArray.tril_indices(3, 3, 1)
    assert_equal(Cumo::Int32[0, 1, 3, 4, 5, 6, 7, 8], idx)

    idx = Cumo::NArray.tril_indices(2, 4)
    assert_equal(Cumo::Int32[0, 4, 5], idx)

    idx = Cumo::NArray.tril_indices(4, 2)
    assert_equal(Cumo::Int32[0, 2, 3, 4, 5, 6, 7], idx)
  end

  def test_triu_indices_error
    a = Cumo::DFloat[1, 2, 3]
    assert_raise(Cumo::NArray::ShapeError) { a.triu_indices }
  end

  def test_tril_indices_error
    a = Cumo::DFloat[1, 2, 3]
    assert_raise(Cumo::NArray::ShapeError) { a.tril_indices }
  end

  def test_triu_bang_with_k
    TYPES.each do |dtype|
      a = dtype.new(3, 3).seq + 1
      a.triu!(1)
      assert_equal(dtype[[0, 2, 3], [0, 0, 6], [0, 0, 0]], a)

      a = dtype.new(3, 3).seq + 1
      a.triu!(-1)
      assert_equal(dtype[[1, 2, 3], [4, 5, 6], [0, 8, 9]], a)
    end
  end

  def test_tril_bang_with_k
    TYPES.each do |dtype|
      a = dtype.new(3, 3).seq + 1
      a.tril!(1)
      assert_equal(dtype[[1, 2, 0], [4, 5, 6], [7, 8, 9]], a)

      a = dtype.new(3, 3).seq + 1
      a.tril!(-1)
      assert_equal(dtype[[0, 0, 0], [4, 0, 0], [7, 8, 0]], a)
    end
  end

  def test_triu_bang_on_noncontiguous
    TYPES.each do |dtype|
      a = dtype.new(4, 4).seq + 1
      view = a[1..2, 1..2]
      refute_predicate(view, :contiguous?)
      view.triu!
      assert_equal(dtype[[6, 7], [0, 11]], view)
    end
  end

  def test_tril_bang_on_noncontiguous
    TYPES.each do |dtype|
      a = dtype.new(4, 4).seq + 1
      view = a[1..2, 1..2]
      refute_predicate(view, :contiguous?)
      view.tril!
      assert_equal(dtype[[6, 0], [10, 11]], view)
    end
  end

  def test_triu_error_on_1d
    TYPES.each do |dtype|
      a = dtype[1, 2, 3]
      assert_raise(Cumo::NArray::ShapeError) { a.triu! }
    end
  end

  def test_tril_error_on_1d
    TYPES.each do |dtype|
      a = dtype[1, 2, 3]
      assert_raise(Cumo::NArray::ShapeError) { a.tril! }
    end
  end

  def test_reshape_bang
    TYPES.each do |dtype|
      a = dtype.new(6).seq
      a.reshape!(2, 3)
      assert_equal([2, 3], a.shape)
      assert_equal(6, a.size)
      assert_equal(dtype[[0, 1, 2], [3, 4, 5]], a)
    end
  end

  def test_reshape_bang_3d
    TYPES.each do |dtype|
      a = dtype.new(24).seq
      a.reshape!(2, 3, 4)
      assert_equal([2, 3, 4], a.shape)
      assert_equal(24, a.size)
    end
  end

  def test_reshape_bang_preserves_data
    TYPES.each do |dtype|
      a = dtype.new(6).seq
      original_data = a.to_a
      a.reshape!(3, 2)
      assert_equal(original_data.flatten, a.to_a.flatten)
    end
  end

  def test_check_axis_valid
    a = Cumo::DFloat.new(2, 3, 4).seq
    assert_equal(0, a.send(:check_axis, 0))
    assert_equal(1, a.send(:check_axis, 1))
    assert_equal(2, a.send(:check_axis, 2))
    assert_equal(0, a.send(:check_axis, -3))
    assert_equal(1, a.send(:check_axis, -2))
    assert_equal(2, a.send(:check_axis, -1))
  end

  def test_check_axis_invalid
    a = Cumo::DFloat.new(2, 3, 4).seq
    assert_raise(ArgumentError) { a.send(:check_axis, 3) }
    assert_raise(ArgumentError) { a.send(:check_axis, -4) }
    assert_raise(ArgumentError) { a.send(:check_axis, '0') }
    assert_raise(ArgumentError) { a.send(:check_axis, 0.5) }
  end
end
