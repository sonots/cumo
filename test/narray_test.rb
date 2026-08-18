# frozen_string_literal: true

require_relative "test_helper"

class NArrayTest < Test::Unit::TestCase
  types = [
    Cumo::DFloat,
    Cumo::SFloat,
    Cumo::DComplex,
    Cumo::SComplex,
    Cumo::Int64,
    Cumo::Int32,
    Cumo::Int16,
    Cumo::Int8,
    Cumo::UInt64,
    Cumo::UInt32,
    Cumo::UInt16,
    Cumo::UInt8,
  ]
  float_types = [
    Cumo::DFloat,
    Cumo::DComplex,
  ]

  if ENV['DTYPE']
    types.select! { |type| type.to_s.downcase.include?(ENV['DTYPE'].downcase) }
    float_types.select! { |type| type.to_s.downcase.include?(ENV['DTYPE'].downcase) }
  end

  def setup
    Cumo::NArray.srand(0)
  end

  types.each do |dtype|
    test dtype do
      assert { dtype < Cumo::NArray }
    end

    test "#{dtype}[]" do
      a = dtype[]

      assert_raise(Cumo::NArray::ShapeError) { a[true] }
      assert_raise(Cumo::NArray::ShapeError) { a[1..-1] }

      assert { a.size == 0 }
      assert { a.ndim == 1 }
      assert { a.shape == [0] }
      assert { !a.inplace? }
      assert { a.row_major? }
      assert { !a.column_major? }
      assert { a.host_order? }
      assert { !a.byte_swapped? }
      assert { a == [] }
      assert { a.to_a == [] }
      assert { a.to_a.is_a?(Array) }
      assert { a.dup == a }
      assert { a.clone == a }
      assert { a.dup.object_id != a.object_id }
      assert { a.clone.object_id != a.object_id }
    end

    types.each do |other_dtype|
      next if dtype == other_dtype

      test "#{dtype}[] == #{other_dtype}[]" do
        assert { dtype[] == other_dtype[] }
      end
    end

    test "#{dtype},free" do
      a = dtype[1, 2, 3, 5, 7, 11]
      assert { a.free }
      assert { !a.free } # return false if already freed
    end

    procs = [
      [proc { |tp, a| tp[*a] }, ""],
      [proc { |tp, a| tp[*a][true] }, "[true]"],
      [proc { |tp, a| tp[*a][0..-1] }, "[0..-1]"]
    ]
    procs.each do |init, ref|

      test "#{dtype},[1,2,3,5,7,11]#{ref}" do
        src = [1, 2, 3, 5, 7, 11]
        a = init.call(dtype, src)

        assert { a.is_a?(dtype) }
        assert { a.size == 6 }
        assert { a.ndim == 1 }
        assert { a.shape == [6] }
        assert { !a.inplace? }
        assert { a.row_major? }
        assert { !a.column_major? }
        assert { a.host_order? }
        assert { !a.byte_swapped? }
        assert { a == [1, 2, 3, 5, 7, 11] }
        assert { a.to_a == [1, 2, 3, 5, 7, 11] }
        assert { a.to_a.is_a?(Array) }
        assert { a.dup == a }
        assert { a.clone == a }
        assert { a.dup.object_id != a.object_id }
        assert { a.clone.object_id != a.object_id }

        assert { a.eq([1, 1, 3, 3, 7, 7]) == [1, 0, 1, 0, 1, 0] }
        assert { a[3..4] == [5, 7] }
        assert { a[5] == 11 }
        assert { a[5].size == 1 }
        assert { a[-1] == 11 }

        assert { a.at([3, 4]) == [5, 7] }
        assert { a.view.at([3, 4]) == [5, 7] }
        assert { a[2..-1].at([1, 2]) == [5, 7] }
        assert { a.at(Cumo::Int32.cast([3, 4])) == [5, 7] }
        assert { a.view.at(Cumo::Int32.cast([3, 4])) == [5, 7] }
        assert { a.at(3..4) == [5, 7] }
        assert { a.view.at(3..4) == [5, 7] }
        assert { a.at([5]) == [11] }
        assert { a.view.at([5]) == [11] }
        assert { a.at([-1]) == [11] }
        assert { a.view.at([-1]) == [11] }

        assert { a[(0..-1).each] == [1, 2, 3, 5, 7, 11] }
        assert { a[(0...-1).each] == [1, 2, 3, 5, 7] }

        if Enumerator.const_defined?(:ArithmeticSequence)
          assert { a[0.step(-1)] == [1, 2, 3, 5, 7, 11] }
          assert { a[0.step(4)] == [1, 2, 3, 5, 7] }
          assert { a[-5.step(-1)] == [2, 3, 5, 7, 11] }
          assert { a[0.step(-1, 2)] == [1, 3, 7] }
          assert { a[0.step(4, 2)] == [1, 3, 7] }
          assert { a[-5.step(-1, 2)] == [2, 5, 11] }

          assert { a[0.step] == [1, 2, 3, 5, 7, 11] }
          assert { a[-5.step] == [2, 3, 5, 7, 11] }
          assert { eval('a[(0..).step(2)]') == [1, 3, 7] }
          assert { eval('a[(0...).step(2)]') == [1, 3, 7] }
          assert { eval('a[(-5..).step(2)]') == [2, 5, 11] }
          assert { eval('a[(-5...).step(2)]') == [2, 5, 11] }
          assert { eval('a[(0..) % 2]') == [1, 3, 7] }
          assert { eval('a[(0...) % 2]') == [1, 3, 7] }
          assert { eval('a[(-5..) % 2]') == [2, 5, 11] }
          assert { eval('a[(-5...) % 2]') == [2, 5, 11] }
        end

        assert { a[(0..-1).step(2)] == [1, 3, 7] }
        assert { a[(0...-1).step(2)] == [1, 3, 7] }
        assert { a[(0..4).step(2)] == [1, 3, 7] }
        assert { a[(0...4).step(2)] == [1, 3] }
        assert { a[(-5..-1).step(2)] == [2, 5, 11] }
        assert { a[(-5...-1).step(2)] == [2, 5] }
        assert { a[(0..-1) % 2] == [1, 3, 7] }
        assert { a[(0...-1) % 2] == [1, 3, 7] }
        assert { a[(0..4) % 2] == [1, 3, 7] }
        assert { a[(0...4) % 2] == [1, 3] }
        assert { a[(-5..-1) % 2] == [2, 5, 11] }
        assert { a[(-5...-1) % 2] == [2, 5] }
        assert { a[[4, 3, 0, 1, 5, 2]] == [7, 5, 1, 2, 11, 3] }
        assert { a.reverse == [11, 7, 5, 3, 2, 1] }
        assert { a.sum == 29 }
        if float_types.include?(dtype)
          assert { a.mean == 29.0 / 6 }
          assert { a.var == 13.76666666666667 }
          assert { a.stddev == 3.710345895825168 }
          assert { a.rms == 5.901977069875258 }
        end
        assert { a.dup.fill(12) == [12] * 6 }
        assert { (a + 1) == [2, 3, 4, 6, 8, 12] }
        assert { (a - 1) == [0, 1, 2, 4, 6, 10] }
        assert { (a * 3) == [3, 6, 9, 15, 21, 33] }
        assert { (a / 0.5) == [2, 4, 6, 10, 14, 22] }
        assert { (-a) == [-1, -2, -3, -5, -7, -11] }
        assert { (a**2) == [1, 4, 9, 25, 49, 121] }
        assert { a.swap_byte.swap_byte == [1, 2, 3, 5, 7, 11] }

        assert { a.contiguous? }
        assert { a.transpose.contiguous? }

        if dtype == Cumo::DComplex || dtype == Cumo::SComplex
          assert { a.real == src }
          assert { a.imag == [0] * 6 }
          assert { a.conj == src }
          assert { a.angle == [0] * 6 }
        else
          assert { a.min == 1 }
          assert { a.max == 11 }
          assert { a.min_index == 0 }
          assert { a.max_index == 5 }
          assert { (a >= 3) == [0, 0, 1, 1, 1, 1] }
          assert { (a >  3) == [0, 0, 0, 1, 1, 1] }
          assert { (a <= 3) == [1, 1, 1, 0, 0, 0] }
          assert { (a <  3) == [1, 1, 0, 0, 0, 0] }
          assert { (a.eq 3) == [0, 0, 1, 0, 0, 0] }
          assert { a.sort == src }
          assert { a.sort_index == (0..5).to_a }
          assert { a.median == 4 }
          assert { dtype.maximum(a, 12 - a) == [11, 10, 9, 7, 7, 11] }
          assert { dtype.minimum(a, 12 - a) == [1, 2, 3, 5, 5, 1] }
          assert { dtype.maximum(a, 5) == [5, 5, 5, 5, 7, 11] }
          assert { dtype.minimum(a, 5) == [1, 2, 3, 5, 5, 5] }
        end
      end
    end

    test "#{dtype},[1..4]" do
      assert { dtype[1..4] == [1, 2, 3, 4] }
    end

    test "#{dtype},[-4..-1]" do
      assert { dtype[-4..-1] == [-4, -3, -2, -1] }
    end

    if Enumerator.const_defined?(:ArithmeticSequence)
      test "#{dtype},[1.step(4)]" do
        assert { dtype[1.step(4)] == [1, 2, 3, 4] }
      end

      test "#{dtype},[-4.step(-1)]" do
        assert { dtype[-4.step(-1)] == [-4, -3, -2, -1] }
      end

      test "#{dtype},[1.step(4, 2)]" do
        assert { dtype[1.step(4, 2)] == [1, 3] }
      end

      test "#{dtype},[-4.step(-1, 2)]" do
        assert { dtype[-4.step(-1, 2)] == [-4, -2] }
      end

      test "#{dtype},[(-4..-1).step(2)]" do
        assert { dtype[(-4..-1).step(2)] == [-4, -2] }
      end
    end

    test "#{dtype},[(1..4) % 2]" do
      assert { dtype[(1..4) % 2] == [1, 3] }
    end

    test "#{dtype},[(-4..-1) % 2]" do
      assert { dtype[(-4..-1) % 2] == [-4, -2] }
    end

    #test "#{dtype}.seq(5)" do
    #  assert { dtype.seq(5) == [0,1,2,3,4] }
    #end

    procs2 = [
      [proc { |tp, src| tp[*src] }, ""],
      [proc { |tp, src| tp[*src][true, true] }, "[true,true]"],
      [proc { |tp, src| tp[*src][0..-1, 0..-1] }, "[0..-1,0..-1]"]
    ]

    procs2.each do |init, ref|

      test "#{dtype},[[1,2,3],[5,7,11]]#{ref}" do
        src = [[1, 2, 3], [5, 7, 11]]
        a = init.call(dtype, src)

        assert { a.is_a?(dtype) }
        assert { a.size == 6 }
        assert { a.ndim == 2 }
        assert { a.shape == [2, 3] }
        assert { !a.inplace? }
        assert { a.row_major? }
        assert { !a.column_major? }
        assert { a.host_order? }
        assert { !a.byte_swapped? }
        assert { a == src }
        assert { a.to_a == src }
        assert { a.to_a.is_a?(Array) }

        assert { a.eq([[1, 1, 3], [3, 7, 7]]) == [[1, 0, 1], [0, 1, 0]] }
        assert { a[5] == 11 }
        assert { a[-1] == 11 }
        assert { a[1, 0] == src[1][0] }
        assert { a[1, 1] == src[1][1] }
        assert { a[1, 2] == src[1][2] }
        assert { a[3..4] == [5, 7] }
        assert { a[0, 1..2] == [2, 3] }

        assert { a.at([0, 1], [1, 2]) == [2, 11] }
        assert { a.view.at([0, 1], [1, 2]) == [2, 11] }
        assert { a.at([0, 1], (0..2) % 2) == [1, 11] }
        assert { a.view.at([0, 1], (0..2) % 2) == [1, 11] }
        assert { a.at((0..1) % 1, [0, 2]) == [1, 11] }
        assert { a.view.at((0..1) % 1, [0, 2]) == [1, 11] }
        assert { a.at(Cumo::Int32.cast([0, 1]), Cumo::Int32.cast([1, 2])) == [2, 11] }
        assert { a.view.at(Cumo::Int32.cast([0, 1]), Cumo::Int32.cast([1, 2])) == [2, 11] }
        assert { a[[0, 1], [0, 2]].at([0, 1], [0, 1]) == [1, 11] }
        assert { a[[0, 1], (0..2) % 2].at([0, 1], [0, 1]) == [1, 11] }
        assert { a[(0..1) % 1, [0, 2]].at([0, 1], [0, 1]) == [1, 11] }
        assert { a[(0..1) % 1, (0..2) % 2].at([0, 1], [0, 1]) == [1, 11] }

        assert { a[0, :*] == src[0] }
        assert { a[1, :*] == src[1] }
        assert { a[:*, 1] == [src[0][1], src[1][1]] }
        assert { a[true, [2, 0, 1]] == [[3, 1, 2], [11, 5, 7]] }
        assert { a.reshape(3, 2) == [[1, 2], [3, 5], [7, 11]] }
        assert { a.reshape(3, nil) == [[1, 2], [3, 5], [7, 11]] }
        assert { a.reshape(nil, 2) == [[1, 2], [3, 5], [7, 11]] }
        assert { a.transpose == [[1, 5], [2, 7], [3, 11]] }
        assert { a.transpose(1, 0) == [[1, 5], [2, 7], [3, 11]] }
        assert { a.triu == [[1, 2, 3], [0, 7, 11]] }
        assert { a.tril == [[1, 0, 0], [5, 7, 0]] }
        assert { a.reverse == [[11, 7, 5], [3, 2, 1]] }
        assert { a.reverse(0, 1) == [[11, 7, 5], [3, 2, 1]] }
        assert { a.reverse(1, 0) == [[11, 7, 5], [3, 2, 1]] }
        assert { a.reverse(0) == [[5, 7, 11], [1, 2, 3]] }
        assert { a.reverse(1) == [[3, 2, 1], [11, 7, 5]] }

        assert { a.sum == 29 }
        assert { a.sum(0) == [6, 9, 14] }
        assert { a.sum(1) == [6, 23] }
        assert { a.prod == 2310 }
        assert { a.prod(0) == [5, 14, 33] }
        assert { a.prod(1) == [6, 385] }
        if float_types.include?(dtype)
          assert { a.mean == 29.0 / 6 }
          assert { a.mean(0) == [3, 4.5, 7] }
          assert { a.mean(1) == [2, 23.0 / 3] }
        end

        assert { a.contiguous? }
        assert { a.reshape(3, 2).contiguous? }
        assert { a[true, 1..2].contiguous? == false }
        assert { a.transpose.contiguous? == false }
        assert { a.fortran_contiguous? == false }
        assert { a.transpose.fortran_contiguous? }
        assert { a.transpose.transpose.fortran_contiguous? == false }
        assert { a.reshape(3, 2).fortran_contiguous? == false }
        assert { a.reshape(3, 2).transpose.fortran_contiguous? }
        assert { a[true, 1..2].fortran_contiguous? == false }
        assert { a[true, 1..2].transpose.fortran_contiguous? == false }

        if dtype == Cumo::DComplex || dtype == Cumo::SComplex
          assert { a.real == src }
          assert { a.imag == [[0] * 3] * 2 }
          assert { a.conj == src }
          assert { a.angle == [[0] * 3] * 2 }
        else
          assert { a.min == 1 }
          assert { a.max == 11 }
          assert { a.min_index == 0 }
          assert { a.min_index(axis: 1) == [0, 3] }
          assert { a.min_index(axis: 0) == [0, 1, 2] }
          assert { a.max_index(axis: 1) == [2, 5] }
          assert { a.max_index(axis: 0) == [3, 4, 5] }
          assert { a.argmin == 0 }
          assert { a.argmax == 5 }
          assert { a.argmin(axis: 1) == [0, 0] }
          assert { a.argmin(axis: 0) == [0, 0, 0] }
          assert { a.argmax(axis: 1) == [2, 2] }
          assert { a.argmax(axis: 0) == [1, 1, 1] }
          assert { (a >= 3) == [[0, 0, 1], [1, 1, 1]] }
          assert { (a >  3) == [[0, 0, 0], [1, 1, 1]] }
          assert { (a <= 3) == [[1, 1, 1], [0, 0, 0]] }
          assert { (a <  3) == [[1, 1, 0], [0, 0, 0]] }
          assert { (a.eq 3) == [[0, 0, 1], [0, 0, 0]] }
          assert { a[a.ne 3] == [1, 2, 5, 7, 11] }
          assert { a[a[true, 2] < 5, true] == [[1, 2, 3]] }
          assert { a[true, a[1, true] > 5] == [[2, 3], [7, 11]] }
          assert { a[:*, (a[0, :*] % 2).eq(1)] == [[1, 3], [5, 11]] }
          assert { a.sort == src }
          assert { a.sort_index == [[0, 1, 2], [3, 4, 5]] }
          assert { a.percentile(0) == 1.0 }
          assert { a.percentile(50) == 4.0 }
          assert { a.percentile(90) == 9.0 }
          assert { a.percentile(100) == 11.0 }
          assert { a.percentile(0, axis: 0) == [1, 2, 3] }
          assert { a.percentile(50, axis: 0) == [3, 4.5, 7] }
          assert { a.percentile(90, axis: 0) == [4.6, 6.5, 10.2] }
          assert { a.percentile(100, axis: 0) == [5, 7, 11] }
          assert { a.percentile(0, axis: 1) == [1, 5] }
          assert { a.percentile(50, axis: 1) == [2, 7] }
          assert { a.percentile(90, axis: 1) == [2.8, 10.2] }
          assert { a.percentile(100, axis: 1) == [3, 11] }
        end
        assert { a.dup.fill(12) == [[12] * 3] * 2 }
        assert { (a + 1) == [[2, 3, 4], [6, 8, 12]] }
        assert { (a + [1, 2, 3]) == [[2, 4, 6], [6, 9, 14]] }
        assert { (a - 1) == [[0, 1, 2], [4, 6, 10]] }
        assert { (a - [1, 2, 3]) == [[0, 0, 0], [4, 5, 8]] }
        assert { (a * 3) == [[3, 6, 9], [15, 21, 33]] }
        assert { (a * [1, 2, 3]) == [[1, 4, 9], [5, 14, 33]] }
        assert { (a / 0.5) == [[2, 4, 6], [10, 14, 22]] }
        assert { (-a) == [[-1, -2, -3], [-5, -7, -11]] }
        assert { (a**2) == [[1, 4, 9], [25, 49, 121]] }
        assert { (dtype[[1, 0], [0, 1]].dot dtype[[4, 1], [2, 2]]) == [[4, 1], [2, 2]] }
        assert { a.swap_byte.swap_byte == src }
      end

      test "#{dtype},[[1,2,3],[5,7,11]]#{ref},aset[]=" do
        src = [[1, 2, 3], [5, 7, 11]]

        a = init.call(dtype, src)
        a[5] = 13
        assert { a[5] == 13 }

        a = init.call(dtype, src)
        a[-1] = 13
        assert { a[-1] == 13 }

        a = init.call(dtype, src)
        a[1, 0] = 13
        assert { a[1, 0] == 13 }

        a = init.call(dtype, src)
        a[1, 1] = 13
        assert { a[1, 1] == 13 }

        a = init.call(dtype, src)
        a[1, 2] = 13
        assert { a[1, 2] == 13 }

        a = init.call(dtype, src)
        a[3..4] = [13, 13]
        assert { a[3..4] == [13, 13] }

        a = init.call(dtype, src)
        a[0, 1..2] = [13, 13]
        assert { a[0, 1..2] == [13, 13] }

        a = init.call(dtype, src)
        a[0, :*] = [13, 13, 13]
        assert { a[0, :*] == [13, 13, 13] }

        a = init.call(dtype, src)
        a[1, :*] = [13, 13, 13]
        assert { a[1, :*] == [13, 13, 13] }

        a = init.call(dtype, src)
        a[:*, 1] = [13, 13]
        assert { a[:*, 1] == [13, 13] }

        a = init.call(dtype, src)
        a[5] = dtype.cast(13)
        assert { a[5] == 13 }
        assert { a[5] == dtype.cast(13) }

        a = init.call(dtype, src)
        a[1, 1] = dtype.cast(13)
        assert { a[1, 1] == 13 }
        assert { a[1, 1] == dtype.cast(13) }

        a = init.call(dtype, src)
        a[3..4] = dtype.cast([13, 13])
        assert { a[3..4] == [13, 13] }
        assert { a[3..4] == dtype.cast([13, 13]) }

        a = init.call(dtype, src)
        a[:*, 1] = dtype.cast([13, 13])
        assert { a[:*, 1] == [13, 13] }
        assert { a[:*, 1] == dtype.cast([13, 13]) }

        a = init.call(dtype, src)
        v = a[0, false]
        v[0] = 13
        assert { v == [13, 2, 3] }
        assert { a == [[13, 2, 3], [5, 7, 11]] }

        a = init.call(dtype, src)
        v = a[1, false]
        v[0] = 13
        assert { v == [13, 7, 11] }
        assert { a == [[1, 2, 3], [13, 7, 11]] }

        a = init.call(dtype, src)
        a[[1, 2, 3]] = 13
        assert { a[[1, 2, 3]] == [13, 13, 13] }
        assert { a == [[1, 13, 13], [13, 7, 11]] }

        a = init.call(dtype, src)
        a[1, [0, 2]] = [13, 13]
        assert { a[1, [0, 2]] == [13, 13] }
        assert { a == [[1, 2, 3], [13, 7, 13]] }

        a = init.call(dtype, src)
        a[1, true] = 13
        assert { a[1, true] == [13, 13, 13] }
        assert { a == [[1, 2, 3], [13, 13, 13]] }
      end

    end

    test "#{dtype},[[[1,2],[3,4]],[[5,6],[7,8]]]" do
      arr = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
      a = dtype[*arr]

      assert { a[0, 1, 1] == 4 }
      assert { a[:rest] == a }
      assert { a[0, :rest] == [[1, 2], [3, 4]] }
      assert { a[0, false] == [[1, 2], [3, 4]] }
      assert { a[0, 1, :rest] == [3, 4] }
      assert { a[0, 1, false] == [3, 4] }
      assert { a[:rest, 0] == [[1, 3], [5, 7]] }
      assert { a[:rest, 0, 1] == [2, 6] }
      assert { a[1, :rest, 0] == [5, 7] }
      assert { a[1, 1, :rest, 0] == 7 }
      assert_raise(IndexError) { a[1, 1, 1, 1, :rest] }
      assert_raise(IndexError) { a[1, 1, 1, :rest, 1] }
      assert_raise(IndexError) { a[:rest, 1, :rest, 0] }

      assert { a.transpose == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]] }
      assert { a.transpose(2, 1, 0) == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]] }
      assert { a.transpose(0, 2, 1) == [[[1, 3], [2, 4]], [[5, 7], [6, 8]]] }

      assert { a.contiguous? }
      assert { a.transpose.contiguous? == false }
      assert { a.fortran_contiguous? == false }
      assert { a.transpose.fortran_contiguous? }
      assert { a.transpose.transpose.fortran_contiguous? == false }
      assert { a.transpose(0, 2, 1).fortran_contiguous? == false }
      assert { a.reshape(2, 4).fortran_contiguous? == false }
      assert { a.reshape(2, 4).transpose.fortran_contiguous? }

      assert { a.at([0, 1], [1, 0], [0, 1]) == [3, 6] }
      assert { a.view.at([0, 1], [1, 0], [0, 1]) == [3, 6] }

      assert { a.transpose == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]] }
      assert { a.transpose(2, 1, 0) == [[[1, 5], [3, 7]], [[2, 6], [4, 8]]] }
      assert { a.transpose(0, 2, 1) == [[[1, 3], [2, 4]], [[5, 7], [6, 8]]] }

      assert { a.reverse == [[[8, 7], [6, 5]], [[4, 3], [2, 1]]] }
      assert { a.reverse(0, 1, 2)    == [[[8, 7], [6, 5]], [[4, 3], [2, 1]]] }
      assert { a.reverse(-3, -2, -1) == [[[8, 7], [6, 5]], [[4, 3], [2, 1]]] }
      assert { a.reverse(0..2)     == [[[8, 7], [6, 5]], [[4, 3], [2, 1]]] }
      assert { a.reverse(-3..-1)   == [[[8, 7], [6, 5]], [[4, 3], [2, 1]]] }
      assert { a.reverse(0...3)    == [[[8, 7], [6, 5]], [[4, 3], [2, 1]]] }
      assert { a.reverse(0)        == [[[5, 6], [7, 8]], [[1, 2], [3, 4]]] }
      assert { a.reverse(1)        == [[[3, 4], [1, 2]], [[7, 8], [5, 6]]] }
      assert { a.reverse(2)        == [[[2, 1], [4, 3]], [[6, 5], [8, 7]]] }
      assert { a.reverse(0, 1) == [[[7, 8], [5, 6]], [[3, 4], [1, 2]]] }
      assert { a.reverse(0..1)     == [[[7, 8], [5, 6]], [[3, 4], [1, 2]]] }
      assert { a.reverse(0...2)    == [[[7, 8], [5, 6]], [[3, 4], [1, 2]]] }
      assert { a.reverse(0, 2) == [[[6, 5], [8, 7]], [[2, 1], [4, 3]]] }
      assert { a.reverse((0..2) % 2) == [[[6, 5], [8, 7]], [[2, 1], [4, 3]]] }
      assert { a.reverse((0..2).step(2)) == [[[6, 5], [8, 7]], [[2, 1], [4, 3]]] }

      enum = arr.flatten.to_enum
      a.each do |e|
        assert { e == enum.next }
      end
      a.each_with_index do |e, *i|
        assert { e == a[*i] }
      end
    end

    unless [Cumo::DComplex, Cumo::SComplex].include?(dtype)
      sub_test_case "#{dtype}, #ptp" do
        test "ptp" do
          assert { dtype[3, 10, 1, 7].ptp == 9 }
          assert { dtype[5].ptp == 0 }
          a = dtype[1..6].reshape(2, 3)
          assert { a.ptp == 5 }
          assert { a.ptp(axis: 0) == [3, 3, 3] }
          assert { a.ptp(axis: 1) == [2, 2] }
          assert { a.ptp(axis: 0, keepdims: true) == [[3, 3, 3]] }
        end

        test "ptp over every axis" do
          [[5], [3, 1], [1, 3], [3, 2], [2, 3], [2, 3, 1], [2, 1, 3], [2, 3, 4]].each do |shape|
            a = dtype.cast(Array.new(shape.reduce(:*)) { |i| (i * 7 + 3) % 11 }).reshape(*shape)
            (-shape.size...shape.size).each do |axis|
              assert { a.ptp(axis: axis) == a.max(axis: axis) - a.min(axis: axis) }
            end
            assert { a.ptp == a.max - a.min }
          end
        end

        # More than one block, so the shared-memory tree reduction runs.
        test "ptp over a large reduction" do
          a = dtype.cast(Array.new(5000) { |i| (i * 37) % 97 })
          assert { a.ptp == a.max - a.min }
          b = a.reshape(50, 100)
          assert { b.ptp(axis: 0) == b.max(axis: 0) - b.min(axis: 0) }
          assert { b.ptp(axis: 1) == b.max(axis: 1) - b.min(axis: 1) }
        end

        test "ptp over views" do
          v = dtype.cast(Array.new(12) { |i| (i * 5 + 1) % 13 }).reshape(4, 3)
          [v[1..2, 0..1], v.transpose, v.reverse(0), v[(0..3).step(2), true]].each do |w|
            assert { w.ptp == w.max - w.min }
            assert { w.ptp(axis: 0) == w.max(axis: 0) - w.min(axis: 0) }
          end
        end

        if [Cumo::DFloat, Cumo::SFloat].include?(dtype)
          test "ptp ignores nan, and ptp(nan: true) propagates it" do
            nan = Float::NAN
            assert { dtype[3, nan, 1, 7].ptp == 6 }
            assert { dtype[nan, 3, 1, 7].ptp == 6 }
            assert { dtype[3, 1, 7, nan].ptp == 6 }
            assert { dtype[nan, nan].ptp.to_f.nan? }
            assert { dtype[3, nan, 1, 7].ptp(nan: true).to_f.nan? }
            assert { dtype[3, 10, 1, 7].ptp(nan: true) == 9 }
          end
        end
      end
    end

    if [Cumo::DFloat, Cumo::SFloat].include?(dtype)
      sub_test_case "#{dtype}, #max/#min corner cases" do
        nan = Float::NAN

        test "max and min ignore nan wherever it sits" do
          assert { dtype[nan, 3, 1, 7].max == 7 }
          assert { dtype[3, nan, 1, 7].max == 7 }
          assert { dtype[3, 1, 7, nan].max == 7 }
          assert { dtype[nan, 3, 1, 7].min == 1 }
          assert { dtype[3, nan, 1, 7].min == 1 }
          assert { dtype[3, 1, 7, nan].min == 1 }
          assert { dtype[-Float::INFINITY, nan].max == -Float::INFINITY }
          assert { dtype[Float::INFINITY, nan].min == Float::INFINITY }
        end

        test "max and min answer nan only when every element is nan" do
          assert { dtype[nan].max.to_f.nan? }
          assert { dtype[nan].min.to_f.nan? }
          assert { dtype[nan, nan].max.to_f.nan? }
          assert { dtype[nan, nan].min.to_f.nan? }

          a = dtype[[1, nan], [nan, nan]]
          assert { a.max(axis: 1).to_a[0] == 1 }
          assert { a.max(axis: 1).to_a[1].nan? }
          assert { a.min(axis: 1).to_a[0] == 1 }
          assert { a.min(axis: 1).to_a[1].nan? }
        end

        # More than one block, so the shared-memory tree reduction runs.
        test "max and min over a large reduction" do
          a = dtype.cast(Array.new(5000) { |i| (i * 37) % 97 })
          a[2500] = nan
          assert { a.max == 96 }
          assert { a.min == 0 }
          assert { dtype.new(5000).fill(nan).max.to_f.nan? }
          assert { dtype.new(5000).fill(nan).min.to_f.nan? }
        end

        test "max and min over views" do
          a = dtype[9, 3, 1, 7, nan]
          assert { a[1..4].max == 7 }
          assert { a[1..4].min == 1 }
          assert { dtype[9, nan, nan][1..2].max.to_f.nan? }
          assert { dtype[9, nan, nan][1..2].min.to_f.nan? }
        end

        test "max(nan: true) and min(nan: true) propagate nan" do
          assert { dtype[3, nan, 1, 7].max(nan: true).to_f.nan? }
          assert { dtype[3, nan, 1, 7].min(nan: true).to_f.nan? }
          assert { dtype[3, 10, 1, 7].max(nan: true) == 10 }
          assert { dtype[3, 10, 1, 7].min(nan: true) == 1 }
        end

        test "max and min keep the earlier of two equal elements" do
          assert { 1.0 / dtype[0.0, -0.0].min.to_f == Float::INFINITY }
          assert { 1.0 / dtype[0.0, -0.0].max.to_f == Float::INFINITY }
          assert { 1.0 / dtype[-0.0, 0.0].min.to_f == -Float::INFINITY }
          assert { 1.0 / dtype[-0.0, 0.0].max.to_f == -Float::INFINITY }
        end
      end
    end

    unless [Cumo::DComplex, Cumo::SComplex].include?(dtype)
      sub_test_case "#{dtype}, #clip" do
        test "clip" do
          a = dtype[0..9]
          assert { a.clip(1, 8) == [1, 1, 2, 3, 4, 5, 6, 7, 8, 8] }
          assert { a.clip(nil, 8) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 8] }
          assert { a.clip(3, nil) == [3, 3, 3, 3, 4, 5, 6, 7, 8, 9] }
        end

        test "clip with array bounds" do
          a = dtype[0..9]
          assert { a.clip(dtype[3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8) == [3, 4, 2, 3, 4, 5, 6, 7, 8, 8] }
          assert { a.clip(1, dtype[9, 9, 9, 5, 5, 5, 5, 5, 5, 5]) == [1, 1, 2, 3, 4, 5, 5, 5, 5, 5] }
          assert { a.clip(dtype.new(10).fill(2), dtype.new(10).fill(7)) == [2, 2, 2, 3, 4, 5, 6, 7, 7, 7] }
        end

        test "clip in place" do
          a = dtype[0..9]
          a.inplace.clip(3, 6)
          assert { a == [3, 3, 3, 3, 4, 5, 6, 6, 6, 6] }
        end

        test "clip over views" do
          a = dtype[0..11].reshape(3, 4)
          assert { a[1..2, 1..2].clip(6, 9) == [[6, 6], [9, 9]] }
          assert { a.transpose.clip(2, 8) == [[2, 4, 8], [2, 5, 8], [2, 6, 8], [3, 7, 8]] }
          assert { a.reverse(1).clip(2, 8) == [[3, 2, 2, 2], [7, 6, 5, 4], [8, 8, 8, 8]] }
        end

        # More than one block, so the whole launch is exercised.
        test "clip over a large array" do
          a = dtype.cast(Array.new(100_000) { |i| i % 100 })
          c = a.clip(20, 80)
          assert { c.min == 20 }
          assert { c.max == 80 }
          assert { c[0..24].to_a == ([20] * 21) + [21, 22, 23, 24] }
        end

        test "clip raises without a usable range" do
          a = dtype[0..9]
          assert_raise(Cumo::NArray::OperationError) { a.clip(6, 3) }
          assert_raise(ArgumentError) { a.clip(nil, nil) }
        end

        test "clip leaves the array untouched when it raises" do
          a = dtype[0..9]
          assert_raise(Cumo::NArray::OperationError) { a.inplace.clip(6, 3) }
          assert { a == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] }
        end
      end
    end

    sub_test_case "#{dtype}, #mulsum" do
      test "vector.mulsum(vector)" do
        a = dtype[1..3]
        b = dtype[2..4]
        assert { a.mulsum(b) == (1 * 2 + 2 * 3 + 3 * 4) }
      end

      if [Cumo::DComplex, Cumo::SComplex, Cumo::DFloat, Cumo::SFloat].include?(dtype)
        test "vector.mulsum(vector, nan: true)" do
          a = dtype[1..3]
          a[0] = 0.0 / 0 / 0
          b = dtype[2..4]
          assert { a.mulsum(b, nan: true) == (0 + 2 * 3 + 3 * 4) }
        end
      end

      # The elementwise reference squares before it sums, so keep the values
      # small enough that the 8-bit dtypes do not wrap.
      small = ->(shape) { dtype.cast(Array.new(shape.reduce(:*)) { |i| i % 4 + 1 }).reshape(*shape) }

      test "mulsum over every axis" do
        [[3], [3, 1], [1, 3], [3, 2], [2, 3], [3, 1, 1], [2, 3, 1], [2, 1, 3], [2, 3, 4]].each do |shape|
          a = small.call(shape)
          (-shape.size...shape.size).each do |axis|
            assert { a.mulsum(a, axis: axis) == (a * a).sum(axis: axis) }
          end
        end
        # mulsum accumulates in its own dtype while sum widens, so compare the
        # two only where the total still fits the narrowest dtype.
        a = small.call([3, 1])
        assert { a.mulsum(a) == (a * a).sum }
      end

      test "mulsum with a broadcast operand" do
        [[3, 1], [3, 2], [2, 3], [2, 3, 1], [2, 3, 4]].each do |shape|
          a = small.call(shape)
          b = dtype.cast([2]).reshape(*shape.map { 1 })
          (0...shape.size).each do |axis|
            assert { a.mulsum(b, axis: axis) == (a * b).sum(axis: axis) }
          end
        end
      end

      test "mulsum over a reversed view" do
        a = small.call([6])
        assert { a.reverse(0).mulsum(a.reverse(0), axis: 0) == (a * a).sum(axis: 0) }
        b = small.call([3, 2])
        assert { b.reverse(0).mulsum(b.reverse(0), axis: 0) == (b * b).sum(axis: 0) }
        c = a[(0..5).step(2)].reverse(0)
        assert { c.mulsum(c, axis: 0) == (c * c).sum(axis: 0) }
      end
    end

    sub_test_case "#{dtype}, #dot" do
      test "scalar.dot(scalar)" do
        a = dtype[1].sum
        b = dtype[3].sum
        assert { a.dot(b) == 1 * 3 }
      end
      test "vector.dot(vector) of 1-elem" do
        a = dtype[1]
        b = dtype[3]
        assert { a.dot(b) == 1 * 3 }
      end
      test "vector.dot(vector)" do
        a = dtype[1..3]
        b = dtype[2..4]
        assert { a.dot(b) == (1 * 2 + 2 * 3 + 3 * 4) }
      end
      test "matrix.dot(vector)" do
        a = dtype[1..6].reshape(3, 2)
        b = dtype[1..2]
        assert { a.dot(b) == [5, 11, 17] }
      end
      test "vector.dot(matrix)" do
        a = dtype[1..2]
        b = dtype[1..6].reshape(2, 3)
        assert { a.dot(b) == [9, 12, 15] }
      end
      test "vector.dot(matrix) of a single column" do
        a = dtype[1..3]
        b = dtype[4..6].reshape(3, 1)
        assert { a.dot(b) == [32] }
      end
      test "matrix.dot(matrix)" do
        a = dtype[1..6].reshape(3, 2)
        b = dtype[1..6].reshape(2, 3)
        assert { a.dot(b) == [[9, 12, 15], [19, 26, 33], [29, 40, 51]] }
        assert { b.dot(a) == [[22, 28], [49, 64]] }
      end
      test "matrix.dot(matrix.transpose)" do
        a = dtype[1..6].reshape(3, 2)
        b = dtype[1..6].reshape(3, 2).transpose
        assert { a.dot(b) == [[5, 11, 17], [11, 25, 39], [17, 39, 61]] }
        assert { b.dot(a) == [[35, 44], [44, 56]] }
      end
      test "matrix.dot(matrix) of contiguous view" do
        a = dtype.new(4, 3).seq(0)[1..2, 0..2] # 2x3
        b = dtype.new(3, 2).seq(0)
        assert { a.dot(b) == [[28, 40], [46, 67]] }
        assert { b.dot(a) == [[6, 7, 8], [24, 29, 34], [42, 51, 60]] }
      end
      test "matrix.dot(matrix) of non-contiguous view" do
        a = dtype.new(4, 4).seq(0)[1..2, 0..2] # 2x3
        b = dtype.new(3, 2).seq(0)
        assert { a.dot(b) == [[34, 49], [58, 85]] }
        assert { b.dot(a) == [[8, 9, 10], [32, 37, 42], [56, 65, 74]] }
      end
      test "matrix.dot(matrix) >= 3 dimensions" do
        a = dtype[1..6 * 2].reshape(2, 3, 2)
        b = dtype[1..6 * 2].reshape(2, 2, 3)
        assert { a.dot(b) ==
                 [[[9, 12, 15],
                   [19, 26, 33],
                   [29, 40, 51]],
                  [[129, 144, 159],
                   [163, 182, 201],
                   [197, 220, 243]]] }
        assert { b.dot(a) ==
                 [[[22, 28],
                   [49, 64]],
                  [[220, 244],
                   [301, 334]]] }
      end
      test "matrix.dot(matrix) >= 4 dimensions" do
        a = dtype[1..6 * 2].reshape(1, 2, 3, 2)
        b = dtype[1..6 * 2].reshape(1, 2, 2, 3)
        assert { a.dot(b) ==
                 [[[[9, 12, 15],
                    [19, 26, 33],
                    [29, 40, 51]],
                   [[129, 144, 159],
                    [163, 182, 201],
                    [197, 220, 243]]]] }
        assert { b.dot(a) ==
                 [[[[22, 28],
                    [49, 64]],
                   [[220, 244],
                    [301, 334]]]] }
      end
      test "matrix.dot(matrix.transpose) >= 3 dimensions" do
        a = dtype[1..6 * 2].reshape(2, 3, 2)
        b = dtype[1..6 * 2].reshape(3, 2, 2).transpose
        assert { a.dot(b) ==
                 [[[7, 19, 31],
                   [15, 43, 71],
                   [23, 67, 111]],
                  [[46, 106, 166],
                   [58, 134, 210],
                   [70, 162, 254]]] }
        assert { b.dot(a) ==
                  [[[61, 76],
                    [79, 100]],
                   [[178, 196],
                    [232, 256]]] }
      end
      test "matrix.dot(matrix) with incorrect shape" do
        a = dtype[1..6].reshape(3, 2)
        b = dtype[1..9].reshape(3, 3)
        assert_raise(Cumo::NArray::ShapeError) { a.dot(b) }
      end
    end

    if [Cumo::DComplex, Cumo::SComplex, Cumo::DFloat, Cumo::SFloat].include?(dtype)
      sub_test_case "#{dtype}, #gemm" do
        test "matrix.gemm(matrix) with alpha" do
          a = dtype[1..6].reshape(2, 3)
          b = dtype[1..6].reshape(2, 3)
          alpha = [Cumo::DComplex, Cumo::SComplex].include?(dtype) ? Complex(3) : 3
          assert { a.gemm(b.transpose) * alpha == a.gemm(b.transpose, alpha: alpha) }
        end
        test "matrix.gemm(matrix) of another dtype" do
          a = dtype[1..6].reshape(2, 3)
          real = [Cumo::SFloat, Cumo::DFloat].include?(dtype)
          [Cumo::Int32, Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex].each do |btype|
            b = btype[1..6].reshape(3, 2)
            if real && [Cumo::SComplex, Cumo::DComplex].include?(btype)
              assert_raise(Cumo::NArray::CastError) { a.gemm(b) }
              next
            end
            c = a.gemm(b)
            assert { c.instance_of?(dtype) }
            assert { c == [[22, 28], [49, 64]] }
          end
        end
        test "matrix.gemm(array)" do
          a = dtype[1..6].reshape(2, 3)
          assert { a.gemm([[1, 2], [3, 4], [5, 6]]) == [[22, 28], [49, 64]] }
        end
      end
    end

    test "#{dtype},eye" do
      assert { dtype.new(3, 3).eye(1) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }
      assert { dtype.new(3, 3).eye(2) == [[2, 0, 0], [0, 2, 0], [0, 0, 2]] }
      assert { dtype.new(3, 3).eye(1, 1) == [[0, 1, 0], [0, 0, 1], [0, 0, 0]] }
      assert { dtype.new(3, 3).eye(1, -1) == [[0, 0, 0], [1, 0, 0], [0, 1, 0]] }
      assert { dtype.new(2, 2, 2).eye(1) == [[[1, 0], [0, 1]], [[1, 0], [0, 1]]] }
      assert { dtype.new(3, 1).eye(1) == [[1], [0], [0]] }
      assert { dtype.new(1, 3).eye(1) == [[1, 0, 0]] }
      assert { dtype.eye(3) == [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }
      assert { dtype.eye(3, 1) == [[1], [0], [0]] }
      assert { dtype.eye(1, 3) == [[1, 0, 0]] }
    end

    test "#{dtype},element-wise" do
      x = dtype[[1, 2, 3], [5, 7, 11]]
      assert { x + x == [[2, 4, 6], [10, 14, 22]] }
      assert { x + 1 == [[2, 3, 4], [6, 8, 12]] }
      assert { x + dtype[1] == [[2, 3, 4], [6, 8, 12]] }
      assert { x + dtype[[1], [2]] == [[2, 3, 4], [7, 9, 13]] }
      assert { x + dtype[1, 2, 3] == [[2, 4, 6], [6, 9, 14]] }
      assert { x + dtype[[1, 2], [3, 4], [5, 6]].transpose == [[2, 5, 8], [7, 11, 17]] }
      assert { x[0, 1..2] + x[1, 0..1] == [7, 10] }
      unless [Cumo::DComplex, Cumo::SComplex].include?(dtype)
        y = x[x > 6] # [7,11]
        assert { y + y == [14, 22] }
        assert { y + 1 == [8, 12] }
        assert { y + dtype[1] == [8, 12] }
        assert { y + dtype[[1, 1], [2, 2]] == [[8, 12], [9, 13]] }
        assert { y.reshape(2, 1) + dtype[[1, 1], [2, 2]] == [[8, 8], [13, 13]] }
      end
    end

    test "#{dtype},reduction" do
      assert { dtype.ones(2, 2, 3, 2).sum(axis: [0, 2, 3]) == [12, 12] }
      assert { dtype.ones(5, 3, 4, 2, 1).sum(axis: [0, 3, 4]) == [[10, 10, 10, 10], [10, 10, 10, 10], [10, 10, 10, 10]] }
      assert { dtype[[1, 2, 3], [4, 5, 6]].sum(axis: 1) == [6, 15] }
      assert { dtype[[1, 2, 3], [4, 5, 6]].sum(axis: 1, keepdims: true) == [[6], [15]] }

      unless [Cumo::DComplex, Cumo::SComplex].include?(dtype)
        assert_nothing_raised { dtype.ones(2, 3, 9, 4, 2).max_index(2) }

        a = dtype[[[6, 8, 5],
                   [2, 5, 6],
                   [4, 5, 5]],
                  [[7, 4, 3],
                   [9, 1, 0],
                   [4, 1, 6]]]
        # the last row of the first plane is [4, 5, 5], and a tie takes the
        # earlier of the two, as numo does
        assert { a.max_index(2) == [[1, 5, 7], [9, 12, 17]] }
        assert { a.max(2) == [[8, 6, 5], [7, 9, 6]] }
        assert { a.argmax(2) == [[1, 2, 1], [0, 0, 2]] }
        assert { a.argmin(2) == [[2, 0, 0], [2, 2, 1]] }

        unless [Cumo::UInt64, Cumo::UInt32, Cumo::UInt16, Cumo::UInt8].include?(dtype)
          a = dtype[[[-6, -8, -5],
                     [-2, 5, 6],
                     [4, -5, 5]],
                    [[-7, -4, -3],
                     [9, 1, 0],
                     [4, -1, -6]]]
          assert { a.max_index(2) == [[2, 5, 8], [11, 12, 15]] }
          assert { a.max(2) == [[-5, 6, 5], [-3, 9, 4]] }
        end

        if [Cumo::DFloat, Cumo::SFloat].include?(dtype)
          assert { dtype[[-Float::INFINITY, 0, 1, Float::INFINITY]].max_index(0) == [0, 1, 2, 3] }
        end
      end
    end

    test "#{dtype},advanced indexing" do
      a = dtype[[1, 2, 3], [4, 5, 6]]
      assert { a[[0, 1], [0, 1]].dup == [[1, 2], [4, 5]] }
      assert { a[[0, 1], [0, 1]].sum == 12 }
      assert { a[[0, 1], [0, 1]].diagonal == [1, 5] }
      diag = a.dup[[0, 1], [0, 1]].diagonal
      diag.inplace - 1
      assert { diag == [0, 4] }

      assert { a.at([0, 1], [0, 1]).dup == [1, 5] }
      at = a.dup
      at.at([0, 1], [0, 1]).inplace - 1
      assert { at == [[0, 2, 3], [4, 4, 6]] }
    end

    sub_test_case "#{dtype}.from_binary" do
      test "frozen string" do
        shape = [2, 5]
        a = dtype.new(*shape)
        a.rand(0, 10)
        original_data = a.to_binary
        data = original_data.dup.freeze
        restored_a = dtype.from_binary(data, shape)
        assert { restored_a == a }
        restored_a[0, 0] += 1
        assert { restored_a != a }
        assert { data == original_data }
      end

      test "not frozen string" do
        shape = [2, 5]
        a = dtype.new(*shape)
        a.rand(0, 10)
        original_data = a.to_binary
        data = original_data.dup
        restored_a = dtype.from_binary(data, shape)
        assert { restored_a == a }
        restored_a[0, 0] += 1
        assert { restored_a != a }
        assert { data == original_data }
      end
    end

    sub_test_case "#{dtype}#store_binary" do
      test "frozen string" do
        shape = [2, 5]
        a = dtype.new(*shape)
        a.rand(0, 10)
        original_data = a.to_binary
        data = original_data.dup.freeze
        restored_a = dtype.new(*shape)
        restored_a.store_binary(data)
        assert { restored_a == a }
        restored_a[0, 0] += 1
        assert { restored_a != a }
        assert { data == original_data }
      end

      test "frozen string into an allocated array" do
        shape = [2, 5]
        a = dtype.new(*shape)
        a.rand(0, 10)
        data = a.to_binary.freeze
        restored_a = dtype.new(*shape).seq
        restored_a.store_binary(data)
        assert { restored_a == a }
        assert { !restored_a.free }
      end

      test "not frozen string" do
        shape = [2, 5]
        a = dtype.new(*shape)
        a.rand(0, 10)
        original_data = a.to_binary
        data = original_data.dup
        restored_a = dtype.new(*shape)
        restored_a.store_binary(data)
        assert { restored_a == a }
        restored_a[0, 0] += 1
        assert { restored_a != a }
        assert { data == original_data }
      end

      test "string with offset" do
        a = dtype.new(2)
        a.rand(0, 10)
        b = dtype.new(1)
        b.store_binary(a.to_binary, dtype::ELEMENT_BYTE_SIZE)
        assert { b == a[1..1] }
      end

      test "non-string argument" do
        a = dtype.new(2)
        # RSTRING_PTR() on a non-String reads whatever the object's layout
        # happens to hold there, so these used to segfault or store garbage.
        [nil, 123, 1.5, :sym, {}, [0x7fffffff, 0x1234].freeze, /abc/].each do |bad|
          assert_raise(TypeError) { a.store_binary(bad) }
        end
      end
    end
  end

  test "cast any object that responds to to_a" do
    object = Struct.new(:to_a).new([1, 2, 3])
    assert { Cumo::NArray.cast(object) == [1, 2, 3] }
  end

  sub_test_case "#dot between types" do
    dot_types = [
      Cumo::Int32,
      Cumo::Int64,
      Cumo::UInt8,
      Cumo::SFloat,
      Cumo::DFloat,
      Cumo::SComplex,
      Cumo::DComplex,
    ]

    dot_types.each do |atype|
      dot_types.each do |btype|
        test "#{atype}.dot(#{btype})" do
          upcast = atype::UPCAST[btype] || btype::UPCAST[atype]
          a = atype[1..6].reshape(2, 3)
          b = btype[1..6].reshape(3, 2)
          assert { a.dot(b).instance_of?(upcast) }
          assert { a.dot(b) == [[22, 28], [49, 64]] }
          assert { a.dot(btype[1..3]) == [14, 32] }
          assert { atype[1..3].dot(b) == [22, 28] }
          assert { atype[1..3].dot(btype[1..3]) == 14 }
        end
      end
    end
  end

  sub_test_case "reshape argument validation" do
    test "a negative size raises" do
      # A negative size used to be cast to size_t, so reshape(-2, -3) produced
      # a shape of [2**64-2, 2**64-3] whose product wraps back to 6 and passes
      # the "total size must be same" check.
      assert_raise(ArgumentError) { Cumo::DFloat.new(6).seq.reshape(-2, -3) }
      assert_raise(ArgumentError) { Cumo::DFloat.new(6).seq.reshape(2, -3) }
      assert_raise(ArgumentError) { Cumo::DFloat.new(6).seq.reshape(-6) }
      assert_raise(ArgumentError) { Cumo::DFloat.new(1).seq.reshape(-1, -1) }
      assert_raise(ArgumentError) { Cumo::DFloat.new(6).seq.reshape!(-2, -3) }
    end

    test "a zero total with an unfixed dimension raises" do
      # The unfixed dimension is solved by dividing by the total, so a zero
      # total used to reach a division by zero and abort with SIGFPE.
      assert_raise(ArgumentError) { Cumo::Int32[1, 2, 3].reshape(0, nil) }
      assert_raise(ArgumentError) { Cumo::DFloat.new(6).seq.reshape(0, true) }
    end

    test "an overflowing total raises" do
      # 2**30 * 2**30 * 16 is 2**64, which wraps to a total of zero.
      assert_raise(ArgumentError) { Cumo::DFloat.new(6).seq.reshape(2**30, 2**30, 16, nil) }
    end

    test "a valid reshape is unaffected" do
      a = Cumo::DFloat.new(6).seq
      assert { a.reshape(2, 3).shape == [2, 3] }
      assert { a.reshape(2, nil).shape == [2, 3] }
      assert { a.reshape(nil, 3).shape == [2, 3] }
      assert { a.reshape(6).to_a == [0, 1, 2, 3, 4, 5] }
      assert_raise(ArgumentError) { a.reshape(4, nil) }
      assert_raise(ArgumentError) { a.reshape(nil, nil) }
      assert_raise(ArgumentError) { a.reshape(0, 6) }
    end
  end

  test "Cumo::DFloat.cast(Cumo::RObject[1, nil, 3])" do
    assert_equal(Cumo::DFloat[1, Float::NAN, 3].format_to_a,
                 Cumo::DFloat.cast(Cumo::RObject[1, nil, 3]).format_to_a)
  end

  test "Cumo::RObject#free" do
    a = Cumo::RObject.new(3)
    a.seq
    assert { a.free }
    assert { !a.free }
  end

  test "GC does not raise with Cumo::RObject" do
    5.times { Cumo::RObject.new(1000).seq.sum }
    assert_nothing_raised { GC.start }
  end

  test "Cumo::RObject#format with a long element" do
    ['A' * 47, 'A' * 48, 'A' * 4096].each do |src|
      a = Cumo::RObject[src]

      formatted = a.format
      assert_kind_of(Cumo::RObject, formatted)
      assert_equal([src], formatted.to_a)

      formatted = a.format_to_a
      assert_kind_of(Array, formatted)
      assert_equal([src], formatted)
    end
  end

  test "Cumo::RObject with a scalar operand" do
    a = Cumo::RObject.new(3)
    a.store(1)
    assert_equal([1, 1, 1], a.to_a)
    assert_equal([1, 2, 3], (Cumo::RObject.new(3).seq + 1).to_a)
  end

  test "Cumo::RObject view" do
    a = Cumo::RObject[0, 1, 2, 3, 4, 5]

    assert_equal([0, 1, 2], a[0..2].copy.to_a)
    assert_equal(3, a[0..2].sum)
    assert_equal(6, a[[0, 2, 4]].sum)
    assert_equal([0, 4, 8], (a[[0, 2, 4]] + a[[0, 2, 4]]).to_a)

    a[[0, 2, 4]] = Cumo::RObject[10, 20, 30]
    assert_equal([10, 1, 20, 3, 30, 5], a.to_a)
  end

  test "Cumo::RObject 2-d view" do
    a = Cumo::RObject.new(3, 4).seq
    b = a[true, [3, 1, 0, 2]]

    assert_equal([[3, 1, 0, 2], [7, 5, 4, 6], [11, 9, 8, 10]], b.to_a)
    assert_equal([[6, 2, 0, 4], [14, 10, 8, 12], [22, 18, 16, 20]], (b + b).to_a)
    assert_equal([21, 15, 12, 18], b.sum(axis: 0).to_a)
  end

  test "single element array" do
    assert { Cumo::SFloat[1].mean == 1.0 }
    assert { Cumo::DFloat[1].mean == 1.0 }
    assert { Cumo::SComplex[1].mean == 1.0 }
    assert { Cumo::DComplex[1].mean == 1.0 }

    assert { Cumo::SFloat[1].var.to_f.nan? }
    assert { Cumo::DFloat[1].var.to_f.nan? }
    assert { Cumo::SComplex[1].var.to_f.nan? }
    assert { Cumo::DComplex[1].var.to_f.nan? }

    assert { Cumo::SFloat[1].stddev.to_f.nan? }
    assert { Cumo::DFloat[1].stddev.to_f.nan? }
    assert { Cumo::SComplex[1].stddev.to_f.nan? }
    assert { Cumo::DComplex[1].stddev.to_f.nan? }

    assert { Cumo::SFloat[1].rms == 1.0 }
    assert { Cumo::DFloat[1].rms == 1.0 }
    assert { Cumo::SComplex[1].rms == 1.0 }
    assert { Cumo::DComplex[1].rms == 1.0 }
  end

  test "concatenate with empty arrays" do
    a = Cumo::DFloat[1, 2, 3]
    empty = Cumo::DFloat[]
    assert { Cumo::NArray.concatenate([empty, a]) == [1, 2, 3] }
    assert { Cumo::NArray.concatenate([a, empty]) == [1, 2, 3] }
    assert { Cumo::NArray.concatenate([empty, empty]) == [] }
    assert { a.concatenate(empty) == [1, 2, 3] }
    assert { empty.concatenate(a) == [1, 2, 3] }
    assert { empty.concatenate(empty) == [] }
  end

  test "empty array leaves no CUDA error behind" do
    {
      Cumo::DFloat => [1.0, 2.0, 3.0, 4.0],
      Cumo::Int32 => [1, 2, 3, 4],
      Cumo::SComplex => [Complex(1), Complex(2), Complex(3), Complex(4)],
    }.each do |dtype, expected|
      empty = dtype.new(0)
      assert_equal([], empty.seq.to_a)
      assert_equal([], (empty + 1).to_a)
      assert_equal([], empty.copy.to_a)
      assert_equal([], (empty * empty).to_a)
      assert_equal([], dtype.new(4).seq[[]].to_a)

      # an error left behind by the launches above surfaces on the next one
      assert_equal(expected, (dtype.new(4).seq + 1).to_a)
    end
  end

  test "parse" do
    assert { Cumo::DFloat.parse("2 -3 5\n4 9 7\n2 -1 6") == Cumo::DFloat[[2, -3, 5], [4, 9, 7], [2, -1, 6]] }
    assert { Cumo::DFloat.parse("1 2; 3 4") == Cumo::DFloat[[1, 2], [3, 4]] }
  end

  test "parse rejects non-literal tokens" do
    ['system("id")', '`id`', 'Kernel.exit', '$stdout', 'self', 'x', 'TRUE', 'True', '[1, 2]'].each do |src|
      assert_raise(ArgumentError) { Cumo::DFloat.parse(src) }
    end
  end

  test "parse rejects expressions" do
    ['1+1', '2*3', '3/4', '3r', '1/3r'].each do |src|
      assert_raise(ArgumentError) { Cumo::DFloat.parse(src) }
    end
  end

  test "parse error names the token" do
    error = assert_raise(ArgumentError) { Cumo::DFloat.parse('1 2 oops 4') }
    assert { error.message.include?('oops') }
  end

  test "parse integer literals" do
    assert { Cumo::Int64.parse('1 -3 +4 1_000') == Cumo::Int64[[1, -3, 4, 1000]] }
    assert { Cumo::Int64.parse('0x1f -0x1f 0b101 0o17 017') == Cumo::Int64[[31, -31, 5, 15, 15]] }
  end

  test "parse float literals" do
    assert { Cumo::DFloat.parse('1.5 -2.5 1e5 1.2e-3') == Cumo::DFloat[[1.5, -2.5, 100_000.0, 0.0012]] }
  end

  test "parse complex literals" do
    expected = Cumo::DComplex[[Complex(0, 2), Complex(2, 3), Complex(0, -2), Complex(0, 1.5)]]
    assert { Cumo::DComplex.parse('2i 2+3i -2i 1.5i') == expected }
  end

  test "parse boolean literals" do
    assert { Cumo::NArray.parse('true false nil') == Cumo::Bit[[1, 0, 0]] }
    assert { Cumo::NArray.parse("true false\nfalse true") == Cumo::Bit[[1, 0], [0, 1]] }
  end

  test "argmax/argmin" do
    [Cumo::DFloat, Cumo::Int32, Cumo::UInt8].each do |dtype|
      a = dtype[3, 4, 1, 2]
      assert { a.argmax == 1 }
      assert { a.argmin == 2 }

      b = dtype[[3, 4, 1], [2, 0, 5]]
      assert { b.argmax == 5 }
      assert { b.argmin == 4 }
      assert { b.argmax(axis: 1) == [1, 2] }
      assert { b.argmax(axis: 0) == [0, 0, 1] }
      assert { b.argmin(axis: 1) == [2, 1] }
      assert { b.argmin(axis: 0) == [1, 1, 0] }
      assert { b.at(b.argmax(axis: 0), 0..-1) == [3, 4, 5] }
      assert { b.at(b.argmin(axis: 0), 0..-1) == [2, 0, 1] }
    end

    # NaN-aware (nan:true returns the NaN position; nan:false ignores NaN)
    c = Cumo::DFloat[3.0, Float::NAN, 1.0, 5.0]
    assert { c.argmax == 3 }
    assert { c.argmax(nan: true) == 1 }
    assert { c.argmin == 2 }
    assert { c.argmin(nan: true) == 1 }
  end

  test "divmod returns quotient and remainder" do
    [Cumo::Int32, Cumo::Int64, Cumo::RObject].each do |dtype|
      a = dtype[17, 20, 23]
      b = dtype[5, 6, 7]
      q, r = a.divmod(b)
      assert { q == [3, 3, 3] }
      assert { r == [2, 2, 2] }
    end

    # floored division semantics with negatives (Cumo::RObject was returning
    # the quotient as the remainder before the fix)
    q, r = Cumo::RObject[-7, 7, 17].divmod(Cumo::RObject[3, 3, 5])
    assert { q == [-3, 2, 3] }
    assert { r == [2, 1, 2] }
  end

  test "indexing a 0-dimensional array" do
    # Indexing a 0-d (scalar) array reaches na_get_strides_nadata with ndim == 0.
    # Before the fix this wrote strides[-1] out of bounds; the result must stay correct.
    [Cumo::DFloat, Cumo::Int32].each do |dtype|
      a = dtype.cast(3)
      assert { a.ndim == 0 }
      assert { a[nil] == [3] }
      assert { a[nil].shape == [1] }
      assert { a[true] == [3] }
    end
  end

  test "adding a new axis to a narray-view" do
    # cumo_na_index_aref_naview read na1->stridx[q[i].orig_dim] before checking
    # orig_dim against na1->base.ndim. A trailing :new keeps orig_dim == ndim,
    # so the read went one element past the end of the view's stridx array.
    a = Cumo::DFloat.new(4).seq
    v = a[1..2]
    assert { v.shape == [2] }

    assert { v[true, :new].shape == [2, 1] }
    assert { v[true, :new] == [[1], [2]] }
    assert { v[:new, true].shape == [1, 2] }
    assert { v[:new, true] == [[1, 2]] }

    b = Cumo::DFloat.new(3, 4).seq
    u = b[true, 1..2]
    assert { u.shape == [3, 2] }

    assert { u[true, true, :new].shape == [3, 2, 1] }
    assert { u[true, true, :new] == [[[1], [2]], [[5], [6]], [[9], [10]]] }
    assert { u[true, :new, true].shape == [3, 1, 2] }
    assert { u[true, :new, true] == [[[1, 2]], [[5, 6]], [[9, 10]]] }
  end

  test "at() rejects :new" do
    # :new leaves q[i].orig_dim == ndim, which made cumo_na_index_at_naview read
    # one element past the end of the view's stridx array. That union member was
    # then used as a device index pointer, so a[..].at([0],[0],:new) died with
    # "an illegal memory access was encountered". at() must reject :new instead.
    a = Cumo::DFloat.new(3, 3).seq
    v = a[true, 0..1]

    # sanity: at() itself still works on both data and view arrays
    assert { a.at([0, 1], [0, 1]) == [0, 4] }
    assert { v.at([0, 1], [0, 1]) == [0, 4] }

    [a, v].each do |x|
      assert_raise(IndexError) { x.at([0], [0], :new) }
      assert_raise(IndexError) { x.at([0], :new, [0]) }
      assert_raise(IndexError) { x.at(:new, [0], [0]) }
    end
  end

  test "map yields once per element" do
    # m_num_to_data used to evaluate its argument twice, so m_map's
    # rb_yield() ran twice for every element.
    [Cumo::DFloat, Cumo::SFloat].each do |dtype|
      a = dtype[1, 2, 3]
      count = 0
      result = a.map { |x| count += 1; x * 2 }
      assert { count == 3 }
      assert { result == [2, 4, 6] }
    end

    # nil from the block still becomes NaN
    assert { Cumo::DFloat[1, 2].map { nil }.to_a.all? { |x| x.nan? } }
  end

  test "Cumo::RObject#rand stays within range" do
    # m_rand multiplied by the raw VALUE instead of NUM2DBL(max),
    # so the generated values fell outside the requested interval.
    Cumo::NArray.srand(0)
    a = Cumo::RObject.new(1000).rand.to_a
    assert { a.min >= 0 && a.max < 1 }

    Cumo::NArray.srand(0)
    b = Cumo::RObject.new(1000).rand(10).to_a
    assert { b.min >= 0 && b.max < 10 }

    Cumo::NArray.srand(0)
    c = Cumo::RObject.new(1000).rand(-2, 3).to_a
    assert { c.min >= -2 && c.max < 3 }
  end

  sub_test_case "#rand / #rand_norm" do
    real_types = [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex, Cumo::RObject]
    signed_types = [Cumo::Int8, Cumo::Int16, Cumo::Int32, Cumo::Int64]
    unsigned_types = [Cumo::UInt8, Cumo::UInt16, Cumo::UInt32, Cumo::UInt64]
    rand_types = real_types.map { |t| [t, []] } +
                 (signed_types + unsigned_types).map { |t| [t, [2, 60]] }

    rand_types.each do |dtype, args|
      test "#{dtype}#rand repeats under the same seed" do
        Cumo::NArray.srand(7)
        a = dtype.new(64).rand(*args).to_a
        Cumo::NArray.srand(7)
        assert_equal(a, dtype.new(64).rand(*args).to_a)
      end

      test "#{dtype}#rand moves on between calls" do
        Cumo::NArray.srand(7)
        a = dtype.new(64).rand(*args).to_a
        b = dtype.new(64).rand(*args).to_a
        assert_not_equal(a, b)
      end

      # Every element draws from its own subsequence, so where the call
      # boundaries fall must not change what any of them gets.
      test "#{dtype}#rand does not depend on how the calls are split" do
        Cumo::NArray.srand(7)
        whole = dtype.new(128).rand(*args).to_a
        Cumo::NArray.srand(7)
        halves = dtype.new(64).rand(*args).to_a + dtype.new(64).rand(*args).to_a
        assert_equal(whole, halves)
      end

      test "#{dtype}#rand stays within range" do
        Cumo::NArray.srand(7)
        low, high = args.empty? ? [0.0, 1.0] : args
        drawn = dtype.new(256).rand(*args).to_a
        values = drawn.flat_map { |x| x.is_a?(Complex) ? [x.real, x.imag] : [x] }
        assert { values.all? { |x| x >= low && x < high } }
      end
    end

    test "rand over a view" do
      a = Cumo::DFloat.new(8, 8).seq
      a[1..6, 1..6].rand
      assert { a[1..6, 1..6].to_a.flatten.all? { |x| x >= 0.0 && x < 1.0 } }

      b = Cumo::DFloat.new(8, 8).seq.transpose
      b.rand
      assert { b.to_a.flatten.all? { |x| x >= 0.0 && x < 1.0 } }
    end

    test "rand is uniform" do
      a = Cumo::DFloat.new(1 << 20).rand
      assert { (a.mean.extract_cpu - 0.5).abs < 0.005 }
      counts = Array.new(10, 0)
      Cumo::Int32.new(1 << 20).rand(0, 10).to_a.each { |x| counts[x] += 1 }
      assert { counts.all? { |c| (c - (1 << 20) / 10).abs < (1 << 20) / 100 } }
    end

    [Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex].each do |dtype|
      test "#{dtype}#rand_norm repeats under the same seed" do
        Cumo::NArray.srand(3)
        a = dtype.new(64).rand_norm.to_a
        Cumo::NArray.srand(3)
        assert_equal(a, dtype.new(64).rand_norm.to_a)
      end

      test "#{dtype}#rand_norm does not depend on how the calls are split" do
        Cumo::NArray.srand(3)
        whole = dtype.new(128).rand_norm.to_a
        Cumo::NArray.srand(3)
        halves = dtype.new(64).rand_norm.to_a + dtype.new(64).rand_norm.to_a
        assert_equal(whole, halves)
      end
    end

    test "rand_norm follows mu and sigma" do
      a = Cumo::DFloat.new(1 << 20).rand_norm
      assert { a.mean.extract_cpu.abs < 0.01 }
      assert { (a.stddev.extract_cpu - 1.0).abs < 0.01 }

      b = Cumo::DFloat.new(1 << 20).rand_norm(10, 0.1)
      assert { (b.mean.extract_cpu - 10).abs < 0.01 }
      assert { (b.stddev.extract_cpu - 0.1).abs < 0.005 }

      c = Cumo::DComplex.new(1 << 19).rand_norm(Complex(1, 2), 0.5)
      assert { (c.real.mean.extract_cpu - 1).abs < 0.01 }
      assert { (c.imag.mean.extract_cpu - 2).abs < 0.01 }
      assert { (c.real.stddev.extract_cpu - 0.5).abs < 0.01 }
    end

    test "rand_norm over a view" do
      a = Cumo::DFloat.new(8, 8).seq
      a[1..6, 1..6].rand_norm
      assert { a[1..6, 1..6].to_a.flatten.size == 36 }

      b = Cumo::DFloat.new(8, 8).seq.transpose
      b.rand_norm
      assert { b.to_a.flatten.size == 64 }
    end
  end

  test "store fills every slot after a Range" do
    assert_equal([9.0, 0.0, 1.0, 5.0], Cumo::DFloat.cast([9, 0...2, 5]).to_a)
    assert_equal([0, 1, 2, 5], Cumo::Int64.cast([(0...3), 5]).to_a)
    assert_equal([9, 0.0, 1.0, 5], Cumo::RObject.cast([9, 0...2, 5]).to_a)
  end

  test "store re-reads a source array the elements rewrite" do
    grow = Class.new do
      def initialize(a) = @a = a
      # the append reallocates the element storage the loop was reading
      def to_f
        @a.concat(Array.new(4096, 7.7))
        1.0
      end
      def to_int = 1
    end

    [Cumo::DFloat, Cumo::Int32, Cumo::RObject].each do |dtype|
      src = Array.new(8) { |i| i + 100 }
      src[0] = grow.new(src)
      assert_nothing_raised { dtype.cast(src) }

      src = Array.new(8) { |i| i + 100 }
      src[0] = grow.new(src)
      assert_nothing_raised { dtype.new(8).store(src) }
    end
  end

  # Shrinking the source keeps the buffer, so this reads past the end rather
  # than through a freed pointer: wrong values, no crash.
  test "store stops where a shrunk source array ends" do
    shrink = Class.new do
      def initialize(a) = @a = a
      def to_int
        4.times { @a.shift }
        1
      end
      def to_f = to_int.to_f
    end

    {
      Cumo::Int64 => [1, 105, 106, 107, 0, 0, 0, 0],
      Cumo::DFloat => [1.0, 105.0, 106.0, 107.0, 0.0, 0.0, 0.0, 0.0],
    }.each do |dtype, expected|
      src = Array.new(8) { |i| i + 100 }
      src[0] = shrink.new(src)
      assert_equal(expected, dtype.cast(src).to_a)

      src = Array.new(8) { |i| i + 100 }
      src[0] = shrink.new(src)
      assert_equal(expected, dtype.new(8).store(src).to_a)
    end
  end

  test "zero-sized multi-dimensional arrays" do
    # The free functions used to skip releasing base.shape when size == 0,
    # leaking the shape array allocated for ndim >= 2.
    [Cumo::DFloat, Cumo::Int32, Cumo::Bit, Cumo::RObject].each do |dtype|
      a = dtype.new(0, 3)
      assert { a.size == 0 }
      assert { a.ndim == 2 }
      assert { a.shape == [0, 3] }
      assert { a.to_a.flatten == [] }
    end

    # exercise the free path explicitly
    assert_nothing_raised do
      1000.times { Cumo::DFloat.new(0, 3) }
      GC.start
    end
  end

  test "repeated reshape! keeps contents" do
    # na_alloc_shape did not free the previous shape before allocating a new
    # one, leaking it on every reshape of an array that already had a shape.
    a = Cumo::DFloat.new(2, 3, 4).seq
    src = a.to_a.flatten

    100.times do
      a.reshape!(4, 3, 2)
      a.reshape!(24)
      a.reshape!(2, 3, 4)
    end

    assert { a.shape == [2, 3, 4] }
    assert { a.to_a.flatten == src }
  end

  test "sort and sort_index over many values" do
    # The partition loop used `for (;;)` with the pointer updates at the end
    # of the body, which some compilers could optimize incorrectly.
    rng = Random.new(42)

    arr = Array.new(100) { rng.rand(-1.0...1.0) }
    [Cumo::DFloat, Cumo::SFloat].each do |dtype|
      a = dtype.cast(arr)
      assert { a.sort.to_a == a.to_a.sort }
      assert { a[a.sort_index].to_a == a.to_a.sort }
    end

    arr = Array.new(100) { rng.rand(-100...100) }
    [Cumo::Int64, Cumo::Int32, Cumo::Int16, Cumo::Int8].each do |dtype|
      a = dtype.cast(arr)
      assert { a.sort.to_a == a.to_a.sort }
      assert { a[a.sort_index].to_a == a.to_a.sort }
    end

    arr = Array.new(100) { rng.rand(0...100) }
    [Cumo::UInt64, Cumo::UInt32, Cumo::UInt16, Cumo::UInt8].each do |dtype|
      a = dtype.cast(arr)
      assert { a.sort.to_a == a.to_a.sort }
      assert { a[a.sort_index].to_a == a.to_a.sort }
    end
  end

  test "flatten on empty arrays" do
    # na_flatten_dim returned the array unchanged whenever size == 0, so a
    # multi-dimensional empty array kept its dimensions instead of becoming 1-D.
    [[0, 3], [2, 0], [0, 0], [0, 2, 3], [2, 0, 3]].each do |shape|
      f = Cumo::DFloat.new(*shape).flatten
      assert { f.ndim == 1 }
      assert { f.shape == [0] }
    end

    # non-empty arrays and views keep working
    a = Cumo::DFloat.new(2, 3).seq
    assert { a.flatten.to_a == [0, 1, 2, 3, 4, 5] }
    assert { a.transpose.flatten.to_a == [0, 3, 1, 4, 2, 5] }
    assert { Cumo::DFloat.cast(5).flatten.shape == [] }
  end

  test "store with a sub-narray longer than the destination" do
    # The zero-fill passed n-i as an unsigned kernel length. When a nested
    # NArray was longer than the destination, i exceeded n and n-i wrapped
    # around, so the kernel wrote far out of bounds (CUDA error 700).
    # Values below match Numo::NArray.
    [Cumo::DFloat, Cumo::Int32].each do |dtype|
      a = dtype.zeros(2, 3)
      a.store([dtype.new(5).seq, dtype.new(5).seq])
      assert { a.to_a == [[0, 1, 2], [0, 1, 2]] }
    end

    # an oversized nested plain Array truncates the same way
    d = Cumo::DFloat.zeros(2, 3)
    d.store([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    assert { d.to_a == [[1, 2, 3], [6, 7, 8]] }
  end

  test "store with a sub-narray shorter than the destination" do
    # The inner copy used the destination length as its element count, so it
    # read past the end of a shorter sub-narray. The stale values only became
    # visible once the memory behind the source had been reused, which needed
    # a previous full-length store plus a GC.
    a = Cumo::DFloat.zeros(2, 3)
    a.store([Cumo::DFloat.new(3).seq, Cumo::DFloat.new(3).seq])
    GC.start

    b = Cumo::DFloat.zeros(2, 3)
    b.store([Cumo::DFloat.new(2).seq, Cumo::DFloat.new(2).seq])
    assert { b.to_a == [[0, 1, 0], [0, 1, 0]] }

    # sanity: the full-length store is still correct
    assert { a.to_a == [[0, 1, 2], [0, 1, 2]] }
  end

  test "store into a strided view is not clobbered by a later allocation" do
    # Storing a Ruby Array into a strided destination stages it in a device
    # buffer and launches a kernel to scatter it. The buffer was released right
    # after the launch, so the memory pool could hand it out again while the
    # kernel was still reading it; from_binary then memcpy'd over it from the
    # host, which does not wait for the stream.
    n = 1_000_000
    src = Array.new(n, 7.0)
    poison = [-1.0].pack("d") * n

    5.times do
      a = Cumo::DFloat.zeros(n, 2)
      a[true, 0] = src
      Cumo::DFloat.from_binary(poison)
      assert { a[true, 0].eq(7.0).count_true == n }
    end
  end

  # cumo_na_parse_narray_index() copied an NArray index straight to the device
  # with no bounds check, so an out-of-range index read out of bounds and a
  # negative one -- which is an ordinary way to index -- became a huge size_t
  # and segfaulted. Every case below is written twice, once for [] and once for
  # at(): they share cumo_na_parse_narray_index(), so one fix closes both, and
  # without both a later at_mode branch could reopen one of them unnoticed.
  #
  # Passing here is not the same as being in bounds: the memory pool rounds
  # allocations up, so a small out-of-range read used to land inside the same
  # chunk and quietly return a value.
  test "an out-of-range NArray index raises" do
    a = Cumo::DFloat.new(4).seq
    v = Cumo::DFloat.new(8).seq[0..3]

    [a, v].each do |x|
      [Cumo::Int32, Cumo::Int64].each do |itype|
        assert_raise(IndexError) { x[itype[4]] }
        assert_raise(IndexError) { x[itype[9999]] }
        assert_raise(IndexError) { x[itype[0, 4]] }
        assert_raise(IndexError) { x.at(itype[4]) }
        assert_raise(IndexError) { x.at(itype[9999]) }
      end
      assert_raise(IndexError) { x[Cumo::Int64[1 << 40]] }
      assert_raise(IndexError) { x.at(Cumo::Int64[1 << 40]) }
    end
  end

  test "a negative NArray index counts from the end" do
    a = Cumo::DFloat.new(4).seq
    v = Cumo::DFloat.new(8).seq[0..3]

    [a, v].each do |x|
      assert { x[Cumo::Int32[-1]] == [3] }
      assert { x[Cumo::Int32[-4]] == [0] }
      assert { x[Cumo::Int32[-1, -2]] == [3, 2] }
      assert { x.at(Cumo::Int32[-1]) == [3] }
      # same index as a Ruby Array, which was always checked
      assert { x[Cumo::Int32[-1]] == x[[-1]] }
      assert { x.at(Cumo::Int32[-1]) == x.at([-1]) }
      # one past the end going backwards
      assert_raise(IndexError) { x[Cumo::Int32[-5]] }
      assert_raise(IndexError) { x.at(Cumo::Int32[-5]) }
    end
  end

  test "an NArray index is checked in every dimension" do
    a = Cumo::DFloat.new(3, 4).seq

    assert { a[Cumo::Int32[0, 2], Cumo::Int32[1, 3]] == [[1, 3], [9, 11]] }
    assert { a[Cumo::Int32[-1], Cumo::Int32[-1]] == [[11]] }
    assert_raise(IndexError) { a[Cumo::Int32[3], Cumo::Int32[0]] }
    assert_raise(IndexError) { a[Cumo::Int32[0], Cumo::Int32[4]] }
    assert_raise(IndexError) { a[Cumo::Int32[0], Cumo::Int32[-5]] }
    assert { a.at(Cumo::Int32[0, 2], Cumo::Int32[1, 3]) == [1, 11] }
    assert_raise(IndexError) { a.at(Cumo::Int32[0, 3], Cumo::Int32[0, 0]) }
  end

  test "indexing by a Bit array is unaffected by the range check" do
    # A Bit index goes through where(), whose result is in range by
    # construction, so the check must not reject it.
    a = Cumo::DFloat.new(6).seq

    assert { a[a > 2.0] == [3, 4, 5] }
    assert { a[a.eq(0.0)] == [0] }
    assert { a[a > 99.0].size == 0 }
    assert { a[Cumo::Bit[1, 0, 1, 0, 1, 0]] == [0, 2, 4] }
  end

  test "a non-Integer step raises" do
    # cumo_na_parse_range() took the step through NUM2SSIZET(), which truncates
    # a Float, and then divided by it: a step under 1 truncated to 0 and killed
    # the process with SIGFPE, which Ruby cannot rescue, while a step above 1
    # was silently truncated and walked with a step the caller never asked for.
    a = Cumo::DFloat.new(10).seq

    assert_raise(ArgumentError) { a[(0..9).step(0.5)] }
    assert_raise(ArgumentError) { a[(0..9).step(1.5)] }
    assert_raise(ArgumentError) { a[(0..9) % 0.5] }
    assert_raise(ArgumentError) { a.at((0..9).step(0.5)) }

    assert { a[(0..9).step(2)] == [0, 2, 4, 6, 8] }
    assert { a[(0..9) % 3] == [0, 3, 6, 9] }
    assert { a[9.step(0, -2)] == [9, 7, 5, 3, 1] }
  end

  test "a new axis on a data array does not read past the strides" do
    # cumo_na_index_aref_nadata() indexes an ALLOCA_N buffer of exactly
    # base.ndim entries with q[i].orig_dim, and a trailing :new leaves
    # orig_dim == base.ndim (cumo_na_index_parse_args() advances j but not k
    # for :new). The view path got this guard in #160; the data path, which is
    # where a plain NArray is routed, did not. The read was 8 bytes past the
    # end and nothing observable depended on it -- the value became the stride
    # of a size-1 dimension whose only index is zero -- so the results below
    # were already correct. They pin the behaviour the guard has to preserve.
    a = Cumo::DFloat.new(4).seq
    assert { a[true, :new].shape == [4, 1] }
    assert { a[true, :new].to_a == [[0], [1], [2], [3]] }
    assert { a[:new, true].shape == [1, 4] }
    assert { a[:new, true].to_a == [[0, 1, 2, 3]] }

    b = Cumo::DFloat.new(2, 3).seq
    assert { b[true, true, :new].shape == [2, 3, 1] }
    assert { b[true, true, :new].to_a == [[[0], [1], [2]], [[3], [4], [5]]] }

    v = Cumo::DFloat.new(6).seq[0..3]
    assert { v[true, :new].to_a == [[0], [1], [2], [3]] }
  end

  sub_test_case "max_index/min_index with nan: true" do
    nan = Float::NAN

    test "one dimension" do
      a = Cumo::DFloat[3, 4, nan, 2]
      assert_equal([2], a.max_index(nan: true).to_a)
      assert_equal([2], a.min_index(nan: true).to_a)

      b = Cumo::DFloat[3, 4, 1, 2]
      assert_equal([1], b.max_index(nan: true).to_a)
      assert_equal([2], b.min_index(nan: true).to_a)
    end

    test "the index is into the whole array, not into the axis" do
      b = Cumo::DFloat[[3, 4, 1], [nan, 2, 5]]
      assert_equal([3], b.max_index(nan: true).to_a)
      assert_equal([3], b.min_index(nan: true).to_a)
      assert_equal([3, 1, 5], b.max_index(axis: 0, nan: true).to_a)
      assert_equal([1, 3], b.max_index(axis: 1, nan: true).to_a)
      assert_equal([3, 4, 2], b.min_index(axis: 0, nan: true).to_a)
      assert_equal([2, 3], b.min_index(axis: 1, nan: true).to_a)
    end

    test "SFloat and a view" do
      assert_equal([2], Cumo::SFloat[3, 4, nan, 2].max_index(nan: true).to_a)
      assert_equal([2], Cumo::DFloat[3, 4, nan, 2, 9][0...4].max_index(nan: true).to_a)
    end

    test "the kernel path is unaffected" do
      a = Cumo::DFloat[[3, 4, 1], [9, 2, 5]]
      assert_equal([3], a.max_index.to_a)
      assert_equal([1, 3], a.max_index(axis: 1).to_a)
      assert_equal([2], a.min_index.to_a)
    end
  end

  sub_test_case "poly" do
    test "evaluates the polynomial" do
      x = Cumo::DFloat.new(4).seq(1)
      assert_equal([3.0, 5.0, 7.0, 9.0], x.poly(1.0, 2.0).to_a)
      assert_equal([6.0, 17.0, 34.0, 57.0], x.poly(1.0, 2.0, 3.0).to_a)
      assert_equal([3.0, 3.0, 3.0, 3.0], x.poly(3.0).to_a)
      assert_equal([1.0, 2.0, 3.0, 4.0], x.poly.to_a)
      assert_equal([5.0], Cumo::DFloat[2.0].poly(1.0, 2.0).to_a)
    end

    test "narray coefficients" do
      x = Cumo::DFloat.new(4).seq(1)
      a0 = Cumo::DFloat[1, 1, 1, 1]
      a1 = Cumo::DFloat[2, 2, 2, 2]
      assert_equal([3.0, 5.0, 7.0, 9.0], x.poly(a0, a1).to_a)
    end

    test "every dtype" do
      assert_equal([3.0, 5.0, 7.0, 9.0], Cumo::SFloat.new(4).seq(1).poly(1.0, 2.0).to_a)
      assert_equal([3, 5, 7, 9], Cumo::Int32.new(4).seq(1).poly(1, 2).to_a)
      assert_equal([3, 5, 7, 9], Cumo::RObject.new(4).seq(1).poly(1, 2).to_a)
    end
  end

  sub_test_case "integer #sum / #prod widen to 64 bits" do
    # The shared-memory tree merges partials that have already outgrown the
    # element type, so anything narrower than 64 bits has to survive the merge.
    sum_widths = {
      Cumo::Int8 => 5,
      Cumo::Int16 => 12,
      Cumo::Int32 => 30,
      Cumo::Int64 => 50,
      Cumo::UInt8 => 5,
      Cumo::UInt16 => 12,
      Cumo::UInt32 => 30,
      Cumo::UInt64 => 50
    }

    sum_widths.each do |dtype, bits|
      test "#{dtype}#sum over partials wider than the element" do
        [4, 8, 64, 1024].each do |n|
          assert_equal(n * (2**bits), dtype.new(n).fill(2**bits).sum)
        end
      end

      test "#{dtype}#sum over a long sequence" do
        a = dtype.new(100_000).seq
        assert_equal(a.to_a.sum, a.sum)
      end

      test "#{dtype}#sum over an axis" do
        a = dtype.new(4, 256).seq
        rows = a.to_a
        assert_equal(rows.map(&:sum), a.sum(axis: 1).to_a)
        assert_equal(rows.transpose.map(&:sum), a.sum(axis: 0).to_a)
      end
    end

    # A merged partial only outgrows a 32 bit element once the whole product has
    # outgrown the 64 bit accumulator, so prod can only show this on 8 and 16 bit.
    prod_values = {
      Cumo::Int8 => 8,
      Cumo::UInt8 => 8,
      Cumo::Int16 => 32,
      Cumo::UInt16 => 32
    }

    prod_values.each do |dtype, value|
      test "#{dtype}#prod over partials wider than the element" do
        assert_equal(value**8, dtype.new(8).fill(value).prod)
      end
    end

    sum_widths.each_key do |dtype|
      test "#{dtype}#prod over small values" do
        a = dtype.new(6).seq(1)
        assert_equal(a.to_a.reduce(:*), a.prod)
      end
    end
  end

  sub_test_case "bincount" do
    int_types = [
      Cumo::Int8, Cumo::Int16, Cumo::Int32, Cumo::Int64,
      Cumo::UInt8, Cumo::UInt16, Cumo::UInt32, Cumo::UInt64,
    ]

    test "counts every value" do
      int_types.each do |dtype|
        assert_equal([1, 3, 1, 1, 0, 0, 0, 1], dtype[0, 1, 1, 3, 2, 1, 7].bincount.to_a)
      end
    end

    test "the result is as long as the largest value plus one" do
      x = Cumo::Int32[0, 1, 1, 3, 2, 1, 7, 23]
      assert_equal(x.max.to_a.first + 1, x.bincount.size)
    end

    test "minlength extends but does not truncate" do
      assert_equal([1, 2, 0, 0, 0, 0], Cumo::Int32[0, 1, 1].bincount(minlength: 6).to_a)
      assert_equal([1, 1, 0, 0, 0, 1], Cumo::Int32[0, 1, 5].bincount(minlength: 2).to_a)
    end

    test "weights" do
      x = Cumo::Int32[0, 1, 1, 2, 2, 2]
      w = Cumo::DFloat[0.3, 0.5, 0.2, 0.7, 1.0, -0.6]
      [0.3, 0.7, 1.1].zip(x.bincount(w).to_a) { |want, got| assert_in_delta(want, got, 1e-12) }
      assert_equal([1.0, 2.0, 3.0], x.bincount(Cumo::SFloat[1, 1, 1, 1, 1, 1]).to_a)
      assert_raise(Cumo::NArray::ShapeError) { Cumo::Int32[0, 1].bincount(Cumo::DFloat[1, 2, 3]) }
    end

    test "a negative element raises" do
      assert_raise(ArgumentError) { Cumo::Int32[0, -1, 2].bincount }
    end

    test "views and multi-dimensional input" do
      assert_equal([1, 2, 0, 1], Cumo::Int32[0, 1, 1, 3, 9][0...4].bincount.to_a)
      assert_equal([[1, 1, 0, 0], [0, 1, 0, 1]], Cumo::Int32[[0, 1], [1, 3]].bincount.to_a)
    end

    test "an item too large to size the output raises" do
      assert_raise(RangeError) { Cumo::UInt64[(2**64) - 1].bincount }
      assert_raise(RangeError) { Cumo::UInt64[(2**64) - 1, (2**62) + 1000].bincount(minlength: 8) }
    end

    test "the arguments are converted before the array is scanned" do
      setter = Class.new do
        def initialize(array, value)
          @array = array
          @value = value
        end

        def to_int
          @array[0] = @value
          3
        end

        def to_f
          @array[0] = @value
          1.0
        end
      end

      a = Cumo::Int32[0, 1, 2]
      r = a.bincount(minlength: setter.new(a, 100_000))
      assert_equal(100_001, r.size)
      assert_equal([1, 1, 1, 3], [r[100_000], r[1], r[2]].map { |x| x.to_a.first } << r.to_a.sum)

      a = Cumo::Int32[0, 1, 2]
      r = a.bincount([setter.new(a, 100_000), 2.0, 3.0])
      assert_equal(100_001, r.size)
      assert_equal([1.0, 2.0, 3.0], [r[100_000], r[1], r[2]].map { |x| x.to_a.first })

      a = Cumo::Int32[0, 1, 2]
      assert_raise(ArgumentError) { a.bincount(minlength: setter.new(a, -1)) }
    end

    test "an array large enough that the filling kernel is still running" do
      r = (Cumo::Int32.new(1 << 20).seq % 997).bincount
      assert_equal(997, r.size)
      assert_equal(1 << 20, r.to_a.sum)
    end
  end

  # These all run a host loop over managed memory, so they read whatever the
  # last kernel has written so far unless they synchronize first. The arrays
  # are large enough that the kernel filling them is still in flight.
  # A host memcpy into managed memory does not wait for the stream, and the pool
  # hands back chunks a still-running kernel may be writing.
  sub_test_case "host writes synchronize before touching device memory" do
    n = 1 << 18
    rounds = 30

    test "from_binary over a recycled chunk" do
      zeros = ("\0" * (n * 8)).freeze
      rounds.times do
        a = Cumo::DFloat.new(n).seq(1)
        a.free
        b = Cumo::DFloat.from_binary(zeros)
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
        assert_equal(0, b.ne(0.0).count_true.to_i)
      end
    end

    test "store_binary over memory a kernel is still writing" do
      rounds.times do
        a = Cumo::DFloat.new(n).seq(1)
        a.store_binary(+("\0" * (n * 8)))
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
        assert_equal(0, a.ne(0.0).count_true.to_i)
      end
    end

    # A frozen string is pointed at rather than copied, so nothing is written.
    test "store_binary of a frozen string" do
      a = Cumo::DFloat.new(n).seq(1)
      a.store_binary(("\0" * (n * 8)).freeze)
      assert_equal(0, a.ne(0.0).count_true.to_i)
    end
  end

  sub_test_case "host loops synchronize before reading device memory" do
    n = 1 << 20
    last = n.to_f
    total = n * (n + 1) / 2.0

    test "minmax" do
      assert_equal([1.0, last], Cumo::DFloat.new(n).seq(1).minmax)
      assert_equal([1, n], Cumo::Int32.new(n).seq(1).minmax)
      assert_equal([1, 3], Cumo::Int32[1, 2, 3].minmax)
    end

    test "nan-aware reductions" do
      a = Cumo::DFloat.new(n).seq(1)
      assert_equal([total], a.sum(nan: true).to_a)
      assert_equal([last], a.max(nan: true).to_a)
      assert_equal([1.0], a.min(nan: true).to_a)
      assert_equal([(1.0 + last) / 2], a.mean(nan: true).to_a)
    end

    test "kahan_sum" do
      assert_equal([total], Cumo::DFloat.new(n).seq(1).kahan_sum.to_a)
    end

    test "median" do
      assert_equal([(last + 1) / 2], Cumo::DFloat.new(n).seq(1).median.to_a)
    end

    test "modf" do
      frac, int = (Cumo::DFloat.new(n).seq(1) / 4).modf
      assert_equal([0.25, 0.5], frac.to_a.first(2))
      assert_equal([0.0, 0.0], int.to_a.first(2))
    end

    test "frexp" do
      frac, exp = Cumo::NMath.frexp(Cumo::DFloat.new(n).seq(1))
      assert_equal([0.5, 0.5], frac.to_a.first(2))
      assert_equal([1, 2], exp.to_a.first(2))
    end

    test "set_real and set_imag" do
      a = Cumo::DComplex.new(n).seq(1)
      a.imag = Cumo::DFloat.new(n).seq(100)
      assert_equal([Complex(1, 100), Complex(2, 101)], a.to_a.first(2))

      b = Cumo::DComplex.new(n).seq(1)
      b.real = Cumo::DFloat.new(n).seq(100)
      assert_equal([Complex(100, 0), Complex(101, 0)], b.to_a.first(2))
    end
  end

  test "a shape whose element count overflows raises" do
    assert_raise(RangeError) { Cumo::Int8.new((2**62) + 1, 4) }
    assert_raise(RangeError) { Cumo::Bit.new((2**62) + 1, 4) }
    assert_raise(RangeError) { Cumo::DFloat.new(2**32, 2**32) }
    assert_raise(RangeError) { Cumo::DFloat.zeros(2**40, 2**40, 2**40) }

    assert_equal(0, Cumo::DFloat.new(0, 2**62).size)
    assert_equal([3, 4], Cumo::DFloat.zeros(3, 4).shape)
  end

  test "a one-dimensional size whose byte count overflows raises" do
    # A one-dimensional shape is not a product, so it reaches the allocator
    # untouched by the check above.
    assert_raise(RangeError) { Cumo::DFloat.new((2**61) + 4).allocate }
    assert_raise(RangeError) { Cumo::DComplex.new((2**60) + 1).allocate }
    assert_raise(RangeError) { Cumo::RObject.new((2**61) + 4).allocate }

    assert_equal(4, Cumo::DFloat.new(4).allocate.size)
  end

  test "from_binary rejects a size whose byte count overflows" do
    assert_raise(ArgumentError) { Cumo::DFloat.from_binary(+"12345678", 2**61) }
    assert_raise(ArgumentError) { Cumo::DFloat.from_binary("12345678", [2**61]) }
    assert_raise(ArgumentError) { Cumo::DFloat.from_binary("12345678", [2**60, 4]) }
    assert_raise(ArgumentError) { Cumo::Int16.from_binary("12345678", [(2**63) + 1]) }
  end

  test "store_binary rejects a size whose byte count overflows" do
    # A frozen String takes cumo_na_set_pointer, which never allocates, so the
    # allocator check does not cover this path.
    assert_raise(ArgumentError) { Cumo::DFloat.new(2**61).store_binary(("A" * 4096).freeze) }
    assert_raise(ArgumentError) { Cumo::DComplex.new(2**60).store_binary(("A" * 4096).freeze) }
  end

  test "marshal_load rejects a non-array shape" do
    [42, "AAAAAAAAAAAAAAAA", nil, {}, 1.5, :abc, true, Object.new].each do |bad|
      assert_raise(ArgumentError) { Cumo::DFloat.allocate.marshal_load([1, bad, 0, +""]) }
      assert_raise(ArgumentError) { Cumo::RObject.allocate.marshal_load([1, bad, 0, []]) }
    end
  end

  test "marshal_load rejects non-string content" do
    [[1, 2], {}, Object.new, 12_345, nil, :abc].each do |bad|
      assert_raise(TypeError) { Cumo::DFloat.allocate.marshal_load([1, [2], 0, bad]) }
    end
  end

  test "marshal_load rejects a malformed stream" do
    # A user-class stream whose payload Marshal.load hands straight to marshal_load.
    body = Marshal.dump([1, 42, 0, ""]).byteslice(2..)
    name = "Cumo::DFloat"
    stream = "\x04\x08U:#{(name.bytesize + 5).chr}#{name}#{body}".b
    assert_raise(ArgumentError) { Marshal.load(stream) }

    a = Cumo::DFloat[1.5, 2.5]
    assert { Marshal.load(Marshal.dump(a)) == a }

    b = Cumo::RObject[1, :x, nil]
    assert { Marshal.load(Marshal.dump(b)) == b }
  end

  test "indexing by an array does not leak host memory" do
    # The index list was staged in pinned host memory released from a CUDA
    # stream callback, but a callback may not call the CUDA API: cudaFreeHost
    # returned cudaErrorNotPermitted and the buffer was never freed, leaking
    # 8 bytes of page-locked memory per element on every indexing.
    omit "needs /proc/self/status" unless File.readable?("/proc/self/status")
    rss = -> { File.read("/proc/self/status")[/VmRSS:\s+(\d+)/, 1].to_i }

    a = Cumo::DFloat.new(1000).seq
    idx = (0...500).to_a
    assert { a[idx].to_a == (0...500).map(&:to_f) }
    GC.start

    before_rss = rss.call
    before_pool = Cumo::CUDA::MemoryPool.total_bytes
    20_000.times { a[idx] }
    GC.start
    # Device memory the pool holds on to also counts towards RSS, and how much
    # it keeps depends on what ran before, so discount it: the pinned host
    # buffers were never in the pool.
    grew = (rss.call - before_rss) - (Cumo::CUDA::MemoryPool.total_bytes - before_pool) / 1024
    # the leak alone was 20_000 * 500 * 8 bytes = 78 MB
    assert { grew < 40_000 }
  end

  test "a store which raises does not leak its staging buffer" do
    # Storing a Ruby Array stages the values in a host buffer as wide as the
    # destination. m_num_to_data raises for a value which is not a number, and
    # the buffer used to be a plain malloc ndloop knew nothing about, so every
    # failed store leaked all of it.
    omit "needs /proc/self/status" unless File.readable?("/proc/self/status")
    # Address space rather than RSS: the buffer is not zero filled, so a store
    # which fails on the second element only ever touches its first page.
    vm_size = -> { File.read("/proc/self/status")[/VmSize:\s+(\d+)/, 1].to_i }

    n = 1 << 19 # 4 MiB of DFloat
    a = Cumo::DFloat.new(n)
    a.store(0)
    bad = [1, "x"]
    assert_raise(TypeError) { a.store(bad) }
    GC.start

    before = vm_size.call
    100.times do
      a.store(bad)
    rescue TypeError
      nil
    end
    GC.start
    # the leak alone was 100 * 4 MiB = 400 MiB
    assert { vm_size.call - before < 100 * 1024 }
  end

  test "abs over a view" do
    # A complex abs writes half as wide as it reads, so the contiguous kernel
    # is only reachable when the output stride matches the real type.
    z = Cumo::DComplex[[3 + 4i, -8 - 6i, 0 + 0i], [0 - 4i, -5 + 12i, 1 + 0i]]
    assert_equal(Cumo::DFloat[[5, 10, 0], [4, 13, 1]], z.abs)
    assert_equal(Cumo::DFloat[5, 4], z[true, 0].abs)
    assert_equal(Cumo::DFloat[[0, 5, 10], [1, 4, 13]], z[true, [2, 0, 1]].abs)

    a = Cumo::DFloat[[3.5, -2.1, 0.0], [-0.7, -0.9, 1.25]]
    assert_equal(Cumo::DFloat[3.5, 0.7], a[true, 0].abs)
    assert_equal(Cumo::DFloat[[0.0, 3.5, 2.1], [1.25, 0.7, 0.9]], a[true, [2, 0, 1]].abs)

    i = Cumo::Int32[[3, -2, 0], [-7, -9, 4]]
    assert_equal(Cumo::Int32[3, 7], i[true, 0].abs)
    assert_equal(Cumo::Int32[[0, 3, 2], [4, 7, 9]], i[true, [2, 0, 1]].abs)
  end

  test "abs of the minimum integer raises" do
    {
      Cumo::Int8 => -128,
      Cumo::Int16 => -32_768,
      Cumo::Int32 => -2_147_483_648,
      Cumo::Int64 => -9_223_372_036_854_775_808
    }.each do |dtype, min|
      assert_raise(Cumo::NArray::ValueError) { dtype[min, 1].abs }
      # the flag the kernel reports through is shared, so it has to be clear
      # again for the next call
      assert_equal(dtype[1, 2], dtype[-1, 2].abs)
    end
  end

  def cond_unary_expect(meth, val)
    case meth
    when :isnan then val.nan?
    when :isinf then val.infinite?
    when :isposinf then val.infinite? == 1
    when :isneginf then val.infinite? == -1
    when :isfinite then val.finite?
    end
  end

  test "cond_unary over long arrays and views" do
    inf = Float::INFINITY
    n = 200
    specials = { 0 => Float::NAN, 1 => inf, 2 => -inf }
    values = Array.new(n) { |i| specials.fetch(i % 7, i - 100.0) }
    a = Cumo::DFloat.cast(values)
    stepped = (0...n).step(3).to_a
    picked = [5, 0, 199, 64, 63, 32, 31, 100]
    # a strided 2-d view runs the loop once per row, so the result starts at a
    # bit offset which is not a multiple of the bit-digit width
    cols = (0...25).step(2).to_a
    grid = (0...8).flat_map { |r| cols.map { |c| (r * 25) + c } }

    [:isnan, :isinf, :isposinf, :isneginf, :isfinite].each do |meth|
      bits = ->(idxs) { Cumo::Bit.cast(idxs.map { |i| cond_unary_expect(meth, values[i]) ? 1 : 0 }) }
      assert_equal(bits.call((0...n).to_a), a.public_send(meth), meth)
      assert_equal(bits.call(stepped), a[(0...n).step(3)].public_send(meth), meth)
      assert_equal(bits.call(picked), a[picked].public_send(meth), meth)
      actual = a.reshape(8, 25)[true, (0...25).step(2)].public_send(meth)
      assert_equal(bits.call(grid).reshape(8, cols.size), actual, meth)
    end
  end

  test "signbit over a view" do
    values = [1.0, -2.0, 0.0, -0.0, 3.5, -0.25] * 20
    a = Cumo::SFloat.cast(values)
    expected = values.map { |x| x.negative? || (x.zero? && (1 / x).negative?) ? 1 : 0 }
    assert_equal(Cumo::Bit.cast(expected), a.signbit)
    every_other = Cumo::Bit.cast(expected.each_slice(2).map(&:first))
    assert_equal(every_other, a[(0...values.size).step(2)].signbit)
  end

  # NaN never equals itself and -0.0 equals 0.0, so map both to a tag before
  # comparing: the sign of a zero is exactly what modf and frexp have to keep
  def float_tags(ary)
    ary.map do |x|
      if x.nan?
        :nan
      elsif x.zero?
        (1 / x).negative? ? :negative_zero : :zero
      else
        x
      end
    end
  end

  def modf_expect(val)
    return [val, val] if val.nan? || val.zero?

    # modf keeps the sign of the argument in the fraction, zero included
    signed_zero = val.negative? ? -(0.0) : 0.0
    return [signed_zero, val] if val.infinite?

    whole = val.abs.floor.to_f
    whole = -whole if val.negative?
    frac = val - whole
    frac = signed_zero if frac.zero?
    [frac, whole]
  end

  test "modf and frexp over long arrays and views" do
    inf = Float::INFINITY
    specials = { 0 => Float::NAN, 1 => inf, 2 => -inf, 3 => -0.0 }
    values = Array.new(200) { |i| specials.fetch(i % 9, (i - 100) / 8.0) }
    stepped = (0...200).step(3).to_a
    reversed = (0...200).to_a.reverse

    [Cumo::DFloat, Cumo::SFloat].each do |dtype|
      a = dtype.cast(values)
      # SFloat rounds on the way in, so the expectation has to start from the
      # stored value rather than the literal
      stored = a.to_a
      cases = [
        [:flat, a, (0...200).to_a],
        [:stepped, a[(0...200).step(3)], stepped],
        [:reversed, a[reversed], reversed]
      ]
      cases.each do |what, view, idxs|
        src = idxs.map { |i| stored[i] }
        expected = src.map { |x| modf_expect(x) }

        frac, whole = view.modf
        assert_equal(float_tags(expected.map(&:first)), float_tags(frac.to_a), "#{dtype} #{what} frac")
        assert_equal(float_tags(expected.map(&:last)), float_tags(whole.to_a), "#{dtype} #{what} whole")

        split = src.map { |x| Math.frexp(x) }
        mantissa, exponent = dtype::Math.frexp(view)
        assert_kind_of(Cumo::Int32, exponent)
        assert_equal(float_tags(split.map(&:first)), float_tags(mantissa.to_a), "#{dtype} #{what} mantissa")
        assert_equal(split.map(&:last), exponent.to_a, "#{dtype} #{what} exponent")
      end
    end
  end

  def reduction_to_a(val)
    val.is_a?(Numeric) ? [val] : val.to_a
  end

  test "minmax answers the same as min and max" do
    # the kernel fuses the two reductions into one pass over the input, so the
    # pair has to stay identical to running min and max separately
    reductions = [{}, { axis: 0 }, { axis: 1 }, { axis: 1, keepdims: true }]
    dtypes = [Cumo::DFloat, Cumo::SFloat, Cumo::Int32, Cumo::Int64, Cumo::Int8, Cumo::UInt8]
    dtypes.each do |dtype|
      # 3000 elements so the reduction spans more than one block
      a = ((dtype.new(3000).seq * 37) % 251).reshape(30, 100)
      { whole: a, strided: a[true, (0...100).step(7)] }.each do |what, view|
        reductions.each do |opts|
          min, max = view.minmax(**opts)
          assert_equal(reduction_to_a(view.min(**opts)), reduction_to_a(min), "#{dtype} #{what} #{opts}")
          assert_equal(reduction_to_a(view.max(**opts)), reduction_to_a(max), "#{dtype} #{what} #{opts}")
        end
      end
    end
  end

  test "minmax over several axes" do
    a = ((Cumo::Int32.new(2 * 3 * 4 * 5).seq * 17) % 61).reshape(2, 3, 4, 5)
    [[0, 2], [1, 3], [0, 1, 2, 3], [3]].each do |axis|
      min, max = a.minmax(axis: axis)
      assert_equal(reduction_to_a(a.min(axis: axis)), reduction_to_a(min), axis.inspect)
      assert_equal(reduction_to_a(a.max(axis: axis)), reduction_to_a(max), axis.inspect)
    end
  end

  test "minmax of an all-NaN reduction is NaN on both sides" do
    # numo answers 0.0 for the max here, which disagrees with its own max.
    # cumo's max returns NaN (PR #219), and minmax now says the same.
    nan = Float::NAN
    [Cumo::DFloat, Cumo::SFloat].each do |dtype|
      a = dtype[[1.0, nan, -3.0, 2.0], [nan, nan, nan, nan]]
      min, max = a.minmax(axis: 1)
      assert_equal([-3.0, :nan], float_tags(min.to_a), dtype.to_s)
      assert_equal([2.0, :nan], float_tags(max.to_a), dtype.to_s)
      assert_equal(float_tags(a.max(axis: 1).to_a), float_tags(max.to_a), dtype.to_s)

      # nan: true is still the host loop, and keeps numo's rule of answering
      # NaN as soon as one element is NaN
      min, max = a.minmax(axis: 1, nan: true)
      assert_equal([:nan, :nan], float_tags(min.to_a), dtype.to_s)
      assert_equal([:nan, :nan], float_tags(max.to_a), dtype.to_s)
    end
  end

  test "nan:true reductions answer what the host rules say" do
    # every nan variant is a host loop reading lp->args[0].iter[0], which the
    # indexer path leaves unset, so the flag has to come off once one is chosen
    nan = Float::NAN
    [Cumo::DFloat, Cumo::SFloat].each do |dtype|
      a = dtype[[1.0, nan, -3.0], [4.0, -5.0, 2.0]]
      assert_equal([-2.0, 1.0], a.sum(axis: 1, nan: true).to_a, dtype.to_s)
      assert_equal([-3.0, -40.0], a.prod(axis: 1, nan: true).to_a, dtype.to_s)
      assert_equal([:nan, -5.0], float_tags(a.min(axis: 1, nan: true).to_a), dtype.to_s)
      assert_equal([:nan, 4.0], float_tags(a.max(axis: 1, nan: true).to_a), dtype.to_s)
      assert_equal([:nan, 9.0], float_tags(a.ptp(axis: 1, nan: true).to_a), dtype.to_s)
      assert_equal([1, 1], a.argmin(axis: 1, nan: true).to_a, dtype.to_s)
      assert_equal([1, 0], a.argmax(axis: 1, nan: true).to_a, dtype.to_s)
      assert_equal([1, 4], a.min_index(axis: 1, nan: true).to_a, dtype.to_s)
      assert_equal([1, 3], a.max_index(axis: 1, nan: true).to_a, dtype.to_s)

      # a reduction over every axis takes the same path with one output
      assert_equal([-1.0], reduction_to_a(a.sum(nan: true)), dtype.to_s)
      assert_equal([120.0], reduction_to_a(a.prod(nan: true)), dtype.to_s)
      assert_equal([:nan], float_tags(reduction_to_a(a.min(nan: true))), dtype.to_s)
      assert_equal([:nan], float_tags(reduction_to_a(a.max(nan: true))), dtype.to_s)
      assert_equal([1], reduction_to_a(a.argmin(nan: true)), dtype.to_s)

      # and the kernel path is untouched: without nan:true a NaN propagates
      # through sum, while min and max skip it
      assert_equal([:nan, 1.0], float_tags(a.sum(axis: 1).to_a), dtype.to_s)
      assert_equal([-3.0, -5.0], float_tags(a.min(axis: 1).to_a), dtype.to_s)
      assert_equal([1.0, 4.0], float_tags(a.max(axis: 1).to_a), dtype.to_s)
    end
  end

  test "a reduction wide enough to be split across blocks" do
    # a reduction gets one block per output, so a single output is split across
    # a second launch; the answer has to be what one block would have said
    n = (1 << 20) + 7 # not a multiple of the split count, so the tail counts
    a = Cumo::DFloat.new(n).seq
    assert_equal([549_762_629_653.0], reduction_to_a(a.sum))
    assert_equal([0.0], reduction_to_a(a.min))
    assert_equal([n - 1.0], reduction_to_a(a.max))
    assert_equal([n - 1.0], reduction_to_a(a.ptp))
    min, max = a.minmax
    assert_equal([0.0], reduction_to_a(min))
    assert_equal([n - 1.0], reduction_to_a(max))

    # a handful of outputs leaves the grid nearly empty too
    rows = 8
    cols = 300_000
    b = Cumo::Int32.new(rows * cols).seq.reshape(rows, cols)
    assert_equal((0...rows).map { |r| r * cols }, b.min(axis: 1).to_a)
    assert_equal((0...rows).map { |r| ((r + 1) * cols) - 1 }, b.max(axis: 1).to_a)
  end

  test "a split reduction keeps the NaN rules" do
    n = 1 << 20
    a = Cumo::DFloat.new(n).seq
    a[n / 2] = Float::NAN
    assert_equal([:nan], float_tags(reduction_to_a(a.sum)))
    # min and max skip a NaN unless every element is one
    assert_equal([0.0], reduction_to_a(a.min))
    assert_equal([n - 1.0], reduction_to_a(a.max))

    all_nan = Cumo::DFloat.new(n).fill(Float::NAN)
    assert_equal([:nan], float_tags(reduction_to_a(all_nan.min)))
    assert_equal([:nan], float_tags(reduction_to_a(all_nan.max)))
    assert_equal([:nan], float_tags(reduction_to_a(all_nan.ptp)))
    min, max = all_nan.minmax
    assert_equal([:nan], float_tags(reduction_to_a(min)))
    assert_equal([:nan], float_tags(reduction_to_a(max)))
  end

  test "an index reduction takes the earlier of two equal elements" do
    # the reduction folds blocks in an order unrelated to the input, so without
    # a tie-break the answer is whichever thread happened to hold the extremum
    [Cumo::DFloat, Cumo::Int32].each do |dtype|
      a = dtype[1, 0, 0, 2, 2]
      assert_equal([1], reduction_to_a(a.min_index), dtype.to_s)
      assert_equal([3], reduction_to_a(a.max_index), dtype.to_s)
      assert_equal([1], reduction_to_a(a.argmin), dtype.to_s)
      assert_equal([3], reduction_to_a(a.argmax), dtype.to_s)

      # more elements than one block can hold at once, so the tie lands in
      # different threads
      long = dtype.cast(Array.new(2000) { |i| i % 3 })
      assert_equal([0], reduction_to_a(long.min_index), dtype.to_s)
      assert_equal([2], reduction_to_a(long.max_index), dtype.to_s)
      assert_equal([0], reduction_to_a(long.argmin), dtype.to_s)
      assert_equal([2], reduction_to_a(long.argmax), dtype.to_s)

      # min_index counts through the whole array, argmin along the axis
      grid = dtype.cast(Array.new(2048) { |i| i % 3 }).reshape(4, 512)
      assert_equal([0, 513, 1026, 1536], grid.min_index(axis: 1).to_a, dtype.to_s)
      assert_equal([2, 512, 1025, 1538], grid.max_index(axis: 1).to_a, dtype.to_s)
      assert_equal([0, 1, 2, 0], grid.argmin(axis: 1).to_a, dtype.to_s)
      assert_equal([2, 0, 1, 2], grid.argmax(axis: 1).to_a, dtype.to_s)
    end
  end

  test "bit operators over long arrays and views" do
    n = 200
    xs = Array.new(n) { |i| (i % 3).zero? ? 1 : 0 }
    ys = Array.new(n) { |i| (i % 5) < 2 ? 1 : 0 }
    a = Cumo::Bit.cast(xs)
    b = Cumo::Bit.cast(ys)
    picked = Array.new(n) { |i| ((i * 7) + 3) % n }
    # an offset that is not a multiple of the bit-digit width has to shift every
    # word into place, and a strided or indexed view drops to a bit at a time
    views = [
      ["flat", (0...n).to_a, ->(v) { v }],
      ["offset 13", (13...n).to_a, ->(v) { v[13...n] }],
      ["offset 33", (33...n).to_a, ->(v) { v[33...n] }],
      ["step 2", (0...n).step(2).to_a, ->(v) { v[(0...n).step(2)] }],
      ["indexed", picked, ->(v) { v[picked] }],
    ]

    views.each do |label, idxs, slice|
      va = slice.call(a)
      vb = slice.call(b)
      assert_equal(Cumo::Bit.cast(idxs.map { |i| 1 - xs[i] }), ~va, label)
      [:&, :|, :^].each do |op|
        want = Cumo::Bit.cast(idxs.map { |i| xs[i].public_send(op, ys[i]) })
        assert_equal(want, va.public_send(op, vb), "#{label} #{op}")
      end
    end

    # a strided 2-d view runs the loop once per row, so each row starts at a
    # different bit offset
    cols = (0...25).step(2).to_a
    grid = (0...8).flat_map { |r| cols.map { |c| (r * 25) + c } }
    ga = a.reshape(8, 25)[true, (0...25).step(2)]
    gb = b.reshape(8, 25)[true, (0...25).step(2)]
    assert_equal(Cumo::Bit.cast(grid.map { |i| 1 - xs[i] }).reshape(8, cols.size), ~ga)
    [:&, :|, :^].each do |op|
      want = Cumo::Bit.cast(grid.map { |i| xs[i].public_send(op, ys[i]) }).reshape(8, cols.size)
      assert_equal(want, ga.public_send(op, gb), op.to_s)
    end
  end

  test "a bit operator in place leaves the bits outside the view alone" do
    n = 200
    xs = Array.new(n) { |i| (i % 3).zero? ? 1 : 0 }
    ys = Array.new(n) { |i| (i % 5) < 2 ? 1 : 0 }
    # the view starts and ends inside a word it shares with elements it must not
    # touch
    view = (10...100)

    a = Cumo::Bit.cast(xs)
    a[view].inplace & Cumo::Bit.cast(ys[view])
    want = xs.each_with_index.map { |x, i| view.cover?(i) ? (x & ys[i]) : x }
    assert_equal(Cumo::Bit.cast(want), a)

    a = Cumo::Bit.cast(xs)
    a[view].inplace.~
    want = xs.each_with_index.map { |x, i| view.cover?(i) ? (1 - x) : x }
    assert_equal(Cumo::Bit.cast(want), a)
  end
  test "a bit view with a negative step reads the elements below its first one" do
    # the view starts at the last element, so every element after it sits in a
    # lower word than the one the loop starts in
    [8, 32, 33, 64, 65, 200].each do |n|
      xs = Array.new(n) { |i| (i % 3).zero? ? 1 : 0 }
      a = Cumo::Bit.cast(xs)
      rev = a[(n - 1).step(0, -1)]
      assert_equal(xs.reverse, rev.to_a, "n=#{n}")
      assert_equal(Cumo::Bit.cast(xs.reverse.map { |x| 1 - x }), ~rev, "n=#{n}")
      ones = xs.reverse.each_index.select { |i| xs.reverse[i] == 1 }
      assert_equal(ones, rev.where.to_a, "n=#{n}")
      assert_equal(false, rev.all?, "n=#{n}")
      assert_equal(true, rev.any?, "n=#{n}")
      assert_equal(xs.count(1), rev.count_true.to_i, "n=#{n}")

      # and as the destination of a store
      dst = Cumo::Bit.zeros(n)
      dst[(n - 1).step(0, -1)].store(a)
      assert_equal(Cumo::Bit.cast(xs.reverse), dst, "n=#{n}")
    end
  end

  test "a bit reduction along an axis of a reversed 2-d view" do
    rows = 7
    cols = 33
    xs = Array.new(rows * cols) { |i| (i % 3).zero? ? 1 : 0 }
    a = Cumo::Bit.cast(xs).reshape(rows, cols)
    rev = a[true, (cols - 1).step(0, -1)]
    grid = (0...rows).map { |r| (0...cols).map { |c| xs[(r * cols) + (cols - 1 - c)] } }
    assert_equal(grid, rev.to_a)
    assert_equal(Cumo::Bit.cast(grid.map { |row| row.all? { |x| x == 1 } ? 1 : 0 }),
                 rev.all?(axis: 1))
    assert_equal(Cumo::Bit.cast(grid.transpose.map { |col| col.any? { |x| x == 1 } ? 1 : 0 }),
                 rev.any?(axis: 0))
  end

  test "bit reductions over long arrays and views" do
    # long enough that the reduction runs on the device, and not a multiple of
    # the bit-digit width so the words at either end are only partly in the view
    n = 20_000
    base = Array.new(n) { 1 }
    picked = Array.new(n) { |i| ((i * 7) + 3) % n }
    views = [
      ["flat", (0...n).to_a, ->(v) { v }],
      ["offset 13", (13...n).to_a, ->(v) { v[13...n] }],
      ["offset 32", (32...n).to_a, ->(v) { v[32...n] }],
      ["step 2", (0...n).step(2).to_a, ->(v) { v[(0...n).step(2)] }],
      ["indexed", picked, ->(v) { v[picked] }],
    ]

    # the deciding element at the start, in the middle, at the end, and nowhere
    [nil, 0, n / 2, n - 1].each do |zero_at|
      xs = base.dup
      xs[zero_at] = 0 if zero_at
      a = Cumo::Bit.cast(xs)
      b = ~a
      views.each do |label, idxs, slice|
        want_all = idxs.all? { |i| xs[i] == 1 }
        lbl = "#{label} zero_at=#{zero_at.inspect}"
        assert_equal(want_all, slice.call(a).all?, lbl)
        assert_equal(!want_all, slice.call(b).any?, lbl)
        assert_equal(want_all, slice.call(b).none?, lbl)
      end
    end
  end

  test "bit reductions along an axis" do
    rows = 6
    cols = 20_000
    xs = Array.new(rows * cols) { |i| (i % 20_001).zero? ? 0 : 1 }
    a = Cumo::Bit.cast(xs).reshape(rows, cols)
    grid = (0...rows).map { |r| xs[r * cols, cols] }

    assert_equal(Cumo::Bit.cast(grid.map { |row| row.all? { |x| x == 1 } ? 1 : 0 }),
                 a.all?(axis: 1))
    assert_equal(Cumo::Bit.cast(grid.map { |row| row.any? { |x| x == 1 } ? 1 : 0 }),
                 a.any?(axis: 1))
    assert_equal(Cumo::Bit.cast(grid.transpose.map { |col| col.all? { |x| x == 1 } ? 1 : 0 }),
                 a.all?(axis: 0))
    assert_equal(Cumo::Bit.cast(grid.map { |row| row.all? { |x| x == 1 } ? 1 : 0 }).reshape(rows, 1),
                 a.all?(axis: 1, keepdims: true))

    # a strided view starts each row at a different bit offset
    sel = (0...cols).step(2).to_a
    v = a[true, (0...cols).step(2)]
    want = grid.map { |row| sel.map { |c| row[c] } }
    assert_equal(Cumo::Bit.cast(want.map { |row| row.all? { |x| x == 1 } ? 1 : 0 }),
                 v.all?(axis: 1))
    assert_equal(Cumo::Bit.cast(want.map { |row| row.none? { |x| x == 1 } ? 1 : 0 }),
                 (~v).all?(axis: 1))
  end

  test "bit where and where2 over long arrays and views" do
    n = 20_000
    xs = Array.new(n) { |i| ((i * 2_654_435_761) >> 11) & 1 }
    picked = Array.new(n) { |i| ((i * 7) + 3) % n }
    views = [
      ["flat", (0...n).to_a, ->(v) { v }],
      ["offset 13", (13...n).to_a, ->(v) { v[13...n] }],
      ["step 2", (0...n).step(2).to_a, ->(v) { v[(0...n).step(2)] }],
      ["reversed", (n - 1).downto(0).to_a, ->(v) { v[(n - 1).step(0, -1)] }],
      ["indexed", picked, ->(v) { v[picked] }],
    ]

    a = Cumo::Bit.cast(xs)
    views.each do |label, idxs, slice|
      seq = idxs.map { |i| xs[i] }
      want_one = seq.each_index.select { |i| seq[i] == 1 }
      want_zero = seq.each_index.select { |i| seq[i] == 0 }
      v = slice.call(a)
      assert_equal(want_one, v.where.to_a, label)
      w1, w0 = v.where2
      assert_equal(want_one, w1.to_a, label)
      assert_equal(want_zero, w0.to_a, label)
    end
  end

  test "bit where along the rows of a 2-d view" do
    rows = 4
    cols = 20_000
    xs = Array.new(rows * cols) { |i| ((i * 40_503) >> 7) & 1 }
    a = Cumo::Bit.cast(xs).reshape(rows, cols)

    # each row is long enough to run on the device, and the view drops a
    # column at either end so the rows stay separate calls
    v = a[true, 1..-2]
    seq = (0...rows).flat_map { |r| xs[r * cols + 1, cols - 2] }
    assert_equal(seq.each_index.select { |i| seq[i] == 1 }, v.where.to_a)
    w1, w0 = v.where2
    assert_equal(seq.each_index.select { |i| seq[i] == 1 }, w1.to_a)
    assert_equal(seq.each_index.select { |i| seq[i] == 0 }, w0.to_a)
  end

  test "a bit count over more than 2**32 bits" do
    # in the window where a grid-stride step of blockDim * gridDim wraps past
    # an unsigned int, an uncapped grid loops forever
    n = 1 << 32
    a = Cumo::Bit.new(n).fill(0)
    ones = [0, n / 2, n - 1]
    ones.each { |i| a[i] = 1 }
    assert_equal(ones.size, a.count_true.to_i)
  end

  # cumsum and cumprod had no tests at all before the scan was moved to the device
  def running(values)
    acc = nil
    values.map { |x| acc = acc.nil? ? x : yield(acc, x) }
  end

  # what the host loop does with nan:true: the first element is copied through,
  # and a nan in the accumulator is replaced by the next value rather than spread
  def running_skip_nan(values)
    acc = values[0]
    out = [acc]
    values[1..-1].each do |y|
      if acc.nan?
        acc = y
      elsif !y.nan?
        acc = yield(acc, y)
      end
      out << acc
    end
    out
  end

  def nan_eq(got, want)
    got.size == want.size &&
      got.each_index.all? { |i| (got[i].nan? && want[i].nan?) || got[i] == want[i] }
  end

  test "cumsum and cumprod over long arrays and views" do
    # long enough that the scan runs on the device, with values that keep every
    # partial result exactly representable, so the answer cannot depend on the
    # order a parallel scan associates in
    n = 20_000
    adds = Array.new(n) { |i| (i % 3) + 1 }
    muls = Array.new(n) { |i| i % 4 < 2 ? 2 : 0.5 }
    views = [
      ["flat", (0...n).to_a, ->(v) { v }],
      ["offset 13", (13...n).to_a, ->(v) { v[13...n] }],
      ["step 2", (0...n).step(2).to_a, ->(v) { v[(0...n).step(2)] }],
      ["reversed", (n - 1).downto(0).to_a, ->(v) { v[(n - 1).step(0, -1)] }],
    ]

    [Cumo::Int32, Cumo::DFloat, Cumo::SFloat].each do |klass|
      a = klass.cast(adds)
      views.each do |label, idxs, slice|
        want = running(idxs.map { |i| adds[i] }) { |x, y| x + y }
        assert_equal(want, slice.call(a).cumsum.to_a, "#{klass} cumsum #{label}")
      end
    end

    [Cumo::DFloat, Cumo::SFloat].each do |klass|
      a = klass.cast(muls)
      views.each do |label, idxs, slice|
        want = running(idxs.map { |i| muls[i] }) { |x, y| x * y }
        assert_equal(want, slice.call(a).cumprod.to_a, "#{klass} cumprod #{label}")
      end
    end
  end

  test "cumsum and cumprod along an axis" do
    rows = 6
    cols = 20_000
    adds = Array.new(rows * cols) { |i| (i % 3) + 1 }
    muls = Array.new(rows * cols) { |i| i % 4 < 2 ? 2 : 0.5 }

    a = Cumo::DFloat.cast(adds).reshape(rows, cols)
    rowwise = (0...rows).map { |r| running(adds[r * cols, cols]) { |x, y| x + y } }
    assert_equal(rowwise, a.cumsum(axis: 1).to_a)

    down = (0...cols).map { |c| running((0...rows).map { |r| adds[r * cols + c] }) { |x, y| x + y } }
    assert_equal((0...rows).map { |r| (0...cols).map { |c| down[c][r] } },
                 a.cumsum(axis: 0).to_a)

    b = Cumo::DFloat.cast(muls).reshape(rows, cols)
    assert_equal((0...rows).map { |r| running(muls[r * cols, cols]) { |x, y| x * y } },
                 b.cumprod(axis: 1).to_a)
  end

  test "cumsum and cumprod carry a nan the same way on either path" do
    # one size below the threshold that keeps the host loop and one above it
    [64, 20_000].each do |n|
      [0, 1, n / 2, n - 1].each do |at|
        adds = Array.new(n) { |i| (i % 3) + 1.0 }
        muls = Array.new(n) { |i| i % 4 < 2 ? 2.0 : 0.5 }
        adds[at] = Float::NAN
        muls[at] = Float::NAN
        lbl = "n=#{n} at=#{at}"

        a = Cumo::DFloat.cast(adds)
        assert(nan_eq(a.cumsum.to_a, running(adds) { |x, y| x + y }), "cumsum #{lbl}")
        assert(nan_eq(a.cumsum(nan: true).to_a, running_skip_nan(adds) { |x, y| x + y }),
               "cumsum nan:true #{lbl}")

        b = Cumo::DFloat.cast(muls)
        assert(nan_eq(b.cumprod.to_a, running(muls) { |x, y| x * y }), "cumprod #{lbl}")
        assert(nan_eq(b.cumprod(nan: true).to_a, running_skip_nan(muls) { |x, y| x * y }),
               "cumprod nan:true #{lbl}")
      end
    end
  end

  # the two-pass variance numo computes, in exact arithmetic
  def exact_var(values)
    mean = values.sum(0r, &:to_r) / values.size
    (values.sum(0r) { |x| (x.to_r - mean)**2 } / (values.size - 1)).to_f
  end

  test "var and stddev over an axis long enough to split the reduction" do
    # one output over a long axis is the shape cumo_reduce_split exists for, and
    # the shifted data is where a one-pass accumulator loses digits if the
    # running mean is not carried along with the deviations
    rows = 5
    cols = 40_000
    xs = Array.new(rows * cols) { |i| 1000.0 + ((i * 7919) % 2003) / 1000.0 }

    flat = Cumo::DFloat.cast(xs)
    assert_in_delta(exact_var(xs), flat.var.to_f, exact_var(xs) * 1e-9)
    assert_in_delta(Math.sqrt(exact_var(xs)), flat.stddev.to_f, Math.sqrt(exact_var(xs)) * 1e-9)

    a = Cumo::DFloat.cast(xs).reshape(rows, cols)
    (0...rows).each do |r|
      want = exact_var(xs[r * cols, cols])
      assert_in_delta(want, a.var(axis: 1).to_a[r], want * 1e-9, "row #{r}")
    end
    assert_equal([rows, 1], a.var(axis: 1, keepdims: true).shape)

    # a column reduction is the many-outputs shape, and a strided view starts
    # each row at a different offset
    down = (0...cols).map { |c| exact_var((0...rows).map { |r| xs[r * cols + c] }) }
    got = a.var(axis: 0).to_a
    [0, 1, cols / 2, cols - 1].each do |c|
      assert_in_delta(down[c], got[c], down[c] * 1e-9, "col #{c}")
    end

    sel = (0...cols).step(3).to_a
    want = exact_var(sel.map { |c| xs[c] })
    assert_in_delta(want, a[true, (0...cols).step(3)].var(axis: 1).to_a[0], want * 1e-9)
  end

  test "mean and rms over an axis long enough to split the reduction" do
    rows = 5
    cols = 40_000
    xs = Array.new(rows * cols) { |i| ((i * 7919) % 2003) / 100.0 - 10.0 }

    flat = Cumo::DFloat.cast(xs)
    want_mean = (xs.sum(0r, &:to_r) / xs.size).to_f
    want_rms = Math.sqrt((xs.sum(0r) { |x| x.to_r**2 } / xs.size).to_f)
    assert_in_delta(want_mean, flat.mean.to_f, want_mean.abs * 1e-9)
    assert_in_delta(want_rms, flat.rms.to_f, want_rms * 1e-9)

    a = Cumo::DFloat.cast(xs).reshape(rows, cols)
    got = a.mean(axis: 1).to_a
    (0...rows).each do |r|
      row = xs[r * cols, cols]
      want = (row.sum(0r, &:to_r) / cols).to_f
      assert_in_delta(want, got[r], want.abs * 1e-9, "row #{r}")
    end

    # the many-outputs shape, and a strided view starting each row elsewhere
    down = (0...cols).map { |c| ((0...rows).sum(0r) { |r| xs[r * cols + c].to_r } / rows).to_f }
    got = a.mean(axis: 0).to_a
    [0, 1, cols / 2, cols - 1].each { |c| assert_in_delta(down[c], got[c], down[c].abs * 1e-9, "col #{c}") }

    sel = (0...cols).step(3).to_a
    want = (sel.sum(0r) { |c| xs[c].to_r } / sel.size).to_f
    assert_in_delta(want, a[true, (0...cols).step(3)].mean(axis: 1).to_a[0], want.abs * 1e-9)
  end

  test "mean of nearly cancelling values stays close to the exact answer" do
    # the mean is tiny next to the elements, so it carries whatever error the
    # sum accumulated and a mis-combined accumulator shows up immediately
    n = 100_000
    xs = Array.new(n) { |i| Math.sin(i * 1.7) * 10 }
    exact = (xs.sum(0r, &:to_r) / n).to_f
    assert_in_delta(exact, Cumo::DFloat.cast(xs).mean.to_f, 1e-12)
    assert_in_delta(exact, Cumo::SFloat.cast(xs).mean.to_f, 1e-4)

    want_rms = Math.sqrt((xs.sum(0r) { |x| x.to_r**2 } / n).to_f)
    assert_in_delta(want_rms, Cumo::DFloat.cast(xs).rms.to_f, want_rms * 1e-12)
  end

  test "var and stddev keep their corner answers" do
    # a single element has no degrees of freedom left, and nan:true drops the
    # nan rather than spreading it
    assert(Cumo::DFloat[3.5].var.to_f.nan?)
    assert(Cumo::DFloat[3.5].stddev.to_f.nan?)
    assert_equal(0.0, Cumo::DFloat[3.5, 3.5].var.to_f)

    n = 20_000
    xs = Array.new(n) { |i| (i % 17) + 1.0 }
    [0, n / 2, n - 1].each do |at|
      ys = xs.dup
      ys[at] = Float::NAN
      a = Cumo::DFloat.cast(ys)
      assert(a.var.to_f.nan?, "nan spreads at=#{at}")
      assert(a.stddev.to_f.nan?, "nan spreads at=#{at}")

      kept = xs.each_with_index.reject { |_, i| i == at }
      want = exact_var(kept.map(&:first))
      assert_in_delta(want, a.var(nan: true).to_f, want * 1e-9, "nan:true at=#{at}")
    end
  end

  test "mulsum over long axes, views and broadcasts" do
    rows = 5
    cols = 40_000
    xs = Array.new(rows * cols) { |i| ((i % 13) - 6).to_f }
    ys = Array.new(rows * cols) { |i| ((i % 7) - 3).to_f }
    a = Cumo::DFloat.cast(xs)
    b = Cumo::DFloat.cast(ys)

    assert_equal(xs.each_index.sum { |i| xs[i] * ys[i] }, a.mulsum(b).to_f)
    assert_equal(xs.each_index.sum { |i| xs[i] * ys[i] }, a.dot(b).to_f)

    sel = (0...xs.size).step(3).to_a
    assert_equal(sel.sum { |i| xs[i] * ys[i] },
                 a[(0...xs.size).step(3)].mulsum(b[(0...xs.size).step(3)]).to_f)
    rev = (xs.size - 1).downto(0).to_a
    want_rev = rev.each_index.sum { |i| xs[rev[i]] * ys[i] }
    assert_equal(want_rev,
                 a[(xs.size - 1).step(0, -1)].mulsum(b).to_f)

    g = a.reshape(rows, cols)
    h = b.reshape(rows, cols)
    assert_equal((0...rows).map { |r| (0...cols).sum { |c| xs[r * cols + c] * ys[r * cols + c] } },
                 g.mulsum(h, axis: 1).to_a)

    # a broadcast operand walks the same element for every row
    v = Array.new(cols) { |i| ((i % 5) - 2).to_f }
    assert_equal((0...rows).map { |r| (0...cols).sum { |c| xs[r * cols + c] * v[c] } },
                 g.mulsum(Cumo::DFloat.cast(v), axis: 1).to_a)
  end

  # the multi-dimensional index of the i-th element of a C-contiguous shape
  def unravel(shape, flat)
    idx = []
    rest = flat
    shape.reverse_each do |d|
      idx.unshift(rest % d)
      rest /= d
    end
    idx
  end

  test "mulsum accumulates over reduce axes that cannot be flattened" do
    # non-adjacent axes leave ndloop no single run to hand the kernel, so it
    # calls it once per piece and each call has to add to what it finds
    shape = [2, 3, 4, 5]
    n = shape.reduce(:*)
    xs = Array.new(n) { |i| ((i % 13) - 6).to_f }
    ys = Array.new(n) { |i| ((i % 7) - 3).to_f }
    a = Cumo::DFloat.cast(xs).reshape(*shape)
    b = Cumo::DFloat.cast(ys).reshape(*shape)

    [[0, 2], [1, 3], [0, 2, 3]].each do |axes|
      kept = (0...shape.size).reject { |d| axes.include?(d) }
      exp = Hash.new(0.0)
      (0...n).each do |i|
        idx = unravel(shape, i)
        exp[kept.map { |d| idx[d] }] += xs[i] * ys[i]
      end
      got = a.mulsum(b, axis: axes).to_a.flatten
      assert_equal(exp.keys.sort.map { |k| exp[k] }, got, "axis=#{axes.inspect}")
    end
  end

  test "mulsum with nan: true drops the pair rather than spreading it" do
    n = 20_000
    xs = Array.new(n) { |i| ((i % 13) - 6).to_f }
    ys = Array.new(n) { |i| ((i % 7) - 3).to_f }
    [0, n / 2, n - 1].each do |at|
      [:x, :y].each do |which|
        a = xs.dup
        b = ys.dup
        (which == :x ? a : b)[at] = Float::NAN
        ca = Cumo::DFloat.cast(a)
        cb = Cumo::DFloat.cast(b)
        assert(ca.mulsum(cb).to_f.nan?, "#{which} at=#{at} spreads")
        pairs = (0...n).reject { |i| i == at }
        want = pairs.sum { |i| a[i] * b[i] }
        assert_equal(want, ca.mulsum(cb, nan: true).to_f, "#{which} at=#{at} nan:true")
      end
    end
  end

  test "store into and out of views that ndloop cannot flatten" do
    # a column slice, a transpose and an index view each leave more than one
    # dimension for the kernel to walk; before the indexer loop the kernel ran
    # once per row of them
    rows = 40
    cols = 12
    xs = Array.new(rows * cols) { |i| (i % 11) + 1.0 }
    src = Cumo::DFloat.cast(xs).reshape(rows, cols)
    grid = (0...rows).map { |r| xs[r * cols, cols] }

    want = grid.map { |row| row[2...(cols - 1)] }
    assert_equal(want, Cumo::DFloat.zeros(rows, cols - 3).store(src[true, 2...(cols - 1)]).to_a)

    assert_equal(grid.transpose, Cumo::DFloat.zeros(cols, rows).store(src.transpose).to_a)

    idx = Array.new(rows) { |i| ((i * 7) + 3) % rows }
    assert_equal(idx.map { |i| grid[i] }, Cumo::DFloat.zeros(rows, cols).store(src[idx, true]).to_a)

    # a store into a slice must leave the columns on either side alone
    dst = Cumo::DFloat.zeros(rows, cols)
    dst[true, 2...(cols - 1)] = src[true, 0...(cols - 3)]
    assert_equal(grid.map { |row| [0.0, 0.0] + row[0...(cols - 3)] + [0.0] }, dst.to_a)

    # a 5-d shape runs past the dimension-specialised kernels
    deep = [2, 2, 3, 2, 5]
    n = deep.reduce(:*)
    ys = Array.new(n) { |i| (i % 7) + 1.0 }
    a = Cumo::DFloat.cast(ys).reshape(*deep)
    assert_equal(ys, Cumo::DFloat.zeros(*deep).store(a).to_a.flatten)
    assert_equal(a.transpose.to_a.flatten,
                 Cumo::DFloat.zeros(*deep.reverse).store(a.transpose).to_a.flatten)

    # broadcast and dtype conversion still reach every element
    assert_equal([[1.0, 2.0, 3.0]] * 4, Cumo::DFloat.zeros(4, 3).store(Cumo::Int32[1, 2, 3]).to_a)
    assert_equal(grid.map { |row| row.map(&:to_i) },
                 Cumo::Int32.zeros(rows, cols).store(src).to_a)
  end

  test "unary operations on views that ndloop cannot flatten" do
    # same shape of view the store tests use: before the indexer loop each of
    # these ran one kernel per row of the view
    rows = 40
    cols = 12
    xs = Array.new(rows * cols) { |i| ((i % 9) - 4).to_f }
    a = Cumo::DFloat.cast(xs).reshape(rows, cols)
    grid = (0...rows).map { |r| xs[r * cols, cols] }
    idx = Array.new(rows) { |i| ((i * 7) + 3) % rows }

    views = [
      [:flat, ->(m) { m }, grid],
      [:colslice, ->(m) { m[true, 2...(cols - 1)] }, grid.map { |row| row[2...(cols - 1)] }],
      [:transpose, :transpose.to_proc, grid.transpose],
      [:idxview, ->(m) { m[idx, true] }, idx.map { |i| grid[i] }]
    ]

    views.each do |label, slice, want|
      v = slice.call(a)
      assert_equal(want.map { |row| row.map(&:-@) }, (-v).to_a, "minus #{label}")
      assert_equal(want.map { |row| row.map { |x| x * x } }, v.square.to_a, "square #{label}")
      assert_equal(want.map { |row| row.map(&:abs) }, v.abs.to_a, "abs #{label}")
      got = Cumo::NMath.exp(v).to_a
      want.each_with_index do |row, r|
        row.each_with_index { |x, c| assert_in_delta(Math.exp(x), got[r][c], 1e-6, "exp #{label} #{r},#{c}") }
      end
    end

    # a 5-d shape runs past the dimension-specialised kernels
    deep = [2, 2, 3, 2, 5]
    ys = Array.new(deep.reduce(:*)) { |i| ((i % 7) - 3).to_f }
    b = Cumo::DFloat.cast(ys).reshape(*deep)
    assert_equal(ys.map(&:-@), (-b).to_a.flatten)
    assert_equal(b.transpose.to_a.flatten.map(&:abs), b.transpose.abs.to_a.flatten)
  end

  test "the unary error flags still fire from the indexer kernel" do
    # the flag is read after the launch, so it has to survive the rewrite
    assert_raise(ZeroDivisionError) { Cumo::Int32.cast([[1, 0], [2, 3]])[true, 0..1].reciprocal }
    assert_raise(Cumo::NArray::ValueError) do
      Cumo::Int32.cast([[-2_147_483_648, 1], [2, 3]])[true, 0..1].abs
    end
    assert_equal([[1, 0], [0, 0]], Cumo::Int32.cast([[1, 2], [3, 4]]).reciprocal.to_a)
  end

  test "a reduction answers the same on a flat layout and a strided one" do
    # the flat layout skips the indexer, so the two paths are different kernels
    # and have to be compared against each other, not just against a literal
    rows = 96
    cols = 130
    xs = Array.new(rows * cols) { |i| ((i * 37) % 101) - 50.0 }
    a = Cumo::DFloat.cast(xs).reshape(rows, cols)
    grid = (0...rows).map { |r| xs[r * cols, cols] }

    assert_equal(grid.map(&:sum), a.sum(axis: 1).to_a)
    assert_equal(grid.transpose.map(&:sum), a.sum(axis: 0).to_a)
    assert_equal(xs.sum, a.sum.to_f)
    assert_equal(grid.map(&:min), a.min(axis: 1).to_a)
    assert_equal(grid.map(&:max), a.max(axis: 1).to_a)
    assert_equal(grid.map { |row| row.max - row.min }, a.ptp(axis: 1).to_a)

    # a transpose leaves the input non-contiguous, which takes the indexer path
    t = a.transpose
    assert_equal(grid.transpose.map(&:sum), t.sum(axis: 1).to_a)
    assert_equal(grid.map(&:sum), t.sum(axis: 0).to_a)
    assert_equal(grid.transpose.map(&:max), t.max(axis: 1).to_a)

    # so does a strided view
    sel = (0...cols).step(3).to_a
    v = a[true, (0...cols).step(3)]
    assert_equal(grid.map { |row| sel.sum { |c| row[c] } }, v.sum(axis: 1).to_a)
    want_min = grid.map { |row| sel.map { |c| row[c] } }
    assert_equal(want_min.map(&:min), v.min(axis: 1).to_a)

    # and a reduction over an axis that is neither first nor last
    deep = [4, 5, 6]
    ys = Array.new(deep.reduce(:*)) { |i| ((i * 13) % 29) - 14.0 }
    b = Cumo::DFloat.cast(ys).reshape(*deep)
    want =
      (0...deep[0]).map do |i|
        (0...deep[2]).map { |k| (0...deep[1]).sum { |j| ys[(i * 30) + (j * 6) + k] } }
      end
    assert_equal(want, b.sum(axis: 1).to_a)
    assert_equal([deep[0], 1, deep[2]], b.sum(axis: 1, keepdims: true).shape)
  end

  test "a split reduction answers the same as an unsplit one" do
    # one output over a long axis is the shape that goes through the two-pass
    # split, and its first pass has a flat variant too
    n = 300_000
    xs = Array.new(n) { |i| ((i * 7) % 13) - 6.0 }
    a = Cumo::DFloat.cast(xs)
    assert_equal(xs.sum, a.sum.to_f)
    assert_equal(xs.min, a.min.to_f)
    assert_equal(xs.max, a.max.to_f)

    # the same values reduced in rows, which is not split
    rows = 300
    b = Cumo::DFloat.cast(xs).reshape(rows, n / rows)
    assert_equal(xs.sum, b.sum(axis: 1).to_a.sum)
    assert_equal(xs.min, b.min(axis: 1).to_a.min)
  end
end
