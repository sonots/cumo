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
          assert { a.var == 13.766666666666666 }
          assert { a.stddev == 3.7103458958251676 }
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
        assert { a.max_index(2) == [[1, 5, 8], [9, 12, 17]] }
        assert { a.max(2) == [[8, 6, 5], [7, 9, 6]] }
        assert { a.argmax(2) == [[1, 2, 2], [0, 0, 2]] }
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
end
