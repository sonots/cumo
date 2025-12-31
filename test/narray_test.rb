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
          assert { (a >= 3) == [[0, 0, 1], [1, 1, 1]] }
          assert { (a >  3) == [[0, 0, 0], [1, 1, 1]] }
          assert { (a <= 3) == [[1, 1, 1], [0, 0, 0]] }
          assert { (a <  3) == [[1, 1, 0], [0, 0, 0]] }
          assert { (a.eq 3) == [[0, 0, 1], [0, 0, 0]] }
          assert { a.sort == src }
          assert { a.sort_index == [[0, 1, 2], [3, 4, 5]] }
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
  end

  test "Cumo::DFloat.cast(Cumo::RObject[1, nil, 3])" do
    assert_equal(Cumo::DFloat[1, Float::NAN, 3].format_to_a,
                 Cumo::DFloat.cast(Cumo::RObject[1, nil, 3]).format_to_a)
  end

  test "one element array" do
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
    assert { Cumo::SFloat[1].rms == 1.0 }
    assert { Cumo::DFloat[1].rms == 1.0 }
  end
end
