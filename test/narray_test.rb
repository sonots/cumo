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

    # A permutation that leaves no pair of axes ndloop can merge, so the kernel
    # sees the full ndim and picks the accessor for it.
    test "#{dtype}, a view permuted past the optimized indexer ndim" do
      (5..8).each do |nd|
        shape = Array.new(nd) { |i| i.even? ? 2 : 3 }
        perm = (1...nd).step(2).to_a + (0...nd).step(2).to_a
        v = dtype.new(*shape).seq.transpose(*perm)
        assert { !v.contiguous? }
        assert { v.dup.to_a == v.to_a }
        assert { (v + v).to_a == (v.dup + v.dup).to_a }
        assert { (-v).to_a == (-v.dup).to_a }
      end
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

        # A numeric bound is now converted by the caller and handed to the
        # kernel, where it used to be cast to a 0-dimensional array.
        test "a numeric bound answers what the same bound as an array answers" do
          a = dtype[0..9]
          assert { a.clip(1.5, 7.5) == a.clip(dtype.new(10).fill(1.5), dtype.new(10).fill(7.5)) }
          assert { a.clip(1.5, nil) == a.clip(dtype.new(10).fill(1.5), nil) }
          assert { a.clip(nil, 7.5) == a.clip(nil, dtype.new(10).fill(7.5)) }
        end

        test "a numeric bound reaches a shape past the optimized indexer ndim" do
          a = dtype.new(*([2] * 9)).seq
          assert { a.clip(1, 2) == a.clip(dtype.new(*([2] * 9)).fill(1), dtype.new(*([2] * 9)).fill(2)) }
          assert { a.clip(1, nil) == a.clip(dtype.new(*([2] * 9)).fill(1), nil) }
        end

        if [Cumo::SFloat, Cumo::DFloat].include?(dtype)
          test "a NaN bound answers what the same bound as an array answers" do
            a = dtype[0..5]
            nan = dtype.new(6).fill(Float::NAN)
            assert { a.clip(Float::NAN, 4) == a.clip(nan, 4) }
            assert { a.clip(1, Float::NAN) == a.clip(1, nan) }
          end
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
        test "matrix.gemm(matrix) broadcasts an operand that holds one matrix" do
          b = dtype.new(3, 4).seq
          # fills the pool next to b, so a read past b shows up as 7
          neighbour = Array.new(2) { dtype.new(200).fill(7) }
          a = dtype.new(10, 2, 3).seq
          c = a.gemm(b)
          assert { c.shape == [10, 2, 4] }
          10.times { |i| assert { c[i, true, true] == a[i, true, true].gemm(b) } }
          assert { neighbour.all? { |n| n == dtype.new(200).fill(7) } }
        end
        test "matrix.gemm(matrix) takes the batch dimensions from either operand" do
          a = dtype.new(2, 3).seq
          b = dtype.new(4, 3, 4).seq
          c = a.gemm(b)
          assert { c.shape == [4, 2, 4] }
          4.times { |i| assert { c[i, true, true] == a.gemm(b[i, true, true]) } }
        end
        test "matrix.gemm(matrix) rejects batch counts that differ" do
          a = dtype.new(4, 2, 3).seq
          b = dtype.new(3, 3, 4).seq
          assert_raise(Cumo::NArray::ShapeError) { a.gemm(b) }
        end
        test "matrix.gemm(matrix) rejects an empty operand" do
          assert_raise(Cumo::NArray::ShapeError) { dtype.new(2, 3).seq.gemm(dtype.new(3, 0)) }
          assert_raise(Cumo::NArray::ShapeError) { dtype.new(0, 3).gemm(dtype.new(3, 4).seq) }
          assert_raise(Cumo::NArray::ShapeError) { dtype.new(2, 0).gemm(dtype.new(0, 4)) }
        end
        test "matrix.gemm(matrix) takes a transposed batch without a copy" do
          a = dtype.new(2, 3, 2, 4).seq
          b = dtype.new(2, 3, 5, 4).seq.transpose(0, 1, 3, 2)
          assert { a.gemm(b) == a.gemm(b.dup) }
          c = dtype.new(3, 2, 4).seq
          d = dtype.new(3, 5, 4).seq.transpose(0, 2, 1)
          assert { c.gemm(d) == c.gemm(d.dup) }
        end
        test "matrix.gemm(matrix) copies a batch that is not a matrix transpose" do
          a = dtype.new(3, 2, 4).seq
          # every axis reversed, so the batch axis is the innermost one
          b = dtype.new(5, 4, 3).seq.transpose
          assert { a.gemm(b) == a.gemm(b.dup) }
          # rows dropped from each matrix, so the batch no longer advances by
          # whole matrices
          c = dtype.new(2, 2, 3, 4).seq
          d = dtype.new(2, 2, 5, 4).seq[true, true, 0..3, true].transpose(0, 1, 3, 2)
          assert { c.gemm(d) == c.gemm(d.dup) }
        end
        test "matrix.gemm(matrix, c) rejects a c whose batch count differs" do
          a = dtype.new(4, 2, 3).seq
          b = dtype.new(3, 4).seq
          assert_raise(Cumo::NArray::ShapeError) { a.gemm(b, dtype.zeros(2, 4)) }
          assert { a.gemm(b, dtype.zeros(4, 2, 4)) == a.gemm(b) }
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

    def bincount_expect(nested, length)
      return nested.map { |row| bincount_expect(row, length) } if nested.first.is_a?(Array)

      counts = Array.new(length, 0)
      nested.each { |x| counts[x] += 1 }
      counts
    end

    def bincount_expect_weighted(nested, weights, length)
      if nested.first.is_a?(Array)
        return nested.zip(weights).map { |row, w| bincount_expect_weighted(row, w, length) }
      end

      sums = Array.new(length, 0.0)
      nested.each_with_index { |x, i| sums[x] += weights[i] }
      sums
    end

    test "every element of a view is counted" do
      # The count ran on the host, one element at a time, behind a device
      # synchronization. The expectations are tallied in Ruby rather than taken
      # from a contiguous copy, so a wrong address cannot cancel out.
      rows = 12
      cols = 25
      lim = 37
      xs = Array.new(rows * cols) { |i| (i * 7) % lim }
      idx = [9, 0, 4, 11, 2]

      views = {
        "contiguous" => ->(a) { a },
        "column slice" => ->(a) { a[true, 3...(cols - 4)] },
        "reversed" => ->(a) { a[true, (cols - 1).step(0, -1)] },
        "row stride" => ->(a) { a[0.step(rows - 1, 3), true] },
        "index view" => ->(a) { a[idx, true] },
        # the index has to be on the last axis for one to reach the kernel; on
        # an outer axis ndloop walks it itself
        "column index" => ->(a) { a[true, [7, 0, 19, 3, 24, 11]] },
        "transpose" => ->(a) { a.transpose },
      }

      int_types.each do |dtype|
        next if dtype == Cumo::Int8 # lim does not fit

        src = dtype.cast(xs).reshape(rows, cols)
        views.each do |what, take|
          v = take.call(src)
          length = v.to_a.flatten.max + 1
          assert_equal(bincount_expect(v.to_a, length), v.bincount.to_a, "#{dtype} #{what}")
          assert_equal(bincount_expect(v.to_a, length + 9), v.bincount(minlength: length + 9).to_a,
                       "#{dtype} #{what} minlength")
        end

        flat = dtype.cast(xs)
        [flat[[13, 2, 40, 7, 99, 0, 61]], flat[(0...(rows * cols)).step(5)],
         flat[((rows * cols) - 1).step(0, -2)]].each_with_index do |v, i|
          length = v.to_a.max + 1
          assert_equal(bincount_expect(v.to_a, length), v.bincount.to_a, "#{dtype} 1-d view #{i}")
        end
      end

      # weights, whose sums stay exact so the order the additions run in cannot
      # show up in the answer
      ws = Array.new(rows * cols) { |i| ((i * 3) % 11) - 5.0 }
      [Cumo::SFloat, Cumo::DFloat].each do |wtype|
        src = Cumo::Int32.cast(xs).reshape(rows, cols)
        weight = wtype.cast(ws).reshape(rows, cols)
        views.each do |what, take|
          v = take.call(src)
          w = take.call(weight)
          length = v.to_a.flatten.max + 1
          got = v.bincount(w).to_a
          want = bincount_expect_weighted(v.to_a, w.to_a, length)
          assert_equal(want, got, "#{wtype} #{what}")
        end
      end
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

  test "copy of a view built from an index array" do
    # ndloop used to walk the outer dimension itself here, and an index array
    # made it synchronize once per row
    rows = 50
    cols = 9
    xs = Array.new(rows * cols) { |i| ((i * 11) % 23) - 11.0 }
    a = Cumo::DFloat.cast(xs).reshape(rows, cols)
    grid = (0...rows).map { |r| xs[r * cols, cols] }
    idx = Array.new(rows) { |i| ((i * 7) + 3) % rows }

    assert_equal(idx.map { |i| grid[i] }, a[idx, true].copy.to_a)
    assert_equal(idx.map { |i| grid[i][2...(cols - 1)] }, a[idx, 2...(cols - 1)].copy.to_a)
    assert_equal(grid, a.copy.to_a)
    assert_equal(grid.transpose, a.transpose.copy.to_a)

    # a reduction over such a view copies it first, so it must agree
    assert_equal(idx.map { |i| grid[i].sum }, a[idx, true].sum(axis: 1).to_a)
    assert_equal((0...cols).map { |c2| idx.sum { |i| grid[i][c2] } },
                 a[idx, true].sum(axis: 0).to_a)

    # the copy must not alias the source, so writing through the source after
    # it was taken must leave it alone
    mutable = Cumo::DFloat.cast(xs).reshape(rows, cols)
    taken = mutable[idx, true].copy
    mutable[idx[0], 0] = 999.0
    assert_equal(grid[idx[0]][0], taken[0, 0].to_f)

    # 5-d runs past the dimension-specialised kernels
    deep = [2, 2, 3, 2, 5]
    ys = Array.new(deep.reduce(:*)) { |i| ((i * 13) % 29) - 14.0 }
    b = Cumo::DFloat.cast(ys).reshape(*deep)
    assert_equal(ys, b.copy.to_a.flatten)
    assert_equal(b.transpose.to_a.flatten, b.transpose.copy.to_a.flatten)
    assert_equal(b[[1, 0], true, true, true, true].to_a.flatten,
                 b[[1, 0], true, true, true, true].copy.to_a.flatten)
  end

  test "elementwise kernels reach every element of a view they cannot flatten" do
    # pow, clip, divmod, maximum, frexp, modf, fill and seq each ran once per
    # row of a view with more than one dimension left before the indexer loop.
    # Each view is checked against a contiguous copy of the same elements, so
    # only the walk is under test and not the operation's own semantics.
    rows = 9
    cols = 7
    xs = Array.new(rows * cols) { |i| ((i * 3) % 13) + 1.0 }
    src = Cumo::DFloat.cast(xs).reshape(rows, cols)
    idx = [5, 0, 3, 8, 1]

    views = {
      "column slice" => ->(a) { a[true, 1...(cols - 1)] },
      "reversed" => ->(a) { a[true, (cols - 1).step(0, -1)] },
      "row stride" => ->(a) { a[0.step(rows - 1, 2), true] },
      "index view" => ->(a) { a[idx, true] },
      "transpose" => ->(a) { a.transpose },
    }

    ops = {
      "pow" => ->(a) { a**2.0 },
      "pow_int32" => ->(a) { a**3 },
      "clip" => ->(a) { a.clip(4.0, 9.0) },
      "clip min" => ->(a) { a.clip(4.0, nil) },
      "clip max" => ->(a) { a.clip(nil, 9.0) },
      "clip array" => ->(a) { a.clip(a - 1.0, a + 1.0) },
      "maximum" => ->(a) { Cumo::DFloat.maximum(a, 6.0) },
      "minimum" => ->(a) { Cumo::DFloat.minimum(a, 6.0) },
      "divmod" => ->(a) { a.divmod(4.0).map(&:to_a) },
      "frexp" => ->(a) { Cumo::NMath.frexp(a).map(&:to_a) },
      "modf" => ->(a) { a.modf.map(&:to_a) },
    }

    views.each do |what, take|
      view = take.call(src)
      flat = view.copy
      assert_equal(flat.to_a, view.to_a, "#{what} itself")
      ops.each do |op, f|
        want = f.call(flat)
        got = f.call(view)
        assert_equal(want.is_a?(Array) ? want : want.to_a,
                     got.is_a?(Array) ? got : got.to_a, "#{op} of #{what}")
      end
    end

    # fill and seq write through the view, so the elements outside it must be
    # left alone and the ones inside must all be reached
    views.each do |what, take|
      cells = take.call(Cumo::DFloat.cast((0...(rows * cols)).to_a).reshape(rows, cols))
                  .to_a.flatten.map(&:to_i)

      dst = Cumo::DFloat.zeros(rows, cols)
      take.call(dst).fill(5.0)
      want = Array.new(rows * cols) { |i| cells.include?(i) ? 5.0 : 0.0 }
      assert_equal(want, dst.to_a.flatten, "fill #{what}")

      dst = Cumo::DFloat.zeros(rows, cols)
      take.call(dst).seq(2.0, 0.5)
      seen = cells.each_with_index.to_h { |cell, i| [cell, 2.0 + (0.5 * i)] }
      want = Array.new(rows * cols) { |i| seen.fetch(i, 0.0) }
      assert_equal(want, dst.to_a.flatten, "seq #{what}")
    end

    # a 5-d shape runs past the dimension-specialised kernels
    deep = [2, 2, 3, 2, 5]
    n = deep.reduce(:*)
    ys = Array.new(n) { |i| (i % 7) + 1.0 }
    a = Cumo::DFloat.cast(ys).reshape(*deep)
    assert_equal(ys.map { |x| x**2 }, (a**2.0).to_a.flatten)
    assert_equal(ys.map { |x| x.clamp(2.0, 5.0) }, a.clip(2.0, 5.0).to_a.flatten)
    assert_equal(a.transpose.copy.modf.map(&:to_a), a.transpose.modf.map(&:to_a))
    d = Cumo::DFloat.zeros(*deep)
    d.transpose.seq(1.0)
    assert_equal((0...n).map { |i| 1.0 + i }, d.transpose.to_a.flatten)

    # the error paths still fire from the kernel
    assert_raise(Cumo::NArray::OperationError) { src[true, 1..-2].clip(9.0, 1.0) }
    ints = Cumo::Int32.cast(xs.map(&:to_i)).reshape(rows, cols)
    assert_raise(ZeroDivisionError) { ints[true, 1..-2].divmod(0) }
  end

  def ints_of(narray)
    narray.to_a.map { |row| row.map(&:to_i) }
  end

  # Combines two nested arrays element by element, whatever their depth
  def zip_deep(a, b, &blk)
    if a.first.is_a?(Array)
      a.zip(b).map { |x, y| zip_deep(x, y, &blk) }
    else
      a.zip(b).map { |x, y| blk.call(x, y) }
    end
  end

  def map_deep(x, &blk)
    x.is_a?(Array) ? x.map { |y| map_deep(y, &blk) } : blk.call(x)
  end

  # Folds a nested array along one axis, so that a reduction can be checked
  # without asking another kernel for the answer
  def fold_axis(nested, axis, &blk)
    return nested.map { |sub| fold_axis(sub, axis - 1, &blk) } if axis.positive?
    nested.reduce { |a, b| a.is_a?(Array) ? zip_deep(a, b, &blk) : blk.call(a, b) }
  end

  def fold_axes(nested, axes, &blk)
    Array(axes).sort.reverse.inject(nested) { |acc, axis| fold_axis(acc, axis, &blk) }
  end

  # The same nesting with each leaf replaced by its place in the flattened
  # array, which is the index max_index and min_index answer with
  def deep_positions(nested, counter = [0])
    nested.map { |x| x.is_a?(Array) ? deep_positions(x, counter) : (counter[0] += 1) - 1 }
  end

  def flat_list(x)
    x.is_a?(Array) ? x.flatten : [x]
  end

  def assert_reductions(view, what, axes: nil)
    nested = view.to_a
    flat = nested.flatten
    shape = view.shape
    strides = shape.each_index.map { |i| shape[(i + 1)..].inject(1, :*) }
    positions = deep_positions(nested)
    axes ||= (0...view.ndim).map { |i| [i] } + [(0...view.ndim).to_a]

    axes.each do |ax|
      tag = "#{what} axis #{ax.inspect}"
      lo = fold_axes(nested, ax) { |x, y| x < y ? x : y }
      hi = fold_axes(nested, ax) { |x, y| x < y ? y : x }
      ptp = lo.is_a?(Array) ? zip_deep(hi, lo) { |x, y| x - y } : hi - lo

      assert_equal(flat_list(fold_axes(nested, ax) { |x, y| x + y }),
                   view.sum(axis: ax).to_a.flatten, "sum of #{tag}")
      assert_equal(flat_list(lo), view.min(axis: ax).to_a.flatten, "min of #{tag}")
      assert_equal(flat_list(hi), view.max(axis: ax).to_a.flatten, "max of #{tag}")
      assert_equal(flat_list(ptp), view.ptp(axis: ax).to_a.flatten, "ptp of #{tag}")

      got_lo, got_hi = view.minmax(axis: ax)
      assert_equal(flat_list(lo), got_lo.to_a.flatten, "minmax low of #{tag}")
      assert_equal(flat_list(hi), got_hi.to_a.flatten, "minmax high of #{tag}")

      at_hi = fold_axes(positions, ax) { |p, q| flat[p] < flat[q] ? q : p }
      at_lo = fold_axes(positions, ax) { |p, q| flat[q] < flat[p] ? q : p }
      assert_equal(flat_list(at_hi), view.max_index(axis: ax).to_a.flatten, "max_index of #{tag}")
      assert_equal(flat_list(at_lo), view.min_index(axis: ax).to_a.flatten, "min_index of #{tag}")

      next unless ax.size == 1

      along = ->(p) { (p / strides[ax.first]) % shape[ax.first] }
      assert_equal(flat_list(map_deep(at_hi, &along)),
                   view.argmax(axis: ax.first).to_a.flatten, "argmax of #{tag}")
      assert_equal(flat_list(map_deep(at_lo, &along)),
                   view.argmin(axis: ax.first).to_a.flatten, "argmin of #{tag}")
    end
  end

  def bit_list(narray)
    narray.to_a.flatten.map(&:to_i)
  end

  # An array of length total with the named positions set, so a fill can be
  # checked without asking another cumo call what it should be
  def bits_set(total, positions, value = 1)
    a = Array.new(total, value ^ 1)
    positions.each { |i| a[i] = value }
    a
  end

  test "Bit results reach every element of a view they cannot flatten" do
    # A Bit element is one bit, so its position and steps are bit counts and
    # the byte indexer cannot address it. Comparisons, the isnan family and the
    # Bit operators each ran once per row of a view before the bit indexer.
    rows = 9
    cols = 7
    xs = Array.new(rows * cols) { |i| ((i * 3) % 13) - 6.0 }
    src = Cumo::DFloat.cast(xs).reshape(rows, cols)
    other = Cumo::DFloat.cast(xs.rotate(5)).reshape(rows, cols)
    m1 = Cumo::Int32.cast(Array.new(rows * cols) { |i| i % 3 }).reshape(rows, cols).gt(1)
    m2 = Cumo::Int32.cast(Array.new(rows * cols) { |i| i % 4 }).reshape(rows, cols).gt(1)
    idx = [5, 0, 3, 8, 1]

    views = {
      "column slice" => ->(a) { a[true, 1...(cols - 1)] },
      "reversed" => ->(a) { a[true, (cols - 1).step(0, -1)] },
      "row stride" => ->(a) { a[0.step(rows - 1, 2), true] },
      "index view" => ->(a) { a[idx, true] },
      "transpose" => ->(a) { a.transpose },
    }

    views.each do |what, take|
      a = take.call(src)
      b = take.call(other)
      p1 = take.call(m1)
      p2 = take.call(m2)
      fa = a.copy
      fb = b.copy
      f1 = p1.copy
      f2 = p2.copy

      %i[gt ge lt le eq ne].each do |op|
        assert_equal(fa.send(op, fb).to_a, a.send(op, b).to_a, "#{op} of #{what}")
      end
      %i[isnan isinf isfinite signbit].each do |op|
        assert_equal(fa.send(op).to_a, a.send(op).to_a, "#{op} of #{what}")
      end
      # The Bit operators are checked against the elements Ruby sees rather
      # than against a copy: copy is the same template, so a copy taken the
      # wrong way would agree with an operator taken the wrong way
      w1 = p1.to_a
      w2 = p2.to_a
      assert_equal(zip_deep(w1, w2) { |u, v| u & v }, (p1 & p2).to_a, "and of #{what}")
      assert_equal(zip_deep(w1, w2) { |u, v| u | v }, (p1 | p2).to_a, "or of #{what}")
      assert_equal(zip_deep(w1, w2) { |u, v| u ^ v }, (p1 ^ p2).to_a, "xor of #{what}")
      assert_equal(zip_deep(w1, w1) { |u, _| u ^ 1 }, (~p1).to_a, "not of #{what}")
      assert_equal(w1.flatten.count(1), p1.count_true, "count_true of #{what}")
      assert_equal(w1, f1.to_a, "copy of #{what}")

      cells = take.call(Cumo::DFloat.cast((0...(rows * cols)).to_a).reshape(rows, cols))
                  .to_a.flatten.map(&:to_i)
      d = Cumo::Bit.new(rows, cols).fill(0)
      take.call(d).fill(1)
      assert_equal(Array.new(rows * cols) { |i| cells.include?(i) ? 1 : 0 }, d.to_a.flatten,
                   "Bit fill of #{what}")
    end

    # One dimension is addressed through the indexer's raw index rather than
    # its per-dimension indices, a path no two-dimensional view reaches
    len = 40
    ys = Array.new(len) { |i| ((i * 5) % 17) - 8.0 }
    flat = Cumo::DFloat.cast(ys)
    flat2 = Cumo::DFloat.cast(ys.rotate(3))
    bits1 = Cumo::Int32.cast(Array.new(len) { |i| i % 3 }).gt(1)
    bits2 = Cumo::Int32.cast(Array.new(len) { |i| i % 4 }).gt(1)

    one_d = {
      "step 3" => ->(a) { a[0.step(len - 1, 3)] },
      "reversed" => ->(a) { a[(len - 1).step(0, -1)] },
      "reversed by 2" => ->(a) { a[(len - 1).step(0, -2)] },
      "fancy" => ->(a) { a[[7, 2, 39, 0, 15, 15]] },
      "tail" => ->(a) { a[3..-4] },
    }

    one_d.each do |what, take|
      a = take.call(flat)
      b = take.call(flat2)
      p1 = take.call(bits1)
      p2 = take.call(bits2)

      assert_equal(a.copy.gt(b.copy).to_a, a.gt(b).to_a, "1-d gt of #{what}")
      assert_equal(a.copy.signbit.to_a, a.signbit.to_a, "1-d signbit of #{what}")
      w1 = p1.to_a
      w2 = p2.to_a
      assert_equal(w1.zip(w2).map { |u, v| u & v }, (p1 & p2).to_a, "1-d and of #{what}")
      assert_equal(w1.map { |u| u ^ 1 }, (~p1).to_a, "1-d not of #{what}")
      assert_equal(w1.count(1), p1.count_true, "1-d count_true of #{what}")
      assert_equal(w1, p1.copy.to_a, "1-d copy of #{what}")

      cells = take.call(Cumo::DFloat.cast((0...len).to_a)).to_a.map(&:to_i)
      d = Cumo::Bit.new(len).fill(0)
      take.call(d).fill(1)
      assert_equal(Array.new(len) { |i| cells.include?(i) ? 1 : 0 }, d.to_a, "1-d Bit fill of #{what}")
    end

    # a 5-d shape runs past the dimension-specialised kernels
    deep = [2, 2, 3, 2, 5]
    n = deep.reduce(:*)
    zs = Array.new(n) { |i| ((i * 7) % 11) - 5.0 }
    a = Cumo::DFloat.cast(zs).reshape(*deep)
    assert_equal(zs.map { |x| x > 0 ? 1 : 0 }, a.gt(0).to_a.flatten)
    assert_equal(a.transpose.copy.gt(0).to_a.flatten, a.transpose.gt(0).to_a.flatten)
    bd = Cumo::Int32.cast(Array.new(n) { |i| i % 3 }).reshape(*deep).gt(1)
    assert_equal(bd.transpose.to_a.flatten.map { |u| u ^ 1 }, (~bd.transpose).to_a.flatten)
  end

  test "Bit stores a Ruby Array into the bits of a view and no others" do
    # The store walked the bits from the host behind a device synchronization.
    # It also declared the step unsigned, which stopped CUMO_INIT_PTR_BIT_IDX
    # from leaving the position alone for a view that walks backwards, so a
    # reversed view starting past the first word walked below the array.
    [1, 31, 32, 33, 64, 65, 100].each do |n|
      vals = Array.new(n) { |i| (i * 7) % 3 == 0 ? 1 : 0 }
      assert_equal(vals, bit_list(Cumo::Bit.cast(vals)), "cast #{n}")

      a = Cumo::Bit.new(n).fill(1)
      a.store(vals[0, n / 2])
      assert_equal(vals[0, n / 2] + Array.new(n - n / 2, 0), bit_list(a), "source shorter than #{n}")

      a = Cumo::Bit.new(n).fill(0)
      a.store(vals + vals)
      assert_equal(vals, bit_list(a), "source longer than #{n}")
    end

    # a run that starts and ends inside a word
    # the last two are a whole number of words that does not start on one, the
    # shape the word-at-a-time path must not take
    [[3, 40], [31, 33], [32, 64], [0, 99], [63, 65], [3, 34], [33, 96]].each do |b, e|
      vals = Array.new(e - b + 1) { |i| i % 2 }
      a = Cumo::Bit.new(100).fill(0)
      a[b..e].store(vals)
      want = Array.new(100, 0)
      vals.each_with_index { |v, i| want[b + i] = v }
      assert_equal(want, bit_list(a), "#{b}..#{e}")
    end

    a = Cumo::Bit.new(8, 9).fill(0)
    a[true, 0.step(8, 3)].store(Array.new(8) { [1, 0, 1] })
    assert_equal(bits_set(72, (0...8).flat_map { |r| [r * 9, r * 9 + 6] }), bit_list(a), "column slice")

    rows = [3, 0, 5]
    a = Cumo::Bit.new(8, 9).fill(0)
    a[rows, true].store(rows.map { Array.new(9, 1) })
    assert_equal(bits_set(72, rows.flat_map { |r| (0...9).map { |j| r * 9 + j } }), bit_list(a), "index view")

    # a reversed view whose first bit is past the first word
    [10, 33, 70, 200].each do |n|
      vals = Array.new(n) { |i| i % 4 == 0 ? 1 : 0 }
      a = Cumo::Bit.new(n).fill(0)
      a[(n - 1).step(0, -1)].store(vals)
      want = Array.new(n)
      vals.each_with_index { |v, i| want[n - 1 - i] = v }
      assert_equal(want, bit_list(a), "reversed #{n}")
    end

    a = Cumo::Bit.new(70).fill(0)
    a[69.step(0, -3)].store(Array.new(24) { |i| i % 2 })
    want = Array.new(70, 0)
    24.times { |i| want[69 - i * 3] = i % 2 }
    assert_equal(want, bit_list(a), "reversed with a step")

    # a Range element expands, and what is left of the row is zeroed
    a = Cumo::Bit.new(8).fill(1)
    a.store([(0..3)])
    assert_equal([0, 1, 1, 1, 0, 0, 0, 0], bit_list(a), "Range")

    a = Cumo::Bit.new(6).fill(1)
    a.store([1, nil, 1])
    assert_equal([1, 0, 1, 0, 0, 0], bit_list(a), "nil")

    a = Cumo::Bit.new(8).fill(1)
    a.store([])
    assert_equal(Array.new(8, 0), bit_list(a), "empty")

    # an element of the Array that is a narray of its own
    a = Cumo::Bit.new(2, 10).fill(1)
    a.store([Cumo::Bit.cast([1, 0, 1]), Cumo::Bit.cast([0, 1])])
    assert_equal([1, 0, 1] + Array.new(7, 0) + [0, 1] + Array.new(8, 0), bit_list(a), "sub-narray")

    # an index array on the innermost dimension, which the loop hands to the
    # store itself rather than walking outside it
    order = [7, 2, 5, 0, 8]
    a = Cumo::Bit.new(3, 9).fill(0)
    a[true, order].store(Array.new(3) { |r| Array.new(order.size) { |j| (r + j) % 2 } })
    want = Array.new(27, 0)
    3.times { |r| order.each_with_index { |col, j| want[r * 9 + col] = (r + j) % 2 } }
    assert_equal(want, bit_list(a), "index on the innermost axis")

    a = Cumo::Bit.new(3, 9).fill(1)
    a[true, order].store(Array.new(3) { [1, 0] })
    want = Array.new(27, 1)
    3.times { |r| order.each_with_index { |col, j| want[r * 9 + col] = (j < 2 ? [1, 0][j] : 0) } }
    assert_equal(want, bit_list(a), "index on the innermost axis, rows shorter than it")

    # the zero fill of a row shorter than the destination goes through the
    # index array as well
    a = Cumo::Bit.new(8, 9).fill(1)
    a[[3, 0], true].store([[1, 0, 1], [0, 1]])
    want = Array.new(72, 1)
    [1, 0, 1, 0, 0, 0, 0, 0, 0].each_with_index { |v, j| want[3 * 9 + j] = v }
    [0, 1, 0, 0, 0, 0, 0, 0, 0].each_with_index { |v, j| want[j] = v }
    assert_equal(want, bit_list(a), "index view with rows shorter than the destination")
  end

  test "Bit#fill reaches the bits of a view and no others" do
    # fill was the last host loop that a plain Cumo::Bit.new(n).fill(1) ran
    # into. It synchronized with the device and wrote the words from the CPU,
    # which left the pages there for the next kernel to fault back.
    [1, 31, 32, 33, 70, 200].each do |n|
      a = Cumo::Bit.new(n).fill(0)
      a.fill(1)
      assert_equal(Array.new(n, 1), bit_list(a), "whole #{n}")
      a.fill(0)
      assert_equal(Array.new(n, 0), bit_list(a), "whole #{n} back to zero")
    end

    # a run that starts and ends inside a word, so the words at either end are
    # shared with elements the fill must not touch
    [[3, 40], [31, 33], [32, 64], [0, 99], [63, 65]].each do |b, e|
      a = Cumo::Bit.new(100).fill(0)
      a[b..e].fill(1)
      assert_equal(bits_set(100, (b..e).to_a), bit_list(a), "#{b}..#{e}")

      a = Cumo::Bit.new(100).fill(1)
      a[b..e].fill(0)
      assert_equal(bits_set(100, (b..e).to_a, 0), bit_list(a), "#{b}..#{e} to zero")
    end

    a = Cumo::Bit.new(8, 9).fill(0)
    a[true, 0.step(8, 3)].fill(1)
    assert_equal(bits_set(72, (0...8).flat_map { |r| [0, 3, 6].map { |c| r * 9 + c } }),
                 bit_list(a), "column slice")

    # rows of an index view share a word with rows outside it
    rows = [3, 0, 5]
    a = Cumo::Bit.new(8, 9).fill(0)
    a[rows, true].fill(1)
    assert_equal(bits_set(72, rows.flat_map { |r| (0...9).map { |c| r * 9 + c } }),
                 bit_list(a), "index view")

    a = Cumo::Bit.new(70).fill(0)
    a[60.step(10, -1)].fill(1)
    assert_equal(bits_set(70, (10..60).to_a), bit_list(a), "reversed")

    a = Cumo::Bit.new(3, 5, 7).fill(0)
    a[true, 2, true].fill(1)
    assert_equal(bits_set(105, (0...3).flat_map { |p| (0...7).map { |c| p * 35 + 14 + c } }),
                 bit_list(a), "3-d row of every plane")

    assert_raise(ArgumentError) { Cumo::Bit.new(4).fill(2) }
  end

  test "storing an Array of narrays zeroes the rest of each row" do
    # The zero fill was placed one whole sub-narray past where it belonged: the
    # branch that copies the sub-narray advanced the pointer, and the kernel it
    # launches was then handed the same offset a second time. On the last row
    # that runs off the end of the array, which Ruby cannot see; what it can see
    # is that the fill lands in the row after the one it belongs to.
    rows = 3
    cols = 7
    dtypes = [Cumo::Int32, Cumo::Int64, Cumo::UInt8, Cumo::SFloat, Cumo::DFloat, Cumo::RObject]

    dtypes.each do |dtype|
      whole = [dtype.new(cols).seq, dtype[1, 2, 3], dtype[4, 5]]
      want = [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 0, 0, 0, 0], [4, 5, 0, 0, 0, 0, 0]]

      dst = dtype.new(rows, cols).fill(9)
      dst.store(whole)
      assert_equal(want, ints_of(dst), "#{dtype} sub-narray rows")

      # the same rows as a Ruby Array take the other branch, which never
      # advanced and so has to keep working unchanged
      dst = dtype.new(rows, cols).fill(9)
      dst.store([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3], [4, 5]])
      assert_equal(want, ints_of(dst), "#{dtype} Array rows")

      dst = dtype.new(rows, cols).fill(9)
      dst[true, (cols - 1).step(0, -1)].store(whole)
      assert_equal([[6, 5, 4, 3, 2, 1, 0], [0, 0, 0, 0, 3, 2, 1], [0, 0, 0, 0, 0, 5, 4]],
                   ints_of(dst), "#{dtype} reversed destination")

      dst = dtype.new(rows, cols).fill(9)
      dst.store([dtype[1, 2, 3], nil, dtype[4, 5]])
      assert_equal([[1, 2, 3, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [4, 5, 0, 0, 0, 0, 0]],
                   ints_of(dst), "#{dtype} nil row")

      dst = dtype.new(rows, cols).fill(9)
      dst.store([dtype[1, 2, 3], 5, dtype[4, 5]])
      assert_equal([[1, 2, 3, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0, 0], [4, 5, 0, 0, 0, 0, 0]],
                   ints_of(dst), "#{dtype} scalar row")

      # a sub-narray longer than the row fills it and nothing else
      dst = dtype.new(2, 3).fill(9)
      dst.store([dtype.new(6).seq, dtype.new(5).seq])
      assert_equal([[0, 1, 2], [0, 1, 2]], ints_of(dst), "#{dtype} row shorter than the sub-narray")
    end
  end

  test "median takes the middle of every row, whatever the axis" do
    # median sorted on the host, one row at a time, behind a device
    # synchronization. The expectations are worked out in Ruby, and the values
    # stay small so that the average of the middle two cannot overflow the
    # narrower integer types.
    rows = 12
    cols = 25
    xs = Array.new(rows * cols) { |i| (i * 37) % 53 }
    middle = lambda do |vals, int|
      sorted = vals.sort
      n = sorted.size
      return sorted[(n - 1) / 2] if n.odd?

      pair = sorted[(n / 2) - 1] + sorted[n / 2]
      int ? pair / 2 : pair / 2.0
    end

    [Cumo::Int8, Cumo::Int32, Cumo::UInt8, Cumo::UInt32, Cumo::SFloat, Cumo::DFloat].each do |dtype|
      int = dtype != Cumo::SFloat && dtype != Cumo::DFloat
      src = dtype.cast(xs).reshape(rows, cols)
      table = src.to_a.map { |r| r.map(&:to_i) }

      assert_equal([middle.call(table.flatten, int)], src.median.to_a, "#{dtype} flat")
      assert_equal(table.map { |r| middle.call(r, int) }, src.median(axis: 1).to_a, "#{dtype} axis 1")
      assert_equal(table.transpose.map { |r| middle.call(r, int) }, src.median(axis: 0).to_a, "#{dtype} axis 0")
      assert_equal(table.map { |r| [middle.call(r, int)] }, src.median(axis: 1, keepdims: true).to_a,
                   "#{dtype} keepdims")

      view = src[true, 0.step(cols - 1, 3)]
      assert_equal(view.to_a.map { |r| middle.call(r.map(&:to_i), int) }, view.median(axis: 1).to_a,
                   "#{dtype} column slice")

      cube = dtype.cast(xs[0, 2 * 5 * 3]).reshape(2, 5, 3)
      want = map_deep(cube.to_a, &:to_i).map { |plane| plane.transpose.map { |r| middle.call(r, int) } }
      assert_equal(want, cube.median(axis: 1).to_a, "#{dtype} 3-d axis 1")

      # an odd row length takes the other branch
      odd = dtype.cast(xs[0, rows * (cols - 2)]).reshape(rows, cols - 2)
      assert_equal(odd.to_a.map { |r| middle.call(r.map(&:to_i), int) }, odd.median(axis: 1).to_a,
                   "#{dtype} odd row length")
    end

    # trailing NaNs are dropped before the middle is taken, which is what numo
    # 0.9 does; numo-narray-alt lost that in a rewrite and answers 3.0 here
    nan = Float::NAN
    [Cumo::SFloat, Cumo::DFloat].each do |dtype|
      assert_equal([2.0], dtype.cast([3.0, nan, 1.0, nan, 2.0]).median.to_a, "#{dtype} NaN dropped")
      assert(dtype.cast([nan, nan]).median.to_a.first.nan?, "#{dtype} all NaN")
      assert_equal([5.0], dtype.cast([5.0]).median.to_a, "#{dtype} one element")
    end
  end

  # Where each element of a row sits once the row is in order, ties left in the
  # order they came and NaN last. The kernel's radix sort is stable, so this is
  # an exact expectation rather than one of several valid answers.
  def sort_index_expect(view, axis)
    flat = view.to_a.flatten
    shape = view.shape
    strides = shape.each_index.map { |i| shape[(i + 1)..].inject(1, :*) }
    rank = ->(p) { [flat[p].is_a?(Float) && flat[p].nan? ? 1 : 0, flat[p].is_a?(Float) && flat[p].nan? ? 0 : flat[p]] }
    want = Array.new(flat.size)
    flat.each_index.group_by { |p|
      shape.each_index.reject { |d| axis.include?(d) }.map { |d| (p / strides[d]) % shape[d] }
    }.each_value do |row|
      order = row.sort_by.with_index { |p, i| rank.call(p) + [i] }
      row.each_with_index { |p, i| want[p] = order[i] }
    end
    want
  end

  def assert_sort_index(view, what, axes: nil)
    axes ||= (0...view.ndim).map { |i| [i] } + [(0...view.ndim).to_a]
    axes.each do |axis|
      got = axis.size == view.ndim ? view.sort_index : view.sort_index(axis: axis)
      assert_equal(sort_index_expect(view, axis), got.to_a.flatten,
                   "#{what} sort_index axis #{axis.inspect}")
    end
  end

  test "sort orders every row, whatever the axis and the layout" do
    # The sort ran on the host, one row at a time, behind a device
    # synchronization. The expectations are sorted in Ruby rather than taken
    # from another cumo call.
    rows = 12
    cols = 25
    n = rows * cols
    xs = Array.new(n) { |i| ((i * 37) % 101) - 50 }
    int_types = [Cumo::Int8, Cumo::Int16, Cumo::Int32, Cumo::Int64, Cumo::UInt8, Cumo::UInt32]

    (int_types + [Cumo::SFloat, Cumo::DFloat]).each do |dtype|
      vals = dtype == Cumo::UInt8 || dtype == Cumo::UInt32 ? xs.map(&:abs) : xs
      src = dtype.cast(vals).reshape(rows, cols)
      table = src.to_a.map { |r| r.map(&:to_i) }

      assert_equal(table.flatten.sort, ints_of(src.sort).flatten, "#{dtype} flat")
      assert_equal(table.map(&:sort), ints_of(src.sort(axis: 1)), "#{dtype} axis 1")
      assert_equal(table.transpose.map(&:sort).transpose, ints_of(src.sort(axis: 0)), "#{dtype} axis 0")

      # a view the rows of which are not laid out end to end
      view = src[true, 0.step(cols - 1, 3)]
      assert_equal(view.to_a.map { |r| r.map(&:to_i).sort }, ints_of(view.sort(axis: 1)), "#{dtype} column slice")

      rev = src[(rows - 1).step(0, -1), true]
      assert_equal(rev.to_a.map { |r| r.map(&:to_i).sort }, ints_of(rev.sort(axis: 1)), "#{dtype} reversed rows")

      idx = src[[7, 0, 4, 11, 2], true]
      assert_equal(idx.to_a.map { |r| r.map(&:to_i).sort }, ints_of(idx.sort(axis: 1)), "#{dtype} index view")

      # sorted in place there is nowhere but the index array to put the answer
      fresh = dtype.cast(vals).reshape(rows, cols)
      fresh[[7, 0, 4, 11, 2], true].inplace.sort(axis: 1)
      assert_equal(idx.to_a.map { |r| r.map(&:to_i).sort },
                   ints_of(fresh[[7, 0, 4, 11, 2], true]), "#{dtype} index view in place")
      assert_equal(ints_of(src[[1, 3, 5, 6, 8, 9, 10], true]),
                   ints_of(fresh[[1, 3, 5, 6, 8, 9, 10], true]), "#{dtype} rows outside the index view")

      # a 3-d shape, and an axis that is neither the first nor the last
      cube = dtype.cast(vals[0, 2 * 5 * 3]).reshape(2, 5, 3)
      want = map_deep(cube.to_a, &:to_i).map { |plane| plane.transpose.map(&:sort).transpose }
      assert_equal(want, map_deep(cube.sort(axis: 1).to_a, &:to_i), "#{dtype} 3-d axis 1")
    end

    # every NaN sorts last whatever its sign, and -0.0 sorts below +0.0
    nan = Float::NAN
    neg_nan = Float::INFINITY - Float::INFINITY
    [Cumo::SFloat, Cumo::DFloat].each do |dtype|
      got = dtype.cast([3.0, nan, 1.0, neg_nan, 2.0]).sort.to_a
      assert_equal([1.0, 2.0, 3.0], got[0, 3], "#{dtype} NaN last")
      assert(got[3].nan? && got[4].nan?, "#{dtype} NaN last")

      zeros = dtype.cast([0.0, -1.0, -0.0, 1.0]).sort.to_a
      assert_equal([-1.0, 0.0, 0.0, 1.0], zeros, "#{dtype} signed zero")
      assert_equal([1, 1, 0, 0], zeros.map { |x| (1.0 / x).negative? ? 1 : 0 }, "#{dtype} signed zero order")
    end

    assert_equal([1, 2, 3], Cumo::Int32[3, 1, 2].sort.to_a, "smallest case")
    assert_equal([7], Cumo::Int32[7].sort.to_a, "one element")
  end

  # Horner from Ruby, so the expectation does not come from another cumo call.
  # A coefficient is either one number or a value per element.
  def poly_expect(values, coefs)
    values.each_with_index.map do |x, i|
      cs = coefs.map { |c| c.is_a?(Array) ? c[i] : c }
      cs.empty? ? x : cs.reverse.reduce { |y, c| y * x + c }
    end
  end

  test "sort_index says where every element of a row came from" do
    # The sort ran on the host, one row at a time, behind a device
    # synchronization, and it read the array through its own index array, which
    # the indexer loop cannot address.
    rows = 12
    cols = 25
    n = rows * cols
    xs = Array.new(n) { |i| ((i * 37) % 101) - 50 }
    int_types = [Cumo::Int8, Cumo::Int16, Cumo::Int32, Cumo::Int64, Cumo::UInt8, Cumo::UInt32]

    (int_types + [Cumo::SFloat, Cumo::DFloat]).each do |dtype|
      vals = dtype == Cumo::UInt8 || dtype == Cumo::UInt32 ? xs.map(&:abs) : xs
      src = dtype.cast(vals).reshape(rows, cols)
      assert_sort_index(src, dtype.to_s)
      assert_sort_index(src[true, 0.step(cols - 1, 3)], "#{dtype} column slice")
      assert_sort_index(src[(rows - 1).step(0, -1), true], "#{dtype} reversed rows")
      assert_sort_index(src[[7, 0, 4, 11, 2], true], "#{dtype} index view")

      cube = dtype.cast(vals[0, 2 * 5 * 3]).reshape(2, 5, 3)
      assert_sort_index(cube, "#{dtype} 3-d", axes: [[0], [1], [2], [0, 1], [1, 2], [0, 1, 2]])
    end

    # every NaN sorts last whatever its sign, and -0.0 sorts below +0.0
    nan = Float::NAN
    neg_nan = Float::INFINITY - Float::INFINITY
    [Cumo::SFloat, Cumo::DFloat].each do |dtype|
      got = dtype.cast([3.0, nan, 1.0, neg_nan, 2.0]).sort_index.to_a
      assert_equal([2, 4, 0], got[0, 3], "#{dtype} NaN last")
      assert_equal([1, 3], got[3, 2].sort, "#{dtype} NaN last")

      assert_equal([1, 2, 0, 3], dtype.cast([0.0, -1.0, -0.0, 1.0]).sort_index.to_a,
                   "#{dtype} signed zero")
    end

    # nan:true keeps the host path, and it answers what numo answers there
    assert_equal([0, 1, 2, 3], Cumo::DFloat[3.0, nan, 1.0, 2.0].sort_index(nan: true).to_a,
                 "nan:true")

    assert_equal([0], Cumo::Int32[7].sort_index.to_a, "one element")
    assert_equal([1, 2, 0], Cumo::Int32[3, 1, 2].sort_index.to_a, "smallest case")
  end

  test "real= and imag= reach every element of a view" do
    # The two ran on the host, one element at a time, behind a device
    # synchronization; they were the last of gen/tmpl to do so.
    rows = 6
    cols = 10
    n = rows * cols
    re = Array.new(n) { |i| (i % 7) - 3.0 }
    im = Array.new(n) { |i| (i % 5) - 2.0 }
    fresh = Array.new(n) { |i| (i % 11) - 5.0 }

    [[Cumo::DComplex, Cumo::DFloat], [Cumo::SComplex, Cumo::SFloat]].each do |ct, rt|
      src = -> { ct.cast(re.each_index.map { |i| Complex(re[i], im[i]) }).reshape(rows, cols) }

      a = src.call
      a.real = rt.cast(fresh).reshape(rows, cols)
      assert_equal(fresh.each_index.map { |i| Complex(fresh[i], im[i]) }, a.to_a.flatten, "#{ct} real= array")

      a = src.call
      a.imag = rt.cast(fresh).reshape(rows, cols)
      assert_equal(fresh.each_index.map { |i| Complex(re[i], fresh[i]) }, a.to_a.flatten, "#{ct} imag= array")

      a = src.call
      a.real = 7.0
      assert_equal(re.each_index.map { |i| Complex(7.0, im[i]) }, a.to_a.flatten, "#{ct} real= scalar")

      # a view whose elements are not laid out end to end
      picked = (0...cols).step(3).to_a
      a = src.call
      a[true, 0.step(cols - 1, 3)].imag = 9.0
      want = re.each_index.map { |i| Complex(re[i], picked.include?(i % cols) ? 9.0 : im[i]) }
      assert_equal(want, a.to_a.flatten, "#{ct} imag= of a column slice")

      order = [3, 0, 5]
      a = src.call
      a[order, true].real = 4.0
      want = re.each_index.map { |i| Complex(order.include?(i / cols) ? 4.0 : re[i], im[i]) }
      assert_equal(want, a.to_a.flatten, "#{ct} real= of an index view")

      a = src.call
      a[(rows - 1).step(0, -1), true].imag = 2.5
      assert_equal(re.each_index.map { |i| Complex(re[i], 2.5) }, a.to_a.flatten, "#{ct} imag= of a reversed view")

      # a shape past the rank the indexer has a specialized accessor for
      deep = ct.cast((0...32).map { |i| Complex(re[i], im[i]) }).reshape(2, 2, 2, 2, 2)
      deep.real = rt.cast(fresh[0, 32]).reshape(2, 2, 2, 2, 2)
      assert_equal((0...32).map { |i| Complex(fresh[i], im[i]) }, deep.to_a.flatten, "#{ct} real= 5-d")

      one = ct.cast([Complex(1, 2)])
      one.imag = 8.0
      assert_equal([Complex(1, 8)], one.to_a, "#{ct} smallest case")
    end
  end

  test "poly evaluates every element, whatever the coefficients and the layout" do
    # poly ran with CUMO_NO_LOOP, so ndloop called the iterator once per
    # element and each call synchronized with the device.
    rows = 6
    cols = 10
    small = Array.new(rows * cols) { |i| (i * 13) % 3 }
    wide = Array.new(rows * cols) { |i| (i * 13) % 5 }

    # every dtype, with values an 8-bit result still holds
    [Cumo::Int8, Cumo::Int32, Cumo::UInt8, Cumo::UInt32,
     Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex, Cumo::RObject].each do |dtype|
      src = dtype.cast(small).reshape(rows, cols)
      assert_equal(poly_expect(small, []), src.poly.to_a.flatten, "#{dtype} no coefficient")
      assert_equal(poly_expect(small, [3]), src.poly(3).to_a.flatten, "#{dtype} one coefficient")
      assert_equal(poly_expect(small, [1, 2, 3]), src.poly(1, 2, 3).to_a.flatten, "#{dtype} three")
      assert_equal(poly_expect(small, [small, 2]), src.poly(src, 2).to_a.flatten,
                   "#{dtype} a coefficient that varies per element")
    end

    [Cumo::Int32, Cumo::Int64, Cumo::SFloat, Cumo::DFloat, Cumo::DComplex].each do |dtype|
      src = dtype.cast(wide).reshape(rows, cols)
      coefs = [1, 2, 3, 4, 5, 6]
      assert_equal(poly_expect(wide, coefs), src.poly(*coefs).to_a.flatten, "#{dtype} six coefficients")

      view = src[true, 0.step(cols - 1, 3)]
      assert_equal(poly_expect(view.to_a.flatten, [1, 2, 3]), view.poly(1, 2, 3).to_a.flatten,
                   "#{dtype} column slice")

      rev = src[(rows - 1).step(0, -1), true]
      assert_equal(poly_expect(rev.to_a.flatten, [1, 2, 3]), rev.poly(1, 2, 3).to_a.flatten,
                   "#{dtype} reversed rows")

      idx = src[[3, 0, 5, 1], true]
      assert_equal(poly_expect(idx.to_a.flatten, [1, 2, 3]), idx.poly(1, 2, 3).to_a.flatten,
                   "#{dtype} index view")
      assert_equal(poly_expect(idx.to_a.flatten, [idx.to_a.flatten, 2]), idx.poly(idx, 2).to_a.flatten,
                   "#{dtype} index view with a coefficient of its own")

      # a shape past the dimension the indexer has a specialized accessor for
      deep = dtype.cast(wide[0, 2 * 2 * 2 * 2 * 2]).reshape(2, 2, 2, 2, 2)
      assert_equal(poly_expect(deep.to_a.flatten, [1, 2, 3]), deep.poly(1, 2, 3).to_a.flatten,
                   "#{dtype} 5-d")
    end

    assert_equal([17], Cumo::Int32[2].poly(1, 2, 3).to_a, "smallest case")
  end

  test "a store from a Ruby Array waits for the kernel that fills an index array" do
    # ndloop finds each row of an index-backed view by reading its index array
    # on the host, and a kernel writes that array. Without the wait the store
    # read whatever the memory pool had left there, and every row of
    # a[[2, 1], true] landed on row 0.
    rows = 3
    cols = 5
    warm = lambda do
      3.times do
        x = Cumo::Int32.zeros(rows, cols)
        x[[0], true].store([[9] * cols])
      end
    end
    values = [[1] * cols, [2] * cols]
    want = [[0] * cols, [2] * cols, [1] * cols]

    warm.call
    a = Cumo::Int32.zeros(rows, cols)
    a[[2, 1], true].store(values)
    assert_equal(want, a.to_a, "store")

    warm.call
    a = Cumo::Int32.zeros(rows, cols)
    a[[2, 1], true] = values
    assert_equal(want, a.to_a, "assignment")

    warm.call
    b = Cumo::Bit.new(rows, cols).fill(0)
    b[[2, 1], true].store([[1] * cols, [0] * cols])
    assert_equal([[0] * cols, [0] * cols, [1] * cols], b.to_a.map { |r| r.map(&:to_i) }, "Bit")

    # a sub-narray of the Array binds its own index array inside the loop,
    # after the entry point has already looked for one
    n = 40
    warm.call
    m = Cumo::Int32.new(4, n).seq
    d = Cumo::Int32.zeros(2, 2, n)
    d.store([m[[3, 1], true], m[[0, 2], true]])
    row = ->(k) { (k * n...(k + 1) * n).to_a }
    assert_equal([[row.call(3), row.call(1)], [row.call(0), row.call(2)]], d.to_a, "sub-narray")
  end

  test "an RObject loop waits for the kernel that fills an index array" do
    # Index arrays are filled by a kernel. Every dtype but RObject hands the
    # index on to a kernel of its own, which the stream already orders, but
    # RObject keeps its data on the host and reads the index there. Without a
    # wait the second such store onwards read an index the kernel had not
    # written yet: zeroes, then whatever the chunk held before, and a bus error
    # when that was not a valid address.
    cols = 7
    order = [6, 0, 5, 1, 4, 2, 3]
    rows = [[0, 1, 1, 0, 1, 1, 0], [1, 0, 1], [0, 1]]
    scatter = lambda do |vals|
      out = Array.new(cols, 0)
      cols.times { |p| out[order[p]] = vals[p] || 0 }
      out
    end
    want = rows.map(&scatter)

    # A fresh view every time, so the index array comes back out of the pool
    # rather than being allocated once and reused. Nothing may run between
    # taking the view and the store, or the kernel finishes on its own and the
    # window closes.
    4.times do |i|
      srcs = rows.map { |v| Cumo::RObject.cast(v) }
      dst = Cumo::RObject.new(rows.size, cols).fill(0)
      dst[true, order].store(srcs)
      assert_equal(want, ints_of(dst), "Array of narrays (#{i})")
    end

    src = Cumo::Int32.new(3, cols).seq
    want_src = src.to_a.map(&scatter)
    4.times do |i|
      dst = Cumo::RObject.new(3, cols).fill(0)
      dst[true, order].store(src)
      assert_equal(want_src, ints_of(dst), "RObject <- Int32 (#{i})")
    end

    bits = src.gt(9)
    want_bits = bits.to_a.map(&scatter)
    4.times do |i|
      dst = Cumo::RObject.new(3, cols).fill(0)
      dst[true, order].store(bits)
      assert_equal(want_bits, ints_of(dst), "RObject <- Bit (#{i})")
    end

    4.times do |i|
      dst = Cumo::RObject.new(3, cols).fill(0)
      dst[true, order].fill(9)
      assert_equal(Array.new(3) { Array.new(cols, 9) }, ints_of(dst), "RObject fill (#{i})")
    end
  end

  test "an index array reaches the store an Array of narrays runs" do
    # store_array runs the same-type store inside a loop of its own, which keeps
    # CUMO_NDF_INDEX_LOOP on, so an index array reaches a kernel whose iarray
    # carries byte steps only. An indexed operand has a step of zero there, so
    # every element of the row went to one address.
    cols = 7
    order = [6, 0, 5, 1, 4, 2, 3]
    rowvals = [[0, 1, 1, 0, 1, 1, 0], [1, 0, 1], [0, 1]]

    # Cumo::RObject keeps its data on the host and takes a loop of its own,
    # which addresses an index already
    [Cumo::Int32, Cumo::Int64, Cumo::UInt8, Cumo::DFloat, Cumo::Bit].each do |dtype|
      srcs = rowvals.map { |v| dtype.cast(v) }
      want = srcs.map do |src|
        vals = src.to_a.map(&:to_i)
        line = Array.new(cols) { |p| vals[p] || 0 }
        out = Array.new(cols)
        cols.times { |p| out[order[p]] = line[p] }
        out
      end

      dst = dtype.new(srcs.size, cols).fill(0)
      dst[true, order].store(srcs)
      assert_equal(want, ints_of(dst), "#{dtype} indexed destination")

      dst = dtype.new(srcs.size, cols).fill(0)
      dst[true, order] = srcs
      assert_equal(want, ints_of(dst), "#{dtype} indexed destination through []=")

      picked = dtype.cast([0, 1, 1, 0, 1])[[2, 0, 1]]
      one = dtype.new(1, 3).fill(0)
      one.store([picked])
      assert_equal([picked.to_a.map(&:to_i)], ints_of(one), "#{dtype} indexed sub-narray")
    end
  end

  test "stores between Bit and another dtype reach every element of a view" do
    # A Bit element is one bit, so the byte iarray cannot address it and ndloop
    # cannot stage it into a buffer either. Both directions ran once per row of
    # a view, and the Bit destination ran on the host. Expectations are built in
    # Ruby, not read off a contiguous copy, which shares the templates here.
    rows = 9
    cols = 7
    n = rows * cols
    xs = Array.new(n) { |i| ((i * 5) % 17) - 8 }
    src = Cumo::Int32.cast(xs).reshape(rows, cols)
    bits = src.gt(0)
    idx = [5, 0, 3, 8, 1]

    views = {
      "contiguous" => ->(a) { a },
      "column slice" => ->(a) { a[true, 1...(cols - 1)] },
      "reversed" => ->(a) { a[true, (cols - 1).step(0, -1)] },
      "row stride" => ->(a) { a[0.step(rows - 1, 2), true] },
      "index view" => ->(a) { a[idx, true] },
      "transpose" => ->(a) { a.transpose },
    }

    views.each do |what, take|
      iv = take.call(src)
      bv = take.call(bits)
      want_bit = map_deep(iv.to_a) { |x| x.positive? ? 1 : 0 }
      cells = take.call(Cumo::Int32.cast((0...n).to_a).reshape(rows, cols)).to_a.flatten

      assert_equal(want_bit, bv.to_a, "bit view #{what}")

      [Cumo::Int32, Cumo::DFloat].each do |k|
        assert_equal(want_bit, k.cast(bv).to_a.then { |a| map_deep(a, &:to_i) }, "#{k}.cast of #{what}")
        dst = k.zeros(*bv.shape)
        dst.store(bv)
        assert_equal(want_bit, map_deep(dst.to_a, &:to_i), "#{k}.store bit of #{what}")

        whole = k.zeros(rows, cols)
        take.call(whole).store(bv)
        want = Array.new(n, 0)
        cells.each_with_index { |cell, i| want[cell] = want_bit.flatten[i] }
        assert_equal(want, map_deep(whole.to_a, &:to_i).flatten, "#{k} view <- bit of #{what}")
      end

      [iv, Cumo::DFloat.cast(iv)].each do |source|
        dst = Cumo::Bit.new(*source.shape)
        dst.store(source)
        assert_equal(map_deep(source.to_a) { |x| x.zero? ? 0 : 1 }, dst.to_a,
                     "bit.store #{source.class} of #{what}")

        whole = Cumo::Bit.new(rows, cols).fill(1)
        take.call(whole).store(source)
        want = Array.new(n, 1)
        flat = map_deep(source.to_a) { |x| x.zero? ? 0 : 1 }.flatten
        cells.each_with_index { |cell, i| want[cell] = flat[i] }
        assert_equal(want, whole.to_a.flatten, "bit view <- #{source.class} of #{what}")
      end

      dst = Cumo::Bit.new(*bv.shape)
      dst.store(bv)
      assert_equal(want_bit, dst.to_a, "bit.store bit of #{what}")

      whole = Cumo::Bit.new(rows, cols).fill(0)
      take.call(whole).store(bv)
      want = Array.new(n, 0)
      cells.each_with_index { |cell, i| want[cell] = want_bit.flatten[i] }
      assert_equal(want, whole.to_a.flatten, "bit view <- bit of #{what}")
    end

    # a Bit view whose first element is not the first bit of its word
    flat = Array.new(200) { |i| ((i * 7) % 3).zero? ? 1 : 0 }
    long = Cumo::Bit.cast(flat)
    [0, 1, 8, 31, 32, 33, 63, 64, 65, 100, 127, 128].each do |i|
      v = long[i..(i + 20)]
      want = flat[i, 21]
      assert_equal(want, v.to_a, "offset #{i}")
      assert_equal(want, Cumo::Int32.cast(v).to_a, "Int32.cast at offset #{i}")

      d = Cumo::Bit.new(200).fill(0)
      d[i..(i + 20)] = v
      assert_equal(Array.new(200) { |k| (k >= i && k <= i + 20) ? flat[k] : 0 }, d.to_a,
                   "bit aset at offset #{i}")

      e = Cumo::Bit.new(200).fill(1)
      e[i..(i + 20)] = Cumo::Int32.new(21).seq % 2
      assert_equal(Array.new(200) { |k| (k >= i && k <= i + 20) ? (k - i) % 2 : 1 }, e.to_a,
                   "int aset at offset #{i}")
    end

    # a 5-d shape runs past the dimension-specialised accessors
    deep = [2, 2, 3, 2, 5]
    m = deep.inject(:*)
    a = Cumo::Int32.cast((0...m).to_a).reshape(*deep)
    b = a.gt(m / 2).transpose
    assert_equal(map_deep(a.transpose.to_a) { |x| x > (m / 2) ? 1 : 0 }, b.to_a, "5-d bit view")
    assert_equal(b.to_a, map_deep(Cumo::Int32.cast(b).to_a, &:to_i), "5-d Int32.cast")
    dst = Cumo::Bit.new(*b.shape)
    dst.store(a.transpose)
    assert_equal(map_deep(a.transpose.to_a) { |x| x.zero? ? 0 : 1 }, dst.to_a, "5-d bit.store int")
    dst2 = Cumo::Bit.new(*b.shape)
    dst2.store(b)
    assert_equal(b.to_a, dst2.to_a, "5-d bit.store bit")

    # a sub-array shorter than the row it is stored into is zero-filled, and the
    # kernel must not read past its end
    short = Cumo::Bit.new(3, 5).fill(1)
    short.store([Cumo::Bit.cast([1, 0, 1]), Cumo::Bit.cast([0, 1]), Cumo::Bit.cast([1, 1, 0, 1, 0])])
    assert_equal([[1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 1, 0]], short.to_a)
    short2 = Cumo::Bit.new(3, 5).fill(1)
    short2.store([Cumo::Int32.cast([1, 0, 5]), Cumo::Int32.cast([0, 2]), Cumo::Int32.cast([1, 1, 0, 3, 0])])
    assert_equal([[1, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 1, 0, 1, 0]], short2.to_a)
  end

  test "reductions reach every element of an axis that is not the innermost" do
    # Reducing over anything but the last axis went through the indexer and put
    # every thread of a block on a different row of the input. The expectations
    # are folded in Ruby rather than read off a contiguous copy, so a wrong
    # address cannot cancel out between the two sides.
    rows = 9
    cols = 7
    xs = (1..(rows * cols)).to_a.shuffle(random: Random.new(4649))
    src = Cumo::Int32.cast(xs).reshape(rows, cols)
    idx = [5, 0, 3, 8, 1]

    views = {
      "contiguous" => ->(a) { a },
      "column slice" => ->(a) { a[true, 1...(cols - 1)] },
      "reversed" => ->(a) { a[true, (cols - 1).step(0, -1)] },
      "row stride" => ->(a) { a[0.step(rows - 1, 2), true] },
      "index view" => ->(a) { a[idx, true] },
      "transpose" => ->(a) { a.transpose },
    }
    views.each { |what, take| assert_reductions(take.call(src), what) }

    # a 5-d shape runs past the dimension-specialised accessors
    deep = [2, 3, 2, 2, 3]
    n = deep.inject(:*)
    a = Cumo::Int32.cast((1..n).to_a.shuffle(random: Random.new(57))).reshape(*deep)
    assert_reductions(a, "5-d")
    assert_reductions(a.transpose, "5-d transpose")
    assert_reductions(a[true, [2, 0], true, true, true], "5-d index view")

    # long enough for the reduce axis to be split across a second launch
    big = Cumo::Int32.new(3000, 33).seq
    assert_reductions(big, "3000x33", axes: [[0], [1]])
    assert_reductions(big[true, 0.step(32, 2)], "3000x33 column slice", axes: [[0], [1]])
    assert_reductions(big.transpose, "3000x33 transpose", axes: [[0], [1]])
  end

  test "an arithmetic operator with a numeric operand allocates only its result" do
    # A numeric operand used to be cast to a 0-dimensional array, which took a
    # whole kernel launch to fill. Measured in a child, or blocks another test
    # left in the pool serve the allocation and total_bytes does not move.
    script = <<~'RUBY'
      require "cumo/narray"
      pool = Cumo::CUDA::MemoryPool
      unless pool.enabled?
        print "no-pool"
        exit
      end
      a = Cumo::SFloat.new(256).seq
      keep = []
      100.times { keep << a + 1.5 }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      pool.free_all_blocks
      keep.clear
      GC.start
      pool.free_all_blocks
      before = pool.total_bytes
      keep = []
      100.times { keep << a + 1.5 }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      print pool.total_bytes - before
    RUBY
    lib = File.expand_path("../lib", __dir__)
    out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], &:read)
    omit("memory pool is disabled") if out == "no-pool"
    # 100 results of 1 KiB each, rounded up to the pool's 512 byte bins. A
    # 0-dimensional operand per operation would add another 100 bins.
    assert_operator(Integer(out), :<, 100 * (1024 + 512))
  end

  test "clip with numeric bounds allocates only its result" do
    # Each numeric bound used to be cast to a 0-dimensional array, and reading
    # the flag that reported min > max synchronized with the device on top of
    # that. Measured in a child, or blocks another test left in the pool serve
    # the allocation and total_bytes does not move.
    script = <<~'RUBY'
      require "cumo/narray"
      pool = Cumo::CUDA::MemoryPool
      unless pool.enabled?
        print "no-pool"
        exit
      end
      a = Cumo::SFloat.new(256).seq
      keep = []
      100.times { keep << a.clip(1.0, 200.0) }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      pool.free_all_blocks
      keep.clear
      GC.start
      pool.free_all_blocks
      before = pool.total_bytes
      keep = []
      100.times { keep << a.clip(1.0, 200.0) }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      print pool.total_bytes - before
    RUBY
    lib = File.expand_path("../lib", __dir__)
    out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], &:read)
    omit("memory pool is disabled") if out == "no-pool"
    # 100 results of 1 KiB each, rounded up to the pool's 512 byte bins. Two
    # 0-dimensional bounds per call would add another 200 bins.
    assert_operator(Integer(out), :<, 100 * (1024 + 512))
  end

  test "a numeric exponent allocates only its result" do
    # The exponent used to be cast to a 0-dimensional array, an Int32 one for
    # an Integer. Measured in a child, or blocks another test left in the pool
    # serve the allocation and total_bytes does not move.
    script = <<~'RUBY'
      require "cumo/narray"
      pool = Cumo::CUDA::MemoryPool
      unless pool.enabled?
        print "no-pool"
        exit
      end
      a = Cumo::SFloat.new(256).seq
      keep = []
      100.times { keep << a**2 }
      100.times { keep << a**2.5 }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      pool.free_all_blocks
      keep.clear
      GC.start
      pool.free_all_blocks
      before = pool.total_bytes
      keep = []
      100.times { keep << a**2 }
      100.times { keep << a**2.5 }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      print pool.total_bytes - before
    RUBY
    lib = File.expand_path("../lib", __dir__)
    out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], &:read)
    omit("memory pool is disabled") if out == "no-pool"
    # 200 results of 1 KiB each, rounded up to the pool's 512 byte bins. A
    # 0-dimensional exponent per call would add another 200 bins.
    assert_operator(Integer(out), :<, 200 * (1024 + 512))
  end

  test "a numeric exponent answers what the same exponent as an array answers" do
    # Int32 as the exponent does not widen any of these, so the array operand
    # is a fair reference for the value the scalar one is handed.
    [Cumo::SFloat, Cumo::DFloat, Cumo::Int32, Cumo::Int64, Cumo::DComplex].each do |dtype|
      a = dtype[1, 2, 3, 4]
      assert_equal((a**Cumo::Int32.new(4).fill(3)).to_a, (a**3).to_a, "#{dtype} ** 3")
      # a 9-d shape runs past the dimension-specialised kernels
      deep = dtype.new(*([2] * 9)).seq(1)
      assert_equal((deep**Cumo::Int32.new(*([2] * 9)).fill(2)).to_a.flatten,
                   (deep**2).to_a.flatten, "#{dtype} 9-d ** 2")
    end

    [Cumo::SFloat, Cumo::DFloat, Cumo::DComplex].each do |dtype|
      a = dtype[1, 2, 3, 4]
      assert_equal((a**dtype.new(4).fill(2.5)).to_a, (a**2.5).to_a, "#{dtype} ** 2.5")
      deep = dtype.new(*([2] * 9)).seq(1)
      assert_equal((deep**dtype.new(*([2] * 9)).fill(0.5)).to_a.flatten,
                   (deep**0.5).to_a.flatten, "#{dtype} 9-d ** 0.5")
    end

    # a narrower dtype keeps its own width, where an Int32 array operand widens it
    assert_equal(Cumo::UInt8, (Cumo::UInt8[1, 2, 16, 17]**2).class)
    assert_equal([1, 4, 0, 33], (Cumo::UInt8[1, 2, 16, 17]**2).to_a)
    assert_equal(Complex, (Cumo::DComplex[2]**Complex(0, 1)).to_a[0].class)
    # a Bignum exponent still falls through to the wider path
    assert_equal([1.0, Float::INFINITY], (Cumo::SFloat[1, 2]**(2**70)).to_a)
    assert_raise(RangeError) { Cumo::SFloat[1, 2]**(2**40) }
    assert_raise(RangeError) { Cumo::Int32[1, 2]**(2**70) }
  end

  test "a numeric operand compared against allocates only its result" do
    # The operand used to be cast to a 0-dimensional array, which took a whole
    # kernel launch to fill. Measured in a child, or blocks another test left
    # in the pool serve the allocation and total_bytes does not move.
    script = <<~'RUBY'
      require "cumo/narray"
      pool = Cumo::CUDA::MemoryPool
      unless pool.enabled?
        print "no-pool"
        exit
      end
      a = Cumo::SFloat.new(8192).seq
      keep = []
      100.times { keep << (a > 1.5) }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      pool.free_all_blocks
      keep.clear
      GC.start
      pool.free_all_blocks
      before = pool.total_bytes
      keep = []
      100.times { keep << (a > 1.5) }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      print pool.total_bytes - before
    RUBY
    lib = File.expand_path("../lib", __dir__)
    out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], &:read)
    omit("memory pool is disabled") if out == "no-pool"
    # 100 Bit results of 1 KiB each. A 0-dimensional operand per comparison
    # would add another 100 of the pool's 512 byte bins.
    assert_operator(Integer(out), :<, 100 * (1024 + 512))
  end

  test "a numeric operand answers what the same operand as an array answers" do
    # The operand keeps the receiver's dtype either way, so the array one is a
    # fair reference for the value the scalar one is handed.
    comparable = [Cumo::Int8, Cumo::Int32, Cumo::Int64, Cumo::UInt8, Cumo::UInt32,
                  Cumo::SFloat, Cumo::DFloat]
    (comparable + [Cumo::SComplex, Cumo::DComplex]).each do |dtype|
      ops = comparable.include?(dtype) ? %i[eq ne gt ge lt le] : %i[eq ne]
      a = dtype[1, 2, 3, 4]
      ops.each do |op|
        assert_equal(a.send(op, dtype.new(4).fill(3)).to_a, a.send(op, 3).to_a,
                     "#{dtype} #{op} 3")
      end
      # a 9-d shape runs past the dimension-specialised kernels
      deep = dtype.new(*([2] * 9)).seq(1)
      assert_equal(deep.eq(dtype.new(*([2] * 9)).fill(7)).to_a.flatten,
                   deep.eq(7).to_a.flatten, "#{dtype} 9-d eq 7")
    end

    [Cumo::SFloat, Cumo::DFloat].each do |dtype|
      a = dtype[-1, 0, 1, Float::NAN, Float::INFINITY]
      %i[eq ne gt ge lt le nearly_eq].each do |op|
        [1.0, Float::NAN, Float::INFINITY].each do |v|
          assert_equal(a.send(op, dtype.new(5).fill(v)).to_a, a.send(op, v).to_a,
                       "#{dtype} #{op} #{v}")
        end
      end
    end

    view = Cumo::Int32.new(3, 4).seq.transpose
    assert_equal(view.gt(Cumo::Int32.new(4, 3).fill(5)).to_a, view.gt(5).to_a)

    # the operand is converted the way casting it converts it
    assert_equal([0, 0, 0, 1], Cumo::UInt8[1, 2, 16, 255].eq(-1).to_a)
    assert_equal([1, 1, 1, 1], Cumo::UInt8[1, 2, 16, 255].gt(256).to_a)
    assert_raise(RangeError) { Cumo::Int32[1, 2].gt(2**40) }
  end

  test "a numeric NMath operand allocates only its result" do
    # The operand used to be cast to a 0-dimensional array, which took a whole
    # kernel launch to fill. Measured in a child, or blocks another test left
    # in the pool serve the allocation and total_bytes does not move.
    script = <<~'RUBY'
      require "cumo/narray"
      pool = Cumo::CUDA::MemoryPool
      unless pool.enabled?
        print "no-pool"
        exit
      end
      m = Cumo::DFloat::Math
      a = Cumo::DFloat.new(128).seq(1)
      keep = []
      100.times { keep << m.atan2(a, 1.5) }
      100.times { keep << m.atan2(1.5, a) }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      pool.free_all_blocks
      keep.clear
      GC.start
      pool.free_all_blocks
      before = pool.total_bytes
      keep = []
      100.times { keep << m.atan2(a, 1.5) }
      100.times { keep << m.atan2(1.5, a) }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      print pool.total_bytes - before
    RUBY
    lib = File.expand_path("../lib", __dir__)
    out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], &:read)
    omit("memory pool is disabled") if out == "no-pool"
    # 200 results of 1 KiB each. A 0-dimensional operand per call would add
    # another 200 of the pool's 512 byte bins.
    assert_operator(Integer(out), :<, 200 * (1024 + 512))
  end

  test "a numeric NMath operand answers what the same operand as an array answers" do
    # Either side of a module function can be the numeric one, so both are
    # compared against the same value spread over an array.
    [Cumo::SFloat, Cumo::DFloat].each do |dtype|
      m = dtype::Math
      a = dtype[1.0, -2.0, 0.5, 3.5]
      spread = dtype.new(4).fill(1.5)
      %i[atan2 hypot ldexp].each do |op|
        assert_equal(m.send(op, a, spread).to_a, m.send(op, a, 1.5).to_a, "#{dtype} #{op}(arr, 1.5)")
        assert_equal(m.send(op, spread, a).to_a, m.send(op, 1.5, a).to_a, "#{dtype} #{op}(1.5, arr)")
      end

      view = dtype.new(3, 4).seq(1).transpose
      assert_equal(m.atan2(view, dtype.new(4, 3).fill(1.5)).to_a, m.atan2(view, 1.5).to_a, "#{dtype} view")
      assert_equal(m.atan2(dtype.new(4, 3).fill(1.5), view).to_a, m.atan2(1.5, view).to_a, "#{dtype} view, reversed")

      deep = dtype.new(*([2] * 9)).seq(1)
      assert_equal(m.hypot(deep, dtype.new(*([2] * 9)).fill(1.5)).to_a.flatten,
                   m.hypot(deep, 1.5).to_a.flatten, "#{dtype} 9-d")
    end

    # nan and the infinities reach the kernel as themselves
    a = Cumo::DFloat[1.0, -2.0, Float::NAN, Float::INFINITY]
    [Float::NAN, Float::INFINITY, -Float::INFINITY, 0.0].each do |v|
      spread = Cumo::DFloat.new(4).fill(v)
      assert_equal(Cumo::DFloat::Math.atan2(a, spread).to_a.map(&:to_s),
                   Cumo::DFloat::Math.atan2(a, v).to_a.map(&:to_s), "atan2(arr, #{v})")
      assert_equal(Cumo::DFloat::Math.atan2(spread, a).to_a.map(&:to_s),
                   Cumo::DFloat::Math.atan2(v, a).to_a.map(&:to_s), "atan2(#{v}, arr)")
    end

    # two numerics still take the two-operand path
    assert_equal(Cumo::DFloat, Cumo::DFloat::Math.atan2(1.0, 2.0).class)
    assert_in_delta(Math.atan2(1.0, 2.0), Cumo::DFloat::Math.atan2(1.0, 2.0).to_a[0], 1e-15)
    assert_raise(Cumo::NArray::CastError) { Cumo::DFloat::Math.atan2(Cumo::DFloat[1.0], "x") }
    assert_raise(Cumo::NArray::CastError) { Cumo::DFloat::Math.atan2("x", Cumo::DFloat[1.0]) }
  end

  test "NMath binary functions on views that ndloop cannot flatten" do
    # same shape of view the unary tests use: before the indexer loop each of
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

    fm = Cumo::DFloat::Math
    views.each do |label, slice, want|
      v = slice.call(a)
      spread = Cumo::DFloat.new(*v.shape).fill(1.5)
      # the scalar operand reads the same elements the array one does
      assert_equal(fm.atan2(v, spread).to_a, fm.atan2(v, 1.5).to_a, "atan2 #{label}, numeric right")
      assert_equal(fm.atan2(spread, v).to_a, fm.atan2(1.5, v).to_a, "atan2 #{label}, numeric left")
      assert_equal(fm.hypot(spread, v).to_a, fm.hypot(1.5, v).to_a, "hypot #{label}, numeric left")

      got = fm.atan2(v, 1.5).to_a
      back = fm.atan2(1.5, v).to_a
      hyp = fm.hypot(v, 1.5).to_a
      want.each_with_index do |row, r|
        row.each_with_index do |x, c|
          assert_in_delta(Math.atan2(x, 1.5), got[r][c], 1e-12, "atan2 #{label} #{r},#{c}")
          assert_in_delta(Math.atan2(1.5, x), back[r][c], 1e-12, "atan2 reversed #{label} #{r},#{c}")
          assert_in_delta(Math.hypot(x, 1.5), hyp[r][c], 1e-12, "hypot #{label} #{r},#{c}")
        end
      end
    end

    # a 9-d shape runs past the dimension-specialised kernels
    ys = Array.new(512) { |i| ((i % 7) - 3).to_f }
    b = Cumo::DFloat.cast(ys).reshape(*([2] * 9))
    got = fm.atan2(b, 1.5).to_a.flatten
    ys.each_with_index { |x, i| assert_in_delta(Math.atan2(x, 1.5), got[i], 1e-12, "atan2 9-d #{i}") }
    assert_equal(fm.atan2(b, Cumo::DFloat.new(*([2] * 9)).fill(1.5)).to_a.flatten, got, "atan2 9-d, numeric right")
  end

  test "an unsigned array raised to a negative power answers instead of spinning" do
    # The unsigned pow_int had no p < 0 guard, and p >>= 1 sticks at -1, so the
    # kernel never finished. Every negative power runs in a child: without the
    # guard the device spins and the process stops only for SIGKILL, which
    # would hang the suite here rather than fail it.
    script = <<~'RUBY'
      require "cumo/narray"
      unsigned = [Cumo::UInt8[1, 2, 3, 4]**-2,
                  Cumo::UInt16[1, 2, 3, 4]**-1,
                  Cumo::UInt32[1, 2, 3, 4]**-3,
                  Cumo::UInt64[1, 2, 3, 4]**(-2**31)]
      signed = [Cumo::Int8[1, 2, 3, 4]**-2,
                Cumo::Int16[1, 2, 3, 4]**-1,
                Cumo::Int32[1, 2, 3, 4]**-3,
                Cumo::Int64[1, 2, 3, 4]**(-2**31)]
      print (unsigned + signed).map { |v| v.to_a.join(",") }.join("|")
    RUBY
    lib = File.expand_path("../lib", __dir__)
    r, w = IO.pipe
    pid = Process.spawn(RbConfig.ruby, "-I#{lib}", "-e", script, out: w, err: File::NULL)
    w.close
    reader = Thread.new { r.read }
    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + 60
    until Process.waitpid(pid, Process::WNOHANG)
      if Process.clock_gettime(Process::CLOCK_MONOTONIC) > deadline
        Process.kill("KILL", pid)
        Process.waitpid(pid)
        reader.kill
        flunk("a negative power of an unsigned array did not finish in 60 seconds")
      end
      sleep 0.05
    end
    # unsigned now answers what signed has always answered
    assert_equal((["0,0,0,0"] * 8).join("|"), reader.value)
  end

  test "the exponents around the negative-power guard are unchanged" do
    %w[UInt8 UInt16 UInt32 UInt64].each do |name|
      u = Cumo.const_get(name)[1, 2, 3, 4]
      assert_equal([1, 1, 1, 1], (u**0).to_a, "#{name} ** 0")
      assert_equal([1, 2, 3, 4], (u**1).to_a, "#{name} ** 1")
      assert_equal([1, 8, 27, 64], (u**3).to_a, "#{name} ** 3")
      assert_equal([1, 16, 81, 256 % (1 << 8)], (Cumo::UInt8[1, 2, 3, 4]**4).to_a) if name == "UInt8"
    end
    # an Int32 array operand upcasts, so a negative exponent there took the
    # signed path all along
    assert_equal([0, 0, 9, 64], (Cumo::UInt8[1, 2, 3, 4]**Cumo::Int32[-2, -1, 2, 3]).to_a)
  end

  test "a float or complex array raised to the INT_MIN power answers instead of spinning" do
    # -p overflows for INT32_MIN and the wrapped negative shifts down to -1, so
    # pow_positive_int would never leave its loop. pow_int takes the magnitude
    # in unsigned now. The exponent reaches it three ways: as a Fixnum, as an
    # Int32 array, and as an Int32 array against a numeric base through coerce.
    # This runs in a child for the same reason as the unsigned test above: a
    # spinning kernel only stops for SIGKILL.
    script = <<~'RUBY'
      require "cumo/narray"
      int_min = Cumo::Int32[-2**31, -2**31, -2**31]
      out = [Cumo::SFloat[0.5, 1.0, 2.0]**(-2**31),
             Cumo::DFloat[0.5, 1.0, 2.0]**(-2**31),
             Cumo::SFloat[0.5, 1.0, 2.0]**(-2**31 + 1),
             Cumo::DFloat[0.5, 1.0, 2.0]**(-2**31 + 1),
             Cumo::SFloat[0.5, 1.0, 2.0]**int_min,
             Cumo::DFloat[0.5, 1.0, 2.0]**int_min].map { |v| v.to_a.join(",") }
      out << (0.5**Cumo::Int32[-2**31, 1, 2]).to_a.join(",")
      # the complex path overflows the same way and only has to finish
      (Cumo::SComplex[Complex(0.5, 0)]**(-2**31)).to_a
      (Cumo::DComplex[Complex(0.5, 0)]**(-2**31)).to_a
      (Cumo::DComplex[Complex(0.5, 0)]**Cumo::Int32[-2**31]).to_a
      print out.join("|")
    RUBY
    lib = File.expand_path("../lib", __dir__)
    r, w = IO.pipe
    pid = Process.spawn(RbConfig.ruby, "-I#{lib}", "-e", script, out: w, err: File::NULL)
    w.close
    reader = Thread.new { r.read }
    deadline = Process.clock_gettime(Process::CLOCK_MONOTONIC) + 60
    until Process.waitpid(pid, Process::WNOHANG)
      if Process.clock_gettime(Process::CLOCK_MONOTONIC) > deadline
        Process.kill("KILL", pid)
        Process.waitpid(pid)
        reader.kill
        flunk("a float power of INT_MIN did not finish in 60 seconds")
      end
      sleep 0.05
    end
    # 0.5 ** -2**31 overflows to infinity and 2.0 ** -2**31 underflows to zero
    assert_equal(((["Infinity,1.0,0.0"] * 6) << "Infinity,0.5,0.25").join("|"), reader.value)
  end

  test "the exponents around the INT_MIN power are unchanged" do
    [Cumo::SFloat, Cumo::DFloat].each do |klass|
      a = klass[0.5, 1.0, 2.0]
      assert_equal([1.0, 1.0, 1.0], (a**0).to_a, "#{klass} ** 0")
      assert_equal([0.5, 1.0, 2.0], (a**1).to_a, "#{klass} ** 1")
      assert_equal([0.25, 1.0, 4.0], (a**2).to_a, "#{klass} ** 2")
      assert_equal([0.125, 1.0, 8.0], (a**3).to_a, "#{klass} ** 3")
      assert_equal([0.0625, 1.0, 16.0], (a**4).to_a, "#{klass} ** 4")
      assert_equal([32.0, 1.0, 0.03125], (a**-5).to_a, "#{klass} ** -5")
      # past 64 the exponent reaches pow() as a double either way
      assert_equal([2.0**-100, 1.0, 2.0**100], (a**100).to_a, "#{klass} ** 100")
      assert_equal([2.0**100, 1.0, 2.0**-100], (a**-100).to_a, "#{klass} ** -100")
    end
    assert_in_delta(1.1**100, (Cumo::SFloat[1.1]**100).to_a[0], 1.0, "SFloat ** 100")
    assert_in_delta(1.1**100, (Cumo::DFloat[1.1]**100).to_a[0], 1e-9, "DFloat ** 100")
    assert_equal(2.0**-100, (Cumo::DComplex[Complex(0.5, 0)]**100).to_a[0].real, "DComplex ** 100")
  end

  test "seq wraps an out-of-range start the way fill and cast do" do
    # f_seq answers a double for the integer types, and the device saturates an
    # out-of-range double, so a negative start collapsed to zero.
    assert_equal([254, 255, 0, 1], Cumo::UInt8.new(4).seq(-2).to_a)
    assert_equal([65534, 65535, 0, 1], Cumo::UInt16.new(4).seq(-2).to_a)
    assert_equal([4294967294, 4294967295, 0, 1], Cumo::UInt32.new(4).seq(-2).to_a)
    assert_equal([18446744073709551614, 18446744073709551615, 0, 1], Cumo::UInt64.new(4).seq(-2).to_a)
    assert_equal([56, 57, 58], Cumo::UInt8.new(3).seq(-200).to_a)
    # a negative step walks the same way
    assert_equal([0, 255, 254, 253], Cumo::UInt8.new(4).seq(0, -1).to_a)

    # every start seq accepts answers what fill and cast answer for it
    [Cumo::UInt8, Cumo::UInt16, Cumo::UInt32, Cumo::UInt64,
     Cumo::Int8, Cumo::Int16, Cumo::Int32, Cumo::Int64].each do |dtype|
      [-2, -1, 0, 1, 3].each do |beg|
        assert_equal(dtype.new(1).fill(beg).to_a, dtype.new(1).seq(beg).to_a, "#{dtype} seq(#{beg})")
        assert_equal(dtype.cast(beg).to_a, dtype.new(1).seq(beg).to_a, "#{dtype} seq(#{beg}) vs cast")
      end
    end

    # UInt64 holds everything from 2**63 up exactly, and fill and cast answer
    # those starts exactly, so seq has to as well
    [2**63, 2**63 + 2**53, 2**64 - 2048].each do |beg|
      assert_equal(Cumo::UInt64.new(1).fill(beg).to_a, Cumo::UInt64.new(1).seq(beg).to_a,
                   "UInt64 seq(#{beg})")
      assert_equal(Cumo::UInt64.cast(beg).to_a, Cumo::UInt64.new(1).seq(beg).to_a,
                   "UInt64 seq(#{beg}) vs cast")
    end

    # a view and a 9-d shape run the other dimension-specialised kernels
    assert_equal([254, 0], Cumo::UInt8.new(2, 2).seq(-2)[true, 0].to_a)
    assert_equal([254, 255, 0, 1], Cumo::UInt8.new(*([2] * 9)).seq(-2).to_a.flatten.first(4))

    # the signed types and the positive overflows are unchanged
    assert_equal([-56, -55, -54], Cumo::Int8.new(3).seq(200).to_a)
    assert_equal([44, 45, 46], Cumo::UInt8.new(3).seq(300).to_a)
    assert_equal([25536, 25537, 25538], Cumo::Int16.new(3).seq(-40000).to_a)
    # the floats never go through the integer conversion
    assert_equal([-2.5, -1.5, -0.5], Cumo::DFloat.new(3).seq(-2.5).to_a)
    assert_equal([-2.5, -1.5, -0.5], Cumo::SFloat.new(3).seq(-2.5).to_a)
  end

  test "a numeric on the left of an operator allocates only its result" do
    # coerce cast the numeric to a 0-dimensional array, which took a whole
    # kernel launch to fill. Measured in a child, or blocks another test left
    # in the pool serve the allocation and total_bytes does not move.
    script = <<~'RUBY'
      require "cumo/narray"
      pool = Cumo::CUDA::MemoryPool
      unless pool.enabled?
        print "no-pool"
        exit
      end
      a = Cumo::SFloat.new(256).seq
      keep = []
      100.times { keep << 0.5 * a }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      pool.free_all_blocks
      keep.clear
      GC.start
      pool.free_all_blocks
      before = pool.total_bytes
      keep = []
      100.times { keep << 0.5 * a }
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      print pool.total_bytes - before
    RUBY
    lib = File.expand_path("../lib", __dir__)
    out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], &:read)
    omit("memory pool is disabled") if out == "no-pool"
    # 100 results of 1 KiB each. The 0-dimensional operand coerce built would
    # add another 100 of the pool's 512 byte bins.
    assert_operator(Integer(out), :<, 100 * (1024 + 512))
  end

  test "a numeric on the left answers what the same value as an array answers" do
    [Cumo::SFloat, Cumo::DFloat, Cumo::Int32, Cumo::Int64].each do |dtype|
      a = dtype[1, 2, 3, 4]
      spread = dtype.new(4).fill(2)
      assert_equal((spread + a).to_a, (2 + a).to_a, "#{dtype} 2 + a")
      assert_equal((spread - a).to_a, (2 - a).to_a, "#{dtype} 2 - a")
      assert_equal((spread * a).to_a, (2 * a).to_a, "#{dtype} 2 * a")
      assert_equal((spread / a).to_a, (2 / a).to_a, "#{dtype} 2 / a")
      assert_equal((spread % a).to_a, (2 % a).to_a, "#{dtype} 2 % a")
      assert_equal((spread**a).to_a, (2**a).to_a, "#{dtype} 2 ** a")
      assert_equal((spread < a).to_a, (2 < a).to_a, "#{dtype} 2 < a")
      assert_equal((spread <= a).to_a, (2 <= a).to_a, "#{dtype} 2 <= a")
    end

    # the operand order is not commuted: a view, a 9-d shape and an index-backed
    # view all reach the same elements
    view = Cumo::SFloat.new(2, 3).seq(1).transpose
    assert_equal((Cumo::SFloat.new(3, 2).fill(2.0) - view).to_a, (2.0 - view).to_a, "view")
    idx = Cumo::SFloat.new(6).seq(1)[[3, 1, 2]]
    assert_equal((Cumo::SFloat.new(3).fill(2.0) - idx).to_a, (2.0 - idx).to_a, "index view")
    deep = Cumo::SFloat.new(*([2] * 9)).seq(1)
    assert_equal((Cumo::SFloat.new(*([2] * 9)).fill(2.0) / deep).to_a.flatten,
                 (2.0 / deep).to_a.flatten, "9-d")
    assert_equal([], (0.5 * Cumo::SFloat.new(0)).to_a, "empty")

    # dividing by an element that is zero still raises from the left
    assert_raise(ZeroDivisionError) { 3 / Cumo::Int32[1, 0, 2] }
    assert_raise(ZeroDivisionError) { 3 % Cumo::Int32[1, 0, 2] }
    # and an out-of-range numeric still raises where casting it raised
    assert_raise(RangeError) { 2**40 * Cumo::Int32[1, 2] }
  end

  test "a numeric cast to a 0-dimensional array still reads back" do
    # the value stays a Ruby numeric until something asks for the element
    [Cumo::SFloat, Cumo::DFloat, Cumo::Int32, Cumo::UInt8, Cumo::DComplex].each do |dtype|
      assert_equal([3], dtype.cast(3).to_a, "#{dtype} cast(3).to_a")
      assert_equal([3], dtype.cast(3).dup.to_a, "#{dtype} cast(3).dup")
      assert_equal([3], Marshal.load(Marshal.dump(dtype.cast(3))).to_a, "#{dtype} marshal")
      assert_equal([9], (dtype.cast(3) * dtype[3]).to_a, "#{dtype} cast * array")
      assert_equal([6], (dtype.cast(3) + dtype.cast(3)).to_a, "#{dtype} cast + cast")
      w = dtype.cast(3)
      w.store(5)
      assert_equal([5], w.to_a, "#{dtype} cast then store")
      assert_not_equal("", dtype.cast(3).inspect, "#{dtype} cast.inspect")
    end
    assert_equal([1], Cumo::Bit.cast(1).to_a)
    assert_equal([1.5], Cumo::RObject.cast(1.5).to_a)
    # the range check still happens where casting it happened
    assert_raise(RangeError) { Cumo::Int32.cast(2**40) }
    assert_raise(RangeError) { Cumo::Int32[1, 2].coerce(2**40) }
    assert_equal([Cumo::SFloat, Cumo::SFloat],
                 Cumo::SFloat[1, 2].coerce(1.0).map(&:class))
  end

  test "every subscript form still answers after the null free was dropped" do
    # aref_md handed every dimension to the device free, even the ones whose
    # subscript is a range or true and never allocated an index array. Freeing
    # NULL costs a pool lookup and a driver call per dimension per subscript.
    # There is no Ruby-visible signal for the call itself, so this only pins
    # that each form of subscript still reaches the elements it should.
    a = Cumo::SFloat.new(4, 4).seq(1)
    assert_equal([[1.0, 2.0], [5.0, 6.0]], a[0..1, 0..1].to_a, "range, range")
    assert_equal(a.to_a, a[true, true].to_a, "true, true")
    assert_equal([[6.0, 8.0], [14.0, 16.0]], a[[1, 3], [1, 3]].to_a, "index, index")
    assert_equal([[5.0, 6.0, 7.0, 8.0], [13.0, 14.0, 15.0, 16.0]], a[[1, 3], true].to_a, "index, true")
    assert_equal([[2.0, 4.0], [10.0, 12.0]], a[[0, 2], [1, 3]].to_a, "index, index again")
    assert_equal([2.0, 6.0, 10.0, 14.0], a[true, 1].to_a, "true, scalar")
    assert_equal([[1.0, 3.0], [9.0, 11.0]], a[(0..3).step(2), (0..3).step(2)].to_a, "step, step")
    assert_equal(6.0, a[1, 1].to_a[0], "scalar, scalar")
    deep = Cumo::SFloat.new(2, 2, 2, 2).seq(1)
    assert_equal(deep.to_a, deep[true, true, true, true].to_a, "4-d all true")
    assert_equal([[2.0, 4.0], [6.0, 8.0]], deep[0, true, true, 1].to_a, "4-d mixed")
    assert_equal(deep[[1, 0], true, true, true].to_a.flatten.sort,
                 deep.to_a.flatten.sort, "4-d index on the outer axis")
  end

  test "an integer divided or reduced by a numeric zero still raises" do
    assert_raise(ZeroDivisionError) { Cumo::Int32[1, 2, 3] / 0 }
    assert_raise(ZeroDivisionError) { Cumo::Int32[[1, 2], [3, 4]][true, 0] / 0 }
    assert_raise(ZeroDivisionError) { Cumo::Int32[1, 2, 3] % 0 }
    assert_equal([0, 1, 1], (Cumo::Int32[1, 2, 3] / 2).to_a)
    assert_equal([1, 0, 1], (Cumo::Int32[1, 2, 3] % 2).to_a)
  end

  test "the operators that take a numeric operand answer what Numo answers" do
    ints = Cumo::Int32[7, 3, 100]
    assert_equal([5, 1, 4], (ints & 5).to_a)
    assert_equal([7, 7, 101], (ints | 5).to_a)
    assert_equal([2, 6, 97], (ints ^ 5).to_a)
    assert_equal([28, 12, 400], (ints << 2).to_a)
    assert_equal([1, 0, 25], (ints >> 2).to_a)
    floats = Cumo::SFloat[7, -3, 100]
    assert_equal([-7, -3, -100], floats.copysign(-1).to_a)
    assert_equal([7, 3, 100], floats.copysign(1).to_a)
  end

  test "a reduction over a strided view does not copy the operand" do
    # In a child, or blocks another test left in the pool serve the copy and it
    # does not show up in total_bytes.
    script = <<~'RUBY'
      require "cumo/narray"
      pool = Cumo::CUDA::MemoryPool
      unless pool.enabled?
        print "no-pool"
        exit
      end
      a = Cumo::SFloat.new(1024, 1024).seq
      a.transpose.sum(axis: -1)
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      pool.free_all_blocks
      before = pool.total_bytes
      a.transpose.sum(axis: -1)
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
      print pool.total_bytes - before
    RUBY
    lib = File.expand_path("../lib", __dir__)
    out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], &:read)
    omit("memory pool is disabled") if out == "no-pool"
    assert_operator(Integer(out), :<, 1024 * 1024 * 4)
  end

  test "a reduction answering an index copies a strided view" do
    # The index the kernel answers counts along memory, so a strided operand
    # would answer an index into the base rather than into the view.
    a = Cumo::Int32.new(6, 5).seq
    v = a[true, 1..3]
    assert_equal([15, 16, 17], v.max_index(axis: 0).to_a)
    assert_equal([2, 5, 8, 11, 14, 17], v.max_index(axis: 1).to_a)
  end

  test "copy of an RObject view stays on the host" do
    # RObject data is host memory, so it must not take the kernel path
    a = Cumo::RObject.cast([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert_equal([[7, 8, 9], [1, 2, 3]], a[[2, 0], true].copy.to_a)
    assert_equal([[1, 4, 7], [2, 5, 8], [3, 6, 9]], a.transpose.copy.to_a)
  end

  test "expand_dims stops at the maximum number of dimensions" do
    max = 12 # CUMO_NA_MAX_DIMENSION
    a = Cumo::DFloat.zeros(*([2] * (max - 1)))
    assert_equal(max, a.expand_dims(0).ndim)
    assert_equal(max, a.expand_dims(max - 1).ndim)
    b = Cumo::DFloat.zeros(*([2] * max))
    assert_raise(Cumo::NArray::DimensionError) { b.expand_dims(0) }
    assert_raise(Cumo::NArray::DimensionError) { b.expand_dims(max) }
    # dimensions no adjacent pair of which ndloop can contract, so the loop
    # reaches the kernel with every one of them
    base = Cumo::DFloat.zeros(*([3] * max))
    view = base[*([0..1] * max)]
    assert_raise(Cumo::NArray::DimensionError) { view.expand_dims(max / 2) }
    view.fill(9.0)
    assert_equal(view.size, base.flatten.eq(9.0).count_true)
  end

  test "kahan_sum keeps what a plain sum loses, in parallel" do
    nan = Float::NAN
    flat = lambda do |v|
      v = v.to_a if v.respond_to?(:ndim)
      v = v.first while v.is_a?(Array) && v.size == 1
      v
    end
    rel_err = lambda do |v, exact|
      ((Rational(flat.call(v)) - exact) / exact).abs
    end

    # one huge term and many tiny ones, which a sequential plain sum drops
    n = 100_000
    src = [1e16] + Array.new(n - 1) { 1.0 }
    exact = Rational(10**16) + (n - 1)
    a = Cumo::DFloat.cast(src)
    assert(rel_err.call(a.kahan_sum, exact) < 1e-15,
           "kahan_sum error #{rel_err.call(a.kahan_sum, exact).to_f}")

    # values spread widely enough that the tree a plain sum walks loses digits
    spread = Array.new(n) { |i| ((i * 2_654_435_761) % 1_000_003) / 1_000_003.0 - 0.5 }
    b = Cumo::DFloat.cast(spread)
    exact_spread = spread.map { |v| Rational(v) }.sum
    assert(rel_err.call(b.kahan_sum, exact_spread) < rel_err.call(b.sum, exact_spread),
           "kahan_sum #{rel_err.call(b.kahan_sum, exact_spread).to_f} should beat " \
           "the plain sum #{rel_err.call(b.sum, exact_spread).to_f}")

    # skipping NaN, and answering what the plain form does when there is none
    with_nan = Cumo::DFloat.cast([1e16, nan, 1.0, 2.0, nan, 3.0])
    kept = Cumo::DFloat.cast([1e16, 1.0, 2.0, 3.0])
    assert_equal(flat.call(kept.kahan_sum), flat.call(with_nan.kahan_sum(nan: true)))
    assert_equal(flat.call(kept.kahan_sum), flat.call(kept.kahan_sum(nan: true)))
    assert_equal(0.0, flat.call(Cumo::DFloat.cast([nan] * 8).kahan_sum(nan: true)))

    # a reduce axis long enough to be split across blocks, over a view
    base = Cumo::DFloat.new(4, 30_000).seq + 1
    [base, base[true, 29_999.step(0, -1)], base.transpose].each_with_index do |v, i|
      [0, 1].each do |ax|
        assert_in_delta(0, (v.kahan_sum(axis: ax) - v.sum(axis: ax)).abs.max, 1e-3,
                        "view #{i} axis #{ax}")
      end
    end

    c = Cumo::DComplex.cast(src.map { |x| Complex(x, -x) })
    got = flat.call(c.kahan_sum)
    assert(((Rational(got.real) - exact) / exact).abs < 1e-15, "complex real part")
    assert(((Rational(got.imag) + exact) / exact).abs < 1e-15, "complex imaginary part")
  end

  test "the index reductions answer the earliest extreme, and the first NaN with nan" do
    nan = Float::NAN
    inf = Float::INFINITY
    got = lambda do |a, op, nan_aware = false|
      v = nan_aware ? a.send(op, nan: true) : a.send(op)
      v = v.to_a if v.respond_to?(:ndim)
      v = v.first while v.is_a?(Array) && v.size == 1
      v
    end
    check = lambda do |label, a, want|
      %i[max_index min_index argmax argmin].each_with_index do |op, i|
        assert_equal(want[i], got.call(a, op), "#{label} #{op}")
      end
    end
    [Cumo::SFloat, Cumo::DFloat].each do |dtype|
      # long enough that the reduce axis is split across blocks
      n = 20_000
      body = (0...n).map { |i| (i * 7 % 101) - 50.0 }
      want = [body.each_with_index.max_by { |v, i| [v, -i] }[1],
              body.each_with_index.min_by { |v, i| [v, i] }[1]]
      check.call("#{dtype} plain", dtype.cast(body), [want[0], want[1], want[0], want[1]])

      # -inf must not lose to the identity, and every element being NaN or the
      # same value answers 0 as numo does
      check.call("#{dtype} all -inf", dtype.cast([-inf] * n), [0, 0, 0, 0])
      check.call("#{dtype} all +inf", dtype.cast([inf] * n), [0, 0, 0, 0])
      check.call("#{dtype} all NaN", dtype.cast([nan] * n), [0, 0, 0, 0])
      check.call("#{dtype} -inf 0 1 inf", dtype.cast([-inf, 0.0, 1.0, inf]), [3, 0, 3, 0])
      check.call("#{dtype} ties", dtype.cast([1.0, 2.0, 2.0, 1.0]), [1, 0, 1, 0])

      # without a NaN anywhere, nan: true has to answer what the plain form does
      clean = dtype.cast(body)
      %i[max_index min_index argmax argmin].each do |op|
        assert_equal(got.call(clean, op), got.call(clean, op, true), "#{dtype} #{op} nan: true, no NaN")
      end

      with_nan = dtype.cast([3.0, -1.0, nan, 1.0, nan])
      %i[max_index min_index argmax argmin].each do |op|
        assert_equal(2, got.call(with_nan, op, true), "#{dtype} #{op} takes the first NaN")
        assert_equal(op.to_s.include?("max") ? 0 : 1, got.call(with_nan, op),
                     "#{dtype} #{op} without nan skips NaN")
      end
    end

    [Cumo::Int32, Cumo::Int64].each do |dtype|
      lo = dtype == Cumo::Int32 ? -(2**31) : -(2**63)
      check.call("#{dtype} all min", dtype.cast([lo] * 100), [0, 0, 0, 0])
      check.call("#{dtype} ties", dtype.cast([1, 2, 2, 1]), [1, 0, 1, 0])
    end

    # a column whose every element is the identity's own value: the identity
    # has to lose to it, or the answer is the identity's index rather than the
    # first element of that column
    {
      "column of -inf" => [[[5.0, -inf], [6.0, -inf]], [[2, 1], [0, 1]]],
      "column of +inf" => [[[5.0, inf], [6.0, inf]], [[2, 1], [0, 1]]],
      "column of NaN" => [[[5.0, nan], [6.0, nan]], [[2, 1], [0, 1]]],
    }.each do |label, (src, want)|
      a = Cumo::DFloat.cast(src)
      assert_equal(want[0], a.max_index(axis: 0).to_a, "#{label} max_index")
      assert_equal(want[1], a.min_index(axis: 0).to_a, "#{label} min_index")
    end

    b = Cumo::DFloat.new(5, 24).seq
    [b, b.transpose, b[true, 23.step(0, -1)]].each_with_index do |v, vi|
      [0, 1].each do |ax|
        %i[max_index min_index argmax argmin].each do |op|
          assert_equal(v.send(op, axis: ax).to_a, v.send(op, axis: ax, nan: true).to_a,
                       "view #{vi} axis #{ax} #{op} over a view with no NaN")
        end
      end
    end
  end

  test "the nan-aware reductions skip NaN, and min and max answer NaN" do
    nan = Float::NAN
    flat = ->(v) { v.respond_to?(:ndim) ? v.to_a.flatten : [v] }
    nan_p = lambda do |v|
      return v.real.nan? || v.imag.nan? if v.is_a?(Complex)
      v.respond_to?(:nan?) && v.nan?
    end
    close = lambda do |got, want, label|
      g = flat.call(got)
      w = flat.call(want)
      assert_equal(w.size, g.size, label)
      g.zip(w).each_with_index do |(x, y), i|
        if nan_p.call(y)
          assert(nan_p.call(x), "#{label}[#{i}]: expected NaN, got #{x.inspect}")
        else
          assert_in_delta(0, (x - y).abs, (y.abs * 1e-5) + 1e-5, "#{label}[#{i}]")
        end
      end
    end

    [Cumo::SFloat, Cumo::DFloat].each do |dtype|
      with = dtype.cast([3.0, -1.0, nan, 1.0, 5.0])
      kept = dtype.cast([3.0, -1.0, 1.0, 5.0])
      %i[sum prod mean var stddev rms].each do |op|
        close.call(with.send(op, nan: true), kept.send(op), "#{dtype} #{op}")
      end
      %i[min max ptp].each { |op| close.call(with.send(op, nan: true), nan, "#{dtype} #{op}") }
      close.call(with.minmax(nan: true)[0], nan, "#{dtype} minmax min")
      close.call(with.minmax(nan: true)[1], nan, "#{dtype} minmax max")

      none = dtype.cast([nan] * 4)
      close.call(none.sum(nan: true), 0, "#{dtype} sum of all NaN")
      close.call(none.prod(nan: true), 1, "#{dtype} prod of all NaN")
      close.call(none.var(nan: true), 0, "#{dtype} var of all NaN")
      close.call(none.stddev(nan: true), 0, "#{dtype} stddev of all NaN")
      %i[mean rms min max ptp].each do |op|
        close.call(none.send(op, nan: true), nan, "#{dtype} #{op} of all NaN")
      end
    end

    # whole rows are NaN, so the answer is the plain reduction over the rows
    # that are left
    [Cumo::DFloat, Cumo::DComplex].each do |dtype|
      base = dtype.new(6, 40).seq + 1
      nan_rows = [1, 4]
      kept_rows = (0...6).to_a - nan_rows
      a = base.copy
      a[nan_rows, true] = nan
      kept = base[kept_rows, true]
      %i[sum mean var stddev rms].each do |op|
        close.call(a.send(op, axis: 0, nan: true), kept.send(op, axis: 0), "#{dtype} #{op} axis 0")
        close.call(a.send(op, nan: true), kept.send(op), "#{dtype} #{op} flat")
        close.call(a.transpose.send(op, axis: 1, nan: true), kept.transpose.send(op, axis: 1),
                   "#{dtype} #{op} transposed")
      end
      next if dtype == Cumo::DComplex

      # every column meets a NaN row, so every one of these is NaN
      all_nan = Cumo::DFloat.new(40).fill(nan)
      %i[min max ptp].each do |op|
        close.call(a.send(op, axis: 0, nan: true), all_nan, "#{dtype} #{op} axis 0")
        close.call(a.transpose.send(op, axis: 1, nan: true), all_nan, "#{dtype} #{op} transposed")
      end
      got = a.minmax(axis: 0, nan: true)
      close.call(got[0], all_nan, "#{dtype} minmax axis 0 min")
      close.call(got[1], all_nan, "#{dtype} minmax axis 0 max")
      # a column with no NaN keeps the plain answer
      clean = base[kept_rows, true]
      %i[min max ptp].each do |op|
        close.call(clean.send(op, axis: 0, nan: true), clean.send(op, axis: 0),
                   "#{dtype} #{op} axis 0 without nan")
      end
    end
  end

  test "a boolean mask reaches every element of a view and no others" do
    mask_expect = lambda do |a, m|
      flat = a.flatten.to_a
      bits = m.flatten.to_a
      flat.each_with_index.select { |_, i| bits[i] == 1 }.map(&:first)
    end
    check = lambda do |label, a, m|
      assert_equal(mask_expect.call(a, m), a[m].to_a, label)
    end
    # around a word of bits, and around the size the loop is split at
    [7, 31, 32, 33, 1000, 8191, 8192, 8193, 40_000].each do |n|
      a = Cumo::DFloat.new(n).seq
      check.call("1d n=#{n} half", a, a.gt(n / 2))
      check.call("1d n=#{n} none", a, a.gt(n * 2))
      check.call("1d n=#{n} all", a, a.ge(0))
      check.call("1d n=#{n} alternating", a, (a % 2).eq(0))
    end
    # rows long enough that the loop reaches the kernel; the transpose below is
    # the short-row case, and only an index on the innermost axis reaches the
    # loop as an index array
    base = Cumo::DFloat.new(8, 20_000).seq
    order = (0...20_000).to_a.shuffle(random: Random.new(3))
    views = {
      "contiguous" => base,
      "column slice" => base[true, 0.step(19_999, 2)],
      "reversed" => base[true, 19_999.step(0, -1)],
      "index array on the outer axis" => base[[7, 0, 3, 5, 1], true],
      "index array on the innermost axis" => base[true, order],
      "index array on both axes" => base[[7, 0, 3, 5, 1], order],
      "transposed" => base.transpose,
    }
    views.each do |label, v|
      check.call(label, v, v.gt(80_000))
      check.call("#{label} on a reversed mask", v, v.reverse.gt(80_000))
    end
  end

  test "at() rejects a scalar subscript" do
    a = Cumo::DFloat.new(3, 3, 3).seq
    assert_raise(IndexError) { a.at(0, 1, 2) }
    assert_raise(IndexError) { a.at([0, 1], 1, 2) }
    assert_raise(IndexError) { a.at(0.0, 1, 2) }
    assert_raise(IndexError) { Cumo::DFloat.new(4).seq.at(1) }
    assert_equal([2.0, 13.0, 24.0], a.at([0, 1, 2], [0, 1, 2], [-1, -2, -3]).to_a)
    assert_equal([0.0, 13.0], a.at(0..1, 0..1, 0..1).to_a)
    assert_equal([5.0], a.at([0], [1], [2]).to_a)
    assert_equal([5.0], a.at(Cumo::Int32[0], Cumo::Int32[1], Cumo::Int32[2]).to_a)
  end

  test "a subclass of RObject keeps its data on the host too" do
    klass = Class.new(Cumo::RObject)
    a = klass.new(6)
    a.allocate
    a.store([1, 2, 3, 4, 5, 6])
    assert_equal([1, 2, 3, 4, 5, 6], a.copy.to_a)
    assert_equal([5, 1, 3], a[[4, 0, 2]].copy.to_a)
    assert_equal(true, a.free)
  end
end
