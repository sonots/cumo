# frozen_string_literal: true

require_relative "test_helper"

class BitTest < Test::Unit::TestCase
  dtype = Cumo::Bit

  test dtype do
    assert { dtype < Cumo::NArray }
  end

  procs = [
    [proc { |tp, a| tp[*a] }, ""],
    [proc { |tp, a| tp[*a][true] }, "[true]"],
    [proc { |tp, a| tp[*a][0..-1] }, "[0..-1]"]
  ]
  procs.each do |init, ref|

    test "#{dtype},[0,1,1,0,1,0,0,1]#{ref}" do
      src = [0, 1, 1, 0, 1, 0, 0, 1]
      n = src.size
      a = init.call(dtype, src)

      assert { a == src }
      assert { (a & 0) == [0] * n }
      assert { (a & 1) == src }
      assert { (a | 0) == src }
      assert { (a | 1) == [1] * n }
      assert { (a ^ 0) == src.map { |x| x ^ 0 } }
      assert { (a ^ 1) == src.map { |x| x ^ 1 } }
      assert { ~a == src.map { |x| 1 - x } }

      assert { a.count_true == 4 }
      assert { a.count_false == 4 }
      assert { a.where == [1, 2, 4, 7] }
      assert { a.where2 == [[1, 2, 4, 7], [0, 3, 5, 6]] }
      assert { a.mask(Cumo::DFloat[1, 2, 3, 4, 5, 6, 7, 8]) == [2, 3, 5, 8] }
      assert { !a.all? }
      assert { a.any? }
      assert { !a.none? }
    end
  end

  procs = [
    [proc { |tp, a| tp[*a] }, ""],
    [proc { |tp, a| tp[*a][true, 0..-1] }, "[true,true]"],
  ]
  procs.each do |init, ref|

    test "#{dtype},[[0,1,1,0],[1,0,0,1]]#{ref}" do
      src = [[0, 1, 1, 0], [1, 0, 0, 1]]
      a = init.call(dtype, src)

      assert { a[5] == 0 }
      assert { a[-1] == 1 }
      assert { a[1, 0] == src[1][0] }
      assert { a[1, 1] == src[1][1] }
      assert { a[1, 2] == src[1][2] }
      assert { a[3..4] == [0, 1] }
      assert { a[0, 1..2] == [1, 1] }
      assert { a[0, :*] == src[0] }
      assert { a[1, :*] == src[1] }
      assert { a[:*, 1] == [src[0][1], src[1][1]] }

      assert { a.count_true == 4 }
      assert { a.count_false == 4 }
      assert { a.where == [1, 2, 4, 7] }
      assert { a.where2 == [[1, 2, 4, 7], [0, 3, 5, 6]] }
      assert { a.mask(Cumo::DFloat[[1, 2, 3, 4], [5, 6, 7, 8]]) == [2, 3, 5, 8] }
      assert { !a.all? }
      assert { a.any? }
      assert { !a.none? }
    end

    test "#{dtype},[[0,1,1,0],[1,0,0,1]]#{ref},aset[]=" do
      src = [[0, 1, 1, 0], [1, 0, 0, 1]]

      a = init.call(dtype, src)
      a[5] = 1
      assert { a[5] == 1 }

      a = init.call(dtype, src)
      a[-1] = 0
      assert { a[-1] == 0 }

      a = init.call(dtype, src)
      a[1, 0] = 0
      assert { a[1, 0] == 0 }

      a = init.call(dtype, src)
      a[1, 1] = 1
      assert { a[1, 1] == 1 }

      a = init.call(dtype, src)
      a[1, 2] = 1
      assert { a[1, 2] == 1 }

      a = init.call(dtype, src)
      a[3..4] = [1, 0]
      assert { a[3..4] == [1, 0] }

      a = init.call(dtype, src)
      a[0, 1..2] = [0, 0]
      assert { a[0, 1..2] == [0, 0] }

      a = init.call(dtype, src)
      a[0, :*] = [1, 0, 0, 1]
      assert { a[0, :*] == [1, 0, 0, 1] }

      a = init.call(dtype, src)
      a[1, :*] = [0, 1, 1, 0]
      assert { a[1, :*] == [0, 1, 1, 0] }

      a = init.call(dtype, src)
      a[:*, 1] = [0, 1]
      assert { a[:*, 1] == [0, 1] }

      a = init.call(dtype, src)
      a[5] = dtype.cast(1)
      assert { a[5] == 1 }
      assert { a[5] == dtype.cast(1) }

      a = init.call(dtype, src)
      a[1, 0] = dtype.cast(0)
      assert { a[1, 0] == 0 }
      assert { a[1, 0] == dtype.cast(0) }

      a = init.call(dtype, src)
      a[3..4] = dtype.cast([1, 0])
      assert { a[3..4] == [1, 0] }
      assert { a[3..4] == dtype.cast([1, 0]) }

      a = init.call(dtype, src)
      a[:*, 1] = dtype.cast([0, 1])
      assert { a[:*, 1] == [0, 1] }
      assert { a[:*, 1] == dtype.cast([0, 1]) }
    end
  end

  procs = [
    [proc{|tp,a| tp[*a] },""],
  ]
  procs.each do |init, ref|

    test "#{dtype},[]#{ref}" do
      src = []
      n = src.size
      a = init.call(dtype, src)

      assert { a == src }
      assert { (a & 0) == [0] * n }
      assert { (a & 1) == src }
      assert { (a | 0) == src }
      assert { (a | 1) == [1] * n }
      assert { (a ^ 0) == src.map {|x| x ^ 0 } }
      assert { (a ^ 1) == src.map {|x| x ^ 1 } }
      assert { ~a == src.map {|x| 1 - x } }

      assert { a.count_true == 0 }
      assert { a.count_false == 0 }
      assert { a.where == [] }
      assert { a.where2 == [[], []] }
      assert { a.mask(Cumo::DFloat[]) == [] }
      assert { !a.all? }
      assert { !a.any? }
      assert { a.none? }
    end

  end

  test "store to view" do
    n = 14
    x = Cumo::Bit.zeros(n + 2, n + 2, 3)
    ~(x[1..-2, 1..-2, 0].inplace)
    assert { x.where.size == n * n }

    x1 = Cumo::Bit.ones(n, n)
    x0 = Cumo::Bit.zeros(n, n)
    y0 = Cumo::Bit.zeros(n + 2, n + 2)
    x = Cumo::NArray.dstack([x1, x0, x0])
    y = Cumo::NArray.dstack([y0, y0, y0])
    y[1..-2, 1..-2, true] = x
    assert { (~y[1..-2, 1..-2, 0]).where.size == 0 }
    assert { y[true, true, 1].where.size == 0 }
  end

  test "assign nil" do
    x = Cumo::RObject.cast([1, 2, 3])
    x[Cumo::Bit.cast([0, 1, 0])] = nil
    assert { x.to_a == [1, nil, 3] }
  end

  test "store a Range element once" do
    assert_equal([0] + [1] * 63, Cumo::Bit.cast([(0...64)]).to_a)
  end

  test "store fills every slot after a Range" do
    assert_equal([1, 0, 1, 0], Cumo::Bit.cast([1, 0...2, 0]).to_a)
    assert_equal([0, 1, 0, 0], Cumo::Bit.ones(4).store([(0...2), 0, 0]).to_a)
  end

  test "store a Range does not write past the destination" do
    a = Cumo::Bit.new(128)
    a.store(0)
    a[0...64].store([(0...64)])
    assert_equal(0, a[64].to_a.first)
  end

  bits = proc { |n| (0...n).map { |i| (i * 7 + 3) % 5 == 0 ? 1 : 0 } }

  [31, 32, 33, 63, 64, 65, 1000].each do |n|
    test "count_true counts a whole array of #{n}" do
      src = bits.call(n)
      a = Cumo::Bit.cast(src)
      assert { a.count_true == src.count(1) }
      assert { a.count_false == src.count(0) }
    end

    test "count_true counts a view of #{n} that starts mid-word" do
      src = bits.call(n)
      a = Cumo::Bit.cast(src)
      [1, 5, 30].each do |off|
        assert { a[off..-1].count_true == src[off..-1].count(1) }
        assert { a[off..-1].count_false == src[off..-1].count(0) }
      end
    end

    test "count_true counts a view of #{n} that is not contiguous" do
      src = bits.call(n)
      a = Cumo::Bit.cast(src)
      [2, 3, 7].each do |step|
        want = (0...n).step(step).map { |i| src[i] }
        assert { a[(0...n).step(step)].count_true == want.count(1) }
        assert { a[(0...n).step(step)].count_false == want.count(0) }
      end
      assert { a[(n - 1).step(0, -1)].count_true == src[1..-1].count(1) }
      idx = (0...n).select { |i| i % 11 == 4 }
      assert { a[idx].count_true == idx.count { |i| src[i] == 1 } }
      assert { a[idx].count_false == idx.count { |i| src[i] == 0 } }
    end
  end

  test "count_true reduces along an axis across word boundaries" do
    rows = 5
    cols = 70
    src = (0...rows).map { |r| bits.call(cols).rotate(r) }
    a = Cumo::Bit.cast(src)
    assert_equal(src.map { |row| row.count(1) }, a.count_true(axis: 1).to_a)
    assert_equal(src.map { |row| row.count(0) }, a.count_false(axis: 1).to_a)
    assert_equal(src.transpose.map { |col| col.count(1) }, a.count_true(axis: 0).to_a)
    assert_equal(src.transpose.map { |col| col.count(1) }, a.transpose.count_true(axis: 1).to_a)
    assert { a.count_true == src.flatten.count(1) }
  end

  # A Bit count ran its whole loop once per output element, so both an axis
  # reduction and a view ndloop cannot flatten cost a kernel launch per row.
  # The reduce axis is addressed in the kernel now, which is what the layouts
  # below check: each row of a column slice starts at a different bit of its
  # word, and a stepped or indexed row is not a run of bits at all.
  test "count_true reduces a view along an axis" do
    rows = 9
    cols = 37
    src = (0...rows).map { |r| bits.call(cols * 2).rotate(r) }
    a = Cumo::Bit.cast(src)

    slice = a[true, 0...cols]
    want = src.map { |row| row[0, cols] }
    assert_equal(want.map { |row| row.count(1) }, slice.count_true(axis: 1).to_a)
    assert_equal(want.map { |row| row.count(0) }, slice.count_false(axis: 1).to_a)
    assert_equal(want.transpose.map { |col| col.count(1) }, slice.count_true(axis: 0).to_a)
    assert_equal(want.flatten.count(1), slice.count_true.to_i)

    stepped = a[true, (0...(cols * 2)).step(3)]
    want = src.map { |row| (0...(cols * 2)).step(3).map { |i| row[i] } }
    assert_equal(want.map { |row| row.count(1) }, stepped.count_true(axis: 1).to_a)
    assert_equal(want.map { |row| row.count(0) }, stepped.count_false(axis: 1).to_a)
    assert_equal(want.flatten.count(1), stepped.count_true.to_i)

    idx = (0...(cols * 2)).select { |i| i % 5 == 2 }
    picked = a[true, idx]
    want = src.map { |row| idx.map { |i| row[i] } }
    assert_equal(want.map { |row| row.count(1) }, picked.count_true(axis: 1).to_a)
    assert_equal(want.transpose.map { |col| col.count(1) }, picked.count_true(axis: 0).to_a)
    assert_equal(want.flatten.count(1), picked.count_true.to_i)
  end

  test "count_true reduces three dimensions over any set of axes" do
    shape = [4, 5, 66]
    src = bits.call(shape.inject(:*))
    a = Cumo::Bit.cast(src).reshape(*shape)

    [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]].each do |axis|
      kept = (0...3).to_a - axis
      counts = Hash.new(0)
      shape[0].times do |i|
        shape[1].times do |j|
          shape[2].times do |k|
            next unless src[(i * shape[1] + j) * shape[2] + k] == 1
            counts[kept.map { |d| [i, j, k][d] }] += 1
          end
        end
      end
      want = kept.map { |d| shape[d] }
                 .inject([[]]) { |acc, n| acc.flat_map { |p| (0...n).map { |v| p + [v] } } }
                 .map { |key| counts[key] }
      assert_equal(want, a.count_true(axis: axis).flatten.to_a, "axis #{axis.inspect}")
    end
  end

  # Few outputs and a long axis leave the grid one block per output, so the
  # count runs the axis in chunks and combines them in a second pass. 200000
  # bits is past the point where that starts.
  test "count_true splits a long axis across blocks" do
    rows = 2
    cols = 200_003
    src = bits.call(rows * cols)
    a = Cumo::Bit.cast(src).reshape(rows, cols)
    want = (0...rows).map { |r| src[r * cols, cols].count(1) }
    assert_equal(want, a.count_true(axis: 1).to_a)
    assert_equal(want.map { |c| cols - c }, a.count_false(axis: 1).to_a)
    assert_equal(src.count(1), a.count_true.to_i)
    assert_equal(src.count(1), a[true, 1...cols].count_true.to_i + a[true, 0].count_true.to_i)
  end

  # all? and any? had the same loop as the count, one launch per output element,
  # and they answer a Bit rather than a number. Both reduce a whole axis now,
  # which is why the patterns here have their odd element out at either end.
  test "all? and any? reduce along an axis" do
    rows = 6
    cols = 70
    [->(r, c) { 1 },
     ->(r, c) { 0 },
     ->(r, c) { c == cols - 1 && r.even? ? 0 : 1 },
     ->(r, c) { c.zero? && r.odd? ? 1 : 0 },
     ->(r, c) { (r + c) % 3 == 0 ? 1 : 0 }].each_with_index do |gen, k|
      src = (0...rows).map { |r| (0...cols).map { |c| gen.call(r, c) } }
      a = Cumo::Bit.cast(src)
      at = "pattern #{k}"

      assert_equal(src.map { |row| row.all?(1) ? 1 : 0 }, a.all?(axis: 1).to_a, "#{at} all? axis 1")
      assert_equal(src.map { |row| row.any?(1) ? 1 : 0 }, a.any?(axis: 1).to_a, "#{at} any? axis 1")
      assert_equal(src.map { |row| row.any?(1) ? 0 : 1 }, a.none?(axis: 1).to_a, "#{at} none? axis 1")
      assert_equal(src.transpose.map { |col| col.all?(1) ? 1 : 0 }, a.all?(axis: 0).to_a, "#{at} all? axis 0")
      assert_equal(src.transpose.map { |col| col.any?(1) ? 1 : 0 }, a.any?(axis: 0).to_a, "#{at} any? axis 0")

      assert_equal(src.flatten.all?(1), a.all?, "#{at} all?")
      assert_equal(src.flatten.any?(1), a.any?, "#{at} any?")
      assert_equal(!src.flatten.any?(1), a.none?, "#{at} none?")

      slice = a[true, 1...(cols - 1)]
      want = src.map { |row| row[1, cols - 2] }
      assert_equal(want.map { |row| row.all?(1) ? 1 : 0 }, slice.all?(axis: 1).to_a, "#{at} view all?")
      assert_equal(want.map { |row| row.any?(1) ? 1 : 0 }, slice.any?(axis: 1).to_a, "#{at} view any?")
      assert_equal(want.flatten.any?(1), slice.any?, "#{at} view any? of the whole")
    end
  end

  # An operand whose rows are runs of bits but which sit apart -- a column slice
  # -- was read one bit at a time through the indexer, two runtime divisions an
  # element, while the output took whole words from one thread each. The word is
  # gathered from the operand's own rows now. A row length that is not a whole
  # number of words makes a word straddle two rows, which is the case that has
  # to come out right.
  test "Bit elementwise ops take a word from a view whose rows are runs" do
    [[5, 70], [9, 37], [7, 31], [4, 96], [2, 33], [3, 200]].each do |rows, cols|
      wide = cols * 2
      s1 = (0...(rows * wide)).map { |i| (i * 7 + 3) % 5 == 0 ? 1 : 0 }
      s2 = (0...(rows * wide)).map { |i| (i * 3 + 1) % 4 == 0 ? 1 : 0 }
      a = Cumo::Bit.cast(s1).reshape(rows, wide)
      b = Cumo::Bit.cast(s2).reshape(rows, wide)
      take = ->(src, off) { (0...rows).map { |r| (0...cols).map { |c| src[r * wide + off + c] } } }
      zip = ->(x, y, &op) { x.zip(y).map { |u, v| u.zip(v).map { |m, n| op.call(m, n) } } }

      [0, 1].each do |off|
        va = a[true, off...(off + cols)]
        vb = b[true, off...(off + cols)]
        wa = take.call(s1, off)
        wb = take.call(s2, off)
        at = "#{rows}x#{cols} off #{off}"

        dst = Cumo::Bit.new(rows, cols).fill(1)
        dst.store(va)
        assert_equal(wa, dst.to_a, "#{at} store")
        assert_equal(wa, va.copy.to_a, "#{at} copy")
        assert_equal(wa.map { |row| row.map { |v| 1 - v } }, (~va).to_a, "#{at} not")
        assert_equal(zip.call(wa, wb) { |u, v| u & v }, (va & vb).to_a, "#{at} and")
        assert_equal(zip.call(wa, wb) { |u, v| u | v }, (va | vb).to_a, "#{at} or")
        assert_equal(zip.call(wa, wb) { |u, v| u ^ v }, (va ^ vb).to_a, "#{at} xor")
        assert_equal(wa.flatten.count(1), va.count_true.to_i, "#{at} count_true")
      end
    end
  end

  test "Bit elementwise ops keep the bit-at-a-time path for a scattered operand" do
    n = 96
    src = (0...(n * n)).map { |i| (i * 7 + 3) % 5 == 0 ? 1 : 0 }
    a = Cumo::Bit.cast(src).reshape(n, n)
    at = ->(i, j) { src[i * n + j] }

    t = a.transpose
    assert_equal((0...n).map { |i| (0...n).map { |j| at.call(j, i) } }, t.copy.to_a, "transposed copy")
    assert_equal((0...n).map { |i| (0...n).map { |j| at.call(j, i) ^ at.call(i, j) } }, (t ^ a).to_a, "transposed xor")

    step = a[true, (0...n).step(2)]
    assert_equal((0...n).map { |i| (0...n).step(2).map { |j| at.call(i, j) } }, step.copy.to_a, "stepped copy")

    idx = (0...n).to_a.rotate(7)
    picked = a[true, idx]
    assert_equal((0...n).map { |i| idx.map { |j| at.call(i, j) } }, picked.copy.to_a, "indexed copy")
  end

  # A view whose rows are each a run of bits but sit apart -- a column slice --
  # is not flat over the whole reduce group, and the fold within a row was given
  # up along with it. These rows start at a different bit of their word from one
  # another and do not end on a word boundary, which is what the fold has to
  # carry now.
  test "count_true folds words within a row of a view it cannot flatten" do
    [[9, 37], [5, 32], [7, 31], [4, 96], [3, 200]].each do |rows, cols|
      src = (0...rows).map { |r| bits.call(cols * 2).rotate(r * 3) }
      a = Cumo::Bit.cast(src)
      slice = a[true, 0...cols]
      want = src.map { |row| row[0, cols] }.flatten
      at = "#{rows}x#{cols}"

      assert_equal(want.count(1), slice.count_true.to_i, "#{at} count_true")
      assert_equal(want.count(0), slice.count_false.to_i, "#{at} count_false")
      assert_equal(want.count(1), slice.count_true(axis: [0, 1]).to_i, "#{at} both axes")
      assert_equal(want.all? { |v| v == 1 }, slice.all?, "#{at} all?")
      assert_equal(want.any? { |v| v == 1 }, slice.any?, "#{at} any?")
    end
  end

  test "count_true folds words within the trailing axes of a 3-D view" do
    rows, mid, cols = 4, 5, 66
    wide = cols * 2
    src = bits.call(rows * mid * wide)
    a = Cumo::Bit.cast(src).reshape(rows, mid, wide)[true, true, 0...cols]
    at = ->(i, j, k) { src[(i * mid + j) * wide + k] }

    want = (0...rows).map do |i|
      (0...mid).sum { |j| (0...cols).count { |k| at.call(i, j, k) == 1 } }
    end
    assert_equal(want, a.count_true(axis: [1, 2]).to_a)
    assert_equal(want.sum, a.count_true.to_i)
    assert_equal(want.map { |c| mid * cols - c }, a.count_false(axis: [1, 2]).to_a)
  end

  # A reduction addresses its operands itself, so mulsum makes one that carries
  # an index array contiguous first -- with a copy that moves whole bytes, where
  # a Bit element is one bit. The sum came out of the wrong bits.
  test "mulsum over an indexed Bit operand reads the bits it was given" do
    n = 12
    src = (0...n).map { |i| i % 3 == 1 ? 1 : 0 }
    a = Cumo::SFloat.new(n).seq(1)
    b = Cumo::Bit.cast(src)

    assert_equal(src.each_index.sum { |i| (i + 1) * src[i] }.to_f, Float(a.mulsum(b)))

    [[1, 4, 7, 10], [0, 1, 2], [11, 10, 9], [4, 4, 4]].each do |idx|
      want = idx.sum { |i| (i + 1) * src[i] }.to_f
      at = "idx #{idx.inspect}"
      assert_equal(idx.map { |i| src[i] }, b[idx].to_a, "#{at} view")
      assert_equal(want, Float(a[idx].mulsum(b[idx])), at)
      assert_equal(want, Float(a[idx].mulsum(b[idx].copy)), "#{at} copied")
    end
  end

  test "NArray#copy of a Bit copies bits, not bytes" do
    src = (0...40).map { |i| i % 3 == 1 ? 1 : 0 }
    b = Cumo::Bit.cast(src)
    na_copy = Cumo::NArray.instance_method(:copy)
    assert_equal(src, na_copy.bind(b).call.to_a)
    idx = (0...40).select { |i| i % 7 == 2 }
    assert_equal(idx.map { |i| src[i] }, na_copy.bind(b[idx]).call.to_a)
    assert_equal(src[5, 20], na_copy.bind(b[5...25]).call.to_a)
  end

  test "all? and any? split a long axis across blocks" do
    rows = 2
    cols = 200_003
    [[cols - 1, 0], [0, 1]].each do |flip, fill|
      src = (0...rows).map { |r| a = Array.new(cols, fill); a[flip] = 1 - fill; a }
      a = Cumo::Bit.cast(src)
      assert_equal(src.map { |row| row.all?(1) ? 1 : 0 }, a.all?(axis: 1).to_a, "fill #{fill}")
      assert_equal(src.map { |row| row.any?(1) ? 1 : 0 }, a.any?(axis: 1).to_a, "fill #{fill}")
      assert_equal(src.flatten.all?(1), a.all?, "fill #{fill} whole")
      assert_equal(src.flatten.any?(1), a.any?, "fill #{fill} whole")
    end
  end

  # where, where2 and mask only reach the device below
  # CUMO_BIT_WHERE_MIN_KERNEL_SIZE, which is 8192, so these are the sizes that
  # exercise the compaction rather than the host loop. 4096 elements is one
  # block's range, so the sizes straddle that too.
  [8192, 8193, 12_289, 100_003].each do |n|
    patterns = {
      "all true" => proc { |i| 1 },
      "all false" => proc { |i| 0 },
      "sparse" => proc { |i| i % 97 == 3 ? 1 : 0 },
      "mixed" => proc { |i| (i * 7 + i / 13) % 3 == 0 ? 1 : 0 }
    }
    patterns.each do |kind, gen|
      test "where and mask over #{n} #{kind} bits" do
        src = (0...n).map { |i| gen.call(i) }
        a = Cumo::Bit.cast(src)
        f = Cumo::SFloat.new(n).seq
        ones = (0...n).select { |i| src[i] == 1 }
        zeros = (0...n).select { |i| src[i].zero? }
        assert_equal(ones, a.where.to_a)
        assert_equal([ones, zeros], a.where2.map(&:to_a))
        assert_equal(ones.map(&:to_f), f[a].to_a)
        assert_equal(ones.size, a.count_true[0])
      end
    end

    test "where and mask over a view of #{n} bits" do
      src = (0...(2 * n)).map { |i| (i * 5 + 2) % 4 == 0 ? 1 : 0 }
      a = Cumo::Bit.cast(src)
      f = Cumo::SFloat.new(2 * n).seq
      [2, 3].each do |st|
        r = (0...(2 * n)).step(st)
        want = r.each_with_index.select { |j, _| src[j] == 1 }.map { |_, k| k }
        assert_equal(want, a[r].where.to_a)
        assert_equal(want.map { |k| (k * st).to_f }, f[r][a[r]].to_a)
      end
      [1, 5, 33].each do |off|
        want = (0...n).select { |i| src[off + i] == 1 }
        assert_equal(want, a[off...(off + n)].where.to_a)
        assert_equal(want.map { |i| (off + i).to_f }, f[off...(off + n)][a[off...(off + n)]].to_a)
      end
      rev = (2 * n - 1).step(0, -1)
      assert_equal((0...(2 * n)).select { |k| src[2 * n - 1 - k] == 1 }, a[rev].where.to_a)
    end
  end

  # The cursor that orders one loop segment after the next lives on the device,
  # so a shape that ndloop walks in more than one segment is its own case.
  test "where and mask over a multi-dimensional mask" do
    r, c = 70, 1000
    src = (0...(r * c)).map { |i| (i * 3 + 1) % 5 == 0 ? 1 : 0 }
    a = Cumo::Bit.cast(src).reshape(r, c)
    f = Cumo::DFloat.new(r * c).seq.reshape(r, c)
    ones = (0...(r * c)).select { |i| src[i] == 1 }
    assert_equal(ones, a.where.to_a)
    assert_equal(ones.map(&:to_f), f[a].to_a)
    t = (0...(r * c)).select { |k| src[(k % r) * c + k / r] == 1 }
    assert_equal(t, a.transpose.where.to_a)
    assert_equal(t.map { |k| ((k % r) * c + k / r).to_f }, f.transpose[a.transpose].to_a)
  end

  test "[]= on a frozen view raises whatever the subscript is" do
    # the store goes to a view derived from self, whose data object is the
    # root, so the check inside get_pointer_for_rw never reaches self
    a = Cumo::Bit.new(5).fill(0)
    v = a[1..3]
    v.freeze
    assert_raise(RuntimeError) { v[0] = 1 }
    assert_raise(RuntimeError) { v[0..1] = 1 }
    assert_raise(RuntimeError) { v[[0, 1]] = 1 }
    assert_raise(RuntimeError) { v.store(1) }
    assert_equal([0, 0, 0, 0, 0], a.to_a)
    # and still writes when nothing is frozen
    w = a[1..3]
    w[0] = 1
    w[1..2] = 1
    assert_equal([0, 1, 1, 1, 0], a.to_a)
  end

  # The word at a time store needs only the output contiguous, so these check
  # the operand shapes that used to fall back to one atomic per element.
  [1, 31, 32, 33, 64, 65, 1000].each do |n|
    make = proc { Cumo::Bit.cast((0...(2 * n)).map { |i| (i * 5 + 1) % 3 == 0 ? 1 : 0 }) }

    test "a bit operation on a strided operand of #{n}" do
      a = make.call
      [2, 3, 7].each do |st|
        v = a[(0...(2 * n)).step(st)]
        rows = v.to_a
        assert_equal(rows.map { |x| 1 - x }, (~v).to_a)
        assert_equal(rows, (v & v).to_a)
        assert_equal([0] * v.size, (v ^ v).to_a)
        assert_equal(rows, Cumo::Bit.new(v.size).store(v).to_a)
      end
      rev = a[(2 * n - 1).step(0, -1)]
      assert_equal(rev.to_a.map { |x| 1 - x }, (~rev).to_a)
      idx = (0...(2 * n)).select { |i| i % 13 == 2 }
      assert_equal(a[idx].to_a.map { |x| 1 - x }, (~a[idx]).to_a)
      assert_equal(a[idx].to_a, Cumo::Bit.new(idx.size).store(a[idx]).to_a)
    end

    test "a bit operation on an operand of #{n} that starts mid-word" do
      a = make.call
      [1, 5, 31, 33].each do |off|
        next if off + n > 2 * n
        v = a[off...(off + n)]
        assert_equal(v.to_a.map { |x| 1 - x }, (~v).to_a)
        assert_equal(v.to_a, Cumo::Bit.new(n).store(v).to_a)
      end
    end

    # The destination is not contiguous here, so the atomic has to stay: the
    # elements it does not own belong to the rest of the array.
    test "a bit operation writing into a view of #{n} leaves the rest alone" do
      a = make.call[0...n]
      want = a.to_a.map { |x| 1 - x }
      [0, 1, 5, 32, 33, 64].each do |off|
        d = Cumo::Bit.ones(n + 128)
        d[off...(off + n)] = ~a
        assert_equal([1] * off + want + [1] * (128 - off), d.to_a)
      end
      d = Cumo::Bit.ones(2 * n)
      d[(0...(2 * n)).step(2)] = ~a
      assert_equal((0...(2 * n)).map { |i| i.odd? ? 1 : want[i / 2] }, d.to_a)
    end
  end

  test "a bit operation on a transposed operand" do
    r, c = 33, 65
    a = Cumo::Bit.cast((0...r).map { |i| (0...c).map { |j| (i + j) % 3 == 0 ? 1 : 0 } })
    t = a.transpose
    assert_equal(t.to_a.map { |row| row.map { |x| 1 - x } }, (~t).to_a)
    assert_equal(t.to_a, (t & t).to_a)
    assert_equal(t.to_a, Cumo::Bit.new(c, r).store(t).to_a)
  end

  # A contiguous bit output is built a word at a time by a whole warp, so the
  # sizes that matter are the ones around a word and the operands whose output
  # is not contiguous and has to fall back.
  [1, 31, 32, 33, 63, 64, 65, 129, 1000].each do |n|
    src = proc { (0...n).map { |i| (i % 7) - 3 } }

    test "a comparison fills #{n} bits" do
      want = src.call.map { |x| x > 0 ? 1 : 0 }
      a = Cumo::Int32.cast(src.call)
      assert_equal(want, (a > 0).to_a)
      assert_equal(want.map { |x| 1 - x }, (a <= 0).to_a)
      assert_equal(want, Cumo::SFloat.cast(src.call).gt(0).to_a)
      assert_equal(src.call.map { |x| x == 1 ? 1 : 0 }, a.eq(1).to_a)
      assert_equal(want, (0 < a).to_a)
    end

    test "a cast to Bit fills #{n} bits" do
      assert_equal(src.call.map { |x| x.zero? ? 0 : 1 }, Cumo::Bit.cast(Cumo::Int32.cast(src.call)).to_a)
    end

    test "a comparison of a non-contiguous operand fills #{n} bits" do
      wide = (0...(2 * n)).map { |i| (i % 7) - 3 }
      a = Cumo::Int32.cast(wide).reshape(n, 2)
      assert_equal((0...n).map { |i| wide[2 * i] > 0 ? 1 : 0 }, (a[true, 0] > 0).to_a)
      assert_equal((0...n).map { |i| wide[2 * i + 1] > 0 ? 1 : 0 }, (a[true, 1] > 0).to_a)
      assert_equal(wide.map { |x| x > 0 ? 1 : 0 }, (a.transpose > 0).transpose.flatten.to_a)
    end

    test "a comparison writing into a bit view fills #{n} bits" do
      a = Cumo::Int32.cast(src.call)
      [0, 1, 5, 32].each do |off|
        b = Cumo::Bit.zeros(n + 64)
        b[off...(off + n)] = (a > 0)
        want = [0] * off + src.call.map { |x| x > 0 ? 1 : 0 } + [0] * (64 - off)
        assert_equal(want, b.to_a)
      end
      b = Cumo::Bit.zeros(2 * n)
      b[(0...(2 * n)).step(2)] = (a > 0)
      assert_equal((0...(2 * n)).map { |i| i.odd? ? 0 : (src.call[i / 2] > 0 ? 1 : 0) }, b.to_a)
    end

    # The elements past the end of the last word belong to the rest of the
    # array, so filling that word must leave them as they were. A destination
    # of ones is what makes an unmasked store show up.
    test "a store into a bit view of #{n} leaves the rest of it alone" do
      a = Cumo::Int32.cast(src.call)
      want = src.call.map { |x| x.zero? ? 0 : 1 }
      [0, 1, 5, 31, 32, 33, 64].each do |off|
        b = Cumo::Bit.ones(n + 128)
        b[off...(off + n)] = a
        assert_equal([1] * off + want + [1] * (128 - off), b.to_a)
        b = Cumo::Bit.ones(n + 128)
        b[off...(off + n)].store(a)
        assert_equal([1] * off + want + [1] * (128 - off), b.to_a)
      end
    end
  end

  # Every element is 0 or 1, so the four statistics come out of the same bit
  # count all? and any? reduce, and DFloat is what says the count is right.
  test "mean, var, stddev and rms of a bit array" do
    [[1000], [64, 129], [7, 33, 17], [1 << 22]].each do |shape|
      srand(7)
      a = Cumo::Bit.cast(Array.new(shape.reduce(:*)) { rand(2) }).reshape(*shape)
      ref = Cumo::DFloat.cast(a)

      ([[]] + (0...shape.size).map { |axis| [axis] }).each do |axis|
        assert { a.mean(*axis).instance_of?(Cumo::DFloat) }
        assert { (a.mean(*axis) - ref.mean(*axis)).abs.max.extract_cpu < 1e-12 }
        assert { (a.var(*axis) - ref.var(*axis)).abs.max.extract_cpu < 1e-12 }
        assert { (a.stddev(*axis) - ref.stddev(*axis)).abs.max.extract_cpu < 1e-12 }
        assert { (a.rms(*axis) - ref.rms(*axis)).abs.max.extract_cpu < 1e-12 }
      end
    end

    a = Cumo::Bit[[1, 0, 1], [1, 1, 0]]
    assert { (a.mean.extract_cpu - (2.0 / 3)).abs < 1e-15 }
    assert { a.mean(0) == [1, 0.5, 0.5] }
    assert { a.mean(axis: 1, keepdims: true).shape == [2, 1] }

    assert { Cumo::Bit[0, 0, 0, 0].mean.extract_cpu.zero? }
    assert { Cumo::Bit[0, 0, 0, 0].var.extract_cpu.zero? }
    assert { (Cumo::Bit[1, 1, 1, 1].mean.extract_cpu - 1).abs < 1e-15 }
    assert { Cumo::Bit[1, 1, 1, 1].var.extract_cpu.zero? }
    assert { (Cumo::Bit[1].mean.extract_cpu - 1).abs < 1e-15 }
    assert { Cumo::Bit[1].var.extract_cpu.nan? }
    assert { Cumo::Bit[1].stddev.extract_cpu.nan? }
    assert { (Cumo::Bit[1].rms.extract_cpu - 1).abs < 1e-15 }
  end

  test "mean, var, stddev and rms of a bit view" do
    srand(11)
    base = Cumo::Bit.cast(Array.new(64 * 65) { rand(2) }).reshape(64, 65)

    [base[true, 1..-2], base[(0..-1).step(2), true], base[[3, 1, 2, 0], true]].each do |v|
      ref = Cumo::DFloat.cast(v)
      [[], [0], [1]].each do |axis|
        assert { (v.mean(*axis) - ref.mean(*axis)).abs.max.extract_cpu < 1e-12 }
        assert { (v.var(*axis) - ref.var(*axis)).abs.max.extract_cpu < 1e-12 }
        assert { (v.stddev(*axis) - ref.stddev(*axis)).abs.max.extract_cpu < 1e-12 }
        assert { (v.rms(*axis) - ref.rms(*axis)).abs.max.extract_cpu < 1e-12 }
      end
    end
  end
end
