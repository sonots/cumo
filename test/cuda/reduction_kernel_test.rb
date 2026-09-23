# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class ReductionKernelTest < Test::Unit::TestCase
    L2NORM = ReductionKernel.new("T x", "T y", "x * x", "a + b", "y = sqrt(a)", "0", "l2norm")
    SUM = ReductionKernel.new("T x", "T y", "x", "a + b", "y = a", "0", "plain_sum")

    def close(a, b, tol = 1e-4)
      (Cumo::DFloat.cast(a) - Cumo::DFloat.cast(b)).abs.max.to_f < tol
    end

    test "the map, reduce and post expressions run along the axis" do
      x = Cumo::SFloat.new(2, 5).seq
      y = L2NORM.call(x, axis: 1)
      assert_equal(Cumo::SFloat, y.class)
      assert_equal([2], y.shape)
      assert { close(y, [Math.sqrt(30), Math.sqrt(255)]) }
      assert { close(L2NORM.call(x), [Math.sqrt(285)]) }
      assert_equal([], L2NORM.call(x).shape)
    end

    test "axis takes a negative, a list or nothing, and keepdims keeps the axes" do
      x = Cumo::DFloat.new(3, 4, 5).seq
      assert { close(SUM.call(x, axis: 0), x.sum(axis: 0)) }
      assert { close(SUM.call(x, axis: -1), x.sum(axis: 2)) }
      assert { close(SUM.call(x, axis: [0, 2]), x.sum(axis: [0, 2])) }
      assert { close(SUM.call(x, axis: [2, 0]), x.sum(axis: [0, 2])) }
      assert { close(SUM.call(x), [x.sum.to_f]) }
      assert_equal([3, 1, 5], SUM.call(x, axis: 1, keepdims: true).shape)
      assert_equal([1, 1, 1], SUM.call(x, keepdims: true).shape)
      assert { close(SUM.call(x, axis: 1, keepdims: true), x.sum(axis: 1, keepdims: true)) }
    end

    test "the sum agrees with Cumo's own on random data over every axis and length" do
      [[7], [1000], [4, 1000], [1000, 4], [3, 17, 33], [513, 2], [2, 513], [1, 2049]].each do |shape|
        x = Cumo::DFloat.new(*shape).rand(-1, 1)
        (0...shape.size).each do |axis|
          assert { close(SUM.call(x, axis: axis), x.sum(axis: axis), 1e-9) }
        end
        assert { close(SUM.call(x), [x.sum.to_f], 1e-9) }
      end
    end

    test "a one letter type follows the argument, and the output decides first" do
      [Cumo::Int32, Cumo::Int64, Cumo::UInt8, Cumo::DFloat].each do |dtype|
        x = dtype.new(4, 6).seq
        y = SUM.call(x, axis: 1)
        assert_equal(dtype, y.class, dtype.to_s)
        assert_equal(x.sum(axis: 1).to_a, y.to_a, dtype.to_s)
      end
      to_s = ReductionKernel.new("T x", "S y", "x", "a + b", "y = a", "0", "sum_to_s")
      y = Cumo::DFloat.new(4)
      assert_same(y, to_s.call(Cumo::Int32.new(4, 6).seq, y, axis: 1))
      assert_equal(Cumo::Int32.new(4, 6).seq.sum(axis: 1).to_a, y.to_a)
      assert_raise(TypeError) { SUM.call(Cumo::Int32.new(4, 6).seq, Cumo::DFloat.new(4), axis: 1) }
      assert_raise(TypeError) { SUM.call(Cumo::SFloat.new(3).seq, Cumo::DFloat.new(3).seq) }
    end

    test "reduce_type decides what the values are accumulated in" do
      x = Cumo::UInt8.new(300).fill(1)
      assert_equal([44], SUM.call(x).to_a)
      wide = ReductionKernel.new("T x", "int32 y", "x", "a + b", "y = a", "0", "wide_sum")
      assert_equal([300], wide.call(x).to_a)
      widest = ReductionKernel.new("T x", "T y", "x", "a + b", "y = a", "0", "widest_sum", reduce_type: "long long")
      assert_equal([44], widest.call(x).to_a)
      named = ReductionKernel.new("T x", "int32 y", "x", "a + b", "y = a", "0", "named_sum", reduce_type: "int64")
      assert_equal([300], named.call(x).to_a)
      lettered = ReductionKernel.new("T x, S w", "S y", "x * w", "a + b", "y = a", "0", "lettered_sum", reduce_type: "S")
      assert_equal([300.0], lettered.call(x, 1.0).to_a)
      mean = ReductionKernel.new("T x", "float32 y", "x", "a + b", "y = a / 300", "0", "mean_of", reduce_type: "double")
      assert_equal([1.0], mean.call(x).to_a)
    end

    test "kmeans' sum_kernel and count_kernel reduce a broadcast mask over the samples" do
      sum_kernel = ReductionKernel.new("T x, S mask", "T out", "mask ? x : 0", "a + b", "out = a", "0", "sum_kernel")
      count_kernel = ReductionKernel.new("T mask", "float32 out", "mask ? 1.0 : 0.0", "a + b", "out = a", "0.0", "count_kernel")
      n = 1000
      k = 3
      x = Cumo::SFloat.new(n, 2).rand
      pred = Cumo::Int32.new(n).rand(k)
      mask = Cumo::UInt8.zeros(k, n)
      k.times { |c| mask[c, true] = pred.eq(c) }
      sums = sum_kernel.call(x, mask[true, true, :new], axis: 1)
      counts = count_kernel.call(mask, axis: 1)
      assert_equal([k, 2], sums.shape)
      assert_equal([k], counts.shape)
      k.times do |c|
        rows = x[pred.eq(c).where, true]
        assert { close(sums[c, true], rows.sum(axis: 0), 1e-3) }
        assert_equal(rows.shape[0].to_f, counts[c].to_f)
      end
    end

    test "a number is a scalar of its declared type and the preamble is in scope" do
      k = ReductionKernel.new("T x, T shift", "T y", "square(x - shift)", "a + b", "y = a", "0", "shifted",
                              preamble: "__device__ inline T square(T v) { return v * v; }")
      x = Cumo::DFloat.new(2, 3).seq
      assert { close(k.call(x, 1.0, axis: 1), ((x - 1)**2).sum(axis: 1)) }
      assert { close(k.call(x, 1, axis: 1), ((x - 1)**2).sum(axis: 1)) }
    end

    test "inputs are broadcast against each other, a view is read where it is, and an axis of length zero gives the identity" do
      x = Cumo::DFloat.new(4, 6).seq
      w = Cumo::DFloat.new(6).seq(1)
      dot = ReductionKernel.new("T x, T w", "T y", "x * w", "a + b", "y = a", "0", "weighted")
      assert { close(dot.call(x, w, axis: 1), (x * w).sum(axis: 1)) }
      assert { close(dot.call(x[true, (0...6).step(2)], w[(0...6).step(2)], axis: 1), (x[true, (0...6).step(2)] * w[(0...6).step(2)]).sum(axis: 1)) }
      assert { close(SUM.call(x.transpose, axis: 0), x.sum(axis: 1)) }
      assert { close(SUM.call(x.freeze, axis: 1), x.sum(axis: 1)) }
      assert_equal([0.0, 0.0], SUM.call(Cumo::DFloat.new(2, 0), axis: 1).to_a)
      assert_equal([0], SUM.call(Cumo::DFloat.new(0, 3), axis: 1).shape)
      assert_equal([3.0], SUM.call(Cumo::DFloat.new.store(3.0)).to_a)
      assert_equal([], SUM.call(Cumo::DFloat.new.store(3.0)).shape)
    end

    test "a transposed or reversed input is read through its strides and not copied" do
      x = Cumo::DFloat.new(64, 48).seq
      [x.transpose, x.reverse(0), x[(63..0).step(-3), (0...48).step(2)], x.transpose.reverse(1)].each do |v|
        [0, 1, nil].each do |axis|
          assert { close(SUM.call(v, axis: axis), axis ? v.sum(axis: axis) : [v.sum.to_f]) }
        end
      end
      omit("needs the memory pool to count with") unless MemoryPool.enabled?
      big = Cumo::SFloat.new(512, 512).seq
      out = Cumo::SFloat.new(512)
      SUM.call(big.transpose, out, axis: 0)
      GC.start
      MemoryPool.free_all_blocks
      before = MemoryPool.total_bytes
      SUM.call(big.transpose, out, axis: 0)
      Runtime.cudaDeviceSynchronize
      assert_equal(0, MemoryPool.total_bytes - before)
      assert { close(out, big.sum(axis: 1)) }
    end

    test "an output that is part of the input does not change what the others read" do
      x = Cumo::DFloat.new(2048, 2048).fill(1)
      out = x[2047, true]
      SUM.call(x.transpose.reverse(0), out, axis: 0)
      assert_equal([2048.0] * 2048, out.to_a)
      y = Cumo::DFloat.new(512, 512).fill(1)
      row = y[0, true]
      SUM.call(y, row, axis: 0)
      assert_equal([512.0] * 512, row.to_a)
    end

    test "every subset of axes agrees with sum, whichever dimensions merge" do
      base = Cumo::DFloat.new(3, 1, 4, 5).seq
      [base, base.transpose(3, 1, 0, 2), base.reverse(2), base[true, true, (3..0).step(-2), true]].each do |x|
        (0..4).each do |r|
          (0...4).to_a.combination(r).each do |axes|
            axis = axes.empty? ? nil : axes
            want = axis ? x.sum(axis: axis) : x.sum
            got = SUM.call(x, axis: axis)
            assert { close(got, want) }
            assert_equal(axis ? want.shape : [], got.shape, "#{x.shape.inspect} axis #{axis.inspect}")
          end
        end
      end
    end

    test "several outputs come back as an Array" do
      k = ReductionKernel.new("T x", "T s, T r", "x", "a + b", "s = a; r = a * 2", "0", "sum_and_double")
      s, r = k.call(Cumo::Int32.new(2, 3).seq, axis: 1)
      assert_equal([3, 12], s.to_a)
      assert_equal([6, 24], r.to_a)
    end

    test "what cannot be reduced is refused before the launch" do
      x = Cumo::DFloat.new(3, 4).seq
      assert_raise(ArgumentError) { SUM.call(x, axis: 2) }
      assert_raise(ArgumentError) { SUM.call(x, axis: -3) }
      assert_raise(ArgumentError) { SUM.call(x, axis: [0, 0]) }
      assert_raise(TypeError) { SUM.call(x, axis: 1.0) }
      assert_raise(ArgumentError) { SUM.call(x, Cumo::DFloat.new(4), axis: 1) }
      assert_raise(ArgumentError) { SUM.call(x, Cumo::DFloat.new(3, 1), axis: 1) }
      assert_raise(ArgumentError) { SUM.call(x, Cumo::DFloat.new(2, 3).transpose, axis: 1) }
      assert_raise(ArgumentError) { SUM.call(2.0) }
      assert_raise(TypeError) { SUM.call(Cumo::Bit.new(3)) }
      assert_raise(TypeError) { SUM.call(Cumo::HFloat.new(3)) }
      assert_raise(ArgumentError) { ReductionKernel.new("raw T x", "T y", "x[i]", "a + b", "y = a", "0", "k") }
      assert_raise(ArgumentError) { ReductionKernel.new("T x", "T a", "x", "a + b", "a = a", "0", "k") }
      assert_raise(ArgumentError) { ReductionKernel.new("T b", "T y", "b", "a + b", "y = a", "0", "k") }
      assert_raise(ArgumentError) { ReductionKernel.new("T x", "T y", "x", "a + b", "y = a", "0", "k", reduce_type: "float32x") }
      assert_raise(ArgumentError) { ReductionKernel.new("T x", "T y", "x", "a + b", "y = a", "0", "k", reduce_type: "R") }
      assert_raise(FrozenError) { SUM.call(x, Cumo::DFloat.zeros(3).freeze, axis: 1) }
      assert_raise(ArgumentError) { ReductionKernel.new("T x", "", "x", "a + b", "y = a", "0", "k") }
      assert_raise(ArgumentError) { ReductionKernel.new("T x", "T y", "x", "a + b", "y = a", "0", "bad name") }
      assert_raise(CompileError) { ReductionKernel.new("T x", "T y", "x +", "a + b", "y = a", "0", "k").call(x) }
    end

    # A few outputs over a long axis go through a second pass that folds the
    # chunks the blocks reduced, so every expression has to survive it.
    test "a long axis reduced to a few values is folded in two passes" do
      x = Cumo::DFloat.new(1_000_000).rand(-1, 1)
      assert { close(SUM.call(x), [x.sum.to_f], 1e-9) }
      y = Cumo::DFloat.new(100_000, 3).rand(-1, 1)
      assert { close(SUM.call(y, axis: 0), y.sum(axis: 0), 1e-9) }
      assert_equal(1, SUM.instance_variable_get(:@finals).size)

      largest = ReductionKernel.new("T x", "T y", "x", "a > b ? a : b", "y = a", -Float::INFINITY, "largest")
      smallest = ReductionKernel.new("T x", "T y", "x", "a < b ? a : b", "y = a", Float::INFINITY, "smallest")
      assert_equal([x.min.to_f], smallest.call(x).to_a)
      assert_equal(y.max(axis: 0).to_a, largest.call(y, axis: 0).to_a)
      assert_equal([x.max.to_f], largest.call(x).to_a)

      count = ReductionKernel.new("T x", "int32 n_pos", "x > 0 ? 1 : 0", "a + b", "n_pos = a", "0", "positives", reduce_type: "long long")
      assert_equal([(x.gt(0)).count_true.to_i], count.call(x).to_a)

      mean_sq = ReductionKernel.new("T x", "T m, T s", "x * x", "a + b", "s = a; m = a / 1000000", "0", "mean_sq")
      m, sq = mean_sq.call(Cumo::SFloat.cast(x))
      assert { close(sq, [(x * x).sum.to_f], 1.0) }
      assert { close(m, [(x * x).sum.to_f / 1_000_000], 1e-4) }

      z = Cumo::SFloat.new(1_000_000).rand
      in_double = ReductionKernel.new("T x", "T y", "x", "a + b", "y = a", "0", "sum_in_double", reduce_type: "double")
      assert { close(in_double.call(z), [Cumo::DFloat.cast(z).sum.to_f], 0.1) }
    end

    test "the kernel is compiled once per set of types and dimensions left after merging" do
      k = ReductionKernel.new("T x", "T y", "x", "a + b", "y = a", "0", "once")
      x = Cumo::SFloat.new(3, 4).seq
      k.call(x, axis: 0)
      k.call(x, axis: 1)
      assert_equal(1, k.instance_variable_get(:@functions).size)
      k.call(x)
      assert_equal(2, k.instance_variable_get(:@functions).size)
      k.call(Cumo::SFloat.new(3, 4, 5).seq, axis: 0)
      assert_equal(2, k.instance_variable_get(:@functions).size)
      k.call(Cumo::SFloat.new(3, 4, 5).seq, axis: 1)
      assert_equal(3, k.instance_variable_get(:@functions).size)
      k.call(Cumo::DFloat.new(3, 4).seq, axis: 1)
      assert_equal(4, k.instance_variable_get(:@functions).size)
    end
  end
end
