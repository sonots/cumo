# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class ElementwiseKernelTest < Test::Unit::TestCase
    SQUARED_DIFF = ElementwiseKernel.new("float32 x, float32 y", "float32 z", "z = (x - y) * (x - y)", "squared_diff")
    GENERIC = ElementwiseKernel.new("T x, T y", "T z", "z = (x - y) * (x - y)", "squared_diff_generic")

    test "the operation runs on every element with broadcasting" do
      x = Cumo::SFloat.new(2, 5).seq
      y = Cumo::SFloat.new(5).seq
      assert_equal([[0] * 5, [25] * 5], SQUARED_DIFF.call(x, y).to_a)
      assert_equal([[25, 16, 9, 4, 1], [0, 1, 4, 9, 16]], SQUARED_DIFF.call(x, 5).to_a)
      assert_equal(Cumo::SFloat, SQUARED_DIFF.call(x, 5).class)
    end

    test "a one letter type takes the dtype of the argument" do
      [Cumo::DFloat, Cumo::Int32, Cumo::UInt8, Cumo::Int64].each do |dtype|
        x = dtype.new(6).seq
        y = dtype.new(6).seq(0, 2)
        z = GENERIC.call(x, y)
        assert_equal(dtype, z.class, dtype.to_s)
        assert_equal(((x - y) * (x - y)).to_a, z.to_a, dtype.to_s)
      end
    end

    test "the same letter cannot be two dtypes, and a named type is held to" do
      assert_raise(TypeError) { GENERIC.call(Cumo::SFloat.new(3).seq, Cumo::DFloat.new(3).seq) }
      assert_raise(TypeError) { SQUARED_DIFF.call(Cumo::DFloat.new(3).seq, Cumo::DFloat.new(3).seq) }
      assert_raise(TypeError) { GENERIC.call(Cumo::SFloat.new(3).seq, Cumo::SFloat.new(3).seq, Cumo::DFloat.new(3)) }
    end

    test "an output given after the inputs is written and answered" do
      x = Cumo::SFloat.new(2, 5).seq
      y = Cumo::SFloat.new(5).seq
      z = Cumo::SFloat.new(2, 5)
      assert_same(z, SQUARED_DIFF.call(x, y, z))
      assert_equal([[0] * 5, [25] * 5], z.to_a)
      assert_raise(ArgumentError) { SQUARED_DIFF.call(x, y, Cumo::SFloat.new(5)) }
      assert_raise(ArgumentError) { SQUARED_DIFF.call(x, y, Cumo::SFloat.new(5, 2).transpose) }
      assert_raise(TypeError) { SQUARED_DIFF.call(x, y, 3) }
      assert_raise(ArgumentError) { SQUARED_DIFF.call(x) }
      assert_raise(ArgumentError) { SQUARED_DIFF.call(x, y, z, z) }
    end

    test "the output decides a type before the inputs do" do
      k = ElementwiseKernel.new("T x", "S y", "y = x * 2", "double_it")
      y = Cumo::DFloat.new(3)
      assert_same(y, k.call(Cumo::Int32.new(3).seq, y))
      assert_equal([0.0, 2.0, 4.0], y.to_a)
    end

    test "a raw argument is indexed by the operation itself" do
      k = ElementwiseKernel.new("T x, raw T y", "T z", "z = x + y[_ind.size() - i - 1]", "add_reverse")
      x = Cumo::Int32.new(5).seq
      y = Cumo::Int32.new(5).seq(10, 10)
      assert_equal([50, 41, 32, 23, 14], k.call(x, y).to_a)
      all_raw = ElementwiseKernel.new("raw T x", "T z", "z = x[i] * 2", "twice")
      assert_raise(ArgumentError) { all_raw.call(x) }
      assert_equal([0, 2, 4], all_raw.call(x, size: 3).to_a)
      assert_raise(ArgumentError) { k.call(x, y, size: 5) }
    end

    test "an input cannot be written and a number is held to its type" do
      assert_raise(CompileError) { ElementwiseKernel.new("T x", "T y", "y = x; x = 99", "writes_input").call(Cumo::SFloat.new(3).seq) }
      assert_raise(CompileError) { ElementwiseKernel.new("raw T x", "T y", "y = x[i]; x[i] = 99", "writes_raw").call(Cumo::SFloat.new(3).seq, size: 3) }
      plus = ElementwiseKernel.new("T x, T c", "T y", "y = x + c", "plus_scalar")
      assert_raise(TypeError) { plus.call(Cumo::Int32[1, 2], 0.9) }
      assert_raise(RangeError) { plus.call(Cumo::UInt8[1, 2], 300) }
      assert_raise(RangeError) { plus.call(Cumo::UInt8[1, 2], -1) }
      assert_raise(RangeError) { plus.call(Cumo::Int32[1, 2], 2**40) }
      assert_raise(RangeError) { plus.call(Cumo::Int64[1, 2], 2**64) }
      assert_nothing_raised { plus.call(Cumo::Int64[1, 2], 2**63 - 1) }
      assert_nothing_raised { plus.call(Cumo::UInt64[1, 2], 2**64 - 1) }
      assert_equal([0, 1], plus.call(Cumo::UInt8[1, 2], 255).to_a)
      assert_equal([3.5, 4.5], plus.call(Cumo::SFloat[1, 2], 2.5).to_a)
      assert_equal([3.0, 4.0], plus.call(Cumo::SFloat[1, 2], 2).to_a)
    end

    test "a subclass of a dtype is read as that dtype" do
      sub = Class.new(Cumo::SFloat)
      assert_equal([0, 1, 4], SQUARED_DIFF.call(sub.new(3).seq, 0).to_a)
      assert_equal(Cumo::SFloat, GENERIC.call(sub.new(3).seq, 0).class)
    end

    test "a number goes as a scalar of the declared type, and can be alone with a raw array" do
      k = ElementwiseKernel.new("raw T x, int32 n_add, int64 seed, T scale", "T out",
                                "out = (x[i] + n_add) * scale + seed", "scaled")
      x = Cumo::DFloat.new(3).seq
      assert_equal([(0 + 2) * 0.5 + 2**33, (1 + 2) * 0.5 + 2**33, (2 + 2) * 0.5 + 2**33],
                   k.call(x, 2, 2**33, 0.5, size: 3).to_a)
      k2 = ElementwiseKernel.new("T x, T c", "T y", "y = x + c", "plus_c")
      assert_equal([1.5, 2.5], k2.call(Cumo::DFloat[1, 2], 0.5).to_a)
      assert_equal(Cumo::DFloat, k2.call(Cumo::DFloat[1, 2], 1).class)
      only_numbers = ElementwiseKernel.new("T a, S b", "T y", "y = a + b", "plus")
      assert_equal(Cumo::DFloat, only_numbers.call(1.5, 2, size: 2).class)
      assert_equal(Cumo::Int64, only_numbers.call(1, 2, size: 2).class)
      assert_equal([3, 3], only_numbers.call(1, 2, size: 2).to_a)
    end

    test "the preamble can define what the operation calls, in terms of the placeholder" do
      k = ElementwiseKernel.new("T x", "T y", "y = cube(x)", "cubed",
                                preamble: "__device__ inline T cube(T v) { return v * v * v; }")
      assert_equal([0.0, 1.0, 8.0], k.call(Cumo::DFloat.new(3).seq).to_a)
      assert_equal([0, 1, 8], k.call(Cumo::Int32.new(3).seq).to_a)
    end

    test "several outputs come back as an Array" do
      k = ElementwiseKernel.new("T a, T b", "T s, T d", "s = a + b; d = a - b", "sum_diff")
      s, d = k.call(Cumo::Int32[5, 7], Cumo::Int32[1, 2])
      assert_equal([6, 9], s.to_a)
      assert_equal([4, 5], d.to_a)
    end

    test "a view is read where it is, whether contiguous or not" do
      base = Cumo::SFloat.new(4, 6).seq
      cols = base[true, (0...6).step(2)]
      assert_equal((cols * cols).to_a, SQUARED_DIFF.call(cols, 0).to_a)
      assert_equal((base.transpose * base.transpose).to_a, SQUARED_DIFF.call(base.transpose, 0).to_a)
      picked = base[[3, 0], true]
      assert_equal((picked * picked).to_a, GENERIC.call(picked, 0).to_a)
      table = Cumo::SFloat.new(3).seq.freeze
      assert_equal([0, 1, 4], SQUARED_DIFF.call(table, 0).to_a)
      assert_raise(FrozenError) { SQUARED_DIFF.call(table, 0, Cumo::SFloat.zeros(3).freeze) }
      zero_dim = Cumo::SFloat.new(2, 3).seq[1, 2]
      assert_equal([[25, 25, 25], [25, 25, 25]], SQUARED_DIFF.call(Cumo::SFloat.zeros(2, 3), zero_dim).to_a)
    end

    test "a reversed, permuted or broadcast view is read through its strides" do
      base = Cumo::DFloat.new(2, 3, 4).seq
      [
        base[true, (2..0).step(-1), (3..0).step(-2)],
        base.transpose(2, 0, 1),
        base[1, true, true].transpose,
        base.reverse(0),
      ].each do |v|
        assert_equal((v * v).to_a, GENERIC.call(v, 0).to_a, v.shape.inspect)
      end
      rows = Cumo::DFloat.new(4, 3).seq.transpose
      col = Cumo::DFloat.new(4).seq(10)
      assert_equal(((rows - col) * (rows - col)).to_a, GENERIC.call(rows, col).to_a)
    end

    test "arrays that walk several dimensions as one give the same answer" do
      x = Cumo::DFloat.new(4, 5, 6).seq
      y = Cumo::DFloat.new(5, 6).seq(1)
      assert_equal(((x - y) * (x - y)).to_a, GENERIC.call(x, y).to_a)
      assert_equal(((x - y[true, 0..0]) * (x - y[true, 0..0])).to_a, GENERIC.call(x, y[true, 0..0]).to_a)
      z = Cumo::DFloat.new(3, 1, 4).seq
      assert_equal(((z - 2) * (z - 2)).to_a, GENERIC.call(z, Cumo::DFloat.new(1, 1, 4).fill(2)).to_a)
      one = Cumo::DFloat.new(1, 1).seq(3)
      assert_equal([[9.0]], GENERIC.call(one, Cumo::DFloat.zeros(1, 1)).to_a)
      r = x.reverse(0)
      assert_equal(((r - y) * (r - y)).to_a, GENERIC.call(r, y).to_a)
      t = x.transpose(0, 2, 1)
      u = Cumo::DFloat.new(4, 6, 5).seq(7)
      assert_equal(((t - u) * (t - u)).to_a, GENERIC.call(t, u).to_a)
    end

    test "arrays laid out end to end run the simple kernel, and a broadcast keeps only the dimensions it needs" do
      k = ElementwiseKernel.new("T x, T y", "T z", "z = x + y", "collapse_probe")
      kinds = -> { k.instance_variable_get(:@functions).keys.map(&:last) }
      k.call(Cumo::SFloat.new(4, 5, 6).seq, Cumo::SFloat.new(4, 5, 6).seq)
      k.call(Cumo::SFloat.new(4, 1, 6).seq, Cumo::SFloat.new(4, 1, 6).seq)
      k.call(Cumo::SFloat.new(1, 1).seq, Cumo::SFloat.new(1, 1).seq)
      assert_equal([:simple], kinds.call)
      k.call(Cumo::SFloat.new(4, 5, 6).seq, Cumo::SFloat.new(5, 6).seq)
      assert_equal([:simple, 2], kinds.call)
      k.call(Cumo::SFloat.new(4, 5, 6).seq, Cumo::SFloat.new(4, 1, 6).seq)
      assert_equal([:simple, 2, 3], kinds.call)
    end

    def pool_growth
      GC.start
      MemoryPool.free_all_blocks
      before = MemoryPool.total_bytes
      yield
      Runtime.cudaDeviceSynchronize
      MemoryPool.total_bytes - before
    end

    test "a view that strides reaches is not copied, and one that walks an index array is" do
      omit("needs the memory pool to count with") unless MemoryPool.enabled?
      base = Cumo::SFloat.new(256, 256).seq
      out = Cumo::SFloat.new(256, 256)
      SQUARED_DIFF.call(base.transpose, 0, out)
      assert_equal(0, pool_growth { SQUARED_DIFF.call(base.transpose, 0, out) }, "transposed")
      assert_equal(0, pool_growth { SQUARED_DIFF.call(base.reverse(1), 0, out) }, "reversed")
      frozen = base.dup.freeze
      assert_equal(0, pool_growth { SQUARED_DIFF.call(frozen, 0, out) }, "frozen")
      assert_equal(0, pool_growth { SQUARED_DIFF.call(out, 0, out) }, "the output itself, element for element")
      picked = base[Cumo::Int32.new(256).seq(255, -1), true]
      assert_operator(pool_growth { SQUARED_DIFF.call(picked, 0, out) }, :>=, 256 * 256 * 4, "index")
      assert_equal((picked * picked).to_a, out.to_a)
    end

    test "an input that shares memory with the output is read before it is written" do
      copy = ElementwiseKernel.new("T x", "T y", "y = x", "copy_in_place")
      a = Cumo::SFloat.new(1 << 20).seq
      copy.call(a.reverse(0), a)
      assert_equal(Cumo::SFloat.new(1 << 20).seq.reverse(0).to_a, a.to_a)
      b = Cumo::SFloat.new(1024, 1024).seq
      copy.call(b.transpose, b)
      assert_equal(Cumo::SFloat.new(1024, 1024).seq.transpose.to_a, b.to_a)
      c = Cumo::SFloat.new(1 << 20).seq
      copy.call(c[0...(1 << 19)], c[(1 << 19)..])
      assert_equal((0...(1 << 19)).map(&:to_f) * 2, c.to_a)
      d = Cumo::SFloat.new(1 << 16).seq
      copy.call(d[0..0], d)
      assert_equal([0.0] * (1 << 16), d.to_a)
      e = Cumo::SFloat.new(1 << 20).seq
      plus = ElementwiseKernel.new("T x", "T y", "y = x + 1", "plus_one_in_place")
      plus.call(e, e)
      assert_equal((1..(1 << 20)).map(&:to_f), e.to_a)
    end

    test "kmeans' var_kernel broadcasts the (N, 1) samples against the (1, K) centers" do
      k = ElementwiseKernel.new("T x0, T x1, T c0, T c1", "T out",
                                "out = (x0 - c0) * (x0 - c0) + (x1 - c1) * (x1 - c1)", "var_kernel")
      x = Cumo::SFloat.new(50, 2).rand
      c = Cumo::SFloat.new(3, 2).rand
      got = k.call(x[true, :new, 0], x[true, :new, 1], c[:new, true, 0], c[:new, true, 1])
      want = (x[true, :new, 0] - c[:new, true, 0])**2 + (x[true, :new, 1] - c[:new, true, 1])**2
      assert_equal([50, 3], got.shape)
      assert { (got - want).abs.max < 1e-6 }
    end

    test "an empty array answers an empty output without a launch" do
      z = SQUARED_DIFF.call(Cumo::SFloat.new(0, 3), Cumo::SFloat.new(3))
      assert_equal([0, 3], z.shape)
    end

    test "the kernel is compiled once per set of types and shapes" do
      k = ElementwiseKernel.new("T x, T y", "T z", "z = x + y", "add_once")
      k.call(Cumo::SFloat.new(4).seq, Cumo::SFloat.new(4).seq)
      k.call(Cumo::SFloat.new(4).seq, Cumo::SFloat.new(4).seq)
      assert_equal(1, k.instance_variable_get(:@functions).size)
      k.call(Cumo::SFloat.new(4).seq, 1)
      assert_equal(2, k.instance_variable_get(:@functions).size)
      k.call(Cumo::SFloat.new(2, 4).seq, Cumo::SFloat.new(4).seq)
      assert_equal(3, k.instance_variable_get(:@functions).size)
      k.call(Cumo::DFloat.new(4).seq, Cumo::DFloat.new(4).seq)
      assert_equal(4, k.instance_variable_get(:@functions).size)
    end

    test "what cannot be a kernel argument is refused" do
      x = Cumo::SFloat.new(3).seq
      assert_raise(TypeError) { GENERIC.call(Cumo::Bit.new(3), Cumo::Bit.new(3)) }
      assert_raise(TypeError) { GENERIC.call(Cumo::HFloat.new(3), Cumo::HFloat.new(3)) }
      assert_raise(TypeError) { GENERIC.call(Cumo::SComplex.new(3), Cumo::SComplex.new(3)) }
      assert_raise(TypeError) { GENERIC.call(Cumo::RObject.new(3), Cumo::RObject.new(3)) }
      assert_raise(TypeError) { GENERIC.call(x, "3") }
      assert_raise(TypeError) { GENERIC.call(x, nil) }
      assert_raise(ArgumentError) { SQUARED_DIFF.call(Cumo::SFloat.new(2, 3), Cumo::SFloat.new(2)) }
    end

    test "a definition that cannot be compiled is refused when it is written" do
      assert_raise(ArgumentError) { ElementwiseKernel.new("T x", "T y", "y = x", "bad name") }
      assert_raise(ArgumentError) { ElementwiseKernel.new("float x", "T y", "y = x", "k") }
      assert_raise(ArgumentError) { ElementwiseKernel.new("T i", "T y", "y = i", "k") }
      assert_raise(ArgumentError) { ElementwiseKernel.new("T n", "T y", "y = n", "k") }
      assert_nothing_raised { ElementwiseKernel.new("T n_items, T index", "T y", "y = n_items + index", "k") }
      assert_raise(ArgumentError) { ElementwiseKernel.new("T _x", "T y", "y = _x", "k") }
      assert_raise(ArgumentError) { ElementwiseKernel.new("const T x", "T y", "y = x", "k") }
      assert_raise(ArgumentError) { ElementwiseKernel.new("T", "T y", "y = 1", "k") }
      assert_raise(ArgumentError) { ElementwiseKernel.new("1 x", "T y", "y = x", "k") }
      assert_raise(ArgumentError) { ElementwiseKernel.new("i x", "i y", "y = x", "k") }
      assert_raise(CompileError) { ElementwiseKernel.new("T x", "T y", "y = x +", "k").call(Cumo::SFloat.new(2)) }
    end
  end
end
