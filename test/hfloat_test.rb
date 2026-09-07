# frozen_string_literal: true

require_relative "test_helper"

class HFloatTest < Test::Unit::TestCase
  dtype = Cumo::HFloat

  MAX = 65504.0
  MIN_SUBNORMAL = 2.0**-24

  test "is an NArray" do
    assert { dtype < Cumo::NArray }
  end

  test "Float16 names the same class" do
    assert_equal dtype, Cumo::Float16
  end

  test "an element is two bytes" do
    assert_equal 2, dtype::ELEMENT_BYTE_SIZE
    assert_equal 16, dtype::ELEMENT_BIT_SIZE
    assert_equal 2, dtype::CONTIGUOUS_STRIDE
    assert_equal 2 * 12, dtype.new(3, 4).seq.to_binary.bytesize
  end

  test "the limits are binary16's" do
    assert_equal MAX, dtype::MAX
    assert_equal 2.0**-14, dtype::MIN
    assert_equal 2.0**-10, dtype::EPSILON
  end

  test "new leaves the array unallocated" do
    a = dtype.new(3)
    assert_match(/\(empty\)/, a.inspect)
    assert_equal [3], a.shape
  end

  test "zeros, ones and fill" do
    assert_equal [0.0, 0.0, 0.0], dtype.zeros(3).to_a
    assert_equal [1.0, 1.0, 1.0], dtype.ones(3).to_a
    assert_equal [2.5, 2.5], dtype.new(2).fill(2.5).to_a
  end

  test "seq counts along the array" do
    assert_equal (0...6).map(&:to_f), dtype.new(6).seq.to_a
    assert_equal [1.0, 1.5, 2.0, 2.5], dtype.new(4).seq(1, 0.5).to_a
    assert_equal [[0.0, 1.0], [2.0, 3.0]], dtype.new(2, 2).seq.to_a
  end

  test "eye" do
    assert_equal [[1.0, 0.0], [0.0, 1.0]], dtype.eye(2).to_a
  end

  test "a literal keeps what half can hold" do
    assert_equal [1.0, -2.5, 0.125], dtype[1, -2.5, 0.125].to_a
  end

  test "inspect names the class and shows the elements" do
    s = dtype[1.5, -2.0].inspect
    assert_match(/Cumo::HFloat/, s)
    assert_match(/1\.5/, s)
    assert_match(/-2/, s)
  end

  test "to_a answers Ruby Floats" do
    assert_equal Float, dtype[1.5].to_a.first.class
  end

  test "an element that half cannot hold is rounded to the nearest" do
    assert_equal 2048.0, dtype[2048].to_a.first
    assert_equal 2048.0, dtype[2049].to_a.first
    assert_equal 2050.0, dtype[2050].to_a.first
    assert_equal 0.0999755859375, dtype[0.1].to_a.first
  end

  test "a double is rounded once, not through float" do
    assert_equal 2050.0, dtype[2050.9999999999].to_a.first
    assert_equal 2050.0, dtype[2049.0000000001].to_a.first
    assert_equal MAX, dtype[65519.999999999].to_a.first
  end

  test "a tie rounds to the even neighbour and the rest to the nearer one" do
    assert_equal 2052.0, dtype[2051].to_a.first
    assert_equal 0.300048828125, dtype[0.3].to_a.first
    assert_equal 0.7001953125, dtype[0.7].to_a.first
    assert_equal 0.333251953125, dtype[1.0 / 3].to_a.first
    assert_equal MAX, dtype[65519].to_a.first
    assert_equal Float::INFINITY, dtype[65520].to_a.first
  end

  test "the host rounds a number the way the GPU does" do
    values = [0.1, 0.3, 0.7, 1.0 / 3, 2051, 65519, 65520, 1e-8, -0.3, 1e-5,
              6.103515625e-05, 2.0**-24, 2.0**-25, 3.0e-8,
              2049.0000000001, 2050.9999999999, 65519.999999999, 1.0000000000000002]
    host = dtype[*values].to_binary.unpack("S*")
    device = dtype.cast(Cumo::DFloat[*values]).to_binary.unpack("S*")
    assert_equal host.map { |x| format("%04x", x) }, device.map { |x| format("%04x", x) }
  end

  test "a magnitude past the largest half is infinite" do
    assert_equal MAX, dtype[65504].to_a.first
    assert_equal Float::INFINITY, dtype[65536].to_a.first
    assert_equal(-Float::INFINITY, dtype[-70000].to_a.first)
    assert_equal Float::INFINITY, dtype[Float::MAX].to_a.first
  end

  test "a magnitude below the smallest subnormal is zero" do
    assert_equal MIN_SUBNORMAL, dtype[MIN_SUBNORMAL].to_a.first
    assert_equal 0.0, dtype[MIN_SUBNORMAL / 4].to_a.first
    assert_equal 0.0, dtype[Float::MIN].to_a.first
  end

  test "nan and infinity survive a round trip" do
    a = dtype[Float::NAN, Float::INFINITY, -Float::INFINITY]
    assert_equal [true, false, false], a.isnan.to_a.map { |x| x == 1 }
    assert_equal [false, true, true], a.isinf.to_a.map { |x| x == 1 }
    assert_equal Float::INFINITY, a[1]
  end

  test "arithmetic answers HFloat" do
    a = dtype[1.0, 2.0, 4.0]
    b = dtype[0.5, 0.5, 0.5]
    assert_equal dtype, (a + b).class
    assert_equal [1.5, 2.5, 4.5], (a + b).to_a
    assert_equal [0.5, 1.5, 3.5], (a - b).to_a
    assert_equal [0.5, 1.0, 2.0], (a * b).to_a
    assert_equal [2.0, 4.0, 8.0], (a / b).to_a
    assert_equal [1.0, 4.0, 16.0], (a**2).to_a
    assert_equal [-1.0, -2.0, -4.0], (-a).to_a
  end

  test "arithmetic against a Ruby number stays in HFloat" do
    a = dtype[1.0, 2.0]
    assert_equal dtype, (a + 1).class
    assert_equal [2.0, 3.0], (a + 1).to_a
    assert_equal [0.5, 1.0], (a * 0.5).to_a
    assert_equal [1.0, 0.5], (1.0 / a).to_a
  end

  test "an addition that overflows half answers infinity" do
    a = dtype[MAX]
    assert_equal Float::INFINITY, (a + a).to_a.first
  end

  test "the four operations round once" do
    a = dtype[1.0009765625]
    assert_equal 1.0009765625, a.to_a.first
    assert_equal 2.001953125, (a + a).to_a.first
    assert_equal 1.001953125, (a * a).to_a.first
  end

  test "divmod and modulo floor the quotient as Ruby does" do
    a = dtype[-7.0]
    b = dtype[3.0]
    q, r = a.divmod(b)
    assert_equal [-3.0], q.to_a
    assert_equal [2.0], r.to_a
    assert_equal [2.0], (a % b).to_a
  end

  test "comparison answers a Bit" do
    a = dtype[1.0, 2.0, 3.0]
    b = dtype[3.0, 2.0, 1.0]
    assert_equal Cumo::Bit, (a > b).class
    assert_equal [0, 0, 1], (a > b).to_a
    assert_equal [0, 1, 0], (a.eq b).to_a
    assert_equal [1, 0, 0], (a < b).to_a
    assert_equal [1, 0, 0], (a.ne(b) & (a < b)).to_a
  end

  test "unary functions" do
    a = dtype[-1.5, 2.5]
    assert_equal [1.5, 2.5], a.abs.to_a
    assert_equal [-2.0, 2.0], a.floor.to_a
    assert_equal [-1.0, 3.0], a.ceil.to_a
    assert_equal [-1.0, 2.0], a.trunc.to_a
    assert_equal [-1.0, 1.0], a.sign.to_a
    assert_equal [2.25, 6.25], a.square.to_a
  end

  test "min and max" do
    a = dtype[3.0, 1.0, 2.0]
    assert_equal 1.0, a.min.to_a.first
    assert_equal 3.0, a.max.to_a.first
    assert_equal 1, a.min_index
    assert_equal 0, a.max_index
    assert_equal [1.0, 3.0], a.minmax.map { |x| x.to_a.first }
  end

  test "clip" do
    assert_equal [1.0, 1.5, 2.0], dtype[0.5, 1.5, 3.0].clip(1, 2).to_a
  end

  test "a view keeps the element size" do
    a = dtype.new(6).seq
    assert_equal [1.0, 3.0, 5.0], a[1.step(-1, 2)].to_a
    assert_equal [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], a.reshape(2, 3).to_a
    assert_equal [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]], a.reshape(2, 3).transpose.to_a
  end

  test "aset writes through a view" do
    a = dtype.new(4).seq
    a[1..2] = 9
    assert_equal [0.0, 9.0, 9.0, 3.0], a.to_a
  end

  test "casting to and from every other dtype" do
    [Cumo::DFloat, Cumo::SFloat, Cumo::Int64, Cumo::Int32, Cumo::Int16, Cumo::Int8,
     Cumo::UInt64, Cumo::UInt32, Cumo::UInt16, Cumo::UInt8, Cumo::RObject].each do |other|
      assert_equal [1.0, 2.0], dtype.cast(other[1, 2]).to_a, other.to_s
      assert_equal [1, 2], other.cast(dtype[1, 2]).to_a.map(&:to_i), other.to_s
    end
    assert_equal [0.0, 1.0], dtype.cast(Cumo::Bit[0, 1]).to_a
    assert_equal [0, 1], Cumo::Bit.cast(dtype[0, 1]).to_a
    assert_equal [1.0, 2.0], Cumo::DComplex.cast(dtype[1, 2]).real.to_a
  end

  test "casting a value the target cannot hold" do
    assert_equal 2048, Cumo::Int32.cast(dtype[2049]).to_a.first
    assert_equal MAX, Cumo::DFloat.cast(dtype[MAX]).to_a.first
    assert_equal Float::INFINITY, dtype.cast(Cumo::DFloat[1e300]).to_a.first
  end

  test "store from another dtype" do
    a = dtype.new(2)
    a.store(Cumo::Int16[3, 4])
    assert_equal [3.0, 4.0], a.to_a
    a.store([5.5, 6.5])
    assert_equal [5.5, 6.5], a.to_a
  end

  test "upcast" do
    assert_equal dtype, dtype.upcast(Cumo::Int32)
    assert_equal dtype, Cumo::Int32.upcast(dtype)
    assert_equal dtype, dtype.upcast(Cumo::Bit)
    assert_equal dtype, Cumo::Bit.upcast(dtype)
    assert_equal Cumo::SFloat, dtype.upcast(Cumo::SFloat)
    assert_equal Cumo::DFloat, dtype.upcast(Cumo::DFloat)
    assert_equal Cumo::SComplex, dtype.upcast(Cumo::SComplex)
    assert_equal Cumo::DComplex, dtype.upcast(Cumo::DComplex)
    assert_equal Cumo::RObject, dtype.upcast(Cumo::RObject)
    assert_equal dtype, dtype.upcast(Float)
    assert_equal dtype, dtype.upcast(Integer)
    assert_equal Cumo::SComplex, dtype.upcast(Complex)
  end

  test "an operand of a wider type wins" do
    assert_equal Cumo::SFloat, (dtype[1] + Cumo::SFloat[1]).class
    assert_equal Cumo::DFloat, (dtype[1] + Cumo::DFloat[1]).class
    assert_equal dtype, (dtype[1] + Cumo::Int32[1]).class
    assert_equal dtype, (Cumo::Int32[1] + dtype[1]).class
  end

  test "to_binary and from_binary keep the bit pattern" do
    a = dtype[1.5, -2.5, 0.25]
    assert_equal a.to_a, dtype.from_binary(a.to_binary).to_a
    assert_equal "\x00\x3e", dtype[1.5].to_binary[0, 2].b
  end

  test "extract answers a zero-dimensional array" do
    a = dtype[1.5, 2.5]
    assert_equal dtype, a[0..0].class
    assert_equal 1.5, a[0]
  end

  test "a large array holds every element" do
    n = 100_000
    a = dtype.new(n).fill(1.5)
    assert_equal 1.5, a[n - 1]
    assert_equal 0, a.ne(1.5).count_true
  end

  test "the methods a later step brings are not claimed yet" do
    assert_raise(NoMethodError) { dtype.new(2, 2).seq.gemm(dtype.new(2, 2).seq) }
    assert_raise(NoMethodError) { dtype.new(1, 1, 2, 2).seq.conv(dtype.new(1, 1, 2, 2).seq) }
  end

  test "sort orders as SFloat does" do
    a = dtype[3.0, -1.0, 2.5, 0.0]
    assert_equal [-1.0, 0.0, 2.5, 3.0], a.sort.to_a
    assert_equal [1, 3, 2, 0], a.sort_index.to_a
    assert_equal(-1.0, a.sort.to_a.first)
    assert_equal [[1.0, 2.0], [3.0, 4.0]], dtype[[2.0, 1.0], [4.0, 3.0]].sort(axis: 1).to_a
  end

  test "sort puts a NaN last and keeps the signed zeros" do
    a = dtype[1.0, Float::NAN, -1.0]
    sorted = a.sort.to_a
    assert_equal [-1.0, 1.0], sorted[0, 2]
    assert_equal true, sorted[2].nan?
    assert_equal [-Float::INFINITY, 0.0, Float::INFINITY],
                 dtype[Float::INFINITY, 0.0, -Float::INFINITY].sort.to_a
    assert_equal [-Float::INFINITY, Float::INFINITY],
                 dtype[0.0, -0.0].sort.to_a.map { |v| 1 / v }
  end

  test "sort agrees with SFloat over a long array" do
    n = 5000
    src = Array.new(n) { |i| ((i * 7919) % 1000 - 500) / 8.0 }
    assert_equal Cumo::SFloat[*src].sort.to_a, dtype[*src].sort.to_a
    assert_equal Cumo::SFloat[*src].sort_index.to_a, dtype[*src].sort_index.to_a
  end

  test "median" do
    assert_equal 2.0, dtype[3.0, 1.0, 2.0].median.to_a.first
    assert_equal 2.5, dtype[1.0, 2.0, 3.0, 4.0].median.to_a.first
    assert_equal [1.5, 3.5], dtype[[1.0, 2.0], [3.0, 4.0]].median(axis: 1).to_a
    assert_equal 2.0, dtype[3.0, 1.0, 2.0, Float::NAN].median.to_a.first
  end

  test "cumsum and cumprod" do
    assert_equal [1.0, 3.0, 6.0, 10.0], dtype[1, 2, 3, 4].cumsum.to_a
    assert_equal [1.0, 2.0, 6.0, 24.0], dtype[1, 2, 3, 4].cumprod.to_a
  end

  test "a running sum is carried wider than half at every length" do
    assert_equal 8192.0, dtype.new(8191).fill(1.0).cumsum.to_a.last
    assert_equal 8192.0, dtype.new(8192).fill(1.0).cumsum.to_a.last
    assert_equal 40_000.0, dtype.new(40_000).fill(1.0).cumsum.to_a.last
    a = dtype.new(3000).fill(1.0).cumsum
    assert_equal 2050.0, a[2049].to_a.first
    assert_equal 3000.0, a[2999].to_a.first
  end

  test "cumsum carries a NaN and skips it when asked" do
    a = dtype[1.0, Float::NAN, 2.0]
    assert_equal true, a.cumsum.to_a[2].nan?
    assert_equal [1.0, 1.0, 3.0], a.cumsum(nan: true).to_a
  end

  test "rand fills the range it names" do
    a = dtype.new(2000).rand(2.0, 5.0)
    assert_equal dtype, a.class
    assert_equal 0, a.lt(2.0).count_true.to_a.first
    assert_equal 0, a.ge(5.0).count_true.to_a.first
    assert_operator a.max.to_a.first, :>, 4.5
    assert_operator a.min.to_a.first, :<, 2.5
  end

  test "a range half cannot hold still answers numbers in it" do
    a = dtype.new(1000).rand(65535.0)
    assert_equal 0, a.isfinite.count_false.to_a.first
    assert_equal 0, a.lt(0.0).count_true.to_a.first
    b = dtype.new(1000).rand(-60_000.0, 60_000.0)
    assert_equal 0, b.isfinite.count_false.to_a.first
    assert_equal 0, b.lt(-60_000.0).count_true.to_a.first
  end

  test "a sigma half cannot hold keeps the body of the distribution" do
    a = dtype.new(2000).rand_norm(0.0, 66_000.0)
    assert_operator a.isfinite.count_true.to_a.first, :>, 500
  end

  test "cumprod keeps half's range, as prod does" do
    a = dtype[300.0, 300.0, 0.0001]
    assert_equal Float::INFINITY, a.cumprod.to_a.last
    assert_equal a.prod.to_a.first, a.cumprod.to_a.last
    assert_equal [1.0, 3.0, 6.0], dtype[1, 2, 3].cumsum.to_a
  end

  test "median of a pair whose sum half cannot hold" do
    assert_equal 44_992.0, dtype[40_000.0, 50_000.0].median.to_a.first
    assert_equal 44_992.0, dtype[40_000.0, 50_000.0].median(nan: true).to_a.first
  end

  test "rand_norm lands around its mean" do
    a = dtype.new(4000).rand_norm(10.0, 1.0)
    assert_in_delta 10.0, a.mean.to_a.first, 0.2
    assert_operator a.stddev.to_a.first, :>, 0.5
  end

  test "the same seed answers the same numbers" do
    Cumo::NArray.srand(42)
    a = dtype.new(100).rand
    Cumo::NArray.srand(42)
    assert_equal a.to_a, dtype.new(100).rand.to_a
  end

  test "dot answers through mulsum until gemm takes it" do
    a = dtype[1.0, 2.0]
    assert_equal 5.0, a.dot(a).to_a.first
    m = dtype.new(2, 2).seq
    assert_equal [[2.0, 3.0], [6.0, 11.0]], m.dot(m).to_a
  end

  test "sum and prod answer HFloat" do
    a = dtype[1.0, 2.0, 4.0]
    assert_equal dtype, a.sum.class
    assert_equal 7.0, a.sum.to_a.first
    assert_equal 8.0, a.prod.to_a.first
    assert_equal [[0.0, 1.0], [2.0, 3.0]], dtype.new(2, 2).seq.to_a
    assert_equal [1.0, 5.0], dtype.new(2, 2).seq.sum(axis: 1).to_a
    assert_equal [2.0, 4.0], dtype.new(2, 2).seq.sum(axis: 0).to_a
  end

  test "a sum longer than half can count is accumulated wider" do
    assert_equal 40_000.0, dtype.new(40_000).fill(1.0).sum.to_a.first
    assert_equal 20_000.0, dtype.new(40_000).fill(0.5).sum.to_a.first
    a = dtype.new(2, 40_000).fill(1.0)
    assert_equal [40_000.0, 40_000.0], a.sum(axis: 1).to_a
  end

  test "mean, var, stddev and rms" do
    a = dtype[1.0, 2.0, 3.0, 4.0]
    assert_equal dtype, a.mean.class
    assert_equal 2.5, a.mean.to_a.first
    assert_in_delta 1.666, a.var.to_a.first, 1e-2
    assert_in_delta 1.291, a.stddev.to_a.first, 1e-2
    assert_in_delta 2.739, a.rms.to_a.first, 1e-2
  end

  test "a mean over more elements than half can count" do
    n = 40_000
    assert_equal 1.0, dtype.new(n).fill(1.0).mean.to_a.first
    assert_equal 0.25, dtype.new(n).fill(0.25).mean.to_a.first
  end

  test "the nan-aware reductions skip a NaN" do
    a = dtype[1.0, Float::NAN, 3.0]
    assert_equal 4.0, a.sum(nan: true).to_a.first
    assert_equal 2.0, a.mean(nan: true).to_a.first
    assert_equal 3.0, a.prod(nan: true).to_a.first
    assert_equal 1.0, a.min(nan: false).to_a.first
    assert_equal true, a.sum.to_a.first.nan?
  end

  test "min and max over a long array" do
    a = dtype.new(40_000).fill(2.0)
    a[12_345] = 100.0
    a[999] = -7.0
    assert_equal 100.0, a.max.to_a.first
    assert_equal(-7.0, a.min.to_a.first)
    assert_equal 12_345, a.max_index
    assert_equal 107.0, a.ptp.to_a.first
  end

  test "mulsum accumulates wider than half" do
    n = 40_000
    assert_equal n.to_f, dtype.new(n).fill(1.0).mulsum(dtype.new(n).fill(1.0)).to_a.first
    assert_equal 20_000.0, dtype.new(n).fill(0.5).mulsum(dtype.new(n).fill(1.0)).to_a.first
    assert_equal 30.0, dtype[1, 2, 3, 4].mulsum(dtype[1, 2, 3, 4]).to_a.first
  end

  test "NMath answers HFloat" do
    assert_equal dtype, Cumo::NMath.sqrt(dtype[4.0]).class
    assert_equal [2.0, 3.0], Cumo::NMath.sqrt(dtype[4.0, 9.0]).to_a
    assert_equal [1.0], Cumo::NMath.exp(dtype[0.0]).to_a
    assert_equal [0.0], Cumo::NMath.log(dtype[1.0]).to_a
    assert_equal [1.0], Cumo::NMath.sin(dtype[Math::PI / 2]).to_a
    assert_equal [1.0], Cumo::NMath.cosh(dtype[0.0]).to_a
    assert_in_delta 0.7853, Cumo::NMath.atan2(dtype[1.0], dtype[1.0]).to_a.first, 1e-3
    assert_equal [5.0], Cumo::NMath.hypot(dtype[3.0], dtype[4.0]).to_a
    assert_equal [8.0], Cumo::NMath.ldexp(dtype[2.0], dtype[2.0]).to_a
    assert_equal [0.5, 2.0], Cumo::NMath.frexp(dtype[2.0]).map { |x| x.to_a.first }
  end

  test "a math function rounds its answer to half" do
    assert_equal 1.4140625, Cumo::NMath.sqrt(dtype[2.0]).to_a.first
    assert_equal 1.0, Cumo::NMath.sinc(dtype[0.0]).to_a.first
  end

  test "prod keeps half's range, as the elementwise multiply does" do
    assert_equal Float::INFINITY, dtype[300.0, 300.0, 0.0001].prod.to_a.first
    assert_equal Float::INFINITY, (dtype[300.0] * dtype[300.0]).to_a.first
    assert_equal 8.0, dtype[1.0, 2.0, 4.0].prod.to_a.first
  end

  test "logseq" do
    assert_equal [1.0, 2.0, 4.0, 8.0], dtype.new(4).logseq(0, 1, 2).to_a
  end
end
