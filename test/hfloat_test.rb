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
    a = dtype[1.0, 2.0]
    assert_raise(NoMethodError) { a.sum }
    assert_raise(NoMethodError) { a.mean }
    assert_raise(NoMethodError) { a.sort }
    assert_raise(NoMethodError) { a.dot(a) }
    assert_raise(NoMethodError) { a.cumsum }
    assert_raise(NoMethodError) { dtype.new(2).rand }
  end

  test "NMath answers through SFloat until HFloat has a module of its own" do
    assert_equal Cumo::SFloat, Cumo::NMath.sqrt(dtype[4.0]).class
    assert_equal [2.0, 3.0], Cumo::NMath.sqrt(dtype[4.0, 9.0]).to_a
    assert_equal [1.0], Cumo::NMath.exp(dtype[0.0]).to_a
  end
end
