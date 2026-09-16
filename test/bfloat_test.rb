# frozen_string_literal: true

require_relative "test_helper"

class BFloatTest < Test::Unit::TestCase
  dtype = Cumo::BFloat

  MAX = 3.38953138925153547590e+38
  MIN_NORMAL = 2.0**-126
  MIN_SUBNORMAL = 2.0**-133

  test "is an NArray" do
    assert { dtype < Cumo::NArray }
  end

  test "BFloat16 names the same class" do
    assert_equal dtype, Cumo::BFloat16
  end

  test "an element is two bytes" do
    assert_equal 2, dtype::ELEMENT_BYTE_SIZE
    assert_equal 16, dtype::ELEMENT_BIT_SIZE
    assert_equal 2, dtype::CONTIGUOUS_STRIDE
    assert_equal 2 * 12, dtype.new(3, 4).seq.to_binary.bytesize
  end

  test "the limits are bfloat16's, which reach as far as a float" do
    assert_equal MAX, dtype::MAX
    assert_equal MIN_NORMAL, dtype::MIN
    assert_equal 2.0**-7, dtype::EPSILON
  end

  test "the exponent range is a float's, so a float magnitude survives" do
    # This is the whole point of the format: 1e38 is finite here and an
    # infinity in binary16.
    # 1e38 lands between two bfloat16 values whose spacing up there is 2**119.
    assert_equal((1 + 22 / 128.0) * 2.0**126, dtype.cast([1.0e38]).to_a[0])
    assert_equal Float::INFINITY, dtype.cast([MAX * 1.01]).to_a[0]
  end

  test "seven mantissa bits, so integers step by two past 256" do
    got = dtype.cast([255.0, 256.0, 257.0, 258.0, 259.0, 260.0]).to_a
    # 257 and 259 are both ties; nearest-even keeps the even mantissa, which
    # rounds 257 down and 259 up.
    assert_equal [255.0, 256.0, 256.0, 258.0, 260.0, 260.0], got
  end

  test "one plus epsilon is representable and half of it is not" do
    assert_equal [1.0 + 2.0**-7, 1.0], dtype.cast([1.0 + 2.0**-7, 1.0 + 2.0**-8]).to_a
  end

  test "subnormals reach 2**-133 and stop" do
    got = dtype.cast([MIN_SUBNORMAL, MIN_SUBNORMAL / 2, MIN_SUBNORMAL / 4]).to_a
    assert_equal [MIN_SUBNORMAL, 0.0, 0.0], got
  end

  test "NaN and both infinities survive a round trip" do
    got = dtype.cast([Float::NAN, Float::INFINITY, -Float::INFINITY]).to_a
    assert { got[0].nan? }
    assert_equal [Float::INFINITY, -Float::INFINITY], got[1, 2]
  end

  test "a NaN keeps its exponent rather than becoming an infinity" do
    # Truncating the top 16 bits of a NaN whose payload is in the low bits
    # leaves an all-ones exponent over a zero mantissa, which reads as an
    # infinity. Both entries of the conversion answer a NaN instead.
    #
    # A Ruby number arrives as a double, so this covers the double entry. The
    # float one is reached only from a binary16 source, which the kernel
    # serves, so the two are exercised separately.
    payload = [0x7f800001].pack("L").unpack1("f")
    assert { dtype.cast([payload]).to_a[0].nan? }
    assert { dtype.cast(Cumo::HFloat.from_binary([0x7e01].pack("S"), [1])).to_a[0].nan? }
  end

  test "a Ruby Float is rounded once, not through float first" do
    # 1.00390625 is the midpoint between 1.0 and 1 + 2**-7, and ties go to the
    # even mantissa, so it answers 1.0. The one above it is what tells the two
    # roundings apart: taken straight it is past the midpoint and answers
    # 1.0078125, while narrowing to float first loses the 2**-30 -- a float
    # near 1.0 steps by 2**-23 -- and lands back on the tie.
    assert_equal 1.0, dtype.cast([1.00390625 - 2.0**-30]).to_a[0]
    assert_equal 1.0, dtype.cast([1.00390625]).to_a[0]
    assert_equal 1.0078125, dtype.cast([1.00390625 + 2.0**-30]).to_a[0]
  end

  test "zeros, ones and fill" do
    assert_equal [0.0, 0.0, 0.0], dtype.zeros(3).to_a
    assert_equal [1.0, 1.0, 1.0], dtype.ones(3).to_a
    assert_equal [2.5, 2.5], dtype.new(2).fill(2.5).to_a
  end

  test "the four operations" do
    a = dtype[6.0, 8.0]
    b = dtype[2.0, 4.0]
    assert_equal [8.0, 12.0], (a + b).to_a
    assert_equal [4.0, 4.0], (a - b).to_a
    assert_equal [12.0, 32.0], (a * b).to_a
    assert_equal [3.0, 2.0], (a / b).to_a
  end

  test "comparisons answer a Bit" do
    a = dtype[1.0, 2.0, 3.0]
    assert_equal [0, 0, 1], a.gt(2.0).to_a
    assert_equal [1, 0, 0], a.lt(2.0).to_a
    assert_equal [0, 1, 0], a.eq(2.0).to_a
  end

  test "minus, abs and square" do
    a = dtype[-1.5, 2.5]
    assert_equal [1.5, -2.5], (-a).to_a
    assert_equal [1.5, 2.5], a.abs.to_a
    assert_equal [2.25, 6.25], (a * a).to_a
  end

  test "a reduction accumulates in float, so it passes 256" do
    # The element type stops moving at 256: a bfloat16 accumulator would
    # answer 256 for every length past that. 39936 is 40000 rounded to the
    # nearest bfloat16, whose spacing is 256 up there.
    assert_equal 39936.0, dtype.ones(40000).sum.to_f
    assert_equal 300.0, dtype.ones(300).sum.to_f
  end

  test "mean, var and stddev" do
    a = dtype[1.0, 2.0, 3.0, 4.0]
    assert_equal 2.5, a.mean.to_f
    assert_in_delta 1.6667, a.var.to_f, 0.01
    assert_in_delta 1.2910, a.stddev.to_f, 0.01
  end

  test "min, max and their indices" do
    a = dtype[3.0, 1.0, 4.0, 2.0]
    assert_equal 1.0, a.min.to_f
    assert_equal 4.0, a.max.to_f
    assert_equal 1, a.min_index
    assert_equal 2, a.max_index
  end

  test "cumsum carries float too" do
    assert_equal 300.0, dtype.ones(300).cumsum[-1].to_f
  end

  test "sort puts every NaN last" do
    got = dtype.cast([3.0, 1.0, Float::NAN, 2.0]).sort.to_a
    assert_equal [1.0, 2.0, 3.0], got[0, 3]
    assert { got[3].nan? }
  end

  test "sort_index answers where each rank came from" do
    assert_equal [1, 2, 0], dtype[3.0, 1.0, 2.0].sort_index.to_a
  end

  test "median drops trailing NaNs" do
    assert_equal 2.5, dtype[3.0, 1.0, 2.0, 4.0].median.to_f
    assert_equal 2.0, dtype[3.0, 1.0, 2.0].median.to_f
  end

  test "NMath computes in float and rounds back" do
    assert_equal [2.0, 3.0], Cumo::NMath.sqrt(dtype[4.0, 9.0]).to_a
    assert_in_delta 1.0, Cumo::NMath.exp(dtype[0.0]).to_a[0], 1e-6
  end

  test "rand stays inside the range it was given" do
    a = dtype.new(2000).rand(1.0)
    assert { a.min.to_f >= 0.0 }
    assert { a.max.to_f < 1.0 }
  end

  test "dot goes through cuBLAS" do
    a = dtype.cast([[1.0, 2.0], [3.0, 4.0]])
    b = dtype.cast([[5.0, 6.0], [7.0, 8.0]])
    assert_equal [[19.0, 22.0], [43.0, 50.0]], a.dot(b).to_a
  end

  test "gemm accumulates in float, so a product past 256 survives" do
    # 300 * 300 is 90000, which a bfloat16 accumulator cannot hold at all.
    a = dtype.cast([[300.0]])
    assert_in_delta 90000.0, a.dot(a).to_a[0][0], 512.0
  end

  test "mixing with HFloat promotes to SFloat, since neither contains the other" do
    # bfloat16 has the wider exponent and binary16 the wider mantissa.
    assert_equal Cumo::SFloat, (dtype[1.0] + Cumo::HFloat[1.0]).class
    assert_equal Cumo::SFloat, (Cumo::HFloat[1.0] + dtype[1.0]).class
  end

  test "mixing with the other float types takes the wider one" do
    assert_equal Cumo::SFloat, (dtype[1.0] + Cumo::SFloat[1.0]).class
    assert_equal Cumo::DFloat, (dtype[1.0] + Cumo::DFloat[1.0]).class
    assert_equal Cumo::SComplex, (dtype[1.0] + Cumo::SComplex[1.0]).class
    assert_equal Cumo::DComplex, (dtype[1.0] + Cumo::DComplex[1.0]).class
  end

  test "mixing with an integer or a Bit stays BFloat" do
    assert_equal dtype, (dtype[1.0] + Cumo::Int32[1]).class
    assert_equal dtype, (dtype[1.0] + Cumo::Int64[1]).class
    assert_equal dtype, (dtype[1.0] + Cumo::UInt8[1]).class
    assert_equal dtype, (dtype[1.0] + 1).class
    assert_equal dtype, (dtype[1.0] + 1.0).class
  end

  test "storing from and into every other float type round trips" do
    [Cumo::HFloat, Cumo::SFloat, Cumo::DFloat].each do |other|
      assert_equal [1.5, 2.5], dtype.cast(other[1.5, 2.5]).to_a, other.name
      assert_equal [1.5, 2.5], other.cast(dtype[1.5, 2.5]).to_a, other.name
    end
  end

  test "narrowing an HFloat rounds rather than truncates" do
    # HFloat is the one source that reaches the float entry of the conversion;
    # every Ruby number takes the double one. 1 + 6 * 2**-10 is exact in
    # binary16 and three quarters of the way to the next bfloat16, so rounding
    # answers the value above and truncating the one below.
    assert_equal 1.0078125, dtype.cast(Cumo::HFloat[1.0 + 6 * 2.0**-10]).to_a[0]
  end

  test "casting to and from a binary string" do
    a = dtype[1.5, -2.5, 0.0]
    assert_equal a.to_a, dtype.from_binary(a.to_binary, [3]).to_a
  end

  test "a view keeps its stride" do
    a = dtype.new(6).seq
    assert_equal [0.0, 2.0, 4.0], a[(0..5).step(2)].to_a
    assert_equal [5.0, 4.0, 3.0, 2.0, 1.0, 0.0], a.reverse.to_a
  end

  test "reshape, transpose and a matrix reduction" do
    a = dtype.new(2, 3).seq
    assert_equal [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], a.to_a
    assert_equal [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]], a.transpose.to_a
    assert_equal [3.0, 12.0], a.sum(axis: 1).to_a
  end

  test "inspect names the class" do
    assert_match(/Cumo::BFloat/, dtype[1.0].inspect)
  end
end
