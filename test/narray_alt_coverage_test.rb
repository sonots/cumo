# frozen_string_literal: true

require_relative "test_helper"

# Ported from numo-narray-alt's test/test_narray.rb: the methods its suite
# covers that cumo's own narray_test.rb never touches.
class NArrayAltCoverageTest < CumoTestBase
  FLOAT_TYPES = [
    Cumo::SFloat,
    Cumo::SComplex,
    Cumo::DFloat,
    Cumo::DComplex,
  ]

  INTEGER_TYPES = [
    Cumo::Int64,
    Cumo::Int32,
    Cumo::Int16,
    Cumo::Int8,
  ]

  UNSIGNED_INTEGER_TYPES = [
    Cumo::UInt64,
    Cumo::UInt32,
    Cumo::UInt16,
    Cumo::UInt8,
  ]

  def test_each
    TYPES.each do |dtype|
      a = dtype[1, 2, 3, 5, 7, 11]
      actual = []
      a.each { |e| actual << e }
      assert_equal([1, 2, 3, 5, 7, 11], actual)
      actual = []
      a[[0, 3, 5]].each { |e| actual << e }
      assert_equal([1, 5, 11], actual)
    end
  end

  def test_map
    TYPES.each do |dtype|
      a = dtype[1, 2, 3, 5, 7, 11]
      calls = 0
      actual = a.map do |e|
        calls += 1
        e * 2
      end
      assert_kind_of(dtype, actual)
      assert_equal(dtype[2, 4, 6, 10, 14, 22], actual)
      assert_equal(6, calls)
      calls = 0
      actual = a[[0, 3, 5]].map do |e|
        calls += 1
        e + 1
      end
      assert_kind_of(dtype, actual)
      assert_equal(dtype[2, 6, 12], actual)
      assert_equal(3, calls)
    end
  end

  def test_each_with_index
    TYPES.each do |dtype|
      a = dtype[1, 2, 3, 5, 7, 11]
      actual = []
      a.each_with_index { |e, i| actual << [i, e] }
      assert_equal([[0, 1], [1, 2], [2, 3], [3, 5], [4, 7], [5, 11]], actual)
      actual = []
      a[[0, 3, 5]].each_with_index { |e, i| actual << [i, e] }
      assert_equal([[0, 1], [1, 5], [2, 11]], actual)
    end
  end

  def test_map_with_index
    TYPES.each do |dtype|
      a = dtype[1, 2, 3, 5, 7, 11]
      calls = 0
      actual = a.map_with_index do |e, i|
        calls += 1
        e + i
      end
      assert_kind_of(dtype, actual)
      assert_equal(dtype[1, 3, 5, 8, 11, 16], actual)
      assert_equal(6, calls)
      calls = 0
      actual = a[[0, 3, 5]].map_with_index do |e, i|
        calls += 1
        e * i
      end
      assert_kind_of(dtype, actual)
      assert_equal(dtype[0, 5, 22], actual)
      assert_equal(3, calls)
    end
  end

  def test_abs
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      actual = dtype[3.5, -2.1, 0.0, -0.7, -0.9].abs
      assert_kind_of(dtype, actual)
      assert_equal(dtype[3.5, 2.1, 0.0, 0.7, 0.9], actual)
    end
    INTEGER_TYPES.each do |dtype|
      actual = dtype[3, -2, 0, -7, -9].abs
      assert_kind_of(dtype, actual)
      assert_equal(dtype[3, 2, 0, 7, 9], actual)
    end
    UNSIGNED_INTEGER_TYPES.each do |dtype|
      actual = dtype[3, 0, 0, 7, 9].abs
      assert_kind_of(dtype, actual)
      assert_equal(dtype[3, 0, 0, 7, 9], actual)
    end
    actual = Cumo::DComplex[3 + 4i, -8 - 6i, 0 + 0i, 0 - 4i, -5 + 12i].abs
    assert_kind_of(Cumo::DFloat, actual)
    assert_equal(Cumo::DFloat[5, 10, 0, 4, 13], actual)
    actual = Cumo::SComplex[3 + 4i, -8 - 6i, 0 + 0i, 0 - 4i, -5 + 12i].abs
    assert_kind_of(Cumo::SFloat, actual)
    assert_equal(Cumo::SFloat[5, 10, 0, 4, 13], actual)
  end

  def test_reciprocal
    FLOAT_TYPES.each do |dtype|
      assert_equal(dtype[0.5, 0.25, 0.125, 0.1], dtype[2, 4, 8, 10].reciprocal)
    end
    INTEGER_TYPES.each do |dtype|
      assert_equal(dtype[1, -1, 0, 0], dtype[1, -1, 2, -2].reciprocal)
    end
    UNSIGNED_INTEGER_TYPES.each do |dtype|
      assert_equal(dtype[1, 0, 0], dtype[1, 2, 4].reciprocal)
    end
    Float.class_eval %{
      def reciprocal
        1.fdiv(self)
      end
    }, __FILE__, __LINE__ - 4
    assert_equal(Cumo::RObject[0.5, 0.25, 0.125, 0.1], Cumo::RObject[2.0, 4.0, 8.0, 10.0].reciprocal)
  end

  def test_sign
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      assert_equal(dtype[1, -1, 0, 1, -1], dtype[3.5, -2.1, 0.0, 0.7, -0.9].sign)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      assert_equal(dtype[1 + 1i, -1 - 1i, 0 + 0i, 0 - 1i, -1 + 0i],
                   dtype[3 + 2i, -2 - 3i, 0 + 0i, 0 - 4i, -5 + 0i].sign)
    end
    INTEGER_TYPES.each do |dtype|
      assert_equal(dtype[1, -1, 0, 1, -1], dtype[3, -2, 0, 7, -9].sign)
    end
    UNSIGNED_INTEGER_TYPES.each do |dtype|
      assert_equal(dtype[1, 0, 0, 1, 0], dtype[3, 0, 0, 7, 0].sign)
    end
  end

  def test_nearly_eq
    a = Cumo::SFloat[1.0, 1.0, 1.0, 1.0]
    b = Cumo::SFloat[1.0 + 1e-7, 1.0 - 1e-7, 1.0 + 1e-6, 1.0 - 1e-6]
    actual = a.nearly_eq(b)
    assert_kind_of(Cumo::Bit, actual)
    assert_equal(Cumo::Bit[1, 1, 0, 0], actual)
    a = Cumo::DFloat[1.0, 1.0, 1.0, 1.0]
    b = Cumo::DFloat[1.0 + 1e-16, 1.0 - 1e-16, 1.0 + 1e-15, 1.0 - 1e-15]
    actual = a.nearly_eq(b)
    assert_kind_of(Cumo::Bit, actual)
    assert_equal(Cumo::Bit[1, 1, 0, 0], actual)
    a = Cumo::RObject[1.0, 1.0, 1.0, 1.0]
    b = Cumo::RObject[1.0 + 1e-16, 1.0 - 1e-16, 1.0 + 1e-15, 1.0 - 1e-15]
    actual = a.nearly_eq(b)
    assert_kind_of(Cumo::Bit, actual)
    assert_equal(Cumo::Bit[1, 1, 0, 0], actual)
    a = Cumo::SComplex[1.0 + 1.0i, 1.0 + 1.0i, 1.0 + 1.0i, 1.0 + 1.0i]
    b = Cumo::SComplex[1.0 + 1.0i + (1e-7 + 1e-7i), 1.0 + 1.0i - (1e-7 + 1e-7i),
                       1.0 + 1.0i + (1e-6 + 1e-6i), 1.0 + 1.0i - (1e-6 + 1e-6i)]
    actual = a.nearly_eq(b)
    assert_kind_of(Cumo::Bit, actual)
    assert_equal(Cumo::Bit[1, 1, 0, 0], actual)
    a = Cumo::DComplex[1.0 + 1.0i, 1.0 + 1.0i, 1.0 + 1.0i, 1.0 + 1.0i]
    b = Cumo::DComplex[1.0 + 1.0i + (1e-16 + 1e-16i), 1.0 + 1.0i - (1e-16 + 1e-16i),
                       1.0 + 1.0i + (1e-15 + 1e-15i), 1.0 + 1.0i - (1e-15 + 1e-15i)]
    actual = a.nearly_eq(b)
    assert_kind_of(Cumo::Bit, actual)
    assert_equal(Cumo::Bit[1, 1, 0, 0], actual)
  end

  def test_floor
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      actual = dtype[1.5, -2.3, 3.0, -4.7, 5.9].floor
      assert_equal(dtype[1.0, -3.0, 3.0, -5.0, 5.0], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1.5 + 2.3i, -2.3 - 3.7i, 3.0 + 4.0i].floor
      assert_equal(dtype[1.0 + 2.0i, -3.0 - 4.0i, 3.0 + 4.0i], actual)
    end
  end

  def test_round
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      actual = dtype[1.5, -2.3, 3.0, -4.7, 5.9].round
      assert_equal(dtype[2.0, -2.0, 3.0, -5.0, 6.0], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1.5 + 2.3i, -2.3 - 3.7i, 3.0 + 4.0i].round
      assert_equal(dtype[2.0 + 2.0i, -2.0 - 4.0i, 3.0 + 4.0i], actual)
    end
  end

  def test_ceil
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      actual = dtype[1.5, -2.3, 3.0, -4.7, 5.9].ceil
      assert_equal(dtype[2.0, -2.0, 3.0, -4.0, 6.0], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1.5 + 2.3i, -2.3 - 3.7i, 3.0 + 4.0i].ceil
      assert_equal(dtype[2.0 + 3.0i, -2.0 - 3.0i, 3.0 + 4.0i], actual)
    end
  end

  def test_trunc
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      actual = dtype[1.5, -2.3, 3.0, -4.7, 5.9].trunc
      assert_equal(dtype[1.0, -2.0, 3.0, -4.0, 5.0], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1.5 + 2.3i, -2.3 - 3.7i, 3.0 + 4.0i].trunc
      assert_equal(dtype[1.0 + 2.0i, -2.0 - 3.0i, 3.0 + 4.0i], actual)
    end
  end

  def test_rint
    [Cumo::DFloat, Cumo::SFloat].each do |dtype|
      actual = dtype[1.5, -2.5, 3.0, -4.5, 5.9].rint
      assert_equal(dtype[2.0, -2.0, 3.0, -4.0, 6.0], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1.5 + 2.5i, -2.5 - 3.5i, 3.0 + 4.0i].rint
      assert_equal(dtype[2.0 + 2.0i, -2.0 - 4.0i, 3.0 + 4.0i], actual)
    end
  end

  def test_copysign
    [Cumo::DFloat, Cumo::SFloat].each do |dtype|
      actual = dtype[1, -2, 3, -4, 5].copysign(dtype[-1, 1, -1, 1, 0])
      assert_kind_of(dtype, actual)
      assert_equal(dtype[-1, 2, -3, 4, 5], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1 + 2i, -2 - 3i, 3 + 4i].copysign(dtype[-1 + 1i, 1 - 1i, -1 + 0i])
      assert_kind_of(dtype, actual)
      assert_equal(dtype[-1 + 2i, 2 - 3i, -3 + 4i], actual)
    end
  end

  def test_signbit
    [Cumo::DFloat, Cumo::SFloat].each do |dtype|
      actual = dtype[1.0, -2.0, 0.0, -0.0, 3.5].signbit
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[0, 1, 0, 1, 0], actual)
    end
  end

  def test_seq
    TYPES.each do |dtype|
      assert_equal(dtype[0, 1, 2, 3, 4], dtype.new(5).seq)
      assert_equal(dtype[2, 3, 4, 5, 6], dtype.new(5).seq(2))
      assert_equal(dtype[5, 8, 11, 14, 17], dtype.new(5).seq(5, 3))
    end
    [Cumo::DFloat, Cumo::SFloat].each do |dtype|
      assert_equal(dtype[1, 10, 100, 1000], dtype.new(4).logseq(0, 1))
      assert_equal(dtype[16, 8, 4, 2, 1], dtype.new(5).logseq(4, -1, 2))
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype.new(3).logseq(2, 1, 1 + 2i)
      assert_equal(3, actual.size)
      [-3 + 4i, -11 - 2i, -7 - 24i].each_with_index do |expected, i|
        assert_operator((expected - actual[i].extract_cpu).abs, :<, 1e-5)
      end
    end
  end

  def test_rand
    INTEGER_TYPES.each do |dtype|
      Cumo::NArray.srand(1_234_567_890)
      if dtype.byte_size < 8
        assert_equal(dtype[3, 4, 1, 4, 0, -2, -2, 2, -2, -3], dtype.new(10).rand(-3, 5))
        assert_equal(dtype[[6, 5, 8, 2, 4], [1, 0, 4, 9, 2]], dtype.new(2, 5).rand(10))
      else
        assert_equal(dtype[3, 1, 4, -2, 2, -2, -3, 2, 1, -1], dtype.new(10).rand(-3, 5))
        assert_equal(dtype[[3, 6, 1, 5, 7], [9, 9, 6, 4, 0]], dtype.new(2, 5).rand(10))
      end
    end
    UNSIGNED_INTEGER_TYPES.each do |dtype|
      Cumo::NArray.srand(1_234_567_890)
      if dtype.byte_size < 8
        assert_equal(dtype[6, 7, 4, 7, 3, 1, 1, 5, 1, 0], dtype.new(10).rand(10))
        assert_equal(dtype[[6, 5, 8, 2, 4], [1, 0, 4, 9, 2]], dtype.new(2, 5).rand(10))
      else
        assert_equal(dtype[6, 4, 7, 1, 5, 1, 0, 5, 8, 4], dtype.new(10).rand(10))
        assert_equal(dtype[[2, 3, 6, 1, 5], [7, 9, 9, 6, 4]], dtype.new(2, 5).rand(10))
      end
    end

    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      Cumo::NArray.srand(1_234_567_890)
      actual1d = dtype.new(5).rand(-1, 1)
      actual2d = dtype.new(2, 3).rand(-1, 1)
      assert_equal(2, actual2d.ndim)
      assert_equal(2, actual2d.shape[0])
      assert_equal(3, actual2d.shape[1])
      if dtype == Cumo::SFloat
        [-0.128008, -0.0780624, -0.470543, 0.808037, -0.00503415].each_with_index do |expected, i|
          assert_in_delta(expected, actual1d[i], 1e-6)
        end
        [0.279245, 0.838221, -0.621664].each_with_index do |expected, i|
          assert_in_delta(expected, actual2d[0, i], 1e-6)
        end
        [-0.860478, 0.933182, 0.592224].each_with_index do |expected, i|
          assert_in_delta(expected, actual2d[1, i], 1e-6)
        end
      elsif dtype == Cumo::DFloat
        [-0.0780623, 0.808037, 0.279245, -0.621664, 0.933182].each_with_index do |expected, i|
          assert_in_delta(expected, actual1d[i], 1e-6)
        end
        [-0.87189, 0.359531, 0.258884].each_with_index do |expected, i|
          assert_in_delta(expected, actual2d[0, i], 1e-6)
        end
        [-0.248656, 0.766334, -0.625959].each_with_index do |expected, i|
          assert_in_delta(expected, actual2d[1, i], 1e-6)
        end
      elsif dtype == Cumo::RObject
        [-0.078062, 0.808036, 0.279245, -0.621664, 0.933181].each_with_index do |expected, i|
          assert_in_delta(expected, actual1d[i], 1e-6)
        end
        [-0.871889, 0.359530, 0.258884].each_with_index do |expected, i|
          assert_in_delta(expected, actual2d[0, i], 1e-6)
        end
        [-0.248655, 0.766333, -0.625958].each_with_index do |expected, i|
          assert_in_delta(expected, actual2d[1, i], 1e-6)
        end
      end
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      Cumo::NArray.srand(1_234_567_890)
      actual1d = dtype.new(5).rand(2 + 3i)
      actual2d = dtype.new(2, 3).rand(-2 - 3i, 1 + 2i)
      assert_equal(2, actual2d.ndim)
      assert_equal(2, actual2d.shape[0])
      assert_equal(3, actual2d.shape[1])
      if dtype == Cumo::SComplex
        [0.871991 + 1.382906i, 0.529456 + 2.712055i, 0.994965 + 1.918868i, 1.838220 + 0.567503i,
         0.139521 + 2.899772i].each_with_index do |expected, i|
          assert_in_delta(expected.real, actual1d[i].real, 1e-6)
          assert_in_delta(expected.imag, actual1d[i].imag, 1e-6)
        end
        [0.388335 - 2.679724i, -1.060235 + 0.398827i, -1.702621 + 0.147211i].each_with_index do |expected, i|
          assert_in_delta(expected.real, actual2d[0, i].real, 1e-6)
          assert_in_delta(expected.imag, actual2d[0, i].imag, 1e-6)
        end
        [-1.871673 - 1.121639i, -0.888626 + 1.415834i, -0.451813 - 2.064896i].each_with_index do |expected, i|
          assert_in_delta(expected.real, actual2d[1, i].real, 1e-6)
          assert_in_delta(expected.imag, actual2d[1, i].imag, 1e-6)
        end
      elsif dtype == Cumo::DComplex
        [0.921937 + 2.712055i, 1.279245 + 0.567503i, 1.933181 + 0.192165i, 1.359530 + 1.888326i,
         0.751344 + 2.649500i].each_with_index do |expected, i|
          assert_in_delta(expected.real, actual1d[i].real, 1e-6)
          assert_in_delta(expected.imag, actual1d[i].imag, 1e-6)
        end
        [-1.438938 - 1.474032i, -1.687581 + 0.280124i, -1.992894 - 0.0294957i].each_with_index do |expected, i|
          assert_in_delta(expected.real, actual2d[0, i].real, 1e-6)
          assert_in_delta(expected.imag, actual2d[0, i].imag, 1e-6)
        end
        [0.795879 - 1.566802i, 0.835083 - 1.379133i, 0.245460 + 1.054923i].each_with_index do |expected, i|
          assert_in_delta(expected.real, actual2d[1, i].real, 1e-6)
          assert_in_delta(expected.imag, actual2d[1, i].imag, 1e-6)
        end
      end
      actual = Cumo::SComplex.new(5).rand
      assert(actual.real.to_a.any? { |v| v != 0.0 })
      assert(actual.imag.to_a.any? { |v| v != 0.0 })
      actual = Cumo::DComplex.new(5).rand
      assert(actual.real.to_a.any? { |v| v != 0.0 })
      assert(actual.imag.to_a.any? { |v| v != 0.0 })
    end

  end

  def test_rand_norm
    [Cumo::DFloat, Cumo::SFloat].each do |dtype|
      Cumo::NArray.srand(1_234_567)
      actual = dtype.new(100_000).rand_norm
      assert_in_delta(0.0, actual.mean, 0.01)
      assert_in_delta(1.0, actual.var, 0.01)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      Cumo::NArray.srand(1_234_567)
      actual = dtype.new(100_000).rand_norm
      assert_in_delta(0.0, actual.mean.real, 0.01)
      assert_in_delta(0.0, actual.mean.imag, 0.01)
      assert_in_delta(1.0, actual.real.var, 0.01)
      assert_in_delta(1.0, actual.imag.var, 0.01)
    end
  end

  def test_isnan
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      actual = dtype[1.0, Float::NAN, 2.0, Float::NAN, 3.0].isnan
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[0, 1, 0, 1, 0], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1.0 + 2.0i, Float::NAN - 2.0i, 4.0 + (Float::NAN * 1i), Float::NAN + (Float::NAN * 1i), 3.0 + 2.0i].isnan
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[0, 1, 1, 1, 0], actual)
    end
  end

  def test_isinf
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      actual = dtype[1.0, Float::INFINITY, -Float::INFINITY, 2.0, Float::NAN].isinf
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[0, 1, 1, 0, 0], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1.0 + 2.0i, Float::INFINITY - 2.0i, 4.0 - (Float::INFINITY * 1i), Float::NAN + (Float::NAN * 1i),
                     3.0 + 2.0i].isinf
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[0, 1, 1, 0, 0], actual)
    end
  end

  def test_isposinf
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      actual = dtype[1.0, Float::INFINITY, -Float::INFINITY, 2.0, Float::NAN].isposinf
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[0, 1, 0, 0, 0], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1.0 + 2.0i, Float::INFINITY - 2.0i, 4.0 - (Float::INFINITY * 1i), Float::NAN + (Float::NAN * 1i),
                     3.0 + 2.0i].isposinf
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[0, 1, 0, 0, 0], actual)
    end
  end

  def test_isneginf
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      actual = dtype[1.0, Float::INFINITY, -Float::INFINITY, 2.0, Float::NAN].isneginf
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[0, 0, 1, 0, 0], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1.0 + 2.0i, Float::INFINITY - 2.0i, 4.0 - (Float::INFINITY * 1i), Float::NAN + (Float::NAN * 1i),
                     3.0 + 2.0i].isneginf
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[0, 0, 1, 0, 0], actual)
    end
  end

  def test_isfinite
    [Cumo::DFloat, Cumo::SFloat, Cumo::RObject].each do |dtype|
      actual = dtype[1.0, Float::INFINITY, -Float::INFINITY, 2.0, Float::NAN].isfinite
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[1, 0, 0, 1, 0], actual)
    end
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      actual = dtype[1.0 + 2.0i, Float::INFINITY - 2.0i, 4.0 - (Float::INFINITY * 1i), Float::NAN + (Float::NAN * 1i),
                     3.0 + 2.0i].isfinite
      assert_kind_of(Cumo::Bit, actual)
      assert_equal(Cumo::Bit[1, 0, 0, 0, 1], actual)
    end
  end

  def test_comparison_gt
    a = Cumo::DFloat[1, 2, 3, 4, 5]
    result = a.gt(3)
    assert_kind_of(Cumo::Bit, result)
    assert_equal(Cumo::Bit[0, 0, 0, 1, 1], result)

    result = a.gt(Cumo::DFloat[3, 3, 3, 3, 3])
    assert_kind_of(Cumo::Bit, result)
    assert_equal(Cumo::Bit[0, 0, 0, 1, 1], result)

    a = Cumo::Int32[-2, -1, 0, 1, 2]
    assert_equal(Cumo::Bit[0, 0, 0, 1, 1], a.gt(0))
    assert_equal(Cumo::Bit[0, 0, 1, 1, 1], a.gt(-1))

    a = Cumo::UInt8[0, 1, 2, 3, 4]
    assert_equal(Cumo::Bit[0, 0, 0, 1, 1], a.gt(2))
  end

  def test_comparison_ge
    a = Cumo::DFloat[1, 2, 3, 4, 5]
    result = a.ge(3)
    assert_kind_of(Cumo::Bit, result)
    assert_equal(Cumo::Bit[0, 0, 1, 1, 1], result)

    result = a.ge(Cumo::DFloat[3, 3, 3, 3, 3])
    assert_kind_of(Cumo::Bit, result)
    assert_equal(Cumo::Bit[0, 0, 1, 1, 1], result)

    a = Cumo::Int32[-2, -1, 0, 1, 2]
    assert_equal(Cumo::Bit[0, 0, 1, 1, 1], a.ge(0))
    assert_equal(Cumo::Bit[0, 1, 1, 1, 1], a.ge(-1))

    a = Cumo::UInt8[0, 1, 2, 3, 4]
    assert_equal(Cumo::Bit[0, 0, 1, 1, 1], a.ge(2))
  end

  def test_comparison_lt
    a = Cumo::DFloat[1, 2, 3, 4, 5]
    result = a.lt(3)
    assert_kind_of(Cumo::Bit, result)
    assert_equal(Cumo::Bit[1, 1, 0, 0, 0], result)

    result = a.lt(Cumo::DFloat[3, 3, 3, 3, 3])
    assert_kind_of(Cumo::Bit, result)
    assert_equal(Cumo::Bit[1, 1, 0, 0, 0], result)

    a = Cumo::Int32[-2, -1, 0, 1, 2]
    assert_equal(Cumo::Bit[1, 1, 0, 0, 0], a.lt(0))
    assert_equal(Cumo::Bit[1, 0, 0, 0, 0], a.lt(-1))

    a = Cumo::UInt8[0, 1, 2, 3, 4]
    assert_equal(Cumo::Bit[1, 1, 0, 0, 0], a.lt(2))
  end

  def test_comparison_le
    a = Cumo::DFloat[1, 2, 3, 4, 5]
    result = a.le(3)
    assert_kind_of(Cumo::Bit, result)
    assert_equal(Cumo::Bit[1, 1, 1, 0, 0], result)

    result = a.le(Cumo::DFloat[3, 3, 3, 3, 3])
    assert_kind_of(Cumo::Bit, result)
    assert_equal(Cumo::Bit[1, 1, 1, 0, 0], result)

    a = Cumo::Int32[-2, -1, 0, 1, 2]
    assert_equal(Cumo::Bit[1, 1, 1, 0, 0], a.le(0))
    assert_equal(Cumo::Bit[1, 1, 0, 0, 0], a.le(-1))

    a = Cumo::UInt8[0, 1, 2, 3, 4]
    assert_equal(Cumo::Bit[1, 1, 1, 0, 0], a.le(2))
  end

  def test_empty?
    TYPES.each do |dtype|
      assert_predicate(dtype[], :empty?)
      refute_predicate(dtype[1, 2, 3], :empty?)
    end
  end

  def test_contiguous
    TYPES.each do |dtype|
      a = dtype[1, 2, 3, 4, 5]
      assert_predicate(a, :contiguous?)
      assert_predicate(a, :row_major?)
      refute_predicate(a, :column_major?)
    end
  end

  def test_fortran_contiguous
    TYPES.each do |dtype|
      a = dtype[1, 2, 3, 4, 5]
      refute_predicate(a, :fortran_contiguous?)
    end
  end

  def test_inplace
    a = Cumo::DFloat[1, 2, 3]
    refute_predicate(a, :inplace?)
    view = a.inplace
    assert_kind_of(Cumo::DFloat, view)
    assert_equal(a, view)
  end

  def test_inplace_bang
    a = Cumo::DFloat[1, 2, 3]
    refute_predicate(a, :inplace?)
    a.inplace!
    assert_predicate(a, :inplace?)
    a.out_of_place!
    refute_predicate(a, :inplace?)
  end

  def test_byte_size
    assert_equal(24, Cumo::DFloat[1, 2, 3].byte_size)
    assert_equal(12, Cumo::SFloat[1, 2, 3].byte_size)
    assert_equal(12, Cumo::Int32[1, 2, 3].byte_size)
    assert_equal(6, Cumo::Int16[1, 2, 3].byte_size)
    assert_equal(3, Cumo::Int8[1, 2, 3].byte_size)
  end

  def test_upcast
    assert_equal(Cumo::DFloat, Cumo::DFloat.upcast(Cumo::SFloat))
    assert_equal(Cumo::DComplex, Cumo::DComplex.upcast(Cumo::DFloat))
    assert_equal(Cumo::DFloat, Cumo::DFloat.upcast(Cumo::Int32))
    assert_equal(Cumo::SFloat, Cumo::SFloat.upcast(Cumo::Int16))
  end

  def test_equality
    TYPES.each do |dtype|
      a = dtype[1, 2, 3]
      b = dtype[1, 2, 3]
      c = dtype[1, 2, 4]
      assert_equal(a, b)
      refute_equal(a, c)
    end
  end

  def test_diagonal
    a = Cumo::DFloat[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert_equal(Cumo::DFloat[1, 5, 9], a.diagonal)
    a = Cumo::Int32[[1, 2, 3, 4], [5, 6, 7, 8]]
    assert_equal(Cumo::Int32[1, 6], a.diagonal)
  end

  def test_linspace
    a = Cumo::DFloat.linspace(0, 1, 5)
    assert_equal(5, a.size)
    assert_in_delta(0.0, a[0], 1e-10)
    assert_in_delta(1.0, a[-1], 1e-10)
    assert_in_delta(0.25, a[1], 1e-10)
  end

  def test_logspace
    a = Cumo::DFloat.logspace(0, 3, 4)
    assert_equal(4, a.size)
    assert_in_delta(1.0, a[0], 1e-10)
    assert_in_delta(1000.0, a[-1], 1e-10)
    assert_in_delta(10.0, a[1], 1e-10)
  end

  def test_complex_conj
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      a = dtype[1 + 2i, 3 - 4i, 0 + 0i, -5 + 6i]
      conj_a = a.conj
      assert_kind_of(dtype, conj_a)
      assert_equal(dtype[1 - 2i, 3 + 4i, 0 - 0i, -5 - 6i], conj_a)
      assert_equal(a, a.conj.conj)
    end
  end

  def test_complex_real
    a = Cumo::DComplex[1 + 2i, 3 - 4i, 0 + 0i, -5 + 6i]
    real_a = a.real
    assert_kind_of(Cumo::DFloat, real_a)
    assert_equal(Cumo::DFloat[1, 3, 0, -5], real_a)

    a = Cumo::SComplex[1 + 2i, 3 - 4i, 0 + 0i, -5 + 6i]
    real_a = a.real
    assert_kind_of(Cumo::SFloat, real_a)
    assert_equal(Cumo::SFloat[1, 3, 0, -5], real_a)
  end

  def test_complex_imag
    a = Cumo::DComplex[1 + 2i, 3 - 4i, 0 + 0i, -5 + 6i]
    imag_a = a.imag
    assert_kind_of(Cumo::DFloat, imag_a)
    assert_equal(Cumo::DFloat[2, -4, 0, 6], imag_a)

    a = Cumo::SComplex[1 + 2i, 3 - 4i, 0 + 0i, -5 + 6i]
    imag_a = a.imag
    assert_kind_of(Cumo::SFloat, imag_a)
    assert_equal(Cumo::SFloat[2, -4, 0, 6], imag_a)
  end

  def test_complex_real_setter
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      a = dtype[1 + 2i, 3 + 4i]
      a.real = 10
      assert_equal(dtype[10 + 2i, 10 + 4i], a)

      a = dtype[1 + 2i, 3 + 4i]
      a.real = [10, 20]
      assert_equal(dtype[10 + 2i, 20 + 4i], a)
    end
  end

  def test_complex_imag_setter
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      a = dtype[1 + 2i, 3 + 4i]
      a.imag = 10
      assert_equal(dtype[1 + 10i, 3 + 10i], a)

      a = dtype[1 + 2i, 3 + 4i]
      a.imag = [10, 20]
      assert_equal(dtype[1 + 10i, 3 + 20i], a)
    end
  end

  def test_complex_im
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      a = dtype[1 + 2i, 3 - 4i]
      result = a.im
      assert_kind_of(dtype, result)
      assert_equal(dtype[-2 + 1i, 4 + 3i], result)
    end
  end

  def test_complex_angle
    a = Cumo::DComplex[1 + 0i, 0 + 1i, -1 + 0i, 0 - 1i]
    angle = a.angle
    assert_kind_of(Cumo::DFloat, angle)
    assert_in_delta(0.0, angle[0], 1e-10)
    assert_in_delta(Math::PI / 2, angle[1], 1e-10)
    assert_in_delta(Math::PI, angle[2], 1e-10)
    assert_in_delta(-Math::PI / 2, angle[3], 1e-10)

    a = Cumo::SComplex[1 + 0i, 0 + 1i, -1 + 0i, 0 - 1i]
    angle = a.angle
    assert_kind_of(Cumo::SFloat, angle)
    assert_in_delta(0.0, angle[0], 1e-6)
    assert_in_delta(Math::PI / 2, angle[1], 1e-6)
  end

  def test_complex_conj_of_purely_real
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      a = dtype[1 + 0i, 2 + 0i, 3 + 0i]
      assert_equal(a, a.conj)
    end
  end

  def test_complex_conj_of_purely_imaginary
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      a = dtype[0 + 1i, 0 + 2i, 0 + 3i]
      conj_a = a.conj
      assert_equal(dtype[0 - 1i, 0 - 2i, 0 - 3i], conj_a)
      assert_equal(-a, conj_a)
    end
  end

  def test_complex_conj_preserves_operations
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      a = dtype[1 + 2i, 3 + 4i]
      b = dtype[5 + 6i, 7 + 8i]
      assert_equal((a + b).conj, a.conj + b.conj)
      assert_equal((a * b).conj, a.conj * b.conj)
    end
  end

  def test_complex_2d_real_imag
    a = Cumo::DComplex[[1 + 2i, 3 + 4i], [5 + 6i, 7 + 8i]]
    assert_equal(Cumo::DFloat[[1, 3], [5, 7]], a.real)
    assert_equal(Cumo::DFloat[[2, 4], [6, 8]], a.imag)
    assert_equal(Cumo::DComplex[[1 - 2i, 3 - 4i], [5 - 6i, 7 - 8i]], a.conj)
  end

  def test_complex_real_imag_of_zero
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      a = dtype[0 + 0i, 0 + 0i]
      assert_equal(0, a.real.sum)
      assert_equal(0, a.imag.sum)
    end
  end

  def test_complex_abs_angle_roundtrip
    [Cumo::DComplex, Cumo::SComplex].each do |dtype|
      a = dtype[1 + 2i, 3 - 4i, -5 + 6i]
      mag = a.abs
      angle = a.angle
      reconstructed = mag * (Cumo::NMath.cos(angle) + (Cumo::NMath.sin(angle) * 1i))
      a.each_with_index do |expected, i|
        assert_in_delta(expected.real, reconstructed[i].real, 1e-6)
        assert_in_delta(expected.imag, reconstructed[i].imag, 1e-6)
      end
    end
  end
  def test_integer_division_by_zero
    omit("Cumo's integer division has no zero check: the guard in binary.c is in the robject branch, and the kernel has none")
    INTEGER_TYPES.each do |dtype|
      assert_raise(ZeroDivisionError) { dtype[1, 0].reciprocal }
      assert_raise(ZeroDivisionError) { dtype[1] / dtype[0] }
    end
  end

  def test_robject_logseq
    assert_equal(Cumo::RObject[1, 10, 100, 1000], Cumo::RObject.new(4).logseq(0, 1))
    assert_equal(Cumo::RObject[16, 8, 4, 2, 1], Cumo::RObject.new(5).logseq(4, -1, 2))
  end
  def test_zero_dimensional_view_is_contiguous
    a = Cumo::DFloat.cast(3.5)
    bin = [3.5].pack("d")

    assert_predicate(a.view, :contiguous?)
    assert_equal(bin, a.view.to_binary)
    assert_equal(bin, a.transpose.to_binary)
    assert_equal(bin, a.flatten.to_binary)
    assert_equal([1, [], 0, bin], a.view.marshal_dump)
    assert_equal(a, Marshal.load(Marshal.dump(a.view)))

    assert_equal([1, [], 0, [:sym]], Cumo::RObject.cast(:sym).view.marshal_dump)
  end

  def test_reshape_bang_on_a_zero_dimensional_view
    a = Cumo::DFloat.cast(3.5)

    assert_equal([3.5], a.view.reshape!(1).to_a)
    assert_equal([[3.5]], a.view.reshape!(1, 1).to_a)
  end
end
