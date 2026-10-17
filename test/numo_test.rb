# frozen_string_literal: true

require "numo/narray"
require_relative "test_helper"

class NumoConversionTest < CumoTestBase
  TYPE_NAMES = %w[Bit Int8 Int16 Int32 Int64 UInt8 UInt16 UInt32 UInt64 SFloat DFloat SComplex DComplex].freeze
  SHAPES = [[], [0], [0, 3], [2, 0, 3], [1], [7], [3, 5], [2, 3, 4]].freeze

  def numo_array(name, *shape)
    a = Numo.const_get(name).new(*shape)
    return a if a.empty?

    case name
    when "Bit" then (Numo::Int32.new(*shape).seq % 3).eq(0)
    when /\AU/ then a.seq(0, 3)
    when /Complex/ then a.seq(-5, 3) * Complex(1, 0.5)
    else a.seq(-5, 3)
    end
  end

  test "from_numo copies every type and shape into the Cumo type of the same name" do
    TYPE_NAMES.each do |name|
      SHAPES.each do |shape|
        src = numo_array(name, *shape)
        a = Cumo::NArray.from_numo(src)
        assert_equal(Cumo.const_get(name), a.class)
        assert_equal(shape, a.shape)
        assert_equal(src.to_a, a.to_a) unless src.empty?
      end
    end
  end

  test "to_numo copies every type and shape into the Numo type of the same name" do
    TYPE_NAMES.each do |name|
      SHAPES.each do |shape|
        src = numo_array(name, *shape)
        a = Cumo::NArray.to_numo(Cumo::NArray.from_numo(src))
        assert_equal(Numo.const_get(name), a.class)
        assert_equal(shape, a.shape)
        assert_equal(src.to_a, a.to_a)
      end
    end
  end

  test "views on either side are copied as the elements they show" do
    TYPE_NAMES.each do |name|
      src = numo_array(name, 9, 17)
      a = Cumo::NArray.from_numo(src)
      [[1..7, 3..14], [true, (1..).step(3)], [4, 5..15]].each do |index|
        expected = src[*index].to_a
        assert_equal(expected, Cumo::NArray.from_numo(src[*index]).to_a, "#{name} #{index}")
        assert_equal(expected, Cumo::NArray.to_numo(a[*index]).to_a, "#{name} #{index}")
      end
    end
  end

  test "empty views on either side are copied with their shape" do
    TYPE_NAMES.each do |name|
      src = numo_array(name, 3, 4)
      a = Cumo::NArray.from_numo(src)
      [[[], true], [[], 1..2]].each do |index|
        assert_equal(src[*index].shape, Cumo::NArray.from_numo(src[*index]).shape, "#{name} #{index}")
        assert_equal(a[*index].shape, Cumo::NArray.to_numo(a[*index]).shape, "#{name} #{index}")
      end
    end
  end

  test "byte-swapped arrays are copied as their values in host order" do
    (TYPE_NAMES - ["Bit"]).each do |name|
      src = numo_array(name, 3, 5)
      a = Cumo::NArray.from_numo(src.to_network)
      assert_false(a.byte_swapped?, name)
      assert_equal(src.to_a, a.to_a, name)
      b = Cumo::NArray.to_numo(a.to_network)
      assert_false(b.byte_swapped?, name)
      assert_equal(src.to_a, b.to_a, name)
    end
  end

  test "a typed class takes and gives only its own type" do
    assert_equal(Cumo::DFloat, Cumo::DFloat.from_numo(Numo::DFloat[1, 2]).class)
    assert_equal(Numo::DFloat, Cumo::DFloat.to_numo(Cumo::DFloat[1, 2]).class)
    assert_raise(TypeError) { Cumo::DFloat.from_numo(Numo::Int32[1, 2]) }
    assert_raise(TypeError) { Cumo::DFloat.to_numo(Cumo::Int32[1, 2]) }
  end

  test "from_numo raises TypeError for what is not a Numo array of a shared type" do
    [Numo::RObject[1], Cumo::DFloat[1], [1, 2], 1].each do |x|
      assert_raise(TypeError) { Cumo::NArray.from_numo(x) }
    end
  end

  test "to_numo raises TypeError for what is not a Cumo array of a shared type" do
    [Cumo::HFloat[1], Cumo::BFloat[1], Cumo::RObject[1], Numo::DFloat[1], nil].each do |x|
      assert_raise(TypeError) { Cumo::NArray.to_numo(x) }
    end
  end
end
