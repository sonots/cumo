# frozen_string_literal: true

require_relative "test_helper"

class CumoTest < Test::Unit::TestCase
  def setup
    @orig_compatible_mode = Cumo.compatible_mode_enabled?
  end

  def teardown
    @orig_compatible_mode ? Cumo.enable_compatible_mode : Cumo.disable_compatible_mode
  end

  def test_enable_compatible_mode
    Cumo.enable_compatible_mode
    assert { Cumo.compatible_mode_enabled? }
  end

  def test_disable_compatible_mode
    Cumo.disable_compatible_mode
    assert { !Cumo.compatible_mode_enabled? }
  end

  def test_compatible_mode_extracts_zero_dimensional_reduction_results
    Cumo.enable_compatible_mode
    a = Cumo::DFloat[3, 1, 7]
    assert_equal([Float, Float], a.minmax.map(&:class))
    assert_equal([1.0, 7.0], a.minmax)
    assert_equal([true, true], Cumo::DFloat[3, Float::NAN, 1].minmax(nan: true).map(&:nan?))
    assert_equal([Integer, Integer], Cumo::Int32[3, 1, 7].minmax.map(&:class))
    assert_equal([1, 7], Cumo::Int32[3, 1, 7].minmax)
    assert_equal([Integer, Integer], Cumo::RObject[3, 1, 7].minmax.map(&:class))
    assert_equal(Float, a.max.class)
    assert_equal(Integer, a.max_index.class)
    assert_equal(Integer, a.argmax.class)
  end

  def test_minmax_returns_zero_dimensional_narrays_without_compatible_mode
    Cumo.disable_compatible_mode
    min, max = Cumo::DFloat[3, 1, 7].minmax
    assert_instance_of(Cumo::DFloat, min)
    assert_instance_of(Cumo::DFloat, max)
    assert_equal(0, min.ndim)
    assert_equal([1.0], min.to_a)
    assert_equal([7.0], max.to_a)
  end

  def test_minmax_over_an_axis_is_unaffected_by_compatible_mode
    a = Cumo::DFloat[[3, 1], [7, 2]]
    [true, false].each do |compatible|
      compatible ? Cumo.enable_compatible_mode : Cumo.disable_compatible_mode
      min, max = a.minmax(axis: 1)
      assert_instance_of(Cumo::DFloat, min)
      assert_equal([1.0, 2.0], min.to_a)
      assert_equal([3.0, 7.0], max.to_a)
    end
  end

  def test_version
    assert_nothing_raised { Cumo::VERSION }
  end
end
