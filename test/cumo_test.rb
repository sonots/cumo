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

  ZERO_DIMENSIONAL_FLOATS = {
    aref: -> { Cumo::DFloat[3, 1, 7][1] },
    extract: -> { Cumo::DFloat.cast(7.0).extract },
    sum: -> { Cumo::DFloat[3, 1, 7].sum },
    prod: -> { Cumo::DFloat[3, 1, 7].prod },
    mean: -> { Cumo::DFloat[3, 1, 7].mean },
    stddev: -> { Cumo::DFloat[3, 1, 7].stddev },
    var: -> { Cumo::DFloat[3, 1, 7].var },
    rms: -> { Cumo::DFloat[3, 1, 7].rms },
    min: -> { Cumo::DFloat[3, 1, 7].min },
    max: -> { Cumo::DFloat[3, 1, 7].max },
    ptp: -> { Cumo::DFloat[3, 1, 7].ptp },
    median: -> { Cumo::DFloat[3, 1, 7].median },
    mulsum: -> { Cumo::DFloat[3, 1, 7].mulsum(Cumo::DFloat[1, 1, 1]) },
    dot: -> { Cumo::DFloat[3, 1, 7].dot(Cumo::DFloat[1, 1, 1]) },
    inner: -> { Cumo::DFloat[3, 1, 7].inner(Cumo::DFloat[1, 1, 1]) },
  }.freeze

  ZERO_DIMENSIONAL_INTEGERS = {
    max_index: -> { Cumo::DFloat[3, 1, 7].max_index },
    min_index: -> { Cumo::DFloat[3, 1, 7].min_index },
    argmax: -> { Cumo::DFloat[3, 1, 7].argmax },
    argmin: -> { Cumo::DFloat[3, 1, 7].argmin },
    count_true: -> { (Cumo::DFloat[3, 1, 7] > 2).count_true },
    count_false: -> { (Cumo::DFloat[3, 1, 7] > 2).count_false },
    bit_aref: -> { (Cumo::DFloat[3, 1, 7] > 2)[1] },
  }.freeze

  def test_compatible_mode_extracts_every_documented_method
    Cumo.enable_compatible_mode
    ZERO_DIMENSIONAL_FLOATS.each do |name, block|
      assert_instance_of(Float, block.call, name)
    end
    ZERO_DIMENSIONAL_INTEGERS.each do |name, block|
      assert_instance_of(Integer, block.call, name)
    end
  end

  def test_every_documented_method_stays_zero_dimensional_without_compatible_mode
    Cumo.disable_compatible_mode
    (ZERO_DIMENSIONAL_FLOATS.merge(ZERO_DIMENSIONAL_INTEGERS)).each do |name, block|
      result = block.call
      assert_kind_of(Cumo::NArray, result, name)
      assert_equal(0, result.ndim, name)
    end
  end

  def test_kernel_conversions_read_a_result_in_either_mode
    [true, false].each do |compatible|
      compatible ? Cumo.enable_compatible_mode : Cumo.disable_compatible_mode
      a = Cumo::DFloat[5, 2]
      assert_equal(7.0, Float(a.sum))
      assert_equal(0, Integer(a.max_index))
      assert_equal(false, a.aref_cpu(0) < 1.0)
      assert_equal(2, (a > 1).count_true_cpu)
      assert_equal(5.0, Cumo::DFloat.cast(5.0).extract_cpu)
    end
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
