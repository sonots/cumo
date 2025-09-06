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

  def test_version
    assert_nothing_raised { Cumo::VERSION }
  end
end
