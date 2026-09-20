# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class ModuleTest < Test::Unit::TestCase
    # get_function answered nil, so a caller found out at whatever it tried to
    # do with what it got back rather than where it asked.
    def test_get_function_says_it_cannot_launch
      assert_raise(NotImplementedError) { Module.new.get_function("foo") }
    end

    def test_unload_without_a_module_does_nothing
      assert_nothing_raised { Module.new.unload }
    end
  end
end
