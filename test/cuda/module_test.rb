# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class ModuleTest < Test::Unit::TestCase
    def test_get_function_needs_a_loaded_module
      assert_raise(ArgumentError) { Module.new.get_function("foo") }
    end

    def test_unload_without_a_module_does_nothing
      assert_nothing_raised { Module.new.unload }
    end
  end
end
