# frozen_string_literal: true

require_relative "test_helper"

class CusolverTest < Test::Unit::TestCase
  def setup
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  test "version" do
    version = Cumo::CUDA::Cusolver.version
    assert_kind_of(Integer, version)
    assert_operator(version, :>=, 11_000)
  end

  test "CusolverError" do
    assert_operator(Cumo::CUDA::CusolverError, :<, StandardError)
  end
end
