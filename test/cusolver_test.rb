# frozen_string_literal: true

require_relative "test_helper"

class CusolverTest < Test::Unit::TestCase
  test "version" do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?

    version = Cumo::CUDA::Cusolver.version
    assert_kind_of(Integer, version)
    assert_operator(version, :>=, 11_000)
  end

  test "version is defined only with cuSOLVER" do
    assert_equal(Cumo::CUDA::Cusolver.available?, Cumo::CUDA::Cusolver.respond_to?(:version))
  end

  test "CusolverError" do
    assert_operator(Cumo::CUDA::CusolverError, :<, StandardError)
  end
end
