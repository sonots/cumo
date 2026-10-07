# frozen_string_literal: true

require "timeout"
require_relative "linalg_test_helper"

class LinalgRactorTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
    omit("no Ractor") unless defined?(Ractor)
  end

  # cuSOLVER runs each call as a sequence of kernels driven from the host, so
  # two calls in flight at once on different threads interleave on the device.
  # Each thread keeps its own workspaces; sharing them gave wrong values, an
  # illegal address or a hang.
  test "two Ractors calling cuSOLVER at once keep their own workspaces" do
    a = Cumo::DFloat.new(512, 512).rand_norm.freeze
    b = Cumo::DFloat.new(480, 480).rand_norm.freeze
    c = Cumo::DFloat.new(128, 128).rand_norm.freeze
    singular = [a, b].map { |m| Cumo::Linalg.svdvals(m).to_a }
    eigen = Cumo::Linalg.eigvals(c).to_a
    ractors = [a, b].map { |m| Ractor.new(m) { |x| 4.times.map { Cumo::Linalg.svdvals(x).to_a } } }
    ractors << Ractor.new(c) { |x| 3.times.map { Cumo::Linalg.eigvals(x).to_a } }
    results = Timeout.timeout(120) { ractors.map(&:value) }
    assert_equal([singular[0]] * 4, results[0])
    assert_equal([singular[1]] * 4, results[1])
    assert_equal([eigen] * 3, results[2])
  end
end
