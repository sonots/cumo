# frozen_string_literal: true

require "json"
require_relative "linalg_test_helper"

class LinalgRactorTest < Test::Unit::TestCase
  include LinalgTestHelper
  include CumoChildProcess

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
    omit("no Ractor") unless defined?(Ractor)
  end

  CONCURRENT_SCRIPT = <<~RUBY
    require "cumo"
    require "cumo/linalg"
    require "json"
    jobs = { 512 => :svdvals, 480 => :svdvals }
    jobs[128] = :eigvals if Cumo::CUDA::Cusolver.respond_to?(:geev, true)
    jobs = jobs.map { |n, f| [Cumo::DFloat.new(n, n).rand_norm.freeze, f] }
    expected = jobs.map { |m, f| Cumo::Linalg.__send__(f, m).to_a }
    ractors = jobs.map { |m, f| Ractor.new(m, f) { |x, g| 4.times.map { Cumo::Linalg.__send__(g, x).to_a } } }
    results = ractors.map { |r| r.respond_to?(:take) ? r.take : r.value }
    worst = results.zip(expected).map do |runs, e|
      scale = e.map(&:abs).max
      runs.map { |run| run.zip(e).map { |x, y| (x - y).abs }.max / scale }.max
    end
    print worst.to_json
  RUBY

  test "two Ractors calling cuSOLVER at once keep their own workspaces" do
    worst = JSON.parse(run_child(CONCURRENT_SCRIPT, timeout: 120).lines.last)
    assert_operator(worst.size, :>=, 2)
    worst.each { |w| assert_operator(w, :<=, 1e-10) }
  end
end
