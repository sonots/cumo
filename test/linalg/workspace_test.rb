# frozen_string_literal: true

require_relative "linalg_test_helper"

class LinalgWorkspaceTest < Test::Unit::TestCase
  include LinalgTestHelper

  setup do
    omit("built without cuSOLVER") unless Cumo::CUDA::Cusolver.available?
  end

  teardown do
    Cumo::CUDA::Stream.null.use
  end

  test "a factorization still running on one stream keeps its workspace from the next stream" do
    streams = Array.new(2) { Cumo::CUDA::Stream.new(non_blocking: true) }
    big = Cumo::DFloat.new(2048, 256).rand(-1, 1)
    small = Cumo::DFloat.new(64, 64).rand(-1, 1)
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    5.times do
      streams[0].use
      q, r = Cumo::Linalg.qr(big, mode: "economic")
      streams[1].use
      3.times { Cumo::Linalg.qr(small, mode: "economic") }
      streams.each(&:synchronize)
      Cumo::CUDA::Stream.null.use
      assert_operator((q.dot(r) - big).abs.max.to_f, :<, 1e-10)
    end
  end

  test "factorizations that grow and shrink on alternating streams stay correct" do
    streams = Array.new(2) { Cumo::CUDA::Stream.new(non_blocking: true) }
    results = [[2048, 256], [16, 4], [4096, 512], [8, 8], [1024, 128]].each_with_index.map do |(m, n), i|
      streams[i % 2].use
      a = Cumo::DFloat.new(m, n).rand(-1, 1)
      [a, *Cumo::Linalg.qr(a, mode: "economic")]
    end
    streams.each(&:synchronize)
    Cumo::CUDA::Stream.null.use
    results.each do |a, q, r|
      assert_operator((q.dot(r) - a).abs.max.to_f, :<, 1e-10, a.shape.inspect)
    end
  end

  test "solves on alternating streams stay correct" do
    streams = Array.new(2) { Cumo::CUDA::Stream.new(non_blocking: true) }
    results = [512, 8, 1024, 16].each_with_index.map do |n, i|
      streams[i % 2].use
      a = Cumo::DFloat.new(n, n).rand(-1, 1) + (Cumo::DFloat.eye(n) * n)
      b = Cumo::DFloat.new(n).rand(-1, 1)
      [a, b, Cumo::Linalg.solve(a, b)]
    end
    streams.each(&:synchronize)
    Cumo::CUDA::Stream.null.use
    results.each do |a, b, x|
      assert_operator((a.dot(x) - b).abs.max.to_f, :<, 1e-10, a.shape.inspect)
    end
  end

  test "a workspace too large to keep is taken for the one call and the next call still works" do
    a = Cumo::DFloat.new(2048, 2048).rand(-1, 1)
    a = a + a.transpose
    w = Cumo::Linalg.eigvalsh(a)
    assert_in_delta(a.trace.to_f, w.sum.to_f, 1e-8 * a.abs.sum.to_f)
    q, r = Cumo::Linalg.qr(a[0...64, 0...8], mode: "economic")
    assert_operator((q.dot(r) - a[0...64, 0...8]).abs.max.to_f, :<, 1e-10)
  end
end
