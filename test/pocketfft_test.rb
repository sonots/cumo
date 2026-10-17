# frozen_string_literal: true

require "numo/pocketfft"
require_relative "test_helper"
require "cumo/pocketfft"

class PocketfftTest < CumoTestBase
  TYPES = [Numo::DFloat, Numo::DComplex, Numo::Int32, Numo::SFloat].freeze

  def setup
    omit("Cumo is built without cuFFT") unless Cumo::CUDA::Cufft.available?
  end

  def numo_input(type, *shape)
    re = Numo::DFloat.new(*shape).rand(-1, 1)
    case type.name
    when "Numo::DComplex" then re + (Complex::I * Numo::DFloat.new(*shape).rand(-1, 1))
    when "Numo::Int32" then Numo::Int32.cast((re * 100).round)
    else type.cast(re)
    end
  end

  def outcome
    r = yield
    r = Cumo::NArray.to_numo(r) if r.is_a?(Cumo::NArray)
    [r.class.name.delete_prefix("Numo::"), r]
  rescue StandardError => e
    [e.class.name.sub(/\A(Numo|Cumo)::/, ""), e.message.gsub("Numo::", "Cumo::")]
  end

  def assert_same_outcome(message, numo_input, cumo_input = nil, &call)
    cumo_input ||= numo_input.map { |x| x.is_a?(Numo::NArray) ? Cumo::NArray.from_numo(x) : x }
    expected = outcome { call.call(Numo::Pocketfft, *numo_input) }
    actual = outcome { call.call(Cumo::Pocketfft, *cumo_input) }
    assert_equal(expected[0], actual[0], message)
    return assert_equal(expected[1], actual[1], message) unless expected[1].is_a?(Numo::NArray)

    assert_equal(expected[1].shape, actual[1].shape, message)
    return if expected[1].empty?

    scale = [Numo::DComplex.cast(expected[1]).abs.max, 1.0].max
    err = (Numo::DComplex.cast(actual[1]) - expected[1]).abs.max / scale
    assert_operator(err, :<=, 1e-12, message)
  end

  test "the 1-dimensional functions answer what numo-pocketfft answers" do
    %i[fft ifft rfft irfft].each do |name|
      TYPES.each do |type|
        [[1], [2], [7], [8], [30], [33], [3, 4]].each do |shape|
          assert_same_outcome("#{name} #{type} #{shape}", [numo_input(type, *shape)]) { |m, a| m.public_send(name, a) }
        end
      end
    end
  end

  test "the 2-dimensional functions answer what numo-pocketfft answers" do
    %i[fft2 ifft2 rfft2 irfft2].each do |name|
      TYPES.each do |type|
        [[3, 5], [4, 6], [1, 4], [5, 1], [6], [2, 3, 4]].each do |shape|
          assert_same_outcome("#{name} #{type} #{shape}", [numo_input(type, *shape)]) { |m, a| m.public_send(name, a) }
        end
      end
    end
  end

  test "the N-dimensional functions answer what numo-pocketfft answers" do
    %i[fftn ifftn rfftn irfftn].each do |name|
      TYPES.each do |type|
        [[7], [4, 6], [3, 4, 5], [2, 3, 4, 5], [4, 1]].each do |shape|
          assert_same_outcome("#{name} #{type} #{shape}", [numo_input(type, *shape)]) { |m, a| m.public_send(name, a) }
        end
      end
    end
  end

  test "what numo-pocketfft refuses raises the same error" do
    %i[fft ifft rfft irfft fft2 ifft2 rfft2 irfft2 fftn ifftn rfftn irfftn].each do |name|
      [nil, 1, [1, 2, 3]].each do |x|
        assert_same_outcome("#{name} #{x.inspect}", [x]) { |m, a| m.public_send(name, a) }
      end
      empty = Numo::DFloat.new(*(name.to_s.include?("2") ? [0, 3] : [0]))
      assert_same_outcome("#{name} empty", [empty]) { |m, a| m.public_send(name, a) }
    end
  end

  test "a 0-dimensional array raises DimensionError, where numo-pocketfft fails in a different way each time" do
    %i[fftn ifftn rfftn irfftn].each do |name|
      [Cumo::DFloat.cast(1.5), Cumo::DComplex.cast(Complex(1, 2))].each do |x|
        assert_raise(Cumo::NArray::DimensionError, "#{name} #{x.class}") { Cumo::Pocketfft.public_send(name, x) }
      end
    end
  end

  test "fftconvolve answers what numo-pocketfft answers, edges included" do
    [
      [[Numo::DFloat, [3]], [Numo::DFloat, [2]]],
      [[Numo::DFloat, [5]], [Numo::DFloat, [4]]],
      [[Numo::Int32, [7]], [Numo::SFloat, [3]]],
      [[Numo::DComplex, [6]], [Numo::DFloat, [5]]],
      [[Numo::DFloat, [3, 4]], [Numo::DFloat, [2, 5]]],
      [[Numo::DComplex, [3, 3, 2]], [Numo::DComplex, [2, 1, 2]]],
      [[Numo::Int32, [1, 1]], [Numo::Int32, [1, 1]]],
      [[Numo::DFloat, [1]], [Numo::DComplex, [1]]]
    ].each do |(ta, sa), (tb, sb)|
      inputs = [numo_input(ta, *sa), numo_input(tb, *sb)]
      assert_same_outcome("fftconvolve #{ta} #{sa} #{tb} #{sb}", inputs) { |m, a, b| m.fftconvolve(a, b) }
    end
    [[Numo::DFloat[1, 2], Numo::DFloat[[1]]], [Numo::DFloat[], Numo::DFloat[1]], [[1, 2], Numo::DFloat[1]]].each do |inputs|
      assert_same_outcome("fftconvolve #{inputs.inspect}", inputs) { |m, a, b| m.fftconvolve(a, b) }
    end
  end
end
