# frozen_string_literal: true

require_relative "../test_helper"
require "cumo/linalg"
require "stringio"

module LinalgTestHelper
  COMPLEX_TYPES = [Cumo::SComplex, Cumo::DComplex].freeze
  private_constant :COMPLEX_TYPES

  def linalg_values(type, values)
    return values unless COMPLEX_TYPES.include?(type)

    map_nested(values) { |x| Complex(x, 2 - x) }
  end

  def linalg_array(type, values)
    type.cast(linalg_values(type, values))
  end

  def host_dot(a, b)
    rows = a.first.is_a?(Array) ? a : [a]
    cols = b.first.is_a?(Array) ? b.transpose : [b]
    c = rows.map { |row| cols.map { |col| row.zip(col).sum { |x, y| x * y } } }
    c = c.map(&:first) unless b.first.is_a?(Array)
    a.first.is_a?(Array) ? c : c.first
  end

  def assert_close(expected, actual, delta)
    expected = [expected].flatten
    actual = [actual.is_a?(Cumo::NArray) ? actual.to_a : actual].flatten
    assert_equal(expected.size, actual.size)
    expected.zip(actual) { |e, a| assert_operator((e - a).abs, :<=, delta) }
  end

  def capture_stderr
    orig = $stderr
    $stderr = StringIO.new
    yield
    $stderr.string
  ensure
    $stderr = orig
  end

  private

  def map_nested(values, &block)
    values.map { |x| x.is_a?(Array) ? map_nested(x, &block) : yield(x) }
  end
end
