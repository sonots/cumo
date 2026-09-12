# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../../lib", __FILE__)
require "cumo"

require "pry"
require "test/unit"
require "timeout"

module CumoChildProcess
  RESHAPING_ALLOCATE = <<~RUBY
    module Reshaping
      def allocate
        unless @done
          @done = true
          send(:initialize, 1)
        end
        super
      end
    end
  RUBY

  MUTATING_ALLOCATE = <<~RUBY
    module Mutating
      def allocate
        unless $done
          $done = true
          $fired = true
          $mutate.call(self)
        end
        super
      end
    end
  RUBY

  def run_child(script, timeout: 120)
    lib = File.expand_path("../lib", __dir__)
    out = nil
    IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], err: [:child, :out]) do |io|
      begin
        Timeout.timeout(timeout) { out = io.read }
      rescue Timeout::Error
        Process.kill(:KILL, io.pid)
        flunk("child did not finish in #{timeout}s")
      end
    end
    assert(Process.last_status.success?, "child failed: #{out}")
    out
  end

  def assert_child_raises(expected, body, prelude: RESHAPING_ALLOCATE)
    script = <<~RUBY
      require "cumo/narray"
      #{prelude}
      begin
        #{body}
        print "no error"
      rescue => e
        print e.message
      end
    RUBY
    assert_equal expected, run_child(script)
  end
end

class CumoTestBase < Test::Unit::TestCase
  include CumoChildProcess

  FLOAT_TYPES = [
    Cumo::DFloat,
    Cumo::DComplex,
  ]

  TYPES = [
    *FLOAT_TYPES,
    Cumo::SFloat,
    Cumo::SComplex,
    Cumo::Int64,
    Cumo::Int32,
    Cumo::Int16,
    Cumo::Int8,
    Cumo::UInt64,
    Cumo::UInt32,
    Cumo::UInt16,
    Cumo::UInt8,

    ## TODO: RObject causes failures on assertion
    # Cumo::RObject,
  ]
end
