# frozen_string_literal: true

require_relative "test_helper"
require "tmpdir"
require "yard"

# A constant defined in C carries its value in a "value: description" comment,
# which is the only way a doc tool can show it: the C expression it is defined
# with means nothing to a reader. This compares every such value with the value
# the constant really holds, so a comment that no longer reaches its definition
# or a limit that a header has moved fails here and not in the published docs.
class DocumentedConstantTest < CumoTestBase
  GEN_DIR = File.expand_path("../ext/cumo/narray/gen", __dir__)
  EXT_DIR = File.expand_path("../ext/cumo", __dir__)
  # @private keeps this one out of the docs, so it documents no value.
  WITHOUT_VALUE = ["Cumo::CUDA::Cusolver::BATCHED_SVD_MAX"].freeze

  class << self
    def documented_values
      @documented_values ||= Dir.mktmpdir("cumo-doc") do |dir|
        parse(generate(dir) + handwritten)
      end
    end

    private

    def generate(dir)
      Dir[File.join(GEN_DIR, "def", "*.rb")].map do |def_rb|
        out = File.join(dir, "#{File.basename(def_rb, '.rb')}.c")
        system(RbConfig.ruby, File.join(GEN_DIR, "cogen.rb"), "-l", "-o", out, def_rb, exception: true)
        out
      end
    end

    def handwritten
      (Dir[File.join(EXT_DIR, "**", "*.c")] + Dir[File.join(EXT_DIR, "**", "*.cpp")])
        .reject { |path| path.start_with?("#{GEN_DIR}/") }
    end

    def parse(files)
      YARD::Registry.clear
      YARD::Logger.instance.level = YARD::Logger::ERROR
      YARD.parse(files)
      YARD::Registry.all(:constant).to_h { |obj| [obj.path, obj.value] }
    end
  end

  test "a constant defined in C documents the value it holds" do
    checked = 0
    self.class.documented_values.each do |path, value|
      next if WITHOUT_VALUE.include?(path)

      names = path.split("::")
      owner = names[0..-2].reduce(Object) { |acc, elem| acc.const_get(elem, false) }
      # The cuDNN constants are defined only in a build with cuDNN.
      next unless owner.const_defined?(names.last, false)

      assert_documented_value(owner.const_get(names.last, false), value, path)
      checked += 1
    end
    assert_operator checked, :>, 100, "the comments no longer reach the definitions"
  end

  test "a constant documents its value on the first line, which is as far as rdoc looks" do
    self.class.documented_values.each do |path, value|
      assert(!value.include?("\n"), "#{path} documents a value that runs past the first line")
    end
  end

  private

  def assert_documented_value(real, value, path)
    case real
    when Hash then assert_equal("{#{real.map { |k, v| "#{k}=>#{v}" }.join(', ')}}", value, path)
    when Module then assert_equal(real.name, value, path)
    when Float then assert_equal(real, Float(value), path)
    when Integer then assert_equal(real, Integer(value), path)
    else flunk("#{path} holds a #{real.class}, which has no documented form")
    end
  end
end
