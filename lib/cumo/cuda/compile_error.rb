# frozen_string_literal: true

module Cumo::CUDA
  class CompileError < StandardError
    def initialize(msg, source, name, options)
      @msg = msg
      @source = source
      @name = name
      @options = options
    end

    def message
      @msg
    end

    def to_s
      @msg
    end

    def dump(io)
      lines = @source.split("\n")
      digits = Math.log10(lines.size).floor + 1
      linum_fmt = "%0#{digits}d "
      io.puts("NVRTC compilation error: #{@msg}")
      io.puts("-----")
      io.puts("Name: #{@name}")
      io.puts("Options: #{@options.join(' ')}")
      io.puts("CUDA source:")
      lines.each.with_index do |line, i|
        io.puts(linum_fmt.sprintf(i + 1) << line.rstrip)
      end
      io.puts("-----")
      io.flush
    end
  end
end
