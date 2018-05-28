require "open3"
require_relative "nvcc"

module MakeMakefileCuda
  class CLI
    attr_reader :argv

    def initialize(argv)
      @argv = argv.map{|e| e.dup }
    end

    def run
      if cu_file?
        puts "[given options]: #{argv.join(' ')}"
        run_command!(*nvcc_command)
      elsif c_file?
        run_command!(*c_command)
      elsif cxx_file?
        run_command!(*cxx_command)
      else
        raise 'something wrong'
      end
    end

    # private

    def run_command!(*args)
      puts colorize(:green, args.join(' '))
      exit system(*args)
    end

    # TODO(sonots): Make it possible to configure "nvcc" and additional arguments
    def nvcc_command
      s = MakeMakefileCuda::Nvcc.generate(argv)
      ["nvcc " << s << " -arch=sm_35"]
    end

    def c_command
      [RbConfig::CONFIG["CC"], *argv[1..-1]]
    end

    def cxx_command
      [RbConfig::CONFIG["CXX"], *argv[1..-1]]
    end

    def src_file
      argv.last # *.{c,cc,cpp,cu}
    end

    def mkmf_cu_ext
      argv.first # --mkmf-cu-ext={c|cxx}
    end

    def cu_file?
      src_file.end_with?('.cu')
    end

    def c_file?
      !cu_file? and mkmf_cu_ext.end_with?('=c')
    end

    def cxx_file?
      !cu_file? and mkmf_cu_ext.end_with?('=cxx')
    end

    COLOR_CODES = {
      red: 31,
      green: 32,
      yellow: 33,
      blue: 34,
      magenta: 35,
      cyan: 36
    }

    def colorize(code, str)
      raise "#{color_code} is not supported" unless COLOR_CODES[code]
      "\e[#{COLOR_CODES[code]}m#{str}\e[0m"
    end
  end
end
