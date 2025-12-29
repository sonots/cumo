# frozen_string_literal: true

require "mkmf"
require "open3"
require_relative "nvcc"

module MakeMakefileCuda
  class CLI
    attr_reader :argv

    def initialize(argv)
      @argv = argv.map { |e| e.dup }
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
      cmd = "nvcc #{s}"
      if ENV['CUMO_NVCC_GENERATE_CODE']
        cmd << " --generate-code=#{ENV['CUMO_NVCC_GENERATE_CODE']}"
      elsif ENV['DEBUG']
        cmd << " -arch=sm_35"
      else
        # Ref. https://en.wikipedia.org/wiki/CUDA
        if cuda_version >= Gem::Version.new("13.0")
          # CUDA 13.0
          capability = [75, 87, 89, 90, 121]
        elsif cuda_version >= Gem::Version.new("12.9")
          # CUDA 12.9
          capability = [50, 60, 70, 75, 87, 89, 90, 121]
        elsif cuda_version >= Gem::Version.new("12.8")
          # CUDA 12.8
          capability = [50, 60, 70, 75, 87, 89, 90, 120]
        elsif cuda_version >= Gem::Version.new("12.0")
          # CUDA 12.0 – 12.6
          capability = [50, 60, 70, 75, 87, 89, 90]
        elsif cuda_version >= Gem::Version.new("11.8")
          # CUDA 11.8
          capability = [35, 50, 60, 70, 75, 87, 89, 90]
        else
          # CUDA 11.0
          capability = [35, 50, 60, 70, 75, 80]
        end

        if find_executable('nvidia-smi')
          arch_version = `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.strip
          capability << (arch_version.to_f * 10).to_i unless arch_version.empty?
        end

        capability.each do |arch|
          cmd << " --generate-code=arch=compute_#{arch},code=sm_#{arch}"
        end
      end
      cmd
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

    def cuda_version
      @cuda_version ||= begin
        output = `nvcc --version`
        if output =~ /Cuda compilation tools, release ([^,]*),/
          Gem::Version.new($1)
        end
      end
    end
  end
end
