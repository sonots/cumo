import hashlib
import math
import os
import re
import shutil
import sys
import tempfile

import six

from cupy.cuda import device
from cupy.cuda import function
from cupy.cuda import nvrtc

_nvrtc_version = nil

require 'tmpdir'
require 'tempfile'
require 'fileutils'
require 'nvrtc.so'

module Numo::NArray::CUDA
  class Compiler
    VALID_KERNEL_NAME_REGEXP = /^[a-zA-Z_][a-zA-Z_0-9]*$/
    DEFAULT_CACHE_DIR = File.expand_path('~/.cupy/kernel_cache')
    @@empty_file_preprocess_cache ||= {}
  
    def self.valid_kernel_name?(name)
      VALID_KERNEL_NAME_REGEXP.match?(name)
    end

    def get_cache_dir
      ENV.fetch('CUPY_CACHE_DIR', DEFAULT_CACHE_DIR)
    end

    def compile_using_nvrtc(source, options=[], arch=nil)
      arch ||= get_arch
      options += ["-arch=#{arch}"]
  
      Dir.mktmpdir do |root_dir|
        path = File.join(root_dir, 'kern')
        cu_path = "#{path}.cu"
  
        File.open(cu_path, 'w') do |cu_file|
          cu_file.write(source)
        end
  
        prog = NVRTCProgram.new(source, cu_path)
        begin
          ptx = prog.compile(options)
        rescue CompileException => e
          if get_bool_env_variable('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', false)
            e.dump($stderr)
          end
          raise e
        end
        return ptx
      end
    end
  
    def compile_with_cache(source, options=[], arch=nil, cache_dir=nil, extra_source=nil)
      # NVRTC does not use extra_source. extra_source is used for cache key.
      cache_dir ||= get_cache_dir()
      arch ||= get_arch()
  
      options += ['-ftz=true']
  
      env = [arch, options, get_nvrtc_version]
      base = @@empty_file_preprocess_cache[env]
      if base.nil?
        # This is checking of NVRTC compiler internal version
        base = preprocess('', options, arch)
        @@empty_file_preprocess_cache[env] = base
      end
      key_src = "#{env} #{base} #{source} #{extra_source}"
  
      key_src = key_src.encode('utf-8')
      # name = '%s_2.cubin' % hashlib.md5(key_src).hexdigest()
  
      unless Dir.exist?(cache_dir)
        FileUtils.mkdir_p(cache_dir)
      end
  
      # mod = function.Module()
      # To handle conflicts in concurrent situation, we adopt lock-free method
      # to avoid performance degradation.
      path = File.join(cache_dir, name)
      if File.exist?(path)
        File.open(path, 'rb') do |file|
          data = file.read
          if data.size >= 32
            hash = data[0...32]
            cubin = data[32..-1]
            # cubin_hash = six.b(hashlib.md5(cubin).hexdigest())
            if hash == cubin_hash
              # mod.load(cubin)
              return mod
            end
          end
        end
      end
  
      ptx = compile_using_nvrtc(source, options, arch)
      # ls = function.LinkState()
      # ls.add_ptr_data(ptx, six.u('cupy.ptx'))
      # cubin = ls.complete()
      # cubin_hash = six.b(hashlib.md5(cubin).hexdigest())
  
      tf = Tempfile.create
      tf.write(cubin_hash)
      tf.write(cubin)
      temp_path = tf.name
      File.mv(temp_path, path)
  
      # Save .cu source file along with .cubin
      if get_bool_env_variable('CUPY_CACHE_SAVE_CUDA_SOURCE', false)
        File.open("#{path}.cu", 'w') do |f|
          f.write(source)
        end
      end
  
      # mod.load(cubin)
      return mod
    end

    private

    def get_nvrtc_version
      $numo_narray_cuda_compiler_nvrtc_version ||= Nvrtc.version()
    end
  
    def get_arch
      # cc = device.Device().compute_capability
      # return 'compute_%s' % cc
      'compute_30'
    end
  
    def get_bool_env_variable(name, default)
      val = ENV[name]
      return default if val.nil? or val.size == 0
      Integer(val) == 1 rescue false
    end

    def _preprocess(source, options, arch):
      options += ["-arch=#{arch}"]
  
      prog = NVRTCProgram.new(source, '')
      begin
        result = prog.compile(options)
      rescue CompileException => e
        if get_bool_env_variable('CUPY_DUMP_CUDA_SOURCE_ON_ERROR', false)
          e.dump(sys.stderr)
        end
        raise e
      end
      # assert isinstance(result, six.text_type)
      result
    end

    class CompileException < StandardError
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

    # class NVRTCProgram
    #   def initialize(src, name="default_program", headers=[], include_names=[])
    #     @ptr = nil

    #     # if isinstance(src, six.binary_type):
    #     #     src = src.decode('UTF-8')
    #     # if isinstance(name, six.binary_type):
    #     #     name = name.decode('UTF-8')

    #     @src = src
    #     @name = name
    #     @ptr = NVRTC.createProgram(src, name, headers, include_names)
    #   end

    #   def __del__(self):
    #     if self.ptr:
    #       NVRTC.destroyProgram(self.ptr)

    #   def compile(self, options=()):
    #       try:
    #           NVRTC.compileProgram(self.ptr, options)
    #           return NVRTC.getPTX(self.ptr)
    #       except NVRTC.NVRTCError:
    #           log = NVRTC.getProgramLog(self.ptr)
    #           raise CompileException(log, self.src, self.name, options)
  end
end
