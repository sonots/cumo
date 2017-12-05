require 'tmpdir'
require 'tempfile'
require 'fileutils'
require 'digest/md5'
require_relative '../cuda'
require_relative 'compile_error'
require_relative 'nvrtc_program'

module Numo::CUDA
  class Compiler
    VALID_KERNEL_NAME = /^[a-zA-Z_][a-zA-Z_0-9]*$/
    DEFAULT_CACHE_DIR = File.expand_path('~/.cumo/kernel_cache')
  
    @@empty_file_preprocess_cache ||= {}
    
    def self.valid_kernel_name?(name)
      VALID_KERNEL_NAME.match?(name)
    end
  
    def compile_using_nvrtc(source, options: [], arch: nil)
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
        rescue CompileError => e
          if get_bool_env_variable('CUMO_DUMP_CUDA_SOURCE_ON_ERROR', false)
            e.dump($stderr)
          end
          raise e
        ensure
          prog.destroy
        end
        return ptx
      end
    end
    
    def compile_with_cache(source, options: [], arch: nil, cache_dir: nil, extra_source: nil)
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
    
      key_src.encode!('utf-8')
      digest = Digest::MD5.hexdigest(key_src)
      name = "#{digest}_2.cubin"
    
      unless Dir.exist?(cache_dir)
        FileUtils.mkdir_p(cache_dir)
      end
   
      # @todo
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
            cubin_hash = Digest::MD5.hexdigest(cubin)
            if hash == cubin_hash
              # @todo
              # mod.load(cubin)
              # return mod
            end
          end
        end
      end
    
      ptx = compile_using_nvrtc(source, options, arch)
      # @todo
      # ls = function.LinkState()
      # ls.add_ptr_data(ptx, six.u('cupy.ptx'))
      # cubin = ls.complete()
      # cubin_hash = Digest::MD5.hexdigest(cubin)
    
      tf = Tempfile.create
      tf.write(cubin_hash)
      tf.write(cubin)
      temp_path = tf.name
      File.mv(temp_path, path)
    
      # Save .cu source file along with .cubin
      if get_bool_env_variable('CUMO_CACHE_SAVE_CUDA_SOURCE', false)
        File.open("#{path}.cu", 'w') do |f|
          f.write(source)
        end
      end
    
      # mod.load(cubin)
      return mod
    end
  
    private
  
    def get_cache_dir
      ENV.fetch('CUMO_CACHE_DIR', DEFAULT_CACHE_DIR)
    end
  
    def get_nvrtc_version
      @@nvrtc_version ||= NVRTC.version
    end
    
    def get_arch
      # @todo
      # cc = device.Device().compute_capability
      # return 'compute_%s' % cc
      'compute_30'
    end
    
    def get_bool_env_variable(name, default)
      val = ENV[name]
      return default if val.nil? or val.size == 0
      Integer(val) == 1 rescue false
    end
  
    def preprocess(source, options, arch):
      options += ["-arch=#{arch}"]
    
      prog = NVRTCProgram.new(source, '')
      begin
        result = prog.compile(options)
        return result
      rescue CompileError => e
        if get_bool_env_variable('CUMO_DUMP_CUDA_SOURCE_ON_ERROR', false)
          e.dump($stderr)
        end
        raise e
      ensure
        prog.destroy
      end
    end
  end
end
