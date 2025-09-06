require 'tmpdir'
require 'tempfile'
require 'fileutils'
require 'digest/md5'

module Cumo::CUDA
  class Compiler
    VALID_KERNEL_NAME = /\A[a-zA-Z_][a-zA-Z_0-9]*\z/
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

        prog = NVRTCProgram.new(source, name: cu_path)
        begin
          ptx = prog.compile(options: options)
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
      cache_dir ||= get_cache_dir
      arch ||= get_arch

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

      # TODO(sonots): thread-safe?
      path = File.join(cache_dir, name)
      cubin = load_cache(path)
      if cubin
        mod = Module.new
        mod.load(cubin)
        return mod
      end

      ptx = compile_using_nvrtc(source, options: options, arch: arch)
      cubin = nil
      cubin_hash = nil
      LinkState.new do |ls|
        ls.add_ptr_data(ptx, 'cumo.ptx')
        cubin = ls.complete()
        cubin_hash = Digest::MD5.hexdigest(cubin)
      end

      save_cache(path, cubin_hash, cubin)

      # Save .cu source file along with .cubin
      if get_bool_env_variable('CUMO_CACHE_SAVE_CUDA_SOURCE', false)
        File.open("#{path}.cu", 'w') do |f|
          f.write(source)
        end
      end

      mod = Module.new
      mod.load(cubin)
      return mod
    end

    private

    def save_cache(path, cubin_hash, cubin)
      tf = Tempfile.create
      tf.write(cubin_hash)
      tf.write(cubin)
      temp_path = tf.path
      File.rename(temp_path, path)
    end

    def load_cache(path)
      return nil unless File.exist?(path)
      File.open(path, 'rb') do |file|
        data = file.read
        return nil unless data.size >= 32
        hash = data[0...32]
        cubin = data[32..-1]
        cubin_hash = Digest::MD5.hexdigest(cubin)
        return nil unless hash == cubin_hash
        return cubin
      end
      nil
    end

    def get_cache_dir
      ENV.fetch('CUMO_CACHE_DIR', DEFAULT_CACHE_DIR)
    end

    def get_nvrtc_version
      @@nvrtc_version ||= NVRTC.nvrtcVersion
    end

    def get_arch
      cc = Device.new.compute_capability
      "compute_#{cc}"
    end

    def get_bool_env_variable(name, default)
      val = ENV[name]
      return default if val.nil? or val.size == 0
      Integer(val) == 1 rescue false
    end

    def preprocess(source, options, arch)
      options += ["-arch=#{arch}"]

      prog = NVRTCProgram.new(source, name: '')
      begin
        result = prog.compile(options: options)
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
