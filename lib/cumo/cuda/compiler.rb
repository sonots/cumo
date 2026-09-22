# frozen_string_literal: true

require 'tmpdir'
require 'tempfile'
require 'fileutils'
require 'digest/md5'
require 'json'

module Cumo::CUDA
  class Compiler
    VALID_KERNEL_NAME = /\A[a-zA-Z_][a-zA-Z_0-9]*\z/
    DEFAULT_CACHE_DIR = File.expand_path('~/.cumo/kernel_cache')

    @@empty_file_preprocess_cache ||= {}

    def self.valid_kernel_name?(name)
      VALID_KERNEL_NAME.match?(name)
    end

    # With name_expressions: answers the PTX and a Hash of each expression
    # to its mangled name; without, the PTX.
    def compile_using_nvrtc(source, options: [], arch: nil, name_expressions: nil)
      arch ||= get_arch
      options += ["-arch=#{arch}"]
      name_expressions = nil if name_expressions && name_expressions.empty?

      Dir.mktmpdir do |root_dir|
        path = File.join(root_dir, 'kern')
        cu_path = "#{path}.cu"

        File.open(cu_path, 'w') do |cu_file|
          cu_file.write(source)
        end

        prog = NVRTCProgram.new(source, name: cu_path, name_expressions: name_expressions || [])
        begin
          ptx = prog.compile(options: options)
          return ptx if name_expressions.nil?
          return [ptx, name_expressions.to_h { |expr| [expr, prog.lowered_name(expr)] }]
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

    # name_expressions: names such as "kernel<float>" that the Module then
    # answers get_function for. Their mangled names are cached with the cubin.
    def compile_with_cache(source, options: [], arch: nil, cache_dir: nil, extra_source: nil, name_expressions: [])
      # NVRTC does not use extra_source. extra_source is used for cache key.
      cache_dir ||= get_cache_dir
      arch ||= get_arch
      name_expressions = name_expressions.uniq

      options += ['-ftz=true']

      env = [arch, options, get_nvrtc_version]
      base = @@empty_file_preprocess_cache[env]
      if base.nil?
        # This is checking of NVRTC compiler internal version
        base = preprocess('', options, arch)
        @@empty_file_preprocess_cache[env] = base
      end
      key_src = "#{env} #{base} #{source} #{extra_source}"
      key_src += " #{name_expressions}" unless name_expressions.empty?

      key_src.encode!('utf-8')
      digest = Digest::MD5.hexdigest(key_src)
      name = "#{digest}_#{name_expressions.empty? ? 2 : 3}.cubin"

      unless Dir.exist?(cache_dir)
        FileUtils.mkdir_p(cache_dir)
      end

      # TODO(sonots): thread-safe?
      path = File.join(cache_dir, name)
      cubin, lowered = load_cache(path, name_expressions)
      if cubin
        mod = Module.new
        mod.load(cubin)
        mod.lowered_names = lowered
        return mod
      end

      ptx = compile_using_nvrtc(source, options: options, arch: arch, name_expressions: name_expressions)
      ptx, lowered = name_expressions.empty? ? [ptx, {}] : ptx
      cubin = nil
      LinkState.new do |ls|
        ls.add_ptr_data(ptx, 'cumo.ptx')
        cubin = ls.complete()
      end

      save_cache(path, cubin, lowered)

      # Save .cu source file along with .cubin
      if get_bool_env_variable('CUMO_CACHE_SAVE_CUDA_SOURCE', false)
        File.open("#{path}.cu", 'w') do |f|
          f.write(source)
        end
      end

      mod = Module.new
      mod.load(cubin)
      mod.lowered_names = lowered
      return mod
    end

    private

    def save_cache(path, cubin_hash, cubin)
      tf = Tempfile.create
      tf.write(cubin_hash)
      tf.write(cubin)
      temp_path = tf.path
      FileUtils.mv(temp_path, path)
    end

    # The file is the MD5 of what follows, then the payload: the cubin, or
    # with name expressions the JSON of their mangled names, its length
    # first as eight hex digits, then the cubin. It is written next to its
    # final place and renamed into it, so a reader sees all of it or none.
    def save_cache(path, cubin, lowered)
      payload = cubin
      unless lowered.empty?
        names = JSON.generate(lowered)
        payload = format('%08x', names.bytesize) + names + cubin
      end
      tf = Tempfile.create('cubin', File.dirname(path), binmode: true)
      tf.write(Digest::MD5.hexdigest(payload))
      tf.write(payload)
      tf.close
      FileUtils.mv(tf.path, path)
    end

    def load_cache(path, name_expressions = [])
      return nil unless File.exist?(path)
      data = File.binread(path)
      return nil unless data.size >= 32
      payload = data[32..-1]
      return nil unless data[0...32] == Digest::MD5.hexdigest(payload)
      return [payload, {}] if name_expressions.empty?
      return nil unless payload.size >= 8 && payload[0, 8].match?(/\A\h{8}\z/)
      length = payload[0, 8].to_i(16)
      return nil unless payload.size >= 8 + length
      lowered = JSON.parse(payload[8, length])
      return nil unless lowered.is_a?(Hash) && lowered.keys == name_expressions && lowered.values.all?(String)
      [payload[(8 + length)..-1], lowered]
    rescue JSON::ParserError
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
