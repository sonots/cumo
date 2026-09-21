# frozen_string_literal: true

require_relative "user_kernel"

module Cumo::CUDA
  # A kernel that maps every element, reduces the mapped values along the
  # given axes and maps the result, the way CuPy's ReductionKernel does.
  #
  #   l2norm = Cumo::CUDA::ReductionKernel.new(
  #     "T x", "T y", "x * x", "a + b", "y = sqrt(a)", "0", "l2norm")
  #   l2norm.call(x, axis: 1)
  #
  # The map expression sees the inputs by name, the reduce expression sees
  # the two values a and b, and the post expression sees the reduced value a
  # and writes the output. The identity starts every reduction.
  class ReductionKernel
    include UserKernel

    BLOCK = 512
    MAX_GRID = 65_535
    # A second pass is worth it once this many blocks would otherwise be
    # idle, and each chunk keeps this many elements per thread.
    WANT_BLOCKS = 256
    MIN_PER_THREAD = 16
    DTYPE_OF_CTYPE = CTYPE.invert.freeze

    attr_reader :name

    def initialize(in_params, out_params, map_expr, reduce_expr, post_map_expr, identity, name,
                   preamble: "", reduce_type: nil)
      check_name(name)
      @in_params = parse_params(in_params)
      @out_params = parse_params(out_params)
      if (raw = (@in_params + @out_params).find(&:raw))
        raise ArgumentError, "#{raw.name} is raw, and a reduction indexes every argument itself"
      end
      raise ArgumentError, "a reduction needs an output" if @out_params.empty?
      if (taken = (@in_params + @out_params).find { |p| %w[a b].include?(p.name) })
        raise ArgumentError, "#{taken.name} is what the reduce expression calls its operands"
      end
      @map_expr = map_expr
      @reduce_expr = reduce_expr
      @post_map_expr = post_map_expr
      @identity = identity_source(identity)
      @name = name
      @preamble = preamble
      @reduce_type = check_reduce_type(reduce_type)
      @functions = {}
      @finals = {}
    end

    # Reduces along axis:, every axis when it is nil, and answers the output
    # or the outputs as an Array. keepdims: keeps the reduced axes as length
    # one. Outputs may be given after the inputs.
    def call(*args, axis: nil, keepdims: false)
      ins, outs = split_args(args, @in_params, @out_params)
      types = resolve_types(@in_params, ins, @out_params, outs)
      arrays = ins.select { |a| a.is_a?(Cumo::NArray) }
      raise ArgumentError, "a reduction needs an NArray to reduce" if arrays.empty?
      in_shape = broadcast_shapes(arrays.map(&:shape))
      axes = normalize_axes(axis, in_shape.size)
      kept = (0...in_shape.size).to_a - axes
      out_shape = kept.map { |d| in_shape[d] }
      full_shape = in_shape.each_with_index.map { |s, d| axes.include?(d) ? 1 : s }
      want = keepdims ? full_shape : out_shape

      outs = @out_params.each_with_index.map { |p, k| outs[k] || types[p.type].new(*want) }
      check_outputs(outs, @out_params, want)

      out_size = out_shape.inject(1, :*)
      red_size = axes.map { |d| in_shape[d] }.inject(1, :*)
      launch(ins, outs, types, in_shape, axes + kept, out_size, red_size) if out_size > 0
      @out_params.size == 1 ? outs[0] : outs
    end

    private

    # NVRTC has no INFINITY or NAN without a header, so they go as bits.
    def identity_source(identity)
      return "__longlong_as_double(0x7ff0000000000000LL)" if identity == Float::INFINITY
      return "-__longlong_as_double(0x7ff0000000000000LL)" if identity == -Float::INFINITY
      return "__longlong_as_double(0x7ff8000000000000LL)" if identity.is_a?(Float) && identity.nan?
      identity.to_s
    end

    # A letter of the parameters, a C type of the table or a Cumo type name.
    def check_reduce_type(t)
      return nil if t.nil?
      return t if t.size == 1 && (@in_params + @out_params).any? { |p| p.type == t }
      return t if CTYPE.value?(t)
      return TYPES[t][1] if TYPES.key?(t)
      raise ArgumentError, "reduce_type #{t.inspect} is not a type of the parameters, a C type or one of #{TYPES.keys.join(', ')}"
    end

    def normalize_axes(axis, ndim)
      return (0...ndim).to_a if axis.nil?
      list = axis.is_a?(Array) ? axis : [axis]
      axes = list.map do |a|
        raise TypeError, "axis: takes Integers, got #{a.inspect}" unless a.is_a?(Integer)
        d = a < 0 ? a + ndim : a
        raise ArgumentError, "axis #{a} is out of range for #{ndim} dimensions" unless (0...ndim).cover?(d)
        d
      end
      raise ArgumentError, "axis: names #{axis.inspect}, and an axis can be reduced once" if axes.uniq.size != axes.size
      axes.sort
    end

    def launch(ins, outs, types, in_shape, order, out_size, red_size)
      params = @in_params + @out_params
      arrays = ins.map { |a| readable(a) } + outs
      kinds = params.zip(arrays).map { |p, a| a.is_a?(Cumo::NArray) ? :array : :scalar }
      nd = in_shape.size
      key = [params.map { |p| CTYPE[types[p.type]] }, kinds, nd]
      fn = (@functions[key] ||= compile(source(types, kinds, nd)))

      per_output = [next_pow2(red_size), BLOCK].min
      block_stride = BLOCK / per_output
      blocks = (out_size + block_stride - 1) / block_stride
      chunks = 1
      partial_dtype = reduce_dtype(types)
      if blocks < WANT_BLOCKS && partial_dtype
        chunks = [(WANT_BLOCKS + blocks - 1) / blocks, red_size / (per_output * MIN_PER_THREAD), MAX_GRID].min
        chunks = 1 if chunks < 2
      end
      partials = chunks > 1 ? partial_dtype.new(out_size * chunks) : nil
      args = []
      @in_params.zip(arrays).each do |p, a|
        if a.is_a?(Cumo::NArray)
          args << a
          st = strides(a, in_shape)
          args << order.map { |d| st[d] }.pack("q*") if nd > 0
        else
          args << pack_scalar(a, types[p.type], p.name)
        end
      end
      arrays[@in_params.size..].each { |a| args << a }
      args << (partials || [0].pack("q"))
      args << order.map { |d| in_shape[d] }.pack("q*") if nd > 0
      args << [out_size * red_size].pack("q")
      args << [out_size].pack("q")
      args << [block_stride].pack("l")
      fn.launch(args, grid: [[blocks, MAX_GRID].min, chunks], block: BLOCK)
      return if partials.nil?

      final = (@finals[key[0, 2]] ||= compile_final(types))
      final.launch([partials] + arrays[@in_params.size..] + [[out_size].pack("q"), [chunks].pack("l")],
                   grid: [(out_size + BLOCK - 1) / BLOCK, MAX_GRID].min, block: BLOCK)
    end

    # The dtype the partial sums of a second pass are kept in, or nil when
    # reduce_type names something no NArray holds.
    def reduce_dtype(types)
      return types[@out_params[0].type] if @reduce_type.nil?
      return types[@reduce_type] if @reduce_type.size == 1 && types.key?(@reduce_type)
      DTYPE_OF_CTYPE[@reduce_type]
    end

    def next_pow2(n)
      p = 1
      p *= 2 while p < n
      p
    end

    def source(types, kinds, nd)
      params = @in_params + @out_params
      placeholders = params.map(&:type).uniq.reject { |t| TYPES.key?(t) }
      reduce_type = @reduce_type || CTYPE[types[@out_params[0].type]]
      decl = []
      read = []
      @in_params.zip(kinds).each do |p, kind|
        ctype = CTYPE[types[p.type]]
        if kind == :scalar
          decl << "const #{ctype} #{p.name}"
        else
          decl << "const #{ctype}* _raw_#{p.name}"
          decl << "_Dims _st_#{p.name}" if nd > 0
          offset = nd > 0 ? (0...nd).map { |d| "_idx[#{d}] * _st_#{p.name}.v[#{d}]" }.join(" + ") : "0"
          read << "const #{ctype} #{p.name} = _raw_#{p.name}[#{offset}];"
        end
      end
      write = @out_params.map do |p|
        ctype = CTYPE[types[p.type]]
        decl << "#{ctype}* _raw_#{p.name}"
        "#{ctype}& #{p.name} = _raw_#{p.name}[_i];"
      end
      decl << "_type_reduce* _partials"
      decl << "_Dims _shape" if nd > 0
      decl << "long long _in_size"
      decl << "long long _out_size"
      decl << "int _block_stride"
      index = nd > 0 ? <<~CUDA : ""
        long long _idx[#{nd}];
        { long long _r = _j; for (int _d = #{nd} - 1; _d >= 0; _d--) { _idx[_d] = _r % _shape.v[_d]; _r /= _shape.v[_d]; } }
      CUDA
      <<~CUDA
        #{placeholders.map { |t| "typedef #{CTYPE[types[t]]} #{t};" }.join("\n")}
        #{nd > 0 ? "struct _Dims { long long v[#{nd}]; };" : ""}
        #{@preamble}
        typedef #{reduce_type} _type_reduce;
        #define REDUCE(a, b) (#{@reduce_expr})
        extern "C" __global__ void #{@name}(#{decl.join(', ')}) {
          __shared__ _type_reduce _sdata[#{BLOCK}];
          const unsigned int _tid = threadIdx.x;
          const long long _j_offset = (long long)(_tid / _block_stride + blockIdx.y * (#{BLOCK} / _block_stride)) * _out_size;
          const long long _j_stride = (long long)(#{BLOCK} / _block_stride) * gridDim.y * _out_size;
          for (long long _i_base = (long long)blockIdx.x * _block_stride; _i_base < _out_size; _i_base += (long long)gridDim.x * _block_stride) {
            _type_reduce _s = _type_reduce(#{@identity});
            const long long _i = _i_base + (_tid % _block_stride);
            #pragma unroll 1
            for (long long _j = _i + _j_offset; _j < _in_size; _j += _j_stride) {
              #{index}
              #{read.join("\n      ")}
              _type_reduce _a = static_cast<_type_reduce>(#{@map_expr});
              _s = REDUCE(_s, _a);
            }
            _sdata[_tid] = _s;
            __syncthreads();
            for (unsigned int _block = #{BLOCK} / 2; _block >= (unsigned int)_block_stride; _block >>= 1) {
              if (_tid < _block) {
                _type_reduce _a = _sdata[_tid], _b = _sdata[_tid + _block];
                _sdata[_tid] = REDUCE(_a, _b);
              }
              __syncthreads();
            }
            if (_tid < (unsigned int)_block_stride && _i < _out_size) {
              if (_partials) {
                _partials[_i * gridDim.y + blockIdx.y] = _sdata[_tid];
              } else {
                const _type_reduce a = _sdata[_tid];
                #{write.join("\n      ")}
                #{@post_map_expr};
              }
            }
            __syncthreads();
          }
        }
      CUDA
    end

    def compile_final(types)
      mod = Compiler.new.compile_with_cache(final_source(types))
      (@modules ||= []) << mod
      mod.get_function("#{@name}_final")
    end

    # The second pass folds the chunks of every output and applies the post
    # expression, one thread per output.
    def final_source(types)
      placeholders = (@in_params + @out_params).map(&:type).uniq.reject { |t| TYPES.key?(t) }
      reduce_type = @reduce_type || CTYPE[types[@out_params[0].type]]
      decl = ["const _type_reduce* _partials"]
      write = @out_params.map do |p|
        ctype = CTYPE[types[p.type]]
        decl << "#{ctype}* _raw_#{p.name}"
        "#{ctype}& #{p.name} = _raw_#{p.name}[_i];"
      end
      decl << "long long _out_size"
      decl << "int _chunks"
      <<~CUDA
        #{placeholders.map { |t| "typedef #{CTYPE[types[t]]} #{t};" }.join("\n")}
        #{@preamble}
        typedef #{reduce_type} _type_reduce;
        #define REDUCE(a, b) (#{@reduce_expr})
        extern "C" __global__ void #{@name}_final(#{decl.join(', ')}) {
          for (long long _i = blockIdx.x * blockDim.x + threadIdx.x; _i < _out_size; _i += gridDim.x * blockDim.x) {
            _type_reduce _s = _type_reduce(#{@identity});
            for (int _c = 0; _c < _chunks; _c++) {
              _type_reduce _b = _partials[_i * _chunks + _c];
              _s = REDUCE(_s, _b);
            }
            const _type_reduce a = _s;
            #{write.join("\n      ")}
            #{@post_map_expr};
          }
        }
      CUDA
    end
  end
end
