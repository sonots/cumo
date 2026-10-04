# frozen_string_literal: true

require_relative "user_kernel"

module Cumo::CUDA
  # A kernel that applies one piece of CUDA C to every element, the way
  # CuPy's ElementwiseKernel does.
  #
  #   squared_diff = Cumo::CUDA::ElementwiseKernel.new(
  #     "float32 x, float32 y", "float32 z", "z = (x - y) * (x - y)", "squared_diff")
  #   squared_diff.call(x, y)
  #
  # A one-letter type is a placeholder that takes the dtype of the argument
  # it is used with. Array arguments are broadcast against each other, a
  # Ruby number goes as a scalar, and an argument marked raw is handed over
  # as a pointer for the operation to index itself, with i the element
  # index and _ind.size() the element count.
  #
  # float16 and bfloat16 are CUDA's __half and __nv_bfloat16, so a kernel
  # that uses them is compiled with the toolkit's headers, found through
  # CUDA_PATH or the nvcc on PATH. They convert to and from every arithmetic
  # type, so mixing one with another type or a literal is ambiguous and needs
  # a cast: x * T(2), or (float)x * 2.
  class ElementwiseKernel
    include UserKernel

    BLOCK = 256
    MAX_GRID = 65_535

    attr_reader :name

    def initialize(in_params, out_params, operation, name, preamble: "")
      check_name(name)
      @in_params = parse_params(in_params)
      @out_params = parse_params(out_params)
      @operation = operation
      @name = name
      @preamble = preamble
      @functions = {}
    end

    # Applies the kernel and answers the output, or the outputs as an Array
    # when there are several. Outputs may be given after the inputs. size:
    # is the element count when no argument decides it.
    def call(*args, size: nil)
      ins, outs = split_args(args, @in_params, @out_params)
      types = resolve_types(@in_params, ins, @out_params, outs)
      shape = broadcast_shape(ins, outs, size)
      outs = @out_params.each_with_index.map { |p, k| outs[k] || types[p.type].new(*shape) }
      check_outputs(outs, @out_params, shape)

      n = shape.inject(1, :*)
      launch(ins, outs, types, shape, n) if n > 0
      @out_params.size == 1 ? outs[0] : outs
    end

    private

    def broadcast_shape(ins, outs, size)
      shapes = @in_params.zip(ins).filter_map { |p, a| a.shape if a.is_a?(Cumo::NArray) && !p.raw }
      shapes += outs.filter_map { |o| o&.shape }
      if shapes.empty?
        raise ArgumentError, "every argument is raw or a number, so size: has to say how many elements there are" if size.nil?
        return [Integer(size)]
      end
      unless size.nil?
        raise ArgumentError, "size: is only for a call whose arguments are all raw or numbers, and here the arrays decide the shape"
      end
      broadcast_shapes(shapes)
    end

    def launch(ins, outs, types, shape, n)
      params = @in_params + @out_params
      arrays = ins.map.with_index { |a, k| @in_params[k].raw ? readable(a) : a } + outs
      layouts = ins.map.with_index do |a, k|
        a.is_a?(Cumo::NArray) && !@in_params[k].raw ? input_layout(a, shape, outs, true) : nil
      end
      walked = params.each_index.select { |k| arrays[k].is_a?(Cumo::NArray) && !params[k].raw }
      cshape, cstrides = collapse(shape, walked.map { |k| layouts[k] ? layouts[k][1] : strides(arrays[k], shape) })
      simple = cshape.size <= 1 && cstrides.all? { |st| st.empty? || st == [1] }
      stride_of = walked.zip(cstrides).to_h
      kinds = arrays.map.with_index { |a, k| !a.is_a?(Cumo::NArray) ? :scalar : params[k].raw ? :raw : :array }
      key = [params.map { |p| CTYPE[types[p.type]] }, kinds, simple ? :simple : cshape.size]
      fn = (@functions[key] ||= compile(source(types, kinds, simple, cshape.size)))

      args = []
      params.zip(arrays).each_with_index do |(p, a), k|
        if !a.is_a?(Cumo::NArray)
          args << pack_scalar(a, types[p.type], p.name)
        else
          args << (layouts[k] ? layouts[k][0] : a)
          args << stride_of[k].pack("q*") unless p.raw || simple
        end
      end
      args << cshape.pack("q*") unless simple
      args << [n].pack("q")
      fn.launch(args, grid: [(n + BLOCK - 1) / BLOCK, MAX_GRID].min, block: BLOCK)
    end

    def source(types, kinds, simple, ndim)
      params = @in_params + @out_params
      placeholders = params.filter_map { |p| p.type unless TYPES.key?(p.type) }
      placeholders.uniq!
      decl = []
      body = []
      params.zip(kinds).each do |p, kind|
        ctype = CTYPE[types[p.type]]
        const = @in_params.include?(p) ? "const " : ""
        case kind
        when :scalar
          decl << "const #{ctype} #{p.name}"
        when :raw
          decl << "#{const}#{ctype}* #{p.name}"
        else
          decl << "#{const}#{ctype}* _raw_#{p.name}"
          decl << "_Dims _st_#{p.name}" unless simple
          offset = simple ? "i" : (0...ndim).map { |d| "_idx[#{d}] * _st_#{p.name}.v[#{d}]" }.join(" + ")
          body << "#{const}#{ctype}& #{p.name} = _raw_#{p.name}[#{offset}];"
        end
      end
      decl << "_Dims _shape" unless simple
      decl << "long long _n"
      index = simple ? "" : <<~CUDA
        long long _idx[#{ndim}];
        { long long _r = i; for (int _d = #{ndim} - 1; _d >= 0; _d--) { _idx[_d] = _r % _shape.v[_d]; _r /= _shape.v[_d]; } }
      CUDA
      <<~CUDA
        #{headers(params.map { |p| CTYPE[types[p.type]] })}
        #{placeholders.map { |t| "typedef #{CTYPE[types[t]]} #{t};" }.join("\n")}
        struct _Ind { long long n; __device__ long long size() const { return n; } };
        #{simple ? "" : "struct _Dims { long long v[#{ndim}]; };"}
        #{@preamble}
        extern "C" __global__ void #{@name}(#{decl.join(', ')}) {
          _Ind _ind; _ind.n = _n;
          #pragma unroll 1
          for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < _n; i += gridDim.x * blockDim.x) {
            #{index}
            #{body.join("\n    ")}
            #{@operation};
          }
        }
      CUDA
    end
  end
end
