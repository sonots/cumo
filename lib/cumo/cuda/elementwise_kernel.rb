# frozen_string_literal: true

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
  class ElementwiseKernel
    Param = Struct.new(:type, :name, :raw)

    TYPES = {
      "float64" => [Cumo::DFloat, "double", "d"],
      "float32" => [Cumo::SFloat, "float", "f"],
      "int64" => [Cumo::Int64, "long long", "q"],
      "int32" => [Cumo::Int32, "int", "l"],
      "int16" => [Cumo::Int16, "short", "s"],
      "int8" => [Cumo::Int8, "signed char", "c"],
      "uint64" => [Cumo::UInt64, "unsigned long long", "Q"],
      "uint32" => [Cumo::UInt32, "unsigned int", "L"],
      "uint16" => [Cumo::UInt16, "unsigned short", "S"],
      "uint8" => [Cumo::UInt8, "unsigned char", "C"],
    }.freeze
    CTYPE = TYPES.values.to_h { |klass, ctype, _| [klass, ctype] }.freeze
    PACK = TYPES.values.to_h { |klass, _, pack| [klass, pack] }.freeze

    BLOCK = 256
    MAX_GRID = 65_535

    attr_reader :name

    def initialize(in_params, out_params, operation, name, preamble: "")
      unless Compiler.valid_kernel_name?(name)
        raise ArgumentError, "#{name.inspect} is not a valid kernel name"
      end
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
      n_in = @in_params.size
      n_out = @out_params.size
      unless args.size == n_in || args.size == n_in + n_out
        raise ArgumentError, "#{@name} takes #{n_in} or #{n_in + n_out} arguments, got #{args.size}"
      end
      ins = args[0, n_in].map { |a| check_arg(a) }
      outs = args[n_in..].map { |a| check_arg(a) }
      outs.each_with_index do |a, k|
        unless a.is_a?(Cumo::NArray)
          raise TypeError, "output #{@out_params[k].name} has to be an NArray, got a #{a.class}"
        end
      end

      types = resolve_types(ins, outs)
      shape = broadcast_shape(ins, outs, size)
      outs = @out_params.each_with_index.map { |p, k| outs[k] || types[p.type].new(*shape) }
      outs.each_with_index do |a, k|
        if a.shape != shape
          raise ArgumentError, "output #{@out_params[k].name} has shape #{a.shape.inspect} where #{shape.inspect} is needed"
        end
        raise ArgumentError, "output #{@out_params[k].name} is not contiguous" unless a.contiguous?
      end

      n = shape.inject(1, :*)
      launch(ins, outs, types, shape, n) if n > 0
      n_out == 1 ? outs[0] : outs
    end

    private

    RESERVED = /\A(i|n)\z|\A_/

    def parse_params(spec)
      spec.split(",").map do |s|
        words = s.split
        raise ArgumentError, "cannot read the parameter #{s.strip.inspect}" unless words.size == 2 || words.size == 3
        raw = words.size == 3
        raise ArgumentError, "cannot read the parameter #{s.strip.inspect}" if raw && words[0] != "raw"
        type, name = words.last(2)
        unless type.match?(/\A[A-Za-z]\z/) || TYPES.key?(type)
          raise ArgumentError, "#{type} is not a type: use one letter or one of #{TYPES.keys.join(', ')}"
        end
        raise ArgumentError, "i is the index, not a type" if type == "i"
        raise ArgumentError, "#{name} is not a valid parameter name" unless name.match?(/\A[A-Za-z_]\w*\z/)
        raise ArgumentError, "#{name} is reserved" if name.match?(RESERVED)
        Param.new(type, name, raw)
      end
    end

    def check_arg(a)
      case a
      when Integer, Float
        a
      when Cumo::NArray
        raise TypeError, "a #{a.class} cannot be handed to a kernel" unless dtype_of(a)
        a
      else
        raise TypeError, "a kernel argument is an NArray, an Integer or a Float, not a #{a.class}"
      end
    end

    def dtype_of(a)
      CTYPE.each_key.find { |klass| a.is_a?(klass) }
    end

    # Outputs first, then inputs, as CuPy does. A placeholder that only a
    # number reaches becomes Int64 or DFloat.
    def resolve_types(ins, outs)
      bound = {}
      (@out_params.zip(outs) + @in_params.zip(ins)).each do |p, a|
        next if a.nil?
        dtype = a.is_a?(Cumo::NArray) ? dtype_of(a) : nil
        if TYPES.key?(p.type)
          want = TYPES[p.type][0]
          if dtype && dtype != want
            raise TypeError, "#{p.name} is declared #{p.type} and got a #{dtype}"
          end
        elsif dtype
          if bound.key?(p.type) && bound[p.type] != dtype
            raise TypeError, "#{p.type} is a #{bound[p.type]} from an earlier argument and a #{dtype} at #{p.name}"
          end
          bound[p.type] ||= dtype
        end
      end
      @in_params.zip(ins).each do |p, a|
        next if TYPES.key?(p.type) || bound.key?(p.type)
        bound[p.type] = a.is_a?(Float) ? Cumo::DFloat : Cumo::Int64
      end
      (@in_params + @out_params).each do |p|
        next if TYPES.key?(p.type) || bound.key?(p.type)
        raise ArgumentError, "nothing decides the type #{p.type} of #{p.name}"
      end
      TYPES.transform_values(&:first).merge(bound)
    end

    def broadcast_shape(ins, outs, size)
      shapes = @in_params.zip(ins).filter_map { |p, a| a.shape if a.is_a?(Cumo::NArray) && !p.raw }
      shapes += outs.compact.map(&:shape)
      if shapes.empty?
        raise ArgumentError, "every argument is raw or a number, so size: has to say how many elements there are" if size.nil?
        return [Integer(size)]
      end
      unless size.nil?
        raise ArgumentError, "size: is only for a call whose arguments are all raw or numbers, and here the arrays decide the shape"
      end
      nd = shapes.map(&:size).max
      Array.new(nd) do |d|
        sizes = shapes.map { |s| d < nd - s.size ? 1 : s[d - (nd - s.size)] }
        m = sizes.reject { |x| x == 1 }.uniq
        if m.size > 1
          raise ArgumentError, "shapes #{shapes.map(&:inspect).join(', ')} cannot be broadcast together"
        end
        m.first || 1
      end
    end

    def strides(a, shape)
      nd = shape.size
      s = [1] * (nd - a.shape.size) + a.shape
      st = Array.new(nd)
      acc = 1
      (nd - 1).downto(0) do |d|
        st[d] = s[d] == 1 ? 0 : acc
        acc *= s[d]
      end
      st
    end

    INT_RANGE = {
      Cumo::Int64 => (-2**63...2**63), Cumo::Int32 => (-2**31...2**31),
      Cumo::Int16 => (-2**15...2**15), Cumo::Int8 => (-2**7...2**7),
      Cumo::UInt64 => (0...2**64), Cumo::UInt32 => (0...2**32),
      Cumo::UInt16 => (0...2**16), Cumo::UInt8 => (0...2**8),
    }.freeze

    def pack_scalar(value, dtype, name)
      if INT_RANGE.key?(dtype)
        raise TypeError, "#{name} is #{CTYPE[dtype]}, and #{value.inspect} is not an Integer" unless value.is_a?(Integer)
        raise RangeError, "#{name} is #{CTYPE[dtype]}, and #{value} does not fit" unless INT_RANGE[dtype].cover?(value)
      end
      [value].pack(PACK[dtype])
    end

    # Every NArray reaches the kernel through a pointer it may write, so a
    # frozen input is read from a copy.
    def launch(ins, outs, types, shape, n)
      params = @in_params + @out_params
      values = ins + outs
      arrays = params.zip(values).map { |p, a| a.is_a?(Cumo::NArray) && (!a.contiguous? || a.frozen?) ? a.dup : a }
      simple = params.zip(arrays).all? { |p, a| !a.is_a?(Cumo::NArray) || p.raw || a.shape == shape }
      kinds = params.zip(arrays).map { |p, a| !a.is_a?(Cumo::NArray) ? :scalar : p.raw ? :raw : :array }
      key = [params.map { |p| CTYPE[types[p.type]] }, kinds, simple ? :simple : shape.size]
      fn = (@functions[key] ||= compile(types, kinds, simple, shape.size))

      args = []
      params.zip(arrays).each do |p, a|
        if a.is_a?(Cumo::NArray)
          args << a
          args << strides(a, shape).pack("q*") unless p.raw || simple
        else
          args << pack_scalar(a, types[p.type], p.name)
        end
      end
      args << shape.pack("q*") unless simple
      args << [n].pack("q")
      fn.launch(args, grid: [(n + BLOCK - 1) / BLOCK, MAX_GRID].min, block: BLOCK)
    end

    def compile(types, kinds, simple, ndim)
      params = @in_params + @out_params
      placeholders = params.map(&:type).uniq.reject { |t| TYPES.key?(t) }
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
          unless simple
            decl << "_Dims _st_#{p.name}"
          end
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
      source = <<~CUDA
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
      mod = Compiler.new.compile_with_cache(source)
      fn = mod.get_function(@name)
      @modules ||= []
      @modules << mod
      fn
    end
  end
end
