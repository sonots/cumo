# frozen_string_literal: true

module Cumo::CUDA
  # What ElementwiseKernel and ReductionKernel share: the parameter syntax,
  # the dtype table, how a type letter is resolved and how arguments are
  # broadcast and handed to the launch.
  module UserKernel
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
    INT_RANGE = {
      Cumo::Int64 => (-2**63...2**63), Cumo::Int32 => (-2**31...2**31),
      Cumo::Int16 => (-2**15...2**15), Cumo::Int8 => (-2**7...2**7),
      Cumo::UInt64 => (0...2**64), Cumo::UInt32 => (0...2**32),
      Cumo::UInt16 => (0...2**16), Cumo::UInt8 => (0...2**8),
    }.freeze

    RESERVED = /\A(i|n)\z|\A_/

    private

    def check_name(name)
      return if Compiler.valid_kernel_name?(name)
      raise ArgumentError, "#{name.inspect} is not a valid kernel name"
    end

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

    def split_args(args, in_params, out_params)
      n_in = in_params.size
      unless args.size == n_in || args.size == n_in + out_params.size
        raise ArgumentError, "#{@name} takes #{n_in} or #{n_in + out_params.size} arguments, got #{args.size}"
      end
      ins = args[0, n_in].map { |a| check_arg(a) }
      outs = args[n_in..].map { |a| check_arg(a) }
      outs.each_with_index do |a, k|
        raise TypeError, "output #{out_params[k].name} has to be an NArray, got a #{a.class}" unless a.is_a?(Cumo::NArray)
      end
      [ins, outs]
    end

    # Outputs first, then inputs, as CuPy does. A placeholder that only a
    # number reaches becomes Int64 or DFloat.
    def resolve_types(in_params, ins, out_params, outs)
      bound = {}
      (out_params.zip(outs) + in_params.zip(ins)).each do |p, a|
        next if a.nil?
        dtype = a.is_a?(Cumo::NArray) ? dtype_of(a) : nil
        if TYPES.key?(p.type)
          want = TYPES[p.type][0]
          raise TypeError, "#{p.name} is declared #{p.type} and got a #{dtype}" if dtype && dtype != want
        elsif dtype
          if bound.key?(p.type) && bound[p.type] != dtype
            raise TypeError, "#{p.type} is a #{bound[p.type]} from an earlier argument and a #{dtype} at #{p.name}"
          end
          bound[p.type] ||= dtype
        end
      end
      in_params.zip(ins).each do |p, a|
        next if TYPES.key?(p.type) || bound.key?(p.type)
        bound[p.type] = a.is_a?(Float) ? Cumo::DFloat : Cumo::Int64
      end
      (in_params + out_params).each do |p|
        next if TYPES.key?(p.type) || bound.key?(p.type)
        raise ArgumentError, "nothing decides the type #{p.type} of #{p.name}"
      end
      TYPES.transform_values(&:first).merge(bound)
    end

    def broadcast_shapes(shapes)
      nd = shapes.map(&:size).max
      Array.new(nd) do |d|
        sizes = shapes.map { |s| d < nd - s.size ? 1 : s[d - (nd - s.size)] }
        m = sizes.reject { |x| x == 1 }.uniq
        raise ArgumentError, "shapes #{shapes.map(&:inspect).join(', ')} cannot be broadcast together" if m.size > 1
        m.first || 1
      end
    end

    # Element strides of a contiguous array read as the broadcast shape, 0
    # where it is broadcast.
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

    # Merges the neighbouring dimensions every array walks with one stride
    # and drops those of length 1; the flat index maps to the same offsets.
    def collapse(shape, strides_list)
      cshape = []
      cstrides = strides_list.map { [] }
      shape.each_with_index do |n, d|
        next if n == 1
        if !cshape.empty? && strides_list.each_index.all? { |k| cstrides[k][-1] == strides_list[k][d] * n }
          cshape[-1] *= n
          strides_list.each_index { |k| cstrides[k][-1] = strides_list[k][d] }
        else
          cshape << n
          strides_list.each_index { |k| cstrides[k] << strides_list[k][d] }
        end
      end
      [cshape, cstrides]
    end

    def pack_scalar(value, dtype, name)
      if INT_RANGE.key?(dtype)
        raise TypeError, "#{name} is #{CTYPE[dtype]}, and #{value.inspect} is not an Integer" unless value.is_a?(Integer)
        raise RangeError, "#{name} is #{CTYPE[dtype]}, and #{value} does not fit" unless INT_RANGE[dtype].cover?(value)
      end
      [value].pack(PACK[dtype])
    end

    # A raw argument is indexed by the operation itself, so it has to be
    # contiguous, and it goes through a pointer the kernel may write.
    def readable(a)
      a.is_a?(Cumo::NArray) && (!a.contiguous? || a.frozen?) ? a.dup : a
    end

    # [address, strides of the broadcast shape, the copy read instead, if any].
    # An input that other threads write while it is read, because it shares
    # memory with an output, is read from a copy; so is an index view, and,
    # for an elementwise kernel, a transposed view, which the copy reorders
    # in tiles where the kernel would read it a row apart per thread.
    def input_layout(a, shape, outs, elementwise)
      layout = Driver.narray_view_layout(a)
      held = nil
      if layout.nil? || (elementwise && transposed?(a, layout[1])) ||
         outs.any? { |o| overlaps?(layout, o) && !(elementwise && same_place?(layout, a, o, shape)) }
        held = a.dup
        layout = Driver.narray_view_layout(held)
      end
      [layout[0], [0] * (shape.size - layout[1].size) + layout[1], held]
    end

    def transposed?(a, st)
      inner = a.shape.rindex { |n| n > 1 }
      return false if inner.nil? || st[inner].abs * a.class::ELEMENT_BYTE_SIZE < 32
      a.shape.each_index.any? { |d| d != inner && a.shape[d] > 1 && st[d].abs == 1 }
    end

    def overlaps?(layout, out)
      o = Driver.narray_view_layout(out)
      layout[2] < o[3] && o[2] < layout[3]
    end

    def same_place?(layout, a, out, shape)
      a.shape == shape && layout[0] == Driver.narray_view_layout(out)[0] && layout[1] == strides(out, shape)
    end

    def check_outputs(outs, out_params, shape)
      outs.each_with_index do |a, k|
        if a.shape != shape
          raise ArgumentError, "output #{out_params[k].name} has shape #{a.shape.inspect} where #{shape.inspect} is needed"
        end
        raise ArgumentError, "output #{out_params[k].name} is not contiguous" unless a.contiguous?
        raise FrozenError, "output #{out_params[k].name} is frozen" if a.frozen?
      end
    end

    def compile(source)
      mod = Compiler.new.compile_with_cache(source)
      (@modules ||= []) << mod
      mod.get_function(@name)
    end
  end
end
