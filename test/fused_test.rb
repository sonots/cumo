# frozen_string_literal: true

require_relative "test_helper"

class FusedTest < CumoTestBase
  # Fires on the next allocate after it is armed, so what it catches is the one
  # the method under test makes rather than the ones the setup makes.
  ARMED_ALLOCATE = <<~RUBY
    $armed = nil
    module Armed
      def allocate
        if $armed
          act = $armed
          $armed = nil
          act.call(self)
        end
        super
      end
    end
  RUBY

  FUSED_FLOAT_TYPES = [
    Cumo::HFloat,
    Cumo::BFloat,
    Cumo::SFloat,
    Cumo::DFloat,
  ]

  # The nine-kernel spelling of the same answer, in double so that it is the
  # reference and not a second implementation of the same rounding.
  def ref_layer_norm(x, gamma, beta, eps)
    x = Cumo::DFloat.cast(x)
    mu = x.mean(axis: -1, keepdims: true)
    d = x - mu
    var = (d * d).mean(axis: -1, keepdims: true)
    d / Cumo::NMath.sqrt(var + eps) * Cumo::DFloat.cast(gamma) + Cumo::DFloat.cast(beta)
  end

  def rtol_for(dtype)
    case dtype.to_s
    when "Cumo::BFloat" then 1e-1
    when "Cumo::HFloat" then 1e-2
    when "Cumo::SFloat" then 1e-5
    else 1e-11
    end
  end

  # exp(x - max) / sum(exp(x - max)), in double for the same reason.
  def ref_softmax(x)
    x = Cumo::DFloat.cast(x)
    e = Cumo::NMath.exp(x - x.max(axis: -1, keepdims: true))
    e / e.sum(axis: -1, keepdims: true)
  end

  def assert_softmax(x)
    actual = x.softmax
    expected = ref_softmax(x)
    assert_kind_of(x.class, actual)
    assert_equal(x.shape, actual.shape)
    rtol = rtol_for(x.class)
    # A softmax over a long row is all small numbers, so a bound that does not
    # scale with them passes for any answer at all: over 20000 columns in half
    # the whole result is five times under a flat 1e-2.
    scale = expected.abs.max.extract_cpu
    assert_operator((expected - Cumo::DFloat.cast(actual)).abs.max.extract_cpu, :<, rtol * scale)
    # Every row is a distribution, which the reference cannot vouch for.
    assert_operator((Cumo::DFloat.cast(actual).sum(axis: -1) - 1.0).abs.max.extract_cpu, :<, rtol)
  end

  # Both normalizations are held to the same bound against the same kind of
  # double reference, so the bound lives in one place.
  def assert_row_norm(x, actual, expected)
    assert_kind_of(x.class, actual)
    assert_equal(x.shape, actual.shape)
    scale = [1.0, expected.abs.max.extract_cpu].max
    error = (expected - Cumo::DFloat.cast(actual)).abs.max.extract_cpu
    assert_operator(error, :<, rtol_for(x.class) * scale)
  end

  def assert_layer_norm(x, gamma, beta, eps: 1e-5)
    actual = eps == 1e-5 ? x.layer_norm(gamma, beta) : x.layer_norm(gamma, beta, eps: eps)
    assert_row_norm(x, actual, ref_layer_norm(x, gamma, beta, eps))
  end

  # The same answer written out of the operators, in double so that it is the
  # reference and not a second implementation of the same rounding. The row is
  # not centred, which is the whole difference from layer_norm.
  def ref_rms_norm(x, gamma, eps)
    x = Cumo::DFloat.cast(x)
    ms = (x * x).mean(axis: -1, keepdims: true)
    x / Cumo::NMath.sqrt(ms + eps) * Cumo::DFloat.cast(gamma)
  end

  def assert_rms_norm(x, gamma, eps: 1e-5)
    actual = eps == 1e-5 ? x.rms_norm(gamma) : x.rms_norm(gamma, eps: eps)
    assert_row_norm(x, actual, ref_rms_norm(x, gamma, eps))
  end

  FUSED_FLOAT_TYPES.each do |dtype|
    # A row shorter than a warp, one that spans a block, and one long enough
    # that a thread walks it many times all take different paths through the
    # reduction, and the trailing axis is the only one that is reduced.
    [[4], [3, 5], [2, 3, 7], [1, 1], [6, 1], [4, 32], [2, 512], [2, 20_000],
     [1, 4096], [70_000, 520], [256, 9_001]].each do |shape|
      test "layer_norm #{shape.inspect} #{dtype}" do
        x = dtype.new(*shape).rand_norm
        gamma = dtype.new(shape.last).rand_norm
        beta = dtype.new(shape.last).rand_norm
        assert_layer_norm(x, gamma, beta)
      end
    end

    test "layer_norm takes eps #{dtype}" do
      x = dtype.new(3, 8).rand_norm
      gamma = dtype.new(8).rand_norm
      beta = dtype.new(8).rand_norm
      assert_layer_norm(x, gamma, beta, eps: 1.0)
      refute_equal(x.layer_norm(gamma, beta).to_a, x.layer_norm(gamma, beta, eps: 1.0).to_a)
    end

    test "layer_norm answers beta for a flat row near the top of the range #{dtype}" do
      big = { Cumo::DFloat => 1e306, Cumo::HFloat => 6e4 }.fetch(dtype, 1e36)
      beta = dtype.new(512).seq * 0.001
      y = dtype.new(4, 512).fill(big).layer_norm(dtype.ones(512), beta)
      assert_equal(Cumo::DFloat.cast(beta.tile(4, 1)).to_a, Cumo::DFloat.cast(y).to_a)
    end

    test "layer_norm reads a row that does not start on sixteen bytes #{dtype}" do
      flat = dtype.new(65).rand_norm
      assert_layer_norm(flat[1..-1], dtype.new(64).rand_norm, dtype.new(64).rand_norm)
    end

    test "layer_norm reads a non-contiguous view #{dtype}" do
      base = dtype.new(5, 4).rand_norm
      x = base.transpose
      gamma = dtype.new(5).rand_norm
      beta = dtype.new(5).rand_norm
      assert_layer_norm(x, gamma, beta)
    end

    test "layer_norm scales by gamma and shifts by beta #{dtype}" do
      x = dtype.new(2, 6).rand_norm
      ones = dtype.ones(6)
      zeros = dtype.zeros(6)
      plain = x.layer_norm(ones, zeros)
      assert_layer_norm(x, dtype.new(6).seq(1), dtype.new(6).seq(-3))
      # gamma and beta are not interchangeable, and a row of one element has no
      # deviation left to scale, so it comes out as beta alone.
      scaled = x.layer_norm(ones * 2, zeros)
      assert_operator((Cumo::DFloat.cast(scaled) - Cumo::DFloat.cast(plain) * 2).abs.max.extract_cpu,
                      :<, rtol_for(dtype))
      one_wide = dtype[[5.0], [7.0]].layer_norm(dtype[2.0], dtype[3.0])
      assert_equal([[3.0], [3.0]], one_wide.to_a)
    end

    # A row too long to walk with one block is reduced through the split
    # machinery and applied by a second kernel instead, and the two sides of
    # that switch have to answer the same thing.
    [[1, 20_000], [2, 9_000], [255, 9_000], [256, 9_000]].each do |shape|
      test "layer_norm #{shape.inspect} #{dtype}" do
        x = dtype.new(*shape).rand_norm
        gamma = dtype.new(shape.last).rand_norm
        beta = dtype.new(shape.last).rand_norm
        assert_layer_norm(x, gamma, beta)
      end
    end

    # The grid is held at 65536 blocks, so this is the only shape where a block
    # takes a second row and the barrier that guards the shared row total
    # between them has anything to guard.
    test "layer_norm walks more rows than the grid holds #{dtype}" do
      x = dtype.new(70_000, 8).rand_norm
      gamma = dtype.new(8).rand_norm
      beta = dtype.new(8).rand_norm
      assert_layer_norm(x, gamma, beta)
    end

    test "layer_norm answers an empty array for an empty one #{dtype}" do
      x = dtype.new(0, 3).seq
      y = x.layer_norm(dtype.new(3).seq, dtype.new(3).seq)
      assert_kind_of(dtype, y)
      assert_equal([0, 3], y.shape)
    end

    # 8192 columns is the last shape one block walks and 8193 the first the split
    # machinery takes, so the pair is the only place the two paths meet.
    [[4], [3, 5], [2, 3, 7], [1, 1], [6, 1], [4, 32], [2, 512],
     [1, 8192], [1, 8193], [2, 9_000], [255, 9_000], [256, 9_000], [70_000, 8],
     [3, 1000], [1, 2000], [1, 4000], [256, 1000], [70_000, 520],
     [256, 8192], [256, 8193]].each do |shape|
      test "softmax #{shape.inspect} #{dtype}" do
        assert_softmax(dtype.new(*shape).rand_norm)
      end
    end

    test "softmax reads a non-contiguous view #{dtype}" do
      assert_softmax(dtype.new(5, 4).rand_norm.transpose)
    end

    # Masking with -Infinity is what attention does with it, and a row of them
    # is what the two sides of a reduction combine into: taking the maximum of
    # two of those and rescaling by exp of their difference asks for exp(NaN).
    [512, 8192, 8193, 20_000].each do |cols|
      test "softmax takes a row masked with -Infinity, #{cols} wide #{dtype}" do
        x = dtype.new(1, cols).fill(-Float::INFINITY)
        x[0, cols - 2] = 1.0
        x[0, cols - 1] = 2.0
        y = x.softmax
        assert_equal(0, y.to_a.flatten.count { |v| v.nan? }, "softmax answered NaN for a masked row")
        assert_in_delta(0.26894142136999516, y[0, cols - 2].extract_cpu, rtol_for(dtype))
        assert_in_delta(0.7310585786300049, y[0, cols - 1].extract_cpu, rtol_for(dtype))
      end
    end

    # Both paths answer what exp(x - max) / sum answers, which is NaN: the
    # largest term is exp(Infinity - Infinity), and a row with nothing finite
    # in it has no maximum to measure against.
    [512, 20_000].each do |cols|
      test "softmax answers NaN where the reference does, #{cols} wide #{dtype}" do
        plus = dtype.new(1, cols).fill(0.0)
        plus[0, 0] = Float::INFINITY
        assert(plus.softmax.to_a.flatten.all? { |v| v.nan? })
        assert(dtype.new(1, cols).fill(-Float::INFINITY).softmax.to_a.flatten.all? { |v| v.nan? })
      end
    end

    # Taking the row maximum out is what keeps exp from overflowing, and it has
    # to be the maximum: softmax is the same answer whatever is subtracted, so
    # only a row whose spread overflows tells the maximum from anything else.
    test "softmax stays finite where a bare exp would not #{dtype}" do
      # bfloat16 carries a float's exponent, so it takes the wide bound with
      # the rest; only binary16 overflows this early.
      far = dtype == Cumo::HFloat ? 20.0 : 1.0e3
      x = dtype[[-far, 0.0, far]]
      y = x.softmax
      assert(y.to_a.flatten.all? { |v| v.finite? }, "softmax answered #{y.to_a.inspect}")
      assert(Cumo::NMath.exp(x).to_a.flatten.any? { |v| v.infinite? },
             "a bare exp is expected to overflow #{dtype} here")
      assert_softmax(x)
    end

    test "softmax keeps the row maximum through a long row #{dtype}" do
      # bfloat16 carries a float's exponent, so it takes the wide bound with
      # the rest; only binary16 overflows this early.
      far = dtype == Cumo::HFloat ? 20.0 : 1.0e3
      [512, 20_000].each do |cols|
        x = dtype.new(1, cols).fill(-far)
        x[0, cols - 1] = far
        y = x.softmax
        assert(y.to_a.flatten.all? { |v| v.finite? })
        assert_in_delta(1.0, y[0, cols - 1].extract_cpu, rtol_for(dtype))
      end
    end

    test "softmax answers one for a row of one #{dtype}" do
      assert_equal([[1.0], [1.0]], dtype[[5.0], [7.0]].softmax.to_a)
    end

    test "softmax answers an empty array for an empty one #{dtype}" do
      y = dtype.new(0, 3).seq.softmax
      assert_kind_of(dtype, y)
      assert_equal([0, 3], y.shape)
    end

    test "softmax takes a NaN through the whole row #{dtype}" do
      y = dtype[[1.0, Float::NAN, 3.0]].softmax
      assert(y.to_a.flatten.all? { |v| v.nan? }, "softmax answered #{y.to_a.inspect}")
    end

    test "softmax refuses what it cannot run along #{dtype}" do
      assert_raise(Cumo::NArray::ShapeError) { dtype.cast(1.0).softmax }
    end

    test "layer_norm refuses what it cannot normalize #{dtype}" do
      x = dtype.new(2, 3).rand_norm
      assert_raise(Cumo::NArray::ShapeError) { x.layer_norm(dtype.new(2).seq, dtype.new(3).seq) }
      assert_raise(Cumo::NArray::ShapeError) { x.layer_norm(dtype.new(3).seq, dtype.new(4).seq) }
      assert_raise(Cumo::NArray::ShapeError) { dtype.cast(1.0).layer_norm(dtype.new(1).seq, dtype.new(1).seq) }
      assert_raise(TypeError) { x.layer_norm(Cumo::Int32.new(3).seq, dtype.new(3).seq) }
      # The right number of elements in the wrong shape is not the right gamma.
      assert_raise(Cumo::NArray::ShapeError) { x.layer_norm(dtype.new(1, 3).seq, dtype.new(3).seq) }
      assert_raise(Cumo::NArray::ShapeError) { x.layer_norm(dtype.new(3).seq, dtype.new(3, 1).seq) }
    end

    # The same shapes layer_norm takes: a row shorter than a warp, one that
    # spans a block, one the split machinery has to take, and more rows than
    # the grid holds.
    [[4], [3, 5], [2, 3, 7], [1, 1], [6, 1], [4, 32], [2, 512], [2, 20_000],
     [1, 8192], [1, 8193], [2, 9_000], [255, 9_000], [256, 9_000], [70_000, 8],
     [1, 4096], [70_000, 520], [256, 9_001]].each do |shape|
      test "rms_norm #{shape.inspect} #{dtype}" do
        x = dtype.new(*shape).rand_norm
        gamma = dtype.new(shape.last).rand_norm
        assert_rms_norm(x, gamma)
      end
    end

    test "rms_norm takes eps #{dtype}" do
      x = dtype.new(3, 8).rand_norm
      gamma = dtype.new(8).rand_norm
      assert_rms_norm(x, gamma, eps: 1.0)
      refute_equal(x.rms_norm(gamma).to_a, x.rms_norm(gamma, eps: 1.0).to_a)
    end

    test "rms_norm reads a row that does not start on sixteen bytes #{dtype}" do
      flat = dtype.new(65).rand_norm
      assert_rms_norm(flat[1..-1], dtype.new(64).rand_norm)
    end

    test "rms_norm reads a non-contiguous view #{dtype}" do
      x = dtype.new(5, 4).rand_norm.transpose
      assert_rms_norm(x, dtype.new(5).rand_norm)
    end

    # A row of one is its own root mean square, so it comes out as gamma
    # whatever it held, and doubling gamma doubles the answer.
    test "rms_norm scales by gamma #{dtype}" do
      x = dtype.new(2, 6).rand_norm
      ones = dtype.ones(6)
      plain = x.rms_norm(ones)
      assert_rms_norm(x, dtype.new(6).seq(1))
      scaled = x.rms_norm(ones * 2)
      assert_operator((Cumo::DFloat.cast(scaled) - Cumo::DFloat.cast(plain) * 2).abs.max.extract_cpu,
                      :<, rtol_for(dtype))
      # A row of one is its own root mean square, so it comes out as gamma with
      # its sign kept. eps is taken out to say that exactly: left in, it is not
      # rounding but part of the answer, 4e-7 of it at this scale.
      one_wide = dtype[[5.0], [-7.0]].rms_norm(dtype[2.0], eps: 0.0)
      assert_operator((Cumo::DFloat.cast(one_wide) - Cumo::DFloat[[2.0], [-2.0]]).abs.max.extract_cpu,
                      :<, rtol_for(dtype))
      assert_rms_norm(dtype[[5.0], [-7.0]], dtype[2.0])
    end

    # Centring is the whole difference from layer_norm, so a row whose mean is
    # not zero is the only thing that tells the two apart. A row that already
    # has a zero mean is the control: there the two agree.
    test "rms_norm does not centre the row #{dtype}" do
      ones = dtype.ones(4)
      offset = dtype[[1.0, 2.0, 3.0, 4.0]]
      assert_operator((Cumo::DFloat.cast(offset.rms_norm(ones)) -
                       Cumo::DFloat.cast(offset.layer_norm(ones, dtype.zeros(4)))).abs.max.extract_cpu,
                      :>, 0.5)
      centred = dtype[[-3.0, -1.0, 1.0, 3.0]]
      assert_operator((Cumo::DFloat.cast(centred.rms_norm(ones)) -
                       Cumo::DFloat.cast(centred.layer_norm(ones, dtype.zeros(4)))).abs.max.extract_cpu,
                      :<, rtol_for(dtype))
    end

    # eps inside the square root is what keeps a row with nothing in it finite.
    test "rms_norm answers zero for a row of zeros #{dtype}" do
      y = dtype.zeros(2, 4).rms_norm(dtype.ones(4))
      assert_equal([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], y.to_a)
    end

    test "rms_norm answers an empty array for an empty one #{dtype}" do
      y = dtype.new(0, 3).seq.rms_norm(dtype.new(3).seq)
      assert_kind_of(dtype, y)
      assert_equal([0, 3], y.shape)
    end

    test "rms_norm refuses what it cannot normalize #{dtype}" do
      x = dtype.new(2, 3).rand_norm
      assert_raise(Cumo::NArray::ShapeError) { x.rms_norm(dtype.new(2).seq) }
      assert_raise(Cumo::NArray::ShapeError) { dtype.cast(1.0).rms_norm(dtype.new(1).seq) }
      assert_raise(TypeError) { x.rms_norm(Cumo::Int32.new(3).seq) }
      # The right number of elements in the wrong shape is not the right gamma.
      assert_raise(Cumo::NArray::ShapeError) { x.rms_norm(dtype.new(1, 3).seq) }
      assert_raise(Cumo::NArray::ShapeError) { x.rms_norm(dtype.new(3, 1).seq) }
    end
  end

  # Half holds 65504, so a row spread this wide squares to a sum of deviations
  # the element type cannot carry. The accumulator is float, so the answer is
  # still the one double gives.
  test "layer_norm accumulates a HFloat row wider than half can square" do
    x = Cumo::HFloat[[-300.0, -100.0, 100.0, 300.0]]
    gamma = Cumo::HFloat.ones(4)
    beta = Cumo::HFloat.zeros(4)
    naive = (x - x.mean(axis: -1, keepdims: true))
    assert((naive * naive).mean(axis: -1).to_a.flatten.all? { |v| v.infinite? },
           "the naive HFloat variance is expected to overflow here")
    assert_layer_norm(x, gamma, beta)
  end

  # rms_norm squares the row rather than its deviations, so a row this wide
  # overflows half one step earlier than layer_norm does.
  test "rms_norm accumulates a HFloat row wider than half can square" do
    x = Cumo::HFloat[[-300.0, -100.0, 100.0, 300.0]]
    assert((x * x).mean(axis: -1).to_a.flatten.all? { |v| v.infinite? },
           "the naive HFloat mean square is expected to overflow here")
    assert_rms_norm(x, Cumo::HFloat.ones(4))
  end

  # bfloat16 carries a float's exponent, so the float accumulator that saves a
  # half row has nothing left over for a bfloat16 one: the square overflows it
  # too and the row comes back as zeros. README says so, and this is what it
  # says it against.
  test "rms_norm cannot save a BFloat row that squares past a float" do
    ones = Cumo::BFloat.ones(4)
    x = Cumo::BFloat[[-3.0e20, 3.0e20, 3.0e20, -3.0e20]]
    assert_equal([[-0.0, 0.0, 0.0, -0.0]], x.rms_norm(ones).to_a)
    assert_equal([[-1.0, 1.0, 1.0, -1.0]],
                 Cumo::DFloat[[-3.0e20, 3.0e20, 3.0e20, -3.0e20]].rms_norm(Cumo::DFloat.ones(4)).to_a)
    # A half row of the same shape is the control: there the accumulator does
    # save it, because half's exponent is far narrower than a float's.
    assert_equal([[-1.0, 1.0, 1.0, -1.0]],
                 Cumo::HFloat[[-300.0, 300.0, 300.0, -300.0]].rms_norm(Cumo::HFloat.ones(4)).to_a)
  end

  # README says all three answer a new array and ignore inplace!, and nothing
  # held that to it.
  test "the fused methods ignore inplace! and leave the receiver alone" do
    before = [[1.0, 2.0, 3.0, 4.0]]
    g = Cumo::SFloat.ones(4)
    b = Cumo::SFloat.zeros(4)
    {"rms_norm" => ->(x) { x.inplace.rms_norm(g) },
     "layer_norm" => ->(x) { x.inplace.layer_norm(g, b) },
     "softmax" => ->(x) { x.inplace.softmax }}.each do |name, call|
      x = Cumo::SFloat[*before]
      y = call.call(x)
      refute_same(x, y, name)
      assert_equal(before, x.to_a, name)
    end
  end

  # The preamble these three share is in row_method.h, so each case below runs
  # against all of them rather than against whichever one it was written for.
  # The operands each one holds, so that a case can arm the one it means to
  # perturb and nothing allocates what it will not use.
  FUSED_CALLS = {
    "layer_norm" => ["x.layer_norm(g, b)", %w[g b]],
    "rms_norm" => ["x.rms_norm(g)", %w[g]],
    "softmax" => ["x.softmax", []],
  }.freeze

  OPERAND_SETUP = {
    "g" => "g = Cumo::DFloat.new(cols).fill(1)",
    "b" => "b = Cumo::DFloat.new(cols).fill(0)",
  }.freeze

  # Taking a pointer runs allocate, and a class is free to redefine it, so the
  # sizes the launch is built from have to be the ones left behind afterwards.
  {"resizes the output under it" =>
     ["Cumo::DFloat.new(1 << 20)", "$armed = ->(a) { a.send(:initialize, 1) }",
      "self or the result was reshaped while it was measured"],
   "frees the input under it" =>
     ["Cumo::DFloat.new(1 << 20)", "$armed = ->(_) { x.free }",
      "cannot read unallocated NArray"],
   "reshapes to the same size" =>
     ["Cumo::DFloat.new(4, 6)", "$armed = ->(a) { a.reshape!(24) }",
      "self or the result was reshaped while it was measured"],
   # The three above all land on the output, which is the only array whose
   # allocate the method itself runs. Reaching self and the operands takes a
   # lambda that ignores what it was handed, and is the other half of what the
   # preamble checks.
   "leaves self reshaped behind it" =>
     ["Cumo::DFloat.new(4, 6)", "$armed = ->(_) { x.reshape!(24) }",
      "self or the result was reshaped while it was measured"],
   "frees an operand under it" =>
     ["Cumo::DFloat.new(4, 6)", "$armed = ->(_) { g.free }",
      "cannot read unallocated NArray", "g"],
   "reshapes an operand under it" =>
     ["Cumo::DFloat.new(4, 6)", "$armed = ->(_) { g.reshape!(3, 2) }",
      "gamma must be 1-dimensional and 6 long", "g"],
  }.each do |what, (make, arm, message, needs)|
    FUSED_CALLS.each do |name, (call, operands)|
      next if needs && !operands.include?(needs)

      test "#{name} refuses an allocate that #{what}" do
        assert_child_raises(message, <<~RUBY, prelude: ARMED_ALLOCATE)
          Cumo::DFloat.prepend(Armed)
          x = #{make}.seq
          cols = x.shape[-1]
          #{operands.map { |o| OPERAND_SETUP[o] }.join("\n          ")}
          Cumo::CUDA::Runtime.cudaDeviceSynchronize
          #{arm}
          #{call}
        RUBY
      end
    end
  end

  # Contiguity, the shape and the class are what cumo_na_as_contiguous_array is
  # asked for and what dup is free not to answer with.
  # The trailing axis of one is the case only the dimension count catches: every
  # axis the original has agrees, and so does the element count.
  {"a contiguous array" => "Cumo::DFloat.new(128, 64).seq.reverse",
   "an array shaped like the one it was given" => "Cumo::DFloat.new(128 * 64).seq",
   "an array shaped like the one it was given, one axis deeper" => "Cumo::DFloat.new(128, 64, 1).seq",
   "a Cumo::DFloat" => "Cumo::Int32.new(128, 64).seq"}.each do |message, body|
    FUSED_CALLS.each do |name, (call, operands)|
      test "#{name} refuses a dup that does not answer #{message}" do
        assert_child_raises("dup did not answer #{message.sub(/, one axis deeper\z/, "")}", <<~RUBY, prelude: "")
          x = Cumo::DFloat.new(64, 128).seq.transpose
          cols = 64
          #{operands.map { |o| OPERAND_SETUP[o] }.join("\n          ")}
          class Cumo::DFloat
            def dup
              #{body}
            end
          end
          #{call}
        RUBY
      end
    end
  end

  # The scale comes back in the class the reduction accumulates in, which is
  # wider than the element for the 16-bit ones. Reading it as the element type
  # answered a scale that was not even positive.
  FUSED_FLOAT_TYPES.each do |dtype|
    test "quantize_symmetric answers the scale of every row #{dtype}" do
      x = dtype.cast(Cumo::SFloat[[1.0, -2.0, 4.0, -8.0], [0.5, 0.25, -1.0, 0.125]])
      q, s = x.quantize_symmetric
      assert_equal(Cumo::Int8, q.class)
      assert_equal([2], s.shape)
      assert_equal([[16, -32, 64, -127], [64, 32, -127, 16]], q.to_a)
      s.to_a.zip([8.0 / 127, 1.0 / 127]).each do |got, want|
        assert_in_delta(want, got, want * 1e-2)
      end
    end

    test "quantize_symmetric answers a positive finite scale for every row #{dtype}" do
      x = dtype.cast(Cumo::SFloat.new(64, 16).seq + 1)
      _, s = x.quantize_symmetric
      assert_equal([64], s.shape)
      s.to_a.each_with_index do |v, i|
        assert(v.finite? && v > 0, "row #{i} answered #{v}")
      end
    end

    test "quantize_symmetric takes a row of zeros and a row with an infinity apart #{dtype}" do
      x = dtype.cast(Cumo::SFloat[[0.0, 0.0], [1.0, Float::INFINITY], [Float::NAN, 1.0]])
      q, s = x.quantize_symmetric
      assert_equal([[0, 0], [0, 0], [0, 0]], q.to_a)
      assert_equal(0.0, s[0].to_f)
      assert_equal(Float::INFINITY, s[1].to_f)
      assert(s[2].to_f.nan?, "a row holding a NaN answers one")
    end
  end

  test "quantize_symmetric matches the spelling it replaces" do
    [[4, 8], [3, 5, 32], [1, 4]].each do |shape|
      x = (Cumo::SFloat.new(*shape).seq % 211) - 105
      q, s = x.quantize_symmetric
      want_s = x.abs.max(axis: -1, keepdims: true) / 127.0
      want_q = (x / want_s.clip(1e-30, nil)).round
      assert_equal(want_q.to_a, Cumo::SFloat.cast(q).to_a, shape.inspect)
      assert_equal(want_s.reshape(*s.shape).to_a, s.to_a, shape.inspect)
    end
  end

  test "quantize_symmetric answers a scale for a row with nothing in it" do
    q, s = Cumo::SFloat.new(2, 0).quantize_symmetric
    assert_equal([2, 0], q.shape)
    assert_equal([0.0, 0.0], s.to_a)
  end

  test "quantize_symmetric drops the last axis from the scale" do
    _, s = Cumo::SFloat.new(2, 3, 4).seq.quantize_symmetric
    assert_equal([2, 3], s.shape)
    _, s1 = Cumo::SFloat.new(4).seq.quantize_symmetric
    assert_equal(0, s1.ndim)
  end

  test "a rejected operand is named, and so are both classes" do
    x = Cumo::SFloat.new(2, 6).seq
    right = Cumo::SFloat.new(6).fill(1.0)
    wrong = Cumo::HFloat.new(6).fill(1.0)
    [["gamma", -> { x.rms_norm(wrong) }],
     ["gamma", -> { x.layer_norm(wrong, right) }],
     ["beta", -> { x.layer_norm(right, wrong) }]].each do |operand, call|
      message = assert_raise(TypeError) { call.call }.message
      assert_equal("#{operand} must be Cumo::SFloat, not Cumo::HFloat", message)
    end
  end

  test "two operands turned down side by side do not read alike" do
    x = Cumo::SFloat.new(2, 6).seq
    right = Cumo::SFloat.new(6).fill(1.0)
    wrong = Cumo::HFloat.new(6).fill(1.0)
    gamma = assert_raise(TypeError) { x.rms_norm(wrong) }.message
    beta = assert_raise(TypeError) { x.layer_norm(right, wrong) }.message
    refute_equal(gamma, beta)
  end

  SUBCLASS = Class.new(Cumo::SFloat)

  test "a rejected receiver is named, and so are both classes" do
    a = SUBCLASS.new(2, 6).seq
    [-> { a.quantize_symmetric },
     -> { a.rms_norm(Cumo::SFloat.new(6).fill(1.0)) },
     -> { a.softmax }].each do |call|
      message = assert_raise(TypeError) { call.call }.message
      assert_equal("self must be Cumo::SFloat, not #{SUBCLASS.name}", message)
    end
  end

  test "a class name outside ASCII survives into the message" do
    klass = Class.new(Cumo::SFloat)
    Object.const_set(:"Cumoテスト", klass)
    message = assert_raise(TypeError) { klass.new(2, 6).seq.softmax }.message
    assert_equal(Encoding::UTF_8, message.encoding)
    assert { message.include?("Cumoテスト") }
  ensure
    Object.send(:remove_const, :"Cumoテスト")
  end
end
