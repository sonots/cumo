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
    when "Cumo::HFloat" then 1e-2
    when "Cumo::SFloat" then 1e-5
    else 1e-11
    end
  end

  def assert_layer_norm(x, gamma, beta, eps: 1e-5)
    actual = eps == 1e-5 ? x.layer_norm(gamma, beta) : x.layer_norm(gamma, beta, eps: eps)
    expected = ref_layer_norm(x, gamma, beta, eps)
    assert_kind_of(x.class, actual)
    assert_equal(x.shape, actual.shape)
    scale = [1.0, expected.abs.max.extract_cpu].max
    error = (expected - Cumo::DFloat.cast(actual)).abs.max.extract_cpu
    assert_operator(error, :<, rtol_for(x.class) * scale)
  end

  FUSED_FLOAT_TYPES.each do |dtype|
    # A row shorter than a warp, one that spans a block, and one long enough
    # that a thread walks it many times all take different paths through the
    # reduction, and the trailing axis is the only one that is reduced.
    [[4], [3, 5], [2, 3, 7], [1, 1], [6, 1], [4, 32], [2, 512], [2, 20_000]].each do |shape|
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
  }.each do |what, (make, arm, message)|
    test "layer_norm refuses an allocate that #{what}" do
      assert_child_raises(message, <<~RUBY, prelude: ARMED_ALLOCATE)
        Cumo::DFloat.prepend(Armed)
        x = #{make}.seq
        cols = x.shape[-1]
        g = Cumo::DFloat.new(cols).fill(1)
        b = Cumo::DFloat.new(cols).fill(0)
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
        #{arm}
        x.layer_norm(g, b)
      RUBY
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
    test "layer_norm refuses a dup that does not answer #{message}" do
      assert_child_raises("dup did not answer #{message.sub(/, one axis deeper\z/, "")}", <<~RUBY, prelude: "")
        class Cumo::DFloat
          def dup
            #{body}
          end
        end
        x = Cumo::DFloat.new(64, 128).seq.transpose
        x.layer_norm(Cumo::DFloat.new(64).fill(1), Cumo::DFloat.new(64).fill(0))
      RUBY
    end
  end
end
