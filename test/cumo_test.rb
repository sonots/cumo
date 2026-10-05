# frozen_string_literal: true

require_relative "test_helper"

class CumoTest < Test::Unit::TestCase
  include CumoChildProcess

  # Each flag used to read only OFF, 0 and NO as a no, so off and false turned
  # it on. A misspelling now keeps the default and says so.
  FLAG_SCRIPT = <<~'RUBY'
    require "cumo/narray"
    $stderr.sync = true
    2.times { Cumo::DFloat[1.0].to_a }
    print [Cumo.compatible_mode_enabled?, Cumo::CUDA::MemoryPool.enabled?, Cumo.allow_tf32?].inspect
  RUBY

  # RUBYOPT is dropped so that a -W0 in the developer's shell cannot silence
  # the warning the misspelling test reads.
  def flags(env)
    run_child(FLAG_SCRIPT, env: { "RUBYOPT" => nil }.merge(env))
  end

  {
    "off"   => [false, false, false], "false" => [false, false, false], "OFF" => [false, false, false], "0" => [false, false, false],
    "on"    => [true, true, true],    "TRUE"  => [true, true, true],    "yes" => [true, true, true],    "1" => [true, true, true],
    ""      => [false, true, false],
  }.each do |word, (compat, pool, tf32)|
    test "CUMO_COMPATIBLE_MODE, CUMO_MEMORY_POOL and CUMO_ALLOW_TF32 read #{word.inspect} for what it says" do
      out = flags("CUMO_COMPATIBLE_MODE" => word, "CUMO_MEMORY_POOL" => word, "CUMO_ALLOW_TF32" => word)
      assert_equal([compat, pool, tf32].inspect, out.lines.last)
      assert_not_include(out, "is not a yes or a no")
    end
  end

  test "a misspelled flag keeps its default and warns" do
    out = flags("CUMO_COMPATIBLE_MODE" => "flase", "CUMO_MEMORY_POOL" => "flase", "CUMO_ALLOW_TF32" => "flase")
    assert_equal([false, true, false].inspect, out.lines.last)
    assert_include(out, "CUMO_COMPATIBLE_MODE=flase is not a yes or a no, leaving it off")
    assert_include(out, "CUMO_MEMORY_POOL=flase is not a yes or a no, leaving it on")
    assert_include(out, "CUMO_ALLOW_TF32=flase is not a yes or a no, leaving it off")
  end

  # The warning about a method that synchronizes is the observable side of the
  # other two flags: shown or not, and once or every time.
  test "CUMO_SHOW_WARNING and CUMO_SHOW_WARNING_ONCE read off and false" do
    sync = "synchronizes with CPU"
    assert_equal(0, flags("CUMO_SHOW_WARNING" => "off").scan(sync).size)
    assert_equal(1, flags("CUMO_SHOW_WARNING" => "true").scan(sync).size)
    assert_equal(2, flags("CUMO_SHOW_WARNING" => "1", "CUMO_SHOW_WARNING_ONCE" => "false").scan(sync).size)
    assert_equal(1, flags("CUMO_SHOW_WARNING" => "1", "CUMO_SHOW_WARNING_ONCE" => "flase").scan(sync).size)
  end

  LOADING_SCRIPT = <<~'RUBY'
    require "cumo"
    require "fiddle"
    mode = Fiddle::Pointer.malloc(4)
    Fiddle::Function.new(Fiddle.dlopen("libcuda.so.1")["cuModuleGetLoadingMode"], [Fiddle::TYPE_VOIDP], Fiddle::TYPE_INT).call(mode)
    print [ENV["CUDA_MODULE_LOADING"], mode[0, 4].unpack1("l")].inspect
  RUBY

  test "modules load lazily unless CUDA_MODULE_LOADING says otherwise" do
    lazy = 2
    eager = 1
    assert_equal(["LAZY", lazy].inspect, run_child(LOADING_SCRIPT, env: { "CUDA_MODULE_LOADING" => nil }).lines.last)
    assert_equal(["EAGER", eager].inspect, run_child(LOADING_SCRIPT, env: { "CUDA_MODULE_LOADING" => "EAGER" }).lines.last)
  end

  NVRTC_SCRIPT = <<~'RUBY'
    nvrtc = -> { File.read("/proc/self/maps").include?("libnvrtc") }
    require "cumo"
    loaded = [nvrtc.()]
    Cumo::CUDA::ElementwiseKernel.new("T x", "T y", "y = x + 1", "nvrtc_on_first_use").call(Cumo::DFloat[1])
    loaded << nvrtc.()
    print loaded.inspect
  RUBY

  CUBLAS_NVRTC_SCRIPT = <<~'RUBY'
    require "fiddle"
    Fiddle.dlopen(ENV.fetch("CUMO_TEST_CUBLAS"))
    print File.read("/proc/self/maps").include?("libnvrtc")
  RUBY

  test "NVRTC is loaded the first time a user kernel is built, not on require" do
    so = $LOADED_FEATURES.grep(%r{/cumo\.so\z}).first
    cublas = IO.popen(["ldd", so], &:read)[%r{=> (\S*/libcublas\.so\S*)}, 1]
    omit("cuBLAS loads NVRTC on its own") if run_child(CUBLAS_NVRTC_SCRIPT, env: { "CUMO_TEST_CUBLAS" => cublas }).lines.last == "true"
    assert_equal([false, true].inspect, run_child(NVRTC_SCRIPT).lines.last)
  end

  HANDLES_SCRIPT = <<~'RUBY'
    require "cumo"
    require "cumo/linalg"
    used = lambda do
      rows = `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`.lines
      mib = rows.to_h { |l| l.split(",").map(&:strip) }[Process.pid.to_s]
      Integer(mib) if mib&.match?(/\A\d+\z/)
    rescue SystemCallError
      nil
    end
    a = Cumo::SFloat.new(64, 64).rand
    w = Cumo::SFloat.new(1, 1, 3, 3).rand
    work = lambda do
      a.dot(a)
      Cumo::Linalg.lu_fact(a) if Cumo::CUDA::Cusolver.available?
      a.reshape(1, 1, 64, 64).conv(w) if Cumo::CUDA::CUDNN.available?
      Cumo::CUDA::Runtime.cudaDeviceSynchronize
    end
    work.call
    Thread.new(&work).join
    before = used.call
    8.times { Thread.new(&work).join }
    after = used.call
    print(before && after ? after - before : "unmeasurable")
  RUBY

  test "a thread that ends leaves its library handles to the next one" do
    out = run_child(HANDLES_SCRIPT)
    omit("nvidia-smi cannot tell this process's device memory") if out.include?("unmeasurable")
    assert_operator(Integer(out.lines.last), :<, 32)
  end

  def setup
    @orig_compatible_mode = Cumo.compatible_mode_enabled?
  end

  def teardown
    @orig_compatible_mode ? Cumo.enable_compatible_mode : Cumo.disable_compatible_mode
  end

  def test_enable_compatible_mode
    Cumo.enable_compatible_mode
    assert { Cumo.compatible_mode_enabled? }
  end

  def test_disable_compatible_mode
    Cumo.disable_compatible_mode
    assert { !Cumo.compatible_mode_enabled? }
  end

  def test_compatible_mode_extracts_zero_dimensional_reduction_results
    Cumo.enable_compatible_mode
    a = Cumo::DFloat[3, 1, 7]
    assert_equal([Float, Float], a.minmax.map(&:class))
    assert_equal([1.0, 7.0], a.minmax)
    assert_equal([true, true], Cumo::DFloat[3, Float::NAN, 1].minmax(nan: true).map(&:nan?))
    assert_equal([Integer, Integer], Cumo::Int32[3, 1, 7].minmax.map(&:class))
    assert_equal([1, 7], Cumo::Int32[3, 1, 7].minmax)
    assert_equal([Integer, Integer], Cumo::RObject[3, 1, 7].minmax.map(&:class))
    assert_equal(Float, a.max.class)
    assert_equal(Integer, a.max_index.class)
    assert_equal(Integer, a.argmax.class)
  end

  ZERO_DIMENSIONAL_FLOATS = {
    aref: -> { Cumo::DFloat[3, 1, 7][1] },
    extract: -> { Cumo::DFloat.cast(7.0).extract },
    sum: -> { Cumo::DFloat[3, 1, 7].sum },
    prod: -> { Cumo::DFloat[3, 1, 7].prod },
    mean: -> { Cumo::DFloat[3, 1, 7].mean },
    stddev: -> { Cumo::DFloat[3, 1, 7].stddev },
    var: -> { Cumo::DFloat[3, 1, 7].var },
    rms: -> { Cumo::DFloat[3, 1, 7].rms },
    min: -> { Cumo::DFloat[3, 1, 7].min },
    max: -> { Cumo::DFloat[3, 1, 7].max },
    ptp: -> { Cumo::DFloat[3, 1, 7].ptp },
    median: -> { Cumo::DFloat[3, 1, 7].median },
    mulsum: -> { Cumo::DFloat[3, 1, 7].mulsum(Cumo::DFloat[1, 1, 1]) },
    dot: -> { Cumo::DFloat[3, 1, 7].dot(Cumo::DFloat[1, 1, 1]) },
    inner: -> { Cumo::DFloat[3, 1, 7].inner(Cumo::DFloat[1, 1, 1]) },
  }.freeze

  ZERO_DIMENSIONAL_INTEGERS = {
    max_index: -> { Cumo::DFloat[3, 1, 7].max_index },
    min_index: -> { Cumo::DFloat[3, 1, 7].min_index },
    argmax: -> { Cumo::DFloat[3, 1, 7].argmax },
    argmin: -> { Cumo::DFloat[3, 1, 7].argmin },
    count_true: -> { (Cumo::DFloat[3, 1, 7] > 2).count_true },
    count_false: -> { (Cumo::DFloat[3, 1, 7] > 2).count_false },
    bit_aref: -> { (Cumo::DFloat[3, 1, 7] > 2)[1] },
    bit_extract: -> { Cumo::Bit.cast(1).extract },
  }.freeze

  def test_compatible_mode_extracts_every_documented_method
    Cumo.enable_compatible_mode
    ZERO_DIMENSIONAL_FLOATS.each do |name, block|
      assert_instance_of(Float, block.call, name)
    end
    ZERO_DIMENSIONAL_INTEGERS.each do |name, block|
      assert_instance_of(Integer, block.call, name)
    end
  end

  def test_every_documented_method_stays_zero_dimensional_without_compatible_mode
    Cumo.disable_compatible_mode
    (ZERO_DIMENSIONAL_FLOATS.merge(ZERO_DIMENSIONAL_INTEGERS)).each do |name, block|
      result = block.call
      assert_kind_of(Cumo::NArray, result, name)
      assert_equal(0, result.ndim, name)
    end
  end

  def test_kernel_conversions_read_a_result_in_either_mode
    [true, false].each do |compatible|
      compatible ? Cumo.enable_compatible_mode : Cumo.disable_compatible_mode
      a = Cumo::DFloat[5, 2]
      assert_equal(7.0, Float(a.sum))
      assert_equal(0, Integer(a.max_index))
      assert_equal(false, a.aref_cpu(0) < 1.0)
      assert_equal(2, (a > 1).count_true_cpu)
      assert_equal(5.0, Cumo::DFloat.cast(5.0).extract_cpu)
    end
  end

  def test_minmax_returns_zero_dimensional_narrays_without_compatible_mode
    Cumo.disable_compatible_mode
    min, max = Cumo::DFloat[3, 1, 7].minmax
    assert_instance_of(Cumo::DFloat, min)
    assert_instance_of(Cumo::DFloat, max)
    assert_equal(0, min.ndim)
    assert_equal([1.0], min.to_a)
    assert_equal([7.0], max.to_a)
  end

  def test_minmax_over_an_axis_is_unaffected_by_compatible_mode
    a = Cumo::DFloat[[3, 1], [7, 2]]
    [true, false].each do |compatible|
      compatible ? Cumo.enable_compatible_mode : Cumo.disable_compatible_mode
      min, max = a.minmax(axis: 1)
      assert_instance_of(Cumo::DFloat, min)
      assert_equal([1.0, 2.0], min.to_a)
      assert_equal([3.0, 7.0], max.to_a)
    end
  end

  def test_version
    assert_nothing_raised { Cumo::VERSION }
  end

  test "every cumo function the extension calls is defined in it" do
    omit("ldd -r is Linux only") unless RUBY_PLATFORM.include?("linux")
    so = $LOADED_FEATURES.grep(%r{/cumo\.so\z}).first
    report = IO.popen(["ldd", "-r", so], err: [:child, :out], &:read)
    assert_equal([], report.scan(/undefined symbol: ((?:cumo|nvrtc)\w+)/).flatten)
  end
end
