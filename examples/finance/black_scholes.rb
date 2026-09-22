# frozen_string_literal: true

require 'optparse'

require_relative '../backend'

# This sample computes call and put prices for European options with
# Black-Scholes equation. It was based on a sample of the financial package
# in CUDA toolkit. For details, please see the corresponding whitepaper.
#
# The following code shows that Cumo enables us to write algorithms for GPUs
# without significantly modifying the existing Numo's code.
# It also briefly describes how to create your own kernel with Cumo.
# If you want to speed up the existing code, please define the kernel
# with Cumo::CUDA::ElementwiseKernel.

def where(xm, condition, a, b)
  mask = xm::DFloat.cast(condition)
  a * mask + b * (1 - mask)
end

# Naive implementation of the pricing of options with Numo and Cumo.
def black_scholes(xm, s, x, t, r, v)
  sqrt_t = xm::NMath.sqrt(t)
  d1 = (xm::NMath.log(s / x) + (r + v * v / 2) * t) / (v * sqrt_t)
  d2 = d1 - v * sqrt_t

  get_cumulative_normal_distribution = lambda do |y|
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    rsqrt2pi = 0.39894228040143267793994605993438
    w = 0.2316419

    k = 1 / (1 + w * y.abs)
    cnd = rsqrt2pi * xm::NMath.exp(-y * y / 2) *
          (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5)))))
    where(xm, y.gt(0), 1 - cnd, cnd)
  end

  cnd_d1 = get_cumulative_normal_distribution.call(d1)
  cnd_d2 = get_cumulative_normal_distribution.call(d2)

  exp_rt = xm::NMath.exp(-r * t)
  call = s * cnd_d1 - x * exp_rt * cnd_d2
  put = x * exp_rt * (1 - cnd_d2) - s * (1 - cnd_d1)
  [call, put]
end

# An example of calling the kernel via Cumo::CUDA::ElementwiseKernel.
# When calling the instance, it automatically compiles the code depending on
# the types of the given arrays, and calls the kernel.
# Other functions used inside the kernel can be defined by 'preamble' option.
def black_scholes_kernel
  @black_scholes_kernel ||= Cumo::CUDA::ElementwiseKernel.new(
    'T s, T x, T t, T r, T v', # Inputs
    'T call, T put', # Outputs
    <<~OPERATION,
      const T sqrt_t = sqrt(t);
      const T d1 = (log(s / x) + (r + v * v / 2) * t) / (v * sqrt_t);
      const T d2 = d1 - v * sqrt_t;

      const T cnd_d1 = get_cumulative_normal_distribution(d1);
      const T cnd_d2 = get_cumulative_normal_distribution(d2);

      const T exp_rt = exp(- r * t);
      call = s * cnd_d1 - x * exp_rt * cnd_d2;
      put = x * exp_rt * (1 - cnd_d2) - s * (1 - cnd_d1);
    OPERATION
    'black_scholes_kernel',
    preamble: <<~PREAMBLE
      __device__
      inline T get_cumulative_normal_distribution(T x) {
          const T A1 = 0.31938153;
          const T A2 = -0.356563782;
          const T A3 = 1.781477937;
          const T A4 = -1.821255978;
          const T A5 = 1.330274429;
          const T RSQRT2PI = 0.39894228040143267793994605993438;
          const T W = 0.2316419;

          const T k = 1 / (1 + W * abs(x));
          T cnd = RSQRT2PI * exp(- x * x / 2) *
              (k * (A1 + k * (A2 + k * (A3 + k * (A4 + k * A5)))));
          if (x > 0) {
              cnd = 1 - cnd;
          }
          return cnd;
      }
    PREAMBLE
  )
end

def assert_allclose(actual, desired, rtol: 1e-7, atol: 0)
  actual = Backend.to_host(actual)
  desired = Backend.to_host(desired)
  mismatch = (actual - desired).abs.gt(atol + rtol * desired.abs).count_true
  raise "Not equal to tolerance rtol=#{rtol}, atol=#{atol}: #{mismatch} / #{actual.size} mismatched" if mismatch.positive?
end

if __FILE__ == $PROGRAM_NAME
  options = { gpu_id: 0, n_options: 10_000_000 }
  OptionParser.new do |parser|
    parser.on('-g', '--gpu-id ID', Integer, 'GPU ID') { |v| options[:gpu_id] = v }
    parser.on('-n', '--n-options N', Integer) { |v| options[:n_options] = v }
  end.parse!

  Cumo::CUDA::Device.new(options[:gpu_id]).use if Backend.gpu?

  rand_range = lambda do |m, mx|
    samples = XM::DFloat.new(options[:n_options]).rand
    m + (mx - m) * samples
  end

  puts 'initializing...'
  stock_price_gpu = rand_range.call(5, 30)
  option_strike_gpu = rand_range.call(1, 100)
  option_years_gpu = rand_range.call(0.25, 10)

  stock_price_cpu = Backend.to_host(stock_price_gpu)
  option_strike_cpu = Backend.to_host(option_strike_gpu)
  option_years_cpu = Backend.to_host(option_years_gpu)

  timer = lambda do |message, &block|
    Backend.synchronize
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    result = block.call
    Backend.synchronize
    finish = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    puts format("%s:\t%f sec", message, finish - start)
    result
  end

  puts 'start computation'
  risk_free = 0.02
  volatility = 0.3
  call_cpu, put_cpu = timer.call(' CPU (Numo, Naive implementation)') do
    black_scholes(Numo, stock_price_cpu, option_strike_cpu, option_years_cpu, risk_free, volatility)
  end

  if Backend.gpu?
    call_gpu1, put_gpu1 = timer.call(' GPU (Cumo, Naive implementation)') do
      black_scholes(Cumo, stock_price_gpu, option_strike_gpu, option_years_gpu, risk_free, volatility)
    end

    call_gpu2, put_gpu2 = timer.call(' GPU (Cumo, Elementwise kernel)') do
      black_scholes_kernel.call(stock_price_gpu, option_strike_gpu, option_years_gpu,
                                risk_free, volatility)
    end

    # Check whether all elements in gpu arrays are equal to those of cpus
    assert_allclose(call_cpu, call_gpu1)
    assert_allclose(call_cpu, call_gpu2)
    assert_allclose(put_cpu, put_gpu1)
    assert_allclose(put_cpu, put_gpu2)
  else
    puts 'Skipping GPU (set GPU=1 to run it on Cumo)'
  end
end
