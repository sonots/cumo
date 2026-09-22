# frozen_string_literal: true

require 'optparse'

require 'cumo/narray'

require_relative 'black_scholes'

# This sample computes call prices for European options with
# Monte-Carlo simulation. It was based on a sample of the financial package
# in CUDA toolkit. For details, please see the corresponding whitepaper.
#
# The present price of an option can also be represented as a discounted
# expectation of the option price under a risk-neutral measure.
# Since it is assumed that the stock price follows a lognormal distribution,
# the call price for European option can be evaluated by approximating the
# risk-neutral expectation at the time of exercise with the Monte-Carlo method.
# Note that as current ElementwiseKernel does not support 'curand'
# due to nvrtc, this sample manually implements a pseudorandom function.

def monte_carlo_kernel
  @monte_carlo_kernel ||= Cumo::CUDA::ElementwiseKernel.new(
    'T s, T x, T t, T r, T v, int32 n_samples, int32 seed', 'T call',
    <<~OPERATION,
      // We can use the special variable i to get the index of the thread.
      // In this case, we used an index as a seed of random sequence.
      uint64_t rand_state[2];
      init_state(rand_state, i, seed);

      T call_sum = 0;
      const T v_by_sqrt_t = v * sqrt(t);
      const T mu_by_t = (r - v * v / 2) * t;

      // compute the price of the call option with Monte Carlo method
      for (int j = 0; j < n_samples; ++j) {
          const T p = sample_normal(rand_state);
          call_sum += get_call_value(s, x, p, mu_by_t, v_by_sqrt_t);
      }
      // convert the future value of the call option to the present value
      const T discount_factor = exp(- r * t);
      call = discount_factor * call_sum / n_samples;
    OPERATION
    'monte_carlo_kernel',
    preamble: <<~PREAMBLE
      typedef unsigned long long uint64_t;

      __device__
      inline T get_call_value(T s, T x, T p, T mu_by_t, T v_by_sqrt_t) {
          const T call_value = s * exp(mu_by_t + v_by_sqrt_t * p) - x;
          return (call_value > 0) ? call_value : 0;
      }

      // Initialize state
      __device__ inline void init_state(uint64_t* a, int i, int seed) {
          a[0] = i + 1;
          a[1] = 0x5c721fd808f616b6 + seed;
      }

      __device__ inline uint64_t xorshift128plus(uint64_t* x) {
          uint64_t s1 = x[0];
          uint64_t s0 = x[1];
          x[0] = s0;
          s1 = s1 ^ (s1 << 23);
          s1 = s1 ^ (s1 >> 17);
          s1 = s1 ^ s0;
          s1 = s1 ^ (s0 >> 26);
          x[1] = s1;
          return s0 + s1;
      }

      // Draw a sample from an uniform distribution in a range of [0, 1]
      __device__ inline T sample_uniform(uint64_t* state) {
          const uint64_t x = xorshift128plus(state);
          // 18446744073709551615 = 2^64 - 1
          return T(x) / T(18446744073709551615);
      }

      // Draw a sample from a normal distribution with N(0, 1)
      __device__ inline T sample_normal(uint64_t* state) {
          T x = sample_uniform(state);
          T s = T(-1.4142135623730950488016887242097);  // = -sqrt(2)
          if (x > 0.5) {
              x = 1 - x;
              s = -s;
          }
          T p = x + T(0.5);
          return s * erfcinv(2 * p);
      }
    PREAMBLE
  )
end

def compute_option_prices(stock_price, option_strike, option_years, risk_free, volatility,
                          n_threads_per_option, n_samples_per_thread, seed = 0)
  n_options = stock_price.size
  call_prices = Cumo::DFloat.new(n_options, n_threads_per_option)
  # Because of the broadcasting rule, in this case this kernel
  # launches n_options * n_threads_per_options threads
  # each of which corresponds to the element of 'call_prices'.
  monte_carlo_kernel.call(
    stock_price[true, :new], option_strike[true, :new], option_years[true, :new],
    risk_free, volatility, n_samples_per_thread, seed, call_prices
  )
  call_prices.mean(axis: 1)
end

def main
  options = { gpu_id: 0, n_options: 1000, n_samples_per_thread: 1000, n_threads_per_option: 100_000 }
  OptionParser.new do |parser|
    parser.on('-g', '--gpu-id ID', Integer, 'GPU ID') { |v| options[:gpu_id] = v }
    parser.on('--n-options N', Integer) { |v| options[:n_options] = v }
    parser.on('--n-samples-per-thread N', Integer) { |v| options[:n_samples_per_thread] = v }
    parser.on('--n-threads-per-option N', Integer) { |v| options[:n_threads_per_option] = v }
  end.parse!

  Cumo::CUDA::Device.new(options[:gpu_id]).use

  rand_range = lambda do |m, mx|
    samples = Cumo::DFloat.new(options[:n_options]).rand
    m + (mx - m) * samples
  end

  puts 'initializing...'
  stock_price = rand_range.call(5, 30)
  option_strike = rand_range.call(1, 100)
  option_years = rand_range.call(0.25, 10)
  risk_free = 0.02
  volatility = 0.3

  timer = lambda do |message, &block|
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    result = block.call
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    finish = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    puts format("%s:\t%f sec", message, finish - start)
    result
  end

  puts 'start computation'
  puts "    # of options: #{options[:n_options]}"
  puts "    # of samples per option: #{options[:n_samples_per_thread] * options[:n_threads_per_option]}"
  call_mc = timer.call('GPU (Cumo, Monte Carlo method)') do
    compute_option_prices(stock_price, option_strike, option_years, risk_free, volatility,
                          options[:n_threads_per_option], options[:n_samples_per_thread])
  end

  # Compute the error between the value of the exact solution
  # and that of the Monte-Carlo simulation
  call_bs, = black_scholes_kernel.call(stock_price, option_strike, option_years, risk_free, volatility)
  error = (call_mc - call_bs).stddev
  puts format('Error: %f', Float(error))
  0
end

exit(main) if __FILE__ == $PROGRAM_NAME
