# frozen_string_literal: true

require 'optparse'

require 'numo/narray'
require 'cumo/narray'

require_relative 'black_scholes'
require_relative 'monte_carlo'

# Cumo also implements a feature to call kernels in different GPUs.
# Through this sample, we will explain how to allocate arrays
# in different devices, and call kernels in parallel.

if __FILE__ == $PROGRAM_NAME
  options = { gpus: [0], n_options: 1000, n_samples_per_thread: 1000, n_threads_per_option: 10_000 }
  OptionParser.new do |parser|
    parser.on('--gpus IDS', Array, 'GPU IDs, comma separated') { |v| options[:gpus] = v.map { |i| Integer(i) } }
    parser.on('--n-options N', Integer) { |v| options[:n_options] = v }
    parser.on('--n-samples-per-thread N', Integer) { |v| options[:n_samples_per_thread] = v }
    parser.on('--n-threads-per-option N', Integer) { |v| options[:n_threads_per_option] = v }
  end.parse!

  gpus = options[:gpus]
  if gpus.empty?
    puts 'At least one GPU is required.'
    exit 1
  end

  rand_range = lambda do |m, mx|
    samples = Numo::DFloat.new(options[:n_options]).rand
    m + (mx - m) * samples
  end

  puts 'initializing...'
  stock_price_cpu = rand_range.call(5, 30)
  option_strike_cpu = rand_range.call(1, 100)
  option_years_cpu = rand_range.call(0.25, 10)
  risk_free = 0.02
  volatility = 0.3

  stock_price_gpus = []
  option_strike_gpus = []
  option_years_gpus = []
  call_prices_gpus = []
  puts 'start computation'
  puts "    # of gpus: #{gpus.size}"
  puts "    # of options: #{options[:n_options]}"
  puts "    # of samples per option: #{gpus.size * options[:n_samples_per_thread] * options[:n_threads_per_option]}"
  # Allocate arrays in different devices
  gpus.each do |gpu_id|
    Cumo::CUDA::Device.new(gpu_id).with do
      stock_price_gpus << Cumo::DFloat.cast(stock_price_cpu)
      option_strike_gpus << Cumo::DFloat.cast(option_strike_cpu)
      option_years_gpus << Cumo::DFloat.cast(option_years_cpu)
      call_prices_gpus << Cumo::DFloat.new(options[:n_options], options[:n_threads_per_option])
    end
  end

  timer = lambda do |message, &block|
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    result = block.call
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    finish = Process.clock_gettime(Process::CLOCK_MONOTONIC)
    puts format("%s:\t%f sec", message, finish - start)
    result
  end

  timer.call('GPU (Cumo, Monte Carlo method)') do
    gpus.each_with_index do |gpu_id, i|
      # Performs Monte-Carlo simulations in parallel
      Cumo::CUDA::Device.new(gpu_id).with do
        monte_carlo_kernel.call(
          stock_price_gpus[i][true, :new],
          option_strike_gpus[i][true, :new],
          option_years_gpus[i][true, :new],
          risk_free, volatility, options[:n_samples_per_thread], i,
          call_prices_gpus[i]
        )
      end
    end
  end

  # Transfer the result from the GPUs
  call_prices = call_prices_gpus.map { |c| Numo::DFloat.cast(c) }
  call_mc = call_prices.reduce { |a, b| a.concatenate(b, axis: 0) }
                       .reshape(gpus.size, options[:n_options], options[:n_threads_per_option])
  call_mc = call_mc.mean(axis: [0, 2])
  # Compute the error between the value of the exact solution
  # and that of the Monte-Carlo simulation
  call_bs = Cumo::CUDA::Device.new(gpus[0]).with do
    Numo::DFloat.cast(black_scholes_kernel.call(stock_price_gpus[0], option_strike_gpus[0],
                                                option_years_gpus[0], risk_free, volatility)[0])
  end
  error = (call_mc - call_bs).stddev
  puts format('Error: %f', error)
end
