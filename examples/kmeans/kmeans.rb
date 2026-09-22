# frozen_string_literal: true

require 'optparse'

require_relative '../backend'

def timer(message)
  Backend.synchronize
  start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  result = yield
  Backend.synchronize
  finish = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  puts format('%s:  %f sec', message, finish - start)
  result
end

def fit_xp(x, n_clusters, max_iter)
  raise ArgumentError, 'X must be 2-dimensional' unless x.ndim == 2

  # Get Numo or Cumo module from the supplied array.
  xm = Backend.array_module(x)

  n_samples = x.shape[0]

  # Make an array to store the labels indicating which cluster each sample is
  # contained.
  pred = xm::Int32.zeros(n_samples)

  # Choose the initial centroid for each cluster.
  # numpy's fancy indexing copies. Numo answers a view here, and numo-narray-alt
  # 0.11.2 gets a broadcast against such a view wrong, so copy explicitly.
  initial_indexes = (0...n_samples).to_a.sample(n_clusters)
  centers = x[initial_indexes, true].dup

  max_iter.times do
    # Compute the new label for each sample.
    distances = xm::NMath.sqrt(((x[true, :new, true] - centers[:new, true, true])**2).sum(axis: 2))
    new_pred = distances.argmin(axis: 1)

    # If the label is not changed for each sample, we suppose the
    # algorithm has converged and exit from the loop.
    break if Integer(new_pred.eq(pred).count_false).zero?

    pred = new_pred

    # Compute the new centroid for each cluster.
    i = xm::Int32.new(n_clusters).seq
    mask = pred[:new, true].eq(i[true, :new])
    sums = (xm::DFloat.cast(mask)[true, true, :new] * x[:new, true, true]).sum(axis: 1)
    counts = mask.count_true(axis: 1).reshape(n_clusters, 1)
    centers = sums / counts
  end

  [centers, pred]
end


def kernels
  @kernels ||= [
    Cumo::CUDA::ElementwiseKernel.new(
      'T x0, T x1, T c0, T c1', 'T out',
      'out = (x0 - c0) * (x0 - c0) + (x1 - c1) * (x1 - c1)',
      'var_kernel'
    ),
    Cumo::CUDA::ReductionKernel.new(
      'T x, S mask', 'T out',
      'mask ? x : 0',
      'a + b', 'out = a', '0',
      'sum_kernel'
    ),
    Cumo::CUDA::ReductionKernel.new(
      'T mask', 'float32 out',
      'mask ? 1.0 : 0.0',
      'a + b', 'out = a', '0.0',
      'count_kernel'
    )
  ]
end

def fit_custom(x, n_clusters, max_iter)
  raise ArgumentError, 'X must be 2-dimensional' unless x.ndim == 2

  var_kernel, sum_kernel, count_kernel = kernels

  n_samples = x.shape[0]

  pred = Cumo::Int32.zeros(n_samples)

  initial_indexes = (0...n_samples).to_a.sample(n_clusters)
  centers = x[initial_indexes, true].dup

  max_iter.times do
    distances = var_kernel.call(x[true, :new, 0], x[true, :new, 1],
                                centers[:new, true, 1], centers[:new, true, 0])
    new_pred = distances.argmin(axis: 1)
    break if Integer(new_pred.eq(pred).count_false).zero?

    pred = new_pred

    i = Cumo::Int32.new(n_clusters).seq
    # A Cumo::Bit cannot be handed to one of these kernels, so the mask is a
    # byte per element, which is what CuPy's bool array is too.
    mask = Cumo::UInt8.cast(pred[:new, true].eq(i[true, :new]))
    sums = sum_kernel.call(x, mask[true, true, :new], axis: 1)
    counts = count_kernel.call(mask, axis: 1).reshape(n_clusters, 1)
    centers = sums / counts
  end

  [centers, pred]
end

def draw(_x, _n_clusters, _centers, _pred, output)
  # Plot the samples and centroids of the fitted clusters into an image file.
  warn "--output-image #{output}: drawing needs matplotlib, which has no Cumo counterpart. Skipped."
end

def run(gpuid, n_clusters, num, max_iter, use_custom_kernel, output)
  samples = Numo::DFloat.new(num, 2).rand_norm
  x_train = (samples + 1).concatenate(samples - 1, axis: 0)

  centers, pred = timer(' CPU ') { fit_xp(x_train, n_clusters, max_iter) }

  unless Backend.gpu?
    puts 'Skipping GPU (set GPU=1 to run it on Cumo)'
    return
  end

  Backend.with_device(gpuid) do
    x_train = Backend.to_device(x_train)

    centers, pred = timer(' GPU ') do
      if use_custom_kernel
        fit_custom(x_train, n_clusters, max_iter)
      else
        fit_xp(x_train, n_clusters, max_iter)
      end
    end

    unless output.nil?
      index = (0...x_train.shape[0]).to_a.sample(300)
      draw(Backend.to_host(x_train[index, true]), n_clusters, Backend.to_host(centers),
           Backend.to_host(pred[index]), output)
    end
  end
end

if __FILE__ == $PROGRAM_NAME
  options = { gpu_id: 0, n_clusters: 2, num: 5_000_000, max_iter: 10, use_custom_kernel: false, output: nil }
  OptionParser.new do |parser|
    parser.on('-g', '--gpu-id ID', Integer, 'ID of GPU.') { |v| options[:gpu_id] = v }
    parser.on('-n', '--n-clusters N', Integer, 'number of clusters') { |v| options[:n_clusters] = v }
    parser.on('--num NUM', Integer, 'number of samples') { |v| options[:num] = v }
    parser.on('-m', '--max-iter N', Integer, 'number of iterations') { |v| options[:max_iter] = v }
    parser.on('--use-custom-kernel', 'use Elementwise kernel') { options[:use_custom_kernel] = true }
    parser.on('-o', '--output-image FILE', String, 'output image file name') { |v| options[:output] = v }
  end.parse!
  run(options[:gpu_id], options[:n_clusters], options[:num], options[:max_iter],
      options[:use_custom_kernel], options[:output])
end
