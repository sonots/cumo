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

def fit(a, b, tol, max_iter)
  # Note that this function works even tensors 'A' and 'b' are Numo or Cumo
  # arrays.
  xm = Backend.array_module(a)
  x = xm::DFloat.zeros(b.shape)
  r0 = b - a.dot(x)
  p_vec = r0
  max_iter.times do
    alpha = r0.inner(r0) / p_vec.inner(a.dot(p_vec))
    x += alpha * p_vec
    r1 = r0 - alpha * a.dot(p_vec)
    return x if Float(xm::NMath.sqrt(r1.inner(r1))) < tol

    beta = r1.inner(r1) / r0.inner(r0)
    p_vec = r1 + beta * p_vec
    r0 = r1
  end
  puts 'Failed to converge. Increase max-iter or tol.'
  x
end

def run(gpu_id, tol, max_iter)
  # Cumo Conjugate gradient example
  #
  # Solve simultaneous linear equations, Ax = b.
  # 'A' and 'x' are created randomly and 'b' is computed by 'Ax' at first.
  # Then, 'x' is computed from 'A' and 'b' in two ways, namely with CPU and
  # GPU. To evaluate the accuracy of computation, the Euclidean distances
  # between the answer 'x' and the reconstructed 'x' are computed.
  3.times do |repeat|
    puts "Trial: #{repeat}"
    # Create the large symmetric matrix 'A'.
    n = 2000
    a = Numo::DFloat.cast(Numo::Int32.new(n, n).rand(-50, 50))
    a = a.dot(a.transpose)
    x_ans = Numo::DFloat.cast(Numo::Int32.new(n).rand(-50, 50))
    b = a.dot(x_ans)

    puts 'Running CPU...'
    x_cpu = timer(' CPU ') { fit(a, b, tol, max_iter) }
    puts Math.sqrt(Float(((x_cpu - x_ans)**2).sum))

    if Backend.gpu?
      Backend.with_device(gpu_id) do
        a_gpu = Backend.to_device(a)
        b_gpu = Backend.to_device(b)
        puts 'Running GPU...'
        x_gpu = timer(' GPU ') { fit(a_gpu, b_gpu, tol, max_iter) }
        puts Math.sqrt(Float(((Backend.to_host(x_gpu) - x_ans)**2).sum))
      end
    else
      puts 'Skipping GPU (set GPU=1 to run it on Cumo)'
    end

    puts
  end
end

if __FILE__ == $PROGRAM_NAME
  options = { gpu_id: 0, tol: 0.1, max_iter: 5000 }
  OptionParser.new do |parser|
    parser.on('-g', '--gpu-id ID', Integer, 'ID of GPU.') { |v| options[:gpu_id] = v }
    parser.on('-t', '--tol TOL', Float, 'tolerance to stop iteration') { |v| options[:tol] = v }
    parser.on('-m', '--max-iter N', Integer, 'number of iterations') { |v| options[:max_iter] = v }
  end.parse!
  run(options[:gpu_id], options[:tol], options[:max_iter])
end
