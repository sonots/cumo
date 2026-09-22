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

def estimate_log_prob(x, inv_cov, means)
  xm = Backend.array_module(x)
  n_features = x.shape[1]
  log_det = xm::NMath.log(inv_cov).sum(axis: 1)
  precisions = inv_cov**2
  log_prob = (means**2 * precisions).sum(axis: 1) -
             2 * x.dot((means * precisions).transpose) + (x**2).dot(precisions.transpose)
  -0.5 * (n_features * Math.log(2 * Math::PI) + log_prob) + log_det
end

def m_step(x, resp)
  nk = resp.sum(axis: 0)
  means = resp.transpose.dot(x) / nk[true, :new]
  x2 = resp.transpose.dot(x * x) / nk[true, :new]
  covariances = x2 - means**2
  [nk / x.shape[0], means, covariances]
end

def e_step(x, inv_cov, means, weights)
  xm = Backend.array_module(x)
  weighted_log_prob = estimate_log_prob(x, inv_cov, means) + xm::NMath.log(weights)
  log_prob_norm = xm::NMath.log(xm::NMath.exp(weighted_log_prob).sum(axis: 1))
  log_resp = weighted_log_prob - log_prob_norm[true, :new]
  [log_prob_norm.mean, log_resp]
end

def train_gmm(x, max_iter, tol, means, covariances)
  xm = Backend.array_module(x)
  lower_bound = -Float::INFINITY
  converged = false
  weights = xm::SFloat[0.5, 0.5]
  inv_cov = 1.0 / xm::NMath.sqrt(covariances)

  max_iter.times do
    prev_lower_bound = lower_bound
    log_prob_norm, log_resp = e_step(x, inv_cov, means, weights)
    weights, means, covariances = m_step(x, xm::NMath.exp(log_resp))
    inv_cov = 1.0 / xm::NMath.sqrt(covariances)
    lower_bound = Float(log_prob_norm)
    change = lower_bound - prev_lower_bound
    if change.abs < tol
      converged = true
      break
    end
  end

  puts 'Failed to converge. Increase max-iter or tol.' unless converged

  [inv_cov, means, weights, covariances]
end

def predict(x, inv_cov, means, weights)
  xm = Backend.array_module(x)
  log_prob = estimate_log_prob(x, inv_cov, means)
  (log_prob + xm::NMath.log(weights)).argmax(axis: 1)
end

def calc_acc(x_train, y_train, x_test, y_test, max_iter, tol, means, covariances)
  inv_cov, means, weights, cov = train_gmm(x_train, max_iter, tol, means, covariances)
  y_train_pred = predict(x_train, inv_cov, means, weights)
  train_accuracy = Float(y_train_pred.eq(y_train).count_true) / y_train.size * 100
  y_test_pred = predict(x_test, inv_cov, means, weights)
  test_accuracy = Float(y_test_pred.eq(y_test).count_true) / y_test.size * 100
  puts format('train_accuracy : %f', train_accuracy)
  puts format('test_accuracy : %f', test_accuracy)
  [y_test_pred, means, cov]
end

def draw(_x, _pred, _means, _covariances, output)
  warn "--output-image #{output}: drawing needs matplotlib and scipy.stats, which have no Cumo counterpart. Skipped."
end

def run(gpuid, num, dim, max_iter, tol, output)
  # Cumo Gaussian Mixture Model example
  #
  # Compute GMM parameters, weights, means and covariance matrix, depending on
  # sampled data. There are two main components, e_step and m_step.
  # In e_step, compute burden rate, which is expressed `resp`, by latest
  # weights, means and covariance matrix.
  # In m_step, compute weights, means and covariance matrix by latest `resp`.
  scale = 1.0
  train1 = Numo::SFloat.new(num, dim).rand_norm(1, scale)
  train2 = Numo::SFloat.new(num, dim).rand_norm(-1, scale)
  x_train = train1.concatenate(train2, axis: 0)
  test1 = Numo::SFloat.new(100, dim).rand_norm(1, scale)
  test2 = Numo::SFloat.new(100, dim).rand_norm(-1, scale)
  x_test = test1.concatenate(test2, axis: 0)
  y_train = Numo::Int32.zeros(num).concatenate(Numo::Int32.ones(num))
  y_test = Numo::Int32.zeros(100).concatenate(Numo::Int32.ones(100))

  mean1 = Numo::DFloat.new(dim).rand_norm(1, scale)
  mean2 = Numo::DFloat.new(dim).rand_norm(-1, scale)
  means = Numo::DFloat.vstack([mean1, mean2])
  covariances = Numo::DFloat.new(2, dim).rand
  puts 'Running CPU...'
  y_test_pred, means, cov = timer(' CPU ') do
    calc_acc(x_train, y_train, x_test, y_test, max_iter, tol, means, covariances)
  end

  unless Backend.gpu?
    puts 'Skipping GPU (set GPU=1 to run it on Cumo)'
    return
  end

  Backend.with_device(gpuid) do
    x_train_gpu = Backend.to_device(x_train)
    y_train_gpu = Backend.to_device(y_train)
    y_test_gpu = Backend.to_device(y_test)
    x_test_gpu = Backend.to_device(x_test)
    means = Backend.to_device(means)
    covariances = Backend.to_device(covariances)
    puts 'Running GPU...'
    y_test_pred, means, cov = timer(' GPU ') do
      calc_acc(x_train_gpu, y_train_gpu, x_test_gpu, y_test_gpu, max_iter, tol, means, covariances)
    end
    draw(x_test_gpu, y_test_pred, means, cov, output) unless output.nil?
  end
end

if __FILE__ == $PROGRAM_NAME
  options = { gpu_id: 0, num: 500_000, dim: 2, max_iter: 30, tol: 1e-3, output: nil }
  OptionParser.new do |parser|
    parser.on('-g', '--gpu-id ID', Integer, 'ID of GPU.') { |v| options[:gpu_id] = v }
    parser.on('-n', '--num NUM', Integer, 'number of train data') { |v| options[:num] = v }
    parser.on('-d', '--dim DIM', Integer, 'dimension of each data') { |v| options[:dim] = v }
    parser.on('-m', '--max-iter N', Integer, 'number of iterations') { |v| options[:max_iter] = v }
    parser.on('-t', '--tol TOL', Float, 'error tolerance to stop iterations') { |v| options[:tol] = v }
    parser.on('-o', '--output-image FILE', String, 'output image file name') { |v| options[:output] = v }
  end.parse!
  run(options[:gpu_id], options[:num], options[:dim], options[:max_iter], options[:tol], options[:output])
end
