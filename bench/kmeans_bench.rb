#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# k-means, on Numo or on Cumo.
#
#   ruby kmeans_bench.rb
#   GPU=1 ruby kmeans_bench.rb
#   GPU=1 N=1000000 D=128 K=64 ITERS=30 ruby kmeans_bench.rb
#
#   UPDATE=gather ruby kmeans_bench.rb   # move the centroids by gathering (default)
#   UPDATE=onehot ruby kmeans_bench.rb   # move them with a one-hot GEMM instead
#   DETAIL=0 ruby kmeans_bench.rb        # drop the per-section synchronization
#
# Three paths known to be weak are taken on purpose:
#   1. min_index(axis: 1)      an axis reduction with many outputs
#   2. x[idx, true]            a gather, which reads the indices back to the host
#   3. C.transpose into a GEMM a non-contiguous operand
#
# gather and onehot solve the same problem by different routes. The first pays
# K synchronizations an iteration, the second one [K, N] GEMM. Which of them
# wins is the same question as whether to replace a gather of embeddings with a
# one-hot matrix multiply.
#
# Correctness is checked by the inertia, the total squared distance from each
# point to its nearest centroid. k-means can only lower it, so a rise means
# something is broken.
#
# The inertia is not comparable between Numo and Cumo: the two draw different
# random data, Cumo having given up matching srand in #223. Compare a backend
# against itself.
# ---------------------------------------------------------------------------

GPU = !%w[0 false].include?(ENV['GPU'].to_s.downcase) && !ENV['GPU'].to_s.empty?

if GPU
  require 'cumo/narray'
  XM = Cumo
else
  require 'numo/narray'
  XM = Numo
end

N      = (ENV['N']      || 100_000).to_i
D      = (ENV['D']      || 64).to_i
K      = (ENV['K']      || 16).to_i
ITERS  = (ENV['ITERS']  || 20).to_i
UPDATE = (ENV['UPDATE'] || 'gather')
DETAIL = ENV['DETAIL'] != '0'
SEED   = (ENV['SEED']   || 42).to_i

raise "N(#{N}) has to divide by K(#{K})" unless (N % K).zero?

TIMES = Hash.new(0.0)

def sync
  XM::CUDA::Runtime.cudaDeviceSynchronize if GPU
end

def scalar(v)
  return v if v.is_a?(Numeric)

  v.respond_to?(:extract_cpu) ? v.extract_cpu : v.extract
end

def timed(key)
  return yield unless DETAIL

  t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  r = yield
  sync
  TIMES[key] += Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
  r
end

def contiguous(a)
  out = XM::SFloat.zeros(*a.shape)
  out.store(a)
  out
end

# --- the data --------------------------------------------------------------
# Points scattered around K true centers, a block at a time so that filling
# them in is a row slice assignment.

def build_data
  x = XM::SFloat.zeros(N, D)
  per = N / K
  spread = 0.35
  K.times do |j|
    center = XM::SFloat.new(1, D).rand * 2.0 - 1.0
    noise = (XM::SFloat.new(per, D).rand * 2.0 - 1.0) * spread
    x[(j * per)...((j + 1) * per), true] = noise + center
  end
  x
end

# Take the initial centroids evenly spread, since the data is laid out in
# blocks and the first K rows would all come from one cluster.
def initial_centroids(x)
  step = N / K
  view = x[(0...N).step(step), true]
  contiguous(view[0...K, true])
rescue StandardError => e
  warn "  ! strided extraction failed, falling back to the first K rows (#{e.class})"
  contiguous(x[0...K, true])
end

# --- assignment ------------------------------------------------------------
# ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2. The first term is constant along a
# row and cannot change the argmin, so the distance matrix leaves it out and
# the inertia adds it back.

def assign(x, c)
  ct = timed(:transpose) { contiguous(c.transpose) }        # [D, K]
  d = timed(:gemm) { x.dot(ct) * -2.0 }                     # [N, K]
  cnorm = timed(:cnorm) { (c * c).sum(axis: 1).reshape(1, K) }
  d = timed(:add_cnorm) { d + cnorm }
  d
end

# min_index answers a flat index into the whole array, unlike numpy's argmin.
# Subtracting the start of each row turns it into a label in 0...K.
def labels_from(d)
  idx = timed(:min_index) { d.min_index(axis: 1) }
  timed(:label_fixup) do
    offset = idx.class.new(N).seq * K
    idx - offset
  end
end

# --- moving the centroids, by gathering ------------------------------------
# One where per cluster to find its members, then a gather and an average.
# Both the where and the gather read something back to the host.

def update_gather(x, labels, cold)
  cnew = XM::SFloat.zeros(K, D)
  K.times do |j|
    idx = timed(:where) { labels.eq(j).where }
    if idx.size.zero?
      cnew[j, true] = cold[j, true] # an empty cluster stays where it is
      next
    end
    pts = timed(:gather) { x[idx, true] }
    cnew[j, true] = timed(:cluster_mean) { pts.mean(axis: 0) }
  end
  cnew
end

# --- moving the centroids, with a one-hot GEMM -----------------------------
# The [K, N] matrix of ones and zeros is built with arithmetic alone, using
# neither Bit nor an index array.

def update_onehot(x, labels, cold)
  onehot = timed(:onehot_build) do
    kidx = labels.class.new(K, 1).seq                        # [K, 1]
    diff = kidx - labels.reshape(1, N)                       # [K, N]
    1.0 - XM::SFloat.cast(diff.abs).clip(0.0, 1.0).ceil      # 1.0 where they match
  end
  sums = timed(:onehot_gemm) { onehot.dot(x) }               # [K, D]
  counts = timed(:onehot_count) { onehot.sum(axis: 1).reshape(K, 1) }
  timed(:onehot_div) do
    alive = counts.clip(0.0, 1.0).ceil                       # 0.0 for an empty cluster
    denom = counts.clip(1.0, 1.0e30)
    sums / denom * alive + cold * (1.0 - alive)
  end
end

# --- inertia ---------------------------------------------------------------

def inertia(d, xnorm)
  dmin = timed(:min) { d.min(axis: 1) }
  scalar(timed(:inertia_sum) { (dmin + xnorm).sum }).to_f
end

# --- main ------------------------------------------------------------------

version = begin
  XM::NArray::VERSION
rescue StandardError
  'unknown'
end

begin
  XM::NArray.srand(SEED)
rescue StandardError
  nil
end

puts "ruby      : #{RUBY_VERSION} (#{RUBY_PLATFORM})"
puts "backend   : #{XM} #{version}"
puts "params    : N=#{N} D=#{D} K=#{K} ITERS=#{ITERS} UPDATE=#{UPDATE} " \
     "DETAIL=#{DETAIL ? 1 : 0}"
puts

x = build_data
c = initial_centroids(x)
xnorm = (x * x).sum(axis: 1)
sync

# Warm up. One iteration of the real thing, so that the first touch of every
# page and the first launch of every kernel stay out of the measurement: the
# timings below are per iteration and one cold one would skew them.
warm = XM::SFloat.new(64, 64).seq
scalar(warm.dot(warm).sum)
inertia(assign(x, c), xnorm)
sync
TIMES.clear
GC.start

t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
prev = nil
ITERS.times do |it|
  d = assign(x, c)
  labels = labels_from(d)
  obj = inertia(d, xnorm)
  c = if UPDATE == 'onehot'
        update_onehot(x, labels, c)
      else
        update_gather(x, labels, c)
      end

  flag = if prev.nil?
           ''
         elsif obj > prev * (1.0 + 1.0e-6)
           '  <== rose, which k-means cannot do'
         else
           ''
         end
  puts format('  iter %3d  inertia %.6e%s', it + 1, obj, flag)
  prev = obj
end
sync
total = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0

puts
if DETAIL
  measured = TIMES.values.sum
  puts 'breakdown (per iteration):'
  TIMES.sort_by { |_, v| -v }.each do |key, v|
    puts format('  %-14s %8.4f s  %5.1f%%', key, v / ITERS, 100.0 * v / measured)
  end
  puts '  (these sections synchronize, so they add up to more than the total below)'
  puts
end

gemm_flops = 2.0 * N * D * K
gemm_flops += 2.0 * K * N * D if UPDATE == 'onehot'

# The total includes reading the inertia back each iteration, which the
# correctness check needs; it is not pure compute time.
puts format('total     : %.4f s  (%.4f s / iter)', total, total / ITERS)
puts format('distance  : %.2f MFLOP / iter', gemm_flops / 1e6)
puts format('assign    : %d points x %d dimensions x %d clusters', N, D, K)
