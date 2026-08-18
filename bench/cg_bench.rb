#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# Conjugate gradient, to price what a host readback costs.
#
#   ruby cg_bench.rb
#   GPU=1 ruby cg_bench.rb                        # the plain way: readbacks every iteration
#   GPU=1 SCALAR=device ruby cg_bench.rb          # keep alpha and beta on the device
#   GPU=1 SCALAR=device CHECK=20 ruby cg_bench.rb # and test convergence rarely
#   GPU=1 OP=dense N=4096 ruby cg_bench.rb        # heavier arithmetic per iteration
#
# CG makes two scalars an iteration, alpha and beta, and multiplies arrays by
# them. Written plainly that reads them back to the host every iteration, and
# the convergence test reads a third, each one waiting for the queue to drain.
# An iteration is not much arithmetic, so the waiting is what shows.
#
#   SCALAR=host   alpha and beta become Ruby Floats, as one would write it
#   SCALAR=device they stay as the 0-dimensional NArray sum answers, and are
#                 never read (Cumo returns 0-dim; on Numo sum is a Float
#                 already, so the setting makes no difference there)
#   CHECK=k       test convergence every k iterations, 0 to never test
#
# A readback does not cost a fixed amount. What it costs is however much work
# is queued when it happens, so it is better read as a cap on how deep the
# pipeline is allowed to get than as a price per call. Several shallow drains
# an iteration cost little each; the last one costs a lot, because until it is
# gone no iteration can overlap the next.
#
# On the host path the floor is two readbacks an iteration whatever CHECK is
# set to, since alpha needs pap and beta needs rs_new as Floats. Keeping the
# scalars on the device is what makes fewer than that possible, and thinning
# the convergence test only pays once they are there.
#
# Correctness is checked against a known solution: x_true is chosen first and
# b = A x_true built from it, so the answer has something to be compared to.
#
# Timings vary by a factor of a few between runs. Take the best of several.
# ---------------------------------------------------------------------------

GPU = !%w[0 false].include?(ENV['GPU'].to_s.downcase) && !ENV['GPU'].to_s.empty?

if GPU
  require 'cumo/narray'
  XM = Cumo
else
  require 'numo/narray'
  XM = Numo
end

OP      = (ENV['OP']      || 'stencil')  # stencil | dense
GRID    = (ENV['GRID']    || 512).to_i   # side of the grid, for stencil
N       = (ENV['N']       || 2048).to_i  # number of unknowns, for dense
MAXITER = (ENV['MAXITER'] || 200).to_i
SCALAR  = (ENV['SCALAR']  || 'host')     # host | device
CHECK   = (ENV['CHECK']   || 1).to_i     # test convergence every CHECK, 0 to never
TOL     = (ENV['TOL']     || 1.0e-6).to_f
DOT     = (ENV['DOT']     || 'sum')      # sum | blas
DETAIL  = ENV['DETAIL'] != '0'
REPEAT  = (ENV['REPEAT']  || 3).to_i     # solve this many times, keep the fastest
SEED    = (ENV['SEED'] || 42).to_i

TIMES = Hash.new(0.0)
READBACKS = [0]

def sync
  XM::CUDA::Runtime.cudaDeviceSynchronize if GPU
end

# A readback, which is where the host waits. Counted, since that is the point.
def scalar(v)
  READBACKS[0] += 1
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

def dotp(a, b)
  if DOT == 'blas'
    a.reshape(a.size).dot(b.reshape(b.size))
  else
    (a * b).sum
  end
end

# --- the operator ----------------------------------------------------------

# The five-point Laplacian, negated to be positive definite. The border stays
# zero and is never touched.
def apply_stencil(p)
  out = XM::SFloat.zeros(GRID, GRID)
  out[1..-2, 1..-2] =
    4.0 * p[1..-2, 1..-2] -
    p[0..-3, 1..-2] - p[2..-1, 1..-2] -
    p[1..-2, 0..-3] - p[1..-2, 2..-1]
  out
end

# A diagonally dominant symmetric matrix. The identity is built with
# arithmetic rather than a fancy index.
def build_dense
  r = XM::SFloat.new(N, N).rand * 2.0 - 1.0
  sym = (r + r.transpose) * 0.5
  i = XM::SFloat.new(N, 1).seq
  j = XM::SFloat.new(1, N).seq
  off = (XM::SFloat.zeros(N, N) + (i - j)).abs.clip(0.0, 1.0).ceil # 0 on the diagonal
  sym + (1.0 - off) * (N * 1.0)
end

# --- setup -----------------------------------------------------------------

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

if OP == 'dense'
  amat = build_dense
  x_true = XM::SFloat.new(N).rand * 2.0 - 1.0
  apply = ->(p) { amat.dot(p) }
  unknowns = N
else
  x_true = XM::SFloat.zeros(GRID, GRID)
  x_true[1..-2, 1..-2] = XM::SFloat.new(GRID - 2, GRID - 2).rand * 2.0 - 1.0
  apply = ->(p) { apply_stencil(p) }
  unknowns = (GRID - 2) * (GRID - 2)
end

b = apply.call(x_true)

puts "ruby      : #{RUBY_VERSION} (#{RUBY_PLATFORM})"
puts "backend   : #{XM} #{version}"
puts "problem   : OP=#{OP} unknowns=#{unknowns} MAXITER=#{MAXITER}"
puts "scalar    : SCALAR=#{SCALAR} CHECK=#{CHECK} DOT=#{DOT}"
puts

# Warm up
warm = XM::SFloat.new(64, 64).seq
warm_sum = (warm * warm).sum
warm_sum = warm_sum.extract if warm_sum.respond_to?(:extract)
sync
GC.start
READBACKS[0] = 0

# --- CG --------------------------------------------------------------------

# Each distinct scalar is read at most once an iteration. Reading rs_old in
# both alpha and beta, and rs_new in both the convergence test and beta, would
# wait twice for a value already in hand.
def solve(apply, b, rs0, shape)
  x = XM::SFloat.zeros(*shape)
  r = b + 0.0 # a copy
  p_dir = r + 0.0
  rs_old = timed(:dot) { dotp(r, r) }
  rs_old_h = SCALAR == 'device' ? nil : rs0
  iters = 0
  converged = nil

  t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  MAXITER.times do |it|
    iters = it + 1

    ap = timed(:apply) { apply.call(p_dir) }
    pap = timed(:dot) { dotp(p_dir, ap) }

    alpha = timed(:alpha) do
      SCALAR == 'device' ? rs_old / pap : rs_old_h / scalar(pap).to_f
    end

    timed(:axpy) do
      x += p_dir * alpha
      r -= ap * alpha
    end

    rs_new = timed(:dot) { dotp(r, r) }
    rs_new_h = SCALAR == 'device' ? nil : scalar(rs_new).to_f

    if CHECK.positive? && (iters % CHECK).zero?
      rel = Math.sqrt((rs_new_h || scalar(rs_new).to_f) / rs0)
      if rel < TOL
        converged = rel
        break
      end
    end

    beta = timed(:beta) do
      SCALAR == 'device' ? rs_new / rs_old : rs_new_h / rs_old_h
    end

    timed(:update_p) { p_dir = r + p_dir * beta }
    rs_old = rs_new
    rs_old_h = rs_new_h
  end
  sync
  { x: x, iters: iters, converged: converged,
    elapsed: Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0 }
end

shape = OP == 'dense' ? [N] : [GRID, GRID]
rs0 = scalar(dotp(b, b)).to_f # the reference for the relative residual, read once

# Solve it a few times and keep the fastest. A single timing here varies by a
# factor of a few, which is enough to invent a trend that is not there.
best = nil
REPEAT.times do
  TIMES.clear
  READBACKS[0] = 0
  run = solve(apply, b, rs0, shape)
  run[:times] = TIMES.dup
  run[:readbacks] = READBACKS[0]
  best = run if best.nil? || run[:elapsed] < best[:elapsed]
end
x = best[:x]
iters = best[:iters]
elapsed = best[:elapsed]
TIMES.replace(best[:times])
READBACKS[0] = best[:readbacks]
puts format('  converged at iter %d, relative residual %.3e', iters, best[:converged]) if best[:converged]

# --- results ---------------------------------------------------------------

res = apply.call(x) - b
rel_res = Math.sqrt(scalar((res * res).sum).to_f / rs0)
err = x - x_true
rel_err = Math.sqrt(scalar((err * err).sum).to_f) /
          Math.sqrt(scalar((x_true * x_true).sum).to_f)

puts format('  iterations     : %d', iters)
puts format('  rel. residual  : %.3e', rel_res)
puts format('  rel. error     : %.3e  (against the known x_true)', rel_err)
puts format('  time           : %.4f s  (%.1f us/iter, best of %d)',
            elapsed, elapsed / iters * 1e6, REPEAT)
puts format('  readbacks      : %d  (%.2f per iteration)',
            READBACKS[0], READBACKS[0].to_f / iters)

if DETAIL
  puts
  measured = TIMES.values.sum
  puts 'breakdown (per iteration):'
  TIMES.sort_by { |_, v| -v }.each do |key, v|
    puts format('  %-10s %9.1f us  %5.1f%%', key, v / iters * 1e6, 100.0 * v / measured)
  end
  puts '  (these sections synchronize, so they add up to more than us/iter above,'
  puts '   and DETAIL=1 is much noisier than DETAIL=0 because of it)'
end

puts
puts '  What to compare:'
puts '    SCALAR=host   CHECK=1   the plain way, several readbacks an iteration'
puts '    SCALAR=device CHECK=1   alpha and beta stay on the device, one readback'
puts '    SCALAR=device CHECK=20  the convergence test thinned out too'
puts '    SCALAR=device CHECK=0   no readback at all inside the loop'
puts '  Thinning the test on its own changes nothing: the host path still reads'
puts '  pap and rs_new every iteration. The two settings only pay together.'
