#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# The 2D Ising model by Metropolis, on Numo or on Cumo.
#
#   ruby ising_bench.rb
#   GPU=1 L=2048 SWEEPS=200 ruby ising_bench.rb
#   GPU=1 COMMIT=mask ruby ising_bench.rb     # commit with a mask instead
#   GPU=1 PGM=1 ruby ising_bench.rb           # write each final lattice as a PGM
#   TEMPS=1.5,2.269,3.0 ruby ising_bench.rb
#
# A checkerboard update touches every other row and column, so a strided view
# is not incidental here: it is where the work happens.
#
# COMMIT=strided writes back through that view (the default).
# COMMIT=mask multiplies the whole lattice by a mask of ones and zeros and
# never touches a non-contiguous operand. The two solve the same physics, so
# the difference between them is what the strided write-back costs.
#
# Correctness is checked by the physics. The 2D Ising model orders below
# Tc = 2/ln(1+sqrt(2)) = 2.269 and disorders above it, so a magnetization that
# falls across that temperature means the update rule is intact.
# ---------------------------------------------------------------------------

GPU = !%w[0 false].include?(ENV['GPU'].to_s.downcase) && !ENV['GPU'].to_s.empty?

if GPU
  require 'cumo/narray'
  XM = Cumo
else
  require 'numo/narray'
  XM = Numo
end

L       = (ENV['L']       || 1024).to_i
SWEEPS  = (ENV['SWEEPS']  || 100).to_i
WARMUP  = (ENV['WARMUP']  || 50).to_i
COMMIT  = (ENV['COMMIT']  || 'strided')
DETAIL  = ENV['DETAIL'] != '0'
PGM     = ENV['PGM'] == '1'
SEED    = (ENV['SEED'] || 42).to_i
TEMPS   = (ENV['TEMPS'] || '1.5,2.0,2.269,2.6,3.2').split(',').map(&:to_f)

raise "L(#{L}) has to be even" unless L.even?

TC = 2.0 / Math.log(1.0 + Math.sqrt(2.0))
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

# --- picking out a sublattice ----------------------------------------------
# Every other row and column, as a step Range so that the result is a view and
# the assignment reaches the lattice. Reshaping to [L/2, 2, L/2, 2] and pinning
# the middle axes would read the same elements, but reshape answers a copy in
# both Numo and Cumo, and the write would go nowhere.
# (r, c) of (0,0) and (1,1) is sublattice A, (0,1) and (1,0) is B.

SUBLATTICE = { a: [[0, 0], [1, 1]], b: [[0, 1], [1, 0]] }.freeze

def commit_strided(s, snew, which)
  SUBLATTICE[which].each do |r, c|
    rows = (r...L).step(2)
    cols = (c...L).step(2)
    s[rows, cols] = snew[rows, cols] # writing back through a strided view
  end
  s
end

# The other route: combine the whole lattice through a mask of ones and zeros,
# touching nothing non-contiguous.
def commit_mask(s, snew, mask)
  snew * mask + s * (1.0 - mask)
end

# 1.0 where (i + j) is even, built with arithmetic alone
def checker_mask
  i = XM::SFloat.new(L, 1).seq
  j = XM::SFloat.new(1, L).seq
  t = (XM::SFloat.zeros(L, L) + i + j) * 0.5
  1.0 - ((t - t.floor) * 2.0)
end

# --- the neighbour sum, with periodic boundaries ----------------------------
# Copy the opposite edges into a one-cell border, then add the four shifted
# slices.

def neighbors(s, pad)
  timed(:halo) do
    pad[1..-2, 1..-2] = s
    pad[0, 1..-2]     = s[-1, true]
    pad[-1, 1..-2]    = s[0, true]
    pad[1..-2, 0]     = s[true, -1]
    pad[1..-2, -1]    = s[true, 0]
  end
  timed(:neighbor) do
    pad[0..-3, 1..-2] + pad[2..-1, 1..-2] + pad[1..-2, 0..-3] + pad[1..-2, 2..-1]
  end
end

# --- one Metropolis step, over one sublattice -------------------------------
# dE = 2 * s * (neighbour sum), and the spin flips when r < exp(-dE/T).
# The accept flag is built from clip and ceil rather than from a Bit.

def half_step(s, pad, mask, which, temp)
  nb = neighbors(s, pad)
  de = timed(:dE) { s * nb * 2.0 }
  pacc = timed(:exp) { XM::NMath.exp((de / -temp).clip(-30.0, 30.0)) }
  rnd = timed(:rand) { XM::SFloat.new(L, L).rand }
  acc = timed(:accept) { (pacc - rnd).clip(0.0, 1.0).ceil } # 1.0 where accepted
  snew = timed(:flip) { s * (1.0 - 2.0 * acc) }

  timed(:commit) do
    if COMMIT == 'mask'
      m = which == :a ? mask : (1.0 - mask)
      commit_mask(s, snew, m)
    else
      commit_strided(s, snew, which)
    end
  end
end

def sweep(s, pad, mask, temp)
  s = half_step(s, pad, mask, :a, temp)
  half_step(s, pad, mask, :b, temp)
end

# --- observables -----------------------------------------------------------

def magnetization(s)
  scalar(s.sum).to_f.abs / (L * L)
end

def energy(s, pad)
  nb = neighbors(s, pad)
  # each bond is counted twice, hence the half
  -scalar((s * nb).sum).to_f / 2.0 / (L * L)
end

def write_pgm(path, s)
  img = XM::UInt8.cast((s + 1.0) * 127.5)
  h, w = img.shape
  bytes = begin
    img.to_binary
  rescue StandardError
    img.to_a.flatten.pack('C*')
  end
  File.open(path, 'wb') do |f|
    f.write("P5\n#{w} #{h}\n255\n")
    f.write(bytes)
  end
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
puts "params    : L=#{L} (#{L * L} sites) SWEEPS=#{SWEEPS} WARMUP=#{WARMUP} " \
     "COMMIT=#{COMMIT} DETAIL=#{DETAIL ? 1 : 0}"
puts format('Tc        : %.4f (exact, for the 2D Ising model)', TC)
puts

mask = checker_mask
pad = XM::SFloat.zeros(L + 2, L + 2)

# Warm up
warm = XM::SFloat.new(64, 64).seq
scalar((warm * warm).sum)
sync
GC.start

puts format('  %8s %10s %10s %10s %10s', 'T', '|m|', 'E/site', 'sec/sweep', 'GLUPS')
puts "  #{'-' * 54}"

TEMPS.each do |temp|
  # Start from all +1, so the ordered phase below Tc is not destroyed first
  s = XM::SFloat.ones(L, L)
  WARMUP.times { s = sweep(s, pad, mask, temp) }
  sync
  TIMES.clear

  t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  m_sum = 0.0
  SWEEPS.times do
    s = sweep(s, pad, mask, temp)
    m_sum += magnetization(s)
  end
  sync
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0

  glups = (L.to_f * L * SWEEPS) / elapsed / 1e9
  puts format('  %8.3f %10.4f %10.4f %10.4f %10.3f',
              temp, m_sum / SWEEPS, energy(s, pad), elapsed / SWEEPS, glups)

  write_pgm(format('ising_T%.3f.pgm', temp), s) if PGM
end

if DETAIL
  puts
  measured = TIMES.values.sum
  puts 'breakdown (last temperature, per sweep):'
  TIMES.sort_by { |_, v| -v }.each do |key, v|
    puts format('  %-10s %8.4f s  %5.1f%%', key, v / SWEEPS, 100.0 * v / measured)
  end
  puts '  (these sections synchronize, so they add up to more than sec/sweep above)'
end

puts
puts format('  |m| falling from near 1 to near 0 across Tc = %.3f means the update', TC)
puts '  rule is intact. A magnetization that stays at 1 whatever the temperature'
puts '  points at the accept test or at a write-back that goes nowhere; one that'
puts '  is 0 everywhere points at how the sublattice is picked out.'
