#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# A particle simulation, to price the layout you reach for first.
#
#   ruby particle_bench.rb
#   GPU=1 ruby particle_bench.rb                    # AoS, the default
#   GPU=1 LAYOUT=soa ruby particle_bench.rb         # SoA
#   GPU=1 N=1000000 STEPS=100 ruby particle_bench.rb
#
# AoS keeps the positions in one [N, 3] array and reaches a component through a
# column slice, p[true, 0]. That is what anyone writes first, and a column
# slice is not contiguous, so every sqrt, clip and ceil applied to it is a
# non-contiguous elementwise operation. SoA keeps px, py and pz as separate
# contiguous arrays, so the same physics touches contiguous memory throughout.
#
# The two do the same operations in the same order, so
#   the difference in time    is what the layout costs,
#   a difference in the answer is a bug.
#
# The physics is a damped vortex field in a box, reflecting off the walls. The
# reflection is built out of clip and ceil rather than comparison operators, so
# the loop stays on the elementwise path and never produces a Bit array.
#
# The initial state is a fixed formula rather than random numbers, so Numo and
# Cumo start from the same values and can be compared with each other. They
# agree to about six significant figures; the rest is that single-precision sin
# and the order a sum is accumulated in are not the same on the two.
#
# Timings vary by a factor of a few between runs, and the first run after the
# GPU has been idle reads far slower than the rest because the clocks are still
# ramping. Hence REPEAT, and hence the best rather than the mean.
# ---------------------------------------------------------------------------

GPU = !%w[0 false].include?(ENV['GPU'].to_s.downcase) && !ENV['GPU'].to_s.empty?

if GPU
  require 'cumo/narray'
  XM = Cumo
else
  require 'numo/narray'
  XM = Numo
end

N      = (ENV['N']      || 200_000).to_i
STEPS  = (ENV['STEPS']  || 50).to_i
LAYOUT = (ENV['LAYOUT'] || 'aos')
REPEAT = (ENV['REPEAT'] || 3).to_i
# Off by default: the breakdown synchronizes at every section, which serializes
# the whole loop and makes the totals above it mean something else.
DETAIL = ENV['DETAIL'] == '1'

DT    = 0.005
GRAV  = 9.8
DRAG  = 0.02
OMEGA = 3.0
LO    = -1.0
HI    = 1.0

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

# --- the two layouts -------------------------------------------------------
# Both answer the same six accessors. Only the memory behind them differs.

COMPONENTS = {
  px: [:@p, 0], py: [:@p, 1], pz: [:@p, 2],
  vx: [:@v, 0], vy: [:@v, 1], vz: [:@v, 2]
}.freeze

# Positions in one [N, 3] array, velocities in another. A component is a
# column slice, which is not contiguous.
class AoS
  def initialize(cols)
    @p = XM::SFloat.zeros(cols[0].size, 3)
    @v = XM::SFloat.zeros(cols[0].size, 3)
    %i[px py pz vx vy vz].each_with_index do |name, i|
      send("#{name}=", cols[i])
    end
  end

  COMPONENTS.each do |name, (ivar, k)|
    define_method(name) { instance_variable_get(ivar)[true, k] }
    define_method("#{name}=") { |val| instance_variable_get(ivar)[true, k] = val }
  end

  def label = 'AoS (a column slice of [N,3], not contiguous)'
end

# One contiguous array per component.
class SoA
  attr_accessor :px, :py, :pz, :vx, :vy, :vz

  def initialize(cols)
    @px, @py, @pz, @vx, @vy, @vz = cols.map { |c| c + 0.0 }
  end

  def label = 'SoA (a contiguous array per component)'
end

# --- one step --------------------------------------------------------------

def step(s)
  # The speed mixes the components, so it needs the accessors whatever the
  # layout is
  speed = timed(:speed) do
    vx = s.vx
    vy = s.vy
    vz = s.vz
    XM::NMath.sqrt(vx * vx + vy * vy + vz * vz + 1.0e-12)
  end

  # a vortex field, gravity, and a drag that grows with the speed
  ax = timed(:field) { XM::NMath.sin(OMEGA * s.py) - DRAG * s.vx * speed }
  ay = timed(:field) { XM::NMath.cos(OMEGA * s.px) - DRAG * s.vy * speed }
  az = timed(:field) { -GRAV - DRAG * s.vz * speed }

  timed(:integrate) do
    s.vx = s.vx + ax * DT
    s.vy = s.vy + ay * DT
    s.vz = s.vz + az * DT
    s.px = s.px + s.vx * DT
    s.py = s.py + s.vy * DT
    s.pz = s.pz + s.vz * DT
  end

  # Reflect off the walls. clip and ceil turn the condition into 0 or 1 without
  # a comparison operator, so no Bit array is built.
  timed(:boundary) do
    %i[px py pz].zip(%i[vx vy vz]).each do |pname, vname|
      p_ = s.send(pname)
      under = (LO - p_).clip(0.0, 1.0).ceil # 1.0 where p < LO
      over  = (p_ - HI).clip(0.0, 1.0).ceil # 1.0 where p > HI
      s.send("#{pname}=", p_ + under * 2.0 * (LO - p_) + over * 2.0 * (HI - p_))
      s.send("#{vname}=", s.send(vname) * (1.0 - 2.0 * (under + over)))
    end
  end

  speed
end

# --- setup -----------------------------------------------------------------

version = begin
  XM::NArray::VERSION
rescue StandardError
  'unknown'
end

# A fixed spread rather than random numbers, so the two backends start from the
# same state and their answers can be held against each other.
def initial_columns
  6.times.map do |i|
    t = XM::SFloat.new(N).seq
    a = XM::NMath.sin(t * (0.7 + (0.13 * i)) + (i * 1.7))
    i < 3 ? a * 0.9 : a * 0.5 # positions inside the box, velocities modest
  end
end

STATE_CLASS = LAYOUT == 'soa' ? SoA : AoS

warm = STATE_CLASS.new(initial_columns)

puts "ruby      : #{RUBY_VERSION} (#{RUBY_PLATFORM})"
puts "backend   : #{XM} #{version}"
puts "layout    : #{warm.label}"
puts "params    : N=#{N} STEPS=#{STEPS} REPEAT=#{REPEAT} DETAIL=#{DETAIL ? 1 : 0}"
puts

# One real step, timed, so the estimate needs no guess about how many array
# operations a step contains. A non-contiguous path that has fallen back to a
# launch per row shows up here, before the whole run is committed to.
step(warm)
sync
t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
step(warm)
sync
one_step = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
est = one_step * STEPS * REPEAT
puts format('  one step %.2f ms / whole run about %.1f s', one_step * 1e3, est)
puts '  ! that is a long run. Lower N or STEPS.' if est > 120.0
puts

warm = nil
GC.start
sync

# --- run -------------------------------------------------------------------

best = nil
REPEAT.times do
  TIMES.clear
  state = STATE_CLASS.new(initial_columns)
  sync
  t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  last_speed = nil
  STEPS.times { last_speed = step(state) }
  sync
  elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
  next if best && best[:elapsed] <= elapsed

  best = { elapsed: elapsed, state: state, speed: last_speed, times: TIMES.dup }
end

state = best[:state]
elapsed = best[:elapsed]
TIMES.replace(best[:times])

# --- results ---------------------------------------------------------------
# The same operations in the same order, so these must not move with LAYOUT.

cx = scalar(state.px.sum).to_f / N
cy = scalar(state.py.sum).to_f / N
cz = scalar(state.pz.sum).to_f / N
vmean = scalar(best[:speed].sum).to_f / N

puts format('  time       : %.4f s  (%.2f ms/step, best of %d)', elapsed, elapsed / STEPS * 1e3, REPEAT)
puts format('  throughput : %.1f M particles/s', N.to_f * STEPS / elapsed / 1e6)
puts
puts '  --- these must not change with LAYOUT ---'
puts format('  centroid   : % .9e  % .9e  % .9e', cx, cy, cz)
puts format('  mean speed : % .9e', vmean)

if DETAIL
  puts
  measured = TIMES.values.sum
  puts 'breakdown (per step):'
  TIMES.sort_by { |_, v| -v }.each do |key, v|
    puts format('  %-10s %9.1f us  %5.1f%%', key, v / STEPS * 1e6, 100.0 * v / measured)
  end
  puts '  (every section synchronizes, so this run is serialized and both the'
  puts '   ms/step above and the sections themselves read differently from a'
  puts '   DETAIL=0 run)'
end

puts
puts '  Run LAYOUT=aos and LAYOUT=soa with the same N to compare.'
puts '  The centroid and the mean speed must agree exactly between the two;'
puts '  they are the same arithmetic in the same order on the same values.'
puts '  Numo and Cumo agree to about six figures, no further: single-precision'
puts '  sin and the order a sum accumulates in differ between them.'
