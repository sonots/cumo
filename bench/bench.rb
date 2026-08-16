#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# Numo / Cumo comparison benchmark
#
#   ruby bench.rb                  # CPU: numo-narray
#   GPU=1 ruby bench.rb            # GPU: cumo
#   SIZE=4096 STEPS=500 ITER=3 ruby bench.rb
#   PGM=0 ruby bench.rb            # skip the PGM output
#
# To compare against numo-narray-alt (yoshoku's fork, SIMD enabled), swap the
# gem; this file needs no change, since the namespace stays Numo:
#   gem uninstall numo-narray && gem install numo-narray-alt
#
# Environment variables that matter on the Cumo side:
#   CUMO_SHOW_WARNING=ON     warn wherever a CPU/GPU synchronization happens
#   CUDA_LAUNCH_BLOCKING=1   synchronize on every launch (for profiling; slow)
#   CUMO_MEMORY_POOL=OFF     turn the memory pool off to measure allocation too
#   CUDA_VISIBLE_DEVICES=0   device selection
#
# Numo::Bit, masked assignment and fancy indexing are deliberately unused. The
# divergence test is written with clip and floor arithmetic alone, so Numo and
# Cumo take exactly the same code path.
# ---------------------------------------------------------------------------

require 'benchmark'

GPU = !%w[0 false].include?(ENV['GPU'].to_s.downcase) && !ENV['GPU'].to_s.empty?

if GPU
  require 'cumo/narray'
  XM = Cumo
else
  require 'numo/narray'
  XM = Numo
end

SIZE  = (ENV['SIZE']  || 1024).to_i   # side of the mandelbrot / diffusion grid
STEPS = (ENV['STEPS'] || 200).to_i    # number of iterations
NBODY = (ENV['NBODY'] || 2048).to_i   # number of n-body particles
MC    = (ENV['MC']    || 10_000_000).to_i
ITER  = (ENV['ITER']  || 3).to_i      # repeats per measurement (3-5 to steady it)
PGM   = ENV['PGM'] != '0'

# --- helpers ---------------------------------------------------------------

# Cumo launches kernels asynchronously, so every measurement boundary syncs.
def sync
  XM::CUDA::Runtime.cudaDeviceSynchronize if GPU
end

# Cumo returns a zero-dimensional NArray from sum / min / max / count_true,
# where Numo returns a Numeric.
def scalar(v)
  return v if v.is_a?(Numeric)
  v.respond_to?(:extract_cpu) ? v.extract_cpu : v.extract
end

def free(*arrays)
  return unless GPU

  arrays.each { |a| a.free if a.respond_to?(:free) }
end

def report(label)
  result = nil
  total = Benchmark.realtime do
    ITER.times do
      result = yield
      sync
    end
  end
  puts format('  %-12s %9.3f s total   %9.3f s/iter', label, total, total / ITER)
  result
end

def write_pgm(path, arr)
  return unless PGM

  lo = scalar(arr.min).to_f
  hi = scalar(arr.max).to_f
  hi = lo + 1.0 if hi <= lo
  img = XM::UInt8.cast((arr - lo) / (hi - lo) * 255.0)
  h, w = img.shape
  bytes =
    begin
      img.to_binary
    rescue StandardError
      # fallback for a Cumo without to_binary (device to host copy)
      img.to_a.flatten.pack('C*')
    end
  File.open(path, 'wb') do |f|
    f.write("P5\n#{w} #{h}\n255\n")
    f.write(bytes)
  end
  free(img)
  puts "  -> #{path}"
end

# --- 1. mandelbrot set (pure elementwise arithmetic) ------------------------
#
# Build an indicator that is 1.0 once diverged and 0.0 while still alive using
# clip and floor alone, then accumulate the survival count. No branches and no
# masks, so it maps straight onto the GPU.
def mandelbrot(n, steps)
  lim = 1.0e10 # bound zr and zi here so neither Inf nor NaN appears
  cr = XM::SFloat.new(1, n).seq / n * 3.0 - 2.0 # real axis [-2.0, 1.0)
  ci = XM::SFloat.new(n, 1).seq / n * 3.0 - 1.5 # imaginary axis [-1.5, 1.5)
  zr = XM::SFloat.zeros(n, n)
  zi = XM::SFloat.zeros(n, n)
  cnt = XM::SFloat.zeros(n, n)
  cr = zr + cr # materialize the broadcast
  ci = zr + ci

  steps.times do
    zr2 = zr * zr
    zi2 = zi * zi
    escaped = ((zr2 + zi2).clip(0.0, 4.0) / 4.0).floor # 1.0 once |z|^2 >= 4
    cnt += 1.0 - escaped
    zi = (2.0 * zr * zi + ci).clip(-lim, lim)
    zr = (zr2 - zi2 + cr).clip(-lim, lim)
  end
  free(cr, ci, zr, zi)
  cnt
end

# --- 2. 2-D heat diffusion (stencil / slice arithmetic) ---------------------
def diffusion(n, steps)
  u = XM::SFloat.zeros(n, n)
  h = [n / 8, 1].max
  u[h...(2 * h), h...(2 * h)] = 255.0
  u[(5 * h)...(7 * h), (3 * h)...(4 * h)] = 255.0
  k = 0.2

  steps.times do
    lap = u[0..-3, 1..-2] + u[2..-1, 1..-2] +
          u[1..-2, 0..-3] + u[1..-2, 2..-1] - 4.0 * u[1..-2, 1..-2]
    u[1..-2, 1..-2] = u[1..-2, 1..-2] + k * lap
  end
  u
end

# --- 3. n-body (2-D, O(N^2) through broadcasting) ---------------------------
def nbody(n, steps)
  dt = 1.0e-3
  eps = 1.0e-3
  x = XM::SFloat.new(n).rand * 2.0 - 1.0
  y = XM::SFloat.new(n).rand * 2.0 - 1.0
  vx = XM::SFloat.zeros(n)
  vy = XM::SFloat.zeros(n)

  steps.times do
    dx = x.reshape(1, n) - x.reshape(n, 1) # [n, n]
    dy = y.reshape(1, n) - y.reshape(n, 1)
    r2 = dx * dx + dy * dy + eps
    inv3 = 1.0 / (r2 * XM::NMath.sqrt(r2))
    vx += (dx * inv3).sum(axis: 1) * dt
    vy += (dy * inv3).sum(axis: 1) * dt
    free(dx, dy, r2, inv3)
    x += vx * dt
    y += vy * dt
  end
  free(y, vx, vy)
  x
end

# --- 4. monte carlo pi (random numbers + reduction) -------------------------
# The sum is SFloat, so a large n costs pi its accuracy to rounding. Speed is
# what this measures.
def monte_carlo_pi(n)
  x = XM::SFloat.new(n).rand
  y = XM::SFloat.new(n).rand
  r2 = x * x + y * y
  inside = (1.0 - r2.clip(0.0, 1.0)).ceil # 1.0 where r2 < 1
  hits = scalar(inside.sum).to_f
  free(x, y, r2, inside)
  4.0 * hits / n
end

# --- 5. compatibility probe (eyeball the incompatibilities the README lists) -
def compat_probe
  a = XM::SFloat[1.0, 2.0, 3.0]
  bit = (a > 1.5)
  count = bit.respond_to?(:count_true_cpu) ? bit.count_true_cpu : bit.count_true
  puts "  (a > 1.5).class     : #{bit.class}"
  puts "  count_true          : #{count.inspect}"
  puts "  a.sum.class         : #{a.sum.class}"   # Numo: Float / Cumo: zero-dimensional NArray
  puts "  a.max.class         : #{a.max.class}"
  puts "  a[0].class          : #{a[0].class}"
  free(a)
end

# --- main ------------------------------------------------------------------

version = begin
  XM::NArray::VERSION
rescue StandardError
  'unknown'
end

puts "ruby      : #{RUBY_VERSION} (#{RUBY_PLATFORM})"
puts "backend   : #{XM} #{version}"
puts "params    : SIZE=#{SIZE} STEPS=#{STEPS} NBODY=#{NBODY} MC=#{MC} ITER=#{ITER}"
puts

# warm up (keep GPU initialization and kernel JIT out of the measurement)
warm = XM::SFloat.new(64, 64).seq
scalar((warm * warm + warm).sum)
sync
free(warm)
GC.start

puts 'compat:'
compat_probe
puts

puts 'bench:'
cnt = report('mandelbrot') { mandelbrot(SIZE, STEPS) }
write_pgm('mandelbrot.pgm', cnt)
free(cnt)
GC.start

u = report('diffusion') { diffusion(SIZE, STEPS) }
write_pgm('diffusion.pgm', u)
free(u)
GC.start

pos = report('nbody') { nbody(NBODY, 10) }
free(pos)
GC.start

pi = nil
report('monte_carlo') { pi = monte_carlo_pi(MC) }
puts format('  pi ~= %.6f', pi)
