#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# Crossover sweep - the element count at which Cumo overtakes Numo, per kind of
# operation.
#
#   ruby crossover_bench.rb                     # both in one process
#   ONLY=cumo JSON=cumo.json ruby crossover_bench.rb
#   ONLY=numo JSON=numo.json ruby crossover_bench.rb
#   MERGE=numo.json,cumo.json ruby crossover_bench.rb   # combine separate runs
#
#   SIZES=1024,4096,16384 TIME_CAP=0.5 ruby crossover_bench.rb
#
# This asks "how small is too small", not "how many times faster on a big
# array". The element count that pays back Cumo's per-operation fixed cost
# (launch + Ruby + ndloop) differs by operation, and that point is the
# guidance a user actually needs.
#
# Every point is measured two ways:
#   pipelined  BATCH operations with a single synchronize at the end. This is
#              the per-operation cost of a run of work, and what the crossover
#              is computed from.
#   per-op     one synchronize after every operation, for code that reads the
#              result straight back.
# Cumo cannot overlap the next launch with the previous operation once a
# synchronize sits between them, so the two differ by 2-4x on the same work.
#
# Timing is best-of-N, and the GPU is spun for a second first so that a
# laptop's clock ramp stays out of the measurement.
#
# Numo's times swing several-fold with the history of the process: above
# glibc's 128 KB mmap threshold every allocation takes page faults. Read the
# crossover as an order of magnitude, and split the processes with ONLY and
# MERGE when it matters.
# ---------------------------------------------------------------------------

require 'json'

ONLY      = ENV['ONLY']
JSON_OUT  = ENV['JSON']
MERGE     = ENV['MERGE']
TIME_CAP  = (ENV['TIME_CAP'] || 1.0).to_f  # stop a sweep once one run exceeds this
WARMUP    = (ENV['WARMUP']   || 1.0).to_f  # seconds of idle work to raise the GPU clock
MIN_REPS  = 3
MAX_REPS  = 30
MIN_TOTAL = 0.15                            # roughly how long to spend on a point
MAX_BATCH = (ENV['BATCH'] || 64).to_i       # most operations to pipeline at once
# The outputs cannot be held on to while pipelining, or the pool has nothing to
# reuse, so each one is freed as it is produced. Allocation and free are part of
# the pipelined number for that reason.
BATCH_BYTES = (ENV['BATCH_BYTES'] || (64 << 20)).to_i

SIZES = (ENV['SIZES'] || '256,1024,4096,16384,65536,262144,1048576,4194304,16777216,67108864')
        .split(',').map(&:to_i)

def clock
  Process.clock_gettime(Process::CLOCK_MONOTONIC)
end

# --- backends --------------------------------------------------------------
# Numo and Cumo have separate namespaces, so both can live in one process.

BACKENDS = {}

unless ONLY == 'cumo'
  begin
    require 'numo/narray'
    BACKENDS[:numo] = { mod: Numo, sync: -> {} }
  rescue LoadError => e
    warn "cannot load numo-narray: #{e.message}"
  end
end

unless ONLY == 'numo'
  begin
    require 'cumo/narray'
    BACKENDS[:cumo] = { mod: Cumo, sync: -> { Cumo::CUDA::Runtime.cudaDeviceSynchronize } }
  rescue LoadError => e
    warn "cannot load cumo: #{e.message}"
  end
end

# --- timing ----------------------------------------------------------------

def best_of(&block)
  best = Float::INFINITY
  reps = 0
  t_start = clock
  loop do
    t0 = clock
    block.call
    dt = clock - t0
    best = dt if dt < best
    reps += 1
    break if reps >= MIN_REPS && (reps >= MAX_REPS || (clock - t_start) > MIN_TOTAL)
  end
  best
end

# Holding on to the outputs leaves the pool nothing to reuse and every
# allocation is a fresh one, so both ways of measuring free alike. The two
# columns then differ by the synchronize and nothing else.
def run_once(callable, frees)
  r = callable.call
  r.free if frees && r.respond_to?(:free)
end

# One synchronize per operation, for code that reads the result straight back.
def measure_sync(callable, sync, frees)
  run_once(callable, frees)
  sync.call
  best_of do
    run_once(callable, frees)
    sync.call
  end
end

# batch operations with a single synchronize at the end, reported per operation.
def measure_pipelined(callable, sync, batch, frees)
  run_once(callable, frees)
  sync.call
  return measure_sync(callable, sync, frees) if batch <= 1

  t = best_of do
    batch.times { run_once(callable, frees) }
    sync.call
  end
  t / batch
end

# Pipeline as many as fit in BATCH_BYTES of live output.
def batch_for(bytes)
  return MAX_BATCH if bytes <= 0

  (BATCH_BYTES / bytes).clamp(1, MAX_BATCH)
end

# --- operations ------------------------------------------------------------
# The two-dimensional ones use a square of side sqrt(n).

OPS = [
  {
    name: 'mul scalar', dims: 1, out_bytes: ->(n) { n * 4 },
    setup: lambda { |xm, n|
      a = xm::SFloat.new(n).seq
      -> { a * 2.0 }
    }
  },
  {
    name: 'add arrays', dims: 1, out_bytes: ->(n) { n * 4 },
    setup: lambda { |xm, n|
      a = xm::SFloat.new(n).seq
      b = xm::SFloat.new(n).seq
      -> { a + b }
    }
  },
  {
    name: 'exp', dims: 1, out_bytes: ->(n) { n * 4 },
    setup: lambda { |xm, n|
      a = xm::SFloat.new(n).seq * 1.0e-6
      -> { xm::NMath.exp(a) }
    }
  },
  {
    name: 'sum all', dims: 1, out_bytes: ->(_n) { 4 },
    setup: lambda { |xm, n|
      a = xm::SFloat.new(n).seq
      -> { a.sum }
    }
  },
  {
    name: 'sum axis1', dims: 2, out_bytes: ->(n) { Integer(Math.sqrt(n)) * 4 },
    setup: lambda { |xm, n|
      s = Integer(Math.sqrt(n))
      a = xm::SFloat.new(s, s).seq
      -> { a.sum(axis: 1) }
    }
  },
  {
    # store returns the destination itself, which must not be freed
    name: 'store colslice', dims: 2, out_bytes: ->(_n) { 0 }, frees: false,
    setup: lambda { |xm, n|
      s = Integer(Math.sqrt(n))
      src = xm::SFloat.new(s, s * 2).seq
      dst = xm::SFloat.zeros(s, s)
      v = src[true, 0...s]
      -> { dst.store(v) }
    }
  },
  {
    name: 'dot (GEMM)', dims: 2, out_bytes: ->(n) { n * 4 },
    setup: lambda { |xm, n|
      s = Integer(Math.sqrt(n))
      a = xm::SFloat.new(s, s).seq * 0.001
      b = xm::SFloat.new(s, s).seq * 0.001
      -> { a.dot(b) }
    }
  }
].freeze

# --- run -------------------------------------------------------------------

results = Hash.new { |h, k| h[k] = Hash.new { |h2, k2| h2[k2] = {} } }

if MERGE
  MERGE.split(',').each do |path|
    JSON.parse(File.read(path)).each do |backend, ops|
      ops.each do |op, points|
        points.each { |n, t| results[backend.to_sym][op][n.to_i] = t }
      end
    end
  end
  puts "merged    : #{MERGE}"
else
  puts "ruby      : #{RUBY_VERSION} (#{RUBY_PLATFORM})"
  BACKENDS.each do |key, be|
    version = begin
      be[:mod]::NArray::VERSION
    rescue StandardError
      'unknown'
    end
    puts format('%-10s: %s %s', key, be[:mod], version)
  end
  puts "sizes     : #{SIZES.first} .. #{SIZES.last} (#{SIZES.size} points)"
  puts "params    : TIME_CAP=#{TIME_CAP}s WARMUP=#{WARMUP}s"
  puts

  # raise the GPU clock before measuring anything
  if BACKENDS[:cumo] && WARMUP.positive?
    a = Cumo::SFloat.new(4_000_000).seq
    t0 = clock
    (a * 1.0001).free while clock - t0 < WARMUP
    BACKENDS[:cumo][:sync].call
    puts format('  warmed up for %.1f s', WARMUP)
    puts
  end

  # Whichever operation is measured first carries the process's one-time setup,
  # so every one of them gets a discarded pass first.
  BACKENDS.each do |key, be|
    OPS.each do |op|
      call = op[:setup].call(be[:mod], SIZES.first)
      3.times { run_once(call, op.fetch(:frees, true)) }
      be[:sync].call
    rescue StandardError => e
      warn format('  discarded pass %s / %s: %s', key, op[:name], e.class)
    end
  end

  BACKENDS.each do |key, be|
    OPS.each do |op|
      frees = op.fetch(:frees, true)
      SIZES.each do |n|
        call = op[:setup].call(be[:mod], n)
        t_sync = measure_sync(call, be[:sync], frees)
        t_pipe = measure_pipelined(call, be[:sync], batch_for(op[:out_bytes].call(n)), frees)
        results[key][op[:name]][n] = [t_sync, t_pipe]
        GC.start
        break if t_sync > TIME_CAP # anything larger takes too long to be useful
      rescue StandardError, NoMemoryError => e
        warn format('  %s / %s / n=%d: %s', key, op[:name], n, e.class)
        break
      end
    end
  end

  if JSON_OUT
    File.write(JSON_OUT, JSON.pretty_generate(results))
    puts "  -> #{JSON_OUT}"
  end
end

# --- report ----------------------------------------------------------------

def fmt_time(t)
  return '        -' unless t

  if t < 1.0e-3
    format('%7.2f us', t * 1e6)
  else
    format('%7.2f ms', t * 1e3)
  end
end

# Interpolate the interval where the ratio crosses 1, in log space.
def crossover(points)
  bracket = points.each_cons(2).find { |(_, r1), (_, r2)| r1 < 1.0 && r2 >= 1.0 }
  return nil unless bracket

  (n1, r1), (n2, r2) = bracket
  f = Math.log(r1) / (Math.log(r1) - Math.log(r2))
  Math.exp(Math.log(n1) + (f * (Math.log(n2) - Math.log(n1))))
end

# A point is [time with a synchronize per operation, time per operation pipelined]
def pipe(points, n)
  v = points[n]
  v.is_a?(Array) ? v[1] : v
end

def syncd(points, n)
  v = points[n]
  v.is_a?(Array) ? v[0] : nil
end

summary = []

OPS.each do |op|
  numo = results[:numo][op[:name]]
  cumo = results[:cumo][op[:name]]
  next if numo.empty? && cumo.empty?

  puts "[#{op[:name]}]"
  puts format('  %12s %11s %11s %10s %13s', 'elements', 'numo', 'cumo', 'numo/cumo', 'cumo per-op')

  ratios = []
  marked = false
  SIZES.each do |n|
    tn = pipe(numo, n)
    tc = pipe(cumo, n)
    ratio = (tn && tc) ? tn / tc : nil
    ratios << [n, ratio] if ratio
    mark = if ratio && ratio >= 1.0 && !marked
             marked = true
             '  <== Cumo wins from here'
           else
             ''
           end
    next unless tn || tc

    puts format('  %12d %11s %11s %10s %13s%s', n, fmt_time(tn), fmt_time(tc),
                ratio ? format('%.2f', ratio) : '-', fmt_time(syncd(cumo, n)), mark)
  end

  cross = crossover(ratios)
  small = cumo.keys.min
  fixed = small ? pipe(cumo, small) : nil
  fixed_sync = small ? syncd(cumo, small) : nil
  best = ratios.map(&:last).compact.max
  summary << [op[:name], cross, fixed, fixed_sync, best]
  puts
end

puts '== summary'
puts format('  %-16s %14s %12s %12s %10s',
            'operation', 'crossover', 'cumo fixed', 'with sync', 'best ratio')
summary.each do |name, cross, fixed, fixed_sync, best|
  puts format('  %-16s %14s %12s %12s %10s',
              name,
              cross ? format('~%d', cross.round) : '-',
              fixed ? format('%.2f us', fixed * 1e6) : '-',
              fixed_sync ? format('%.2f us', fixed_sync * 1e6) : '-',
              best ? format('%.1fx', best) : '-')
end

puts
puts '  The cumo column and the crossover are the pipelined numbers.'
puts '  "cumo fixed" is the per-operation cost at the smallest size, which is'
puts '  the floor of launch + Ruby + ndloop. "with sync" waits for each'
puts '  operation to finish; the difference is not what a synchronize costs but'
puts '  what it prevents, namely overlapping the next launch with this one.'
puts '  The crossover is where the CPU takes longer than that fixed cost, so it'
puts '  moves left as the fixed cost falls or the work per operation grows.'
puts '  An operation with no crossover had one side ahead over the whole range.'
puts
if results[:numo].key?('dot (GEMM)') && !defined?(Numo::Linalg)
  puts '  Note: without numo-linalg, dot runs against the built-in Numo'
  puts '  implementation. With BLAS its crossover moves far to the right.'
end
puts '  Note: Numo swings several-fold with the history of the process, since'
puts '  above glibc\'s 128 KB mmap threshold every allocation takes page faults.'
puts '  Read the crossover as an order of magnitude, and split the processes'
puts '  with ONLY and MERGE when the number has to be tight.'
