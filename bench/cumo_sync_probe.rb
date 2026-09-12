#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# An inventory of where Cumo synchronizes behind your back.
#
#   CUMO_SHOW_WARNING=ON CUMO_SHOW_WARNING_ONCE=OFF ruby cumo_sync_probe.rb
#
#   NAME='a[0,0]' ruby cumo_sync_probe.rb      # one probe, in its own process
#   GROUP=readback ruby cumo_sync_probe.rb
#   ruby cumo_sync_probe.rb --list             # the probe names
#   BIG_MB=64 BACKLOG=60 ruby cumo_sync_probe.rb
#
# Two detectors run independently and their answers are compared.
#
#   (a) Cumo's own warning: whatever CUMO_SHOW_WARNING prints for the call.
#   (b) Timing: queue kernels without synchronizing, then make the call. One
#       that is asynchronous returns immediately; one that synchronizes waits
#       for everything queued.
#
# (b) owes nothing to Cumo's instrumentation, so it finds the calls that
# synchronize without saying so -- and, the other way round, the ones that
# warn without actually synchronizing.
# ---------------------------------------------------------------------------

require 'cumo/narray'
require 'tempfile'

BIG_MB   = (ENV['BIG_MB']  || 32).to_f
BACKLOG  = (ENV['BACKLOG'] || 60).to_i
REPS     = (ENV['REPS']    || 5).to_i
NAME     = ENV['NAME']
GROUP    = ENV['GROUP']
SYNC_RATIO = (ENV['SYNC_RATIO'] || 0.3).to_f # waiting this fraction of the backlog counts as a sync

F32 = 4
BIG_N = (BIG_MB * 1024 * 1024 / F32).to_i
SMALL_N = 1024

def clock
  Process.clock_gettime(Process::CLOCK_MONOTONIC)
end

def sync
  Cumo::CUDA::Runtime.cudaDeviceSynchronize
end

# --- capturing stdout and stderr -------------------------------------------
# The warnings come from the C side, so replacing Ruby's $stderr does not catch
# them; the file descriptors have to be redirected. Which of the two a warning
# lands on is not fixed, so both are captured.

def capture_stdio
  tmp = Tempfile.new('cumo_sync')
  old_out = STDOUT.dup
  old_err = STDERR.dup
  STDOUT.flush
  STDERR.flush
  begin
    STDOUT.reopen(tmp.path, 'a')
    STDERR.reopen(tmp.path, 'a')
    yield
  ensure
    STDOUT.flush
    STDERR.flush
    STDOUT.reopen(old_out)
    STDERR.reopen(old_err)
    old_out.close
    old_err.close
  end
  tmp.rewind
  text = tmp.read
  tmp.close!
  text
end

# --- the backlog -----------------------------------------------------------
# Queue as much as possible without synchronizing. A contiguous store allocates
# nothing, so the queue is pure device work.

BIG_SRC = Cumo::SFloat.new(BIG_N).seq
BIG_DST = Cumo::SFloat.zeros(BIG_N)
sync

def enqueue!
  BACKLOG.times { BIG_DST.store(BIG_SRC) }
end

# --- the probes ------------------------------------------------------------

PROBES = []

def probe(group, name, counts_only: false, &setup)
  PROBES << { group: group, name: name, setup: setup, counts_only: counts_only }
end

# Reading data back is meant to synchronize. The question is whether it says so.
probe('readback', 'a[0,0]') do
  a = Cumo::SFloat.new(SMALL_N).seq
  -> { a[0] }
end

probe('readback', 'sum (returns 0-dim)') do
  a = Cumo::SFloat.new(SMALL_N).seq
  -> { a.sum }
end

probe('readback', 'sum.extract') do
  a = Cumo::SFloat.new(SMALL_N).seq
  -> { a.sum.extract }
end

probe('readback', 'max') do
  a = Cumo::SFloat.new(SMALL_N).seq
  -> { a.max }
end

probe('readback', 'to_a') do
  a = Cumo::SFloat.new(SMALL_N).seq
  -> { a.to_a }
end

probe('readback', 'to_binary') do
  a = Cumo::SFloat.new(SMALL_N).seq
  -> { a.to_binary }
end

probe('readback', 'inspect') do
  a = Cumo::SFloat.new(SMALL_N).seq
  -> { a.inspect }
end

probe('readback', 'to_s') do
  a = Cumo::SFloat.new(SMALL_N).seq
  -> { a.to_s }
end

probe('readback', 'count_true') do
  bit = (Cumo::SFloat.new(SMALL_N).seq > 0.5)
  -> { bit.count_true }
end

probe('readback', 'count_true_cpu') do
  bit = (Cumo::SFloat.new(SMALL_N).seq > 0.5)
  bit.respond_to?(:count_true_cpu) ? -> { bit.count_true_cpu } : nil
end

probe('readback', 'where') do
  bit = (Cumo::SFloat.new(SMALL_N).seq > 0.5)
  -> { bit.where }
end

# Metadata. A synchronization here would be pure waste.
probe('meta', 'shape') do
  a = Cumo::SFloat.new(SMALL_N).seq
  -> { a.shape }
end

probe('meta', 'size / ndim') do
  a = Cumo::SFloat.new(SMALL_N).seq
  lambda {
    a.size
    a.ndim
  }
end

probe('meta', 'reshape') do
  a = Cumo::SFloat.new(SMALL_N).seq
  -> { a.reshape(32, 32) }
end

probe('meta', 'transpose') do
  a = Cumo::SFloat.new(32, 32).seq
  -> { a.transpose }
end

probe('meta', 'view [true, 0...16]') do
  a = Cumo::SFloat.new(32, 32).seq
  -> { a[true, 0...16] }
end

# Allocation. A request the pool cannot satisfy falls through to cudaMalloc,
# which synchronizes.
probe('alloc', 'zeros same size (warm)') do
  n = BIG_N / 16
  x = Cumo::SFloat.zeros(n)
  x.free if x.respond_to?(:free)
  -> { Cumo::SFloat.zeros(n) }
end

probe('alloc', 'zeros a new size each time') do
  n = BIG_N / 16
  i = 0
  lambda {
    i += 1
    Cumo::SFloat.zeros(n + (i * 4096))
  }
end

probe('alloc', 'free') do
  n = BIG_N / 16
  lambda {
    x = Cumo::SFloat.zeros(n)
    x.free if x.respond_to?(:free)
  }
end

# Computation. All of it should be asynchronous.
probe('compute', 'a + b') do
  a = Cumo::SFloat.new(BIG_N).seq
  b = Cumo::SFloat.new(BIG_N).seq
  -> { a + b }
end

probe('compute', 'a * 2.0') do
  a = Cumo::SFloat.new(BIG_N).seq
  -> { a * 2.0 }
end

probe('compute', 'NMath.exp') do
  a = Cumo::SFloat.new(BIG_N).seq * 1.0e-6
  -> { Cumo::NMath.exp(a) }
end

probe('compute', 'dot (cuBLAS)') do
  a = Cumo::SFloat.new(512, 512).seq * 0.001
  b = Cumo::SFloat.new(512, 512).seq * 0.001
  -> { a.dot(b) }
end

probe('compute', 'sum(axis: 1)') do
  a = Cumo::SFloat.new(512, 512).seq
  -> { a.sum(axis: 1) }
end

probe('compute', 'max(axis: 1)') do
  a = Cumo::SFloat.new(512, 512).seq
  -> { a.max(axis: 1) }
end

probe('compute', 'rand (cuRAND)') do
  a = Cumo::SFloat.new(BIG_N / 4)
  -> { a.rand }
end

probe('compute', 'seq') do
  a = Cumo::SFloat.new(BIG_N / 4)
  -> { a.seq }
end

probe('compute', 'fill') do
  a = Cumo::SFloat.new(BIG_N / 4)
  -> { a.fill(1.0) }
end

probe('compute', 'cmp (> 0.5)') do
  a = Cumo::SFloat.new(BIG_N).seq
  -> { a > 0.5 }
end

probe('compute', 'cast f32->f64') do
  a = Cumo::SFloat.new(BIG_N / 4).seq
  -> { Cumo::DFloat.cast(a) }
end

probe('compute', 'store colslice') do
  src = Cumo::SFloat.new(1024, 1024).seq
  dst = Cumo::SFloat.zeros(1024, 512)
  v = src[true, 0...512]
  -> { dst.store(v) }
end

probe('compute', 'masked aset a[bit] = 0') do
  a = Cumo::SFloat.new(BIG_N / 16).seq
  bit = (a > 0.5)
  -> { a[bit] = 0.0 }
end

probe('compute', 'index aref a[idx, true]') do
  a = Cumo::SFloat.new(512, 512).seq
  idx = Cumo::Int32.new(512).seq
  -> { a[idx, true] }
end

# Host to device
# A block walks the array on the host and can queue work on it, so the walk has
# to wait between elements. It only has to wait when the block queued something,
# which is what the idle count says: nothing for a block that stays off the
# device, once per touched element for one that does not.
BLOCK_N = 32

probe('block', 'each (block off the device)', counts_only: true) do
  a = Cumo::SFloat.new(BLOCK_N).seq
  -> { a.each { |x| x } }
end

probe('block', 'each (block writes)', counts_only: true) do
  a = Cumo::SFloat.new(BLOCK_N).seq
  -> { i = 0; a.each { |_x| a[i] = 1; i += 1 } }
end

probe('block', 'each_with_index (block writes)', counts_only: true) do
  a = Cumo::SFloat.new(BLOCK_N).seq
  -> { a.each_with_index { |_x, i| a[i] = 1 } }
end

probe('block', 'map (block writes)', counts_only: true) do
  a = Cumo::SFloat.new(BLOCK_N).seq
  -> { i = 0; a.map { |x| a[i] = 1; i += 1; x } }
end

probe('block', 'map_with_index (block writes)', counts_only: true) do
  a = Cumo::SFloat.new(BLOCK_N).seq
  -> { a.map_with_index { |x, i| a[i] = 1; x } }
end

probe('block', 'Bit each (block writes)', counts_only: true) do
  a = Cumo::Bit.new(BLOCK_N).fill(1)
  -> { i = 0; a.each { |_x| a[i] = 0; i += 1 } }
end

probe('block', 'Bit each_with_index (block writes)', counts_only: true) do
  a = Cumo::Bit.new(BLOCK_N).fill(1)
  -> { a.each_with_index { |_x, i| a[i] = 0 } }
end

probe('h2d', 'from_binary') do
  bin = Array.new(BIG_N / 4, 1.0).pack('f*')
  -> { Cumo::SFloat.from_binary(bin, [BIG_N / 4]) }
end

probe('h2d', 'store (Ruby Array)') do
  a = Cumo::SFloat.zeros(SMALL_N)
  src = Array.new(SMALL_N, 1.0)
  -> { a.store(src) }
end

# --- main ------------------------------------------------------------------

if ARGV.include?('--list')
  PROBES.each { |p| puts p[:name] }
  exit
end

warn_on = ENV['CUMO_SHOW_WARNING'].to_s.upcase == 'ON'

puts "ruby        : #{RUBY_VERSION} (#{RUBY_PLATFORM})"
puts "cumo        : #{begin
  Cumo::NArray::VERSION
rescue StandardError
  'unknown'
end}"
puts "warning     : CUMO_SHOW_WARNING=#{ENV['CUMO_SHOW_WARNING'] || '(unset)'} " \
     "CUMO_SHOW_WARNING_ONCE=#{ENV['CUMO_SHOW_WARNING_ONCE'] || '(unset)'}"
puts "params      : BIG=#{BIG_MB}MB BACKLOG=#{BACKLOG} REPS=#{REPS}"
unless warn_on
  puts
  puts '  ! Run with CUMO_SHOW_WARNING=ON CUMO_SHOW_WARNING_ONCE=OFF.'
  puts '    (the timing detector works either way)'
end

# Calibrate the backlog. The first run touches pages for the first time, which
# on managed memory costs more than the copies do, so warm up before measuring
# and take the best of a few runs.
3.times do
  enqueue!
  sync
end
t_backlog = Float::INFINITY
3.times do
  t0 = clock
  enqueue!
  sync
  dt = clock - t0
  t_backlog = dt if dt < t_backlog
end
sync
t0 = clock
enqueue!
t_launch = clock - t0
sync
puts format('backlog     : queue %.2f ms / run %.2f ms (%d calls)',
            t_launch * 1e3, t_backlog * 1e3, BACKLOG)
if t_launch > t_backlog * 0.5
  puts '  ! Queueing itself is taking too long. Raise BACKLOG or BIG_MB.'
end
puts

results = []

puts format('  %-34s %10s %6s %12s  %-9s %s', 'probe', 'idle', 'waits', 'behind queue', 'verdict', 'warning')
puts "  #{'-' * 99}"

current_group = nil
PROBES.each do |p|
  next if NAME && p[:name] != NAME
  next if GROUP && p[:group] != GROUP

  if p[:group] != current_group
    current_group = p[:group]
    puts "  [#{current_group}]"
  end

  begin
    call = p[:setup].call
    if call.nil?
      puts format('  %-34s %10s %6s %12s  %-9s %s', p[:name], '-', '-', '-', 'skip', '(no such API)')
      next
    end

    # (1b) how many times it waits with nothing queued. With the warning left
    # on every occurrence this is the wait count, not a yes or no.
    sync
    idle_text = capture_stdio { call.call }
    idle_warns = idle_text.to_s.lines.grep(/warn|sync|Sync|WARN/i).size

    if p[:counts_only]
      # Timing these would measure the warning writes, not the call.
      puts format('  %-34s %10s %6d %12s  %-9s %s',
                  p[:name], '-', idle_warns, '-', 'waits', '-')
      results << { name: p[:name], group: p[:group], sync: idle_warns.positive?,
                   warned: idle_warns.positive? }
      next
    end

    # (1) how long the call takes with nothing queued
    sync
    call.call
    sync
    idle = Float::INFINITY
    REPS.times do
      t = clock
      call.call
      sync
      dt = clock - t
      idle = dt if dt < idle
    end

    # (2) how long it takes behind the queue, and what it printed
    sync
    GC.start
    # The capture redirects two file descriptors through a temporary file, which
    # costs more than an asynchronous call does, so it is measured separately
    # and taken off the reading rather than left in it.
    sync
    capture_overhead = clock
    capture_stdio { nil }
    capture_overhead = clock - capture_overhead

    sync
    enqueue!
    t = clock
    text = capture_stdio { call.call }
    blocked = [clock - t - capture_overhead, 0.0].max
    sync

    warns = text.to_s.lines.grep(/warn|sync|Sync|WARN/i)
    delta = blocked - idle
    ratio = t_backlog.positive? ? delta / t_backlog : 0.0
    is_sync = ratio > SYNC_RATIO

    results << { name: p[:name], group: p[:group], sync: is_sync, warned: !warns.empty? }

    note =
      if warns.empty?
        '-'
      else
        format('x%d %s', warns.size, warns.first.to_s.strip[0, 40])
      end

    puts format('  %-34s %8.1f us %6d %10.2f ms  %-9s %s',
                p[:name], idle * 1e6, idle_warns, blocked * 1e3,
                is_sync ? 'SYNC' : 'async', note)
  rescue StandardError, NotImplementedError => e
    msg = e.message.to_s.split("\n").first.to_s[0, 40]
    puts format('  %-34s %10s %6s %12s  %-9s %s', p[:name], '-', '-', '-', 'ERROR', "#{e.class}: #{msg}")
  end
end

# --- the cross-tabulation --------------------------------------------------

puts
puts '== cross-tabulation'

quadrant = lambda do |title, sel|
  list = results.select(&sel).map { |r| r[:name] }
  puts "  #{title}: #{list.empty? ? 'none' : list.join(', ')}"
end

quadrant.call('synchronizes, silent  <- a gap in the instrumentation', ->(r) { r[:sync] && !r[:warned] })
quadrant.call('synchronizes, warned   <- as intended', ->(r) { r[:sync] && r[:warned] })
quadrant.call('asynchronous, warned   <- warns for nothing', ->(r) { !r[:sync] && r[:warned] })

puts
puts '  The first line is what this is for: those calls stall the pipeline and'
puts '  CUMO_SHOW_WARNING will not tell you. Either add the warning or, where'
puts '  the synchronization is avoidable, remove it.'
puts '  A SYNC in the meta group is waste with nothing to show for it.'
puts
puts '  The block group counts waits rather than timing them: a walk that hands'
puts '  every element to a Ruby block has to wait between elements, since the'
puts '  block can queue work on the array being walked. A block that stays off'
puts '  the device should cost none, and one that writes should cost about one'
puts '  per element. A count of one there means the walk waits once for the whole'
puts '  row, which is what made Bit#each and map answer differently from numo.'
