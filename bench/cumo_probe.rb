#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# Cumo diagnostic probe
#
# Sweeps operations against data layouts to find the paths that launch one
# kernel per row. The test is the one the store investigation used: if the
# time per row barely moves as the number of rows changes, the path is bound
# by launch overhead rather than by bandwidth.
#
#   GPU=1 ruby cumo_probe.rb
#   GPU=1 GROUP=store ruby cumo_probe.rb          # one group only
#   GPU=1 COLS=4096 ROWS=512,1024,2048 REPS=7 ruby cumo_probe.rb
#   ruby cumo_probe.rb                            # the same table from Numo
#
# An unsupported API is reported as an ERROR row rather than stopping the
# sweep: that it raises at all is a result worth seeing.
# ---------------------------------------------------------------------------

GPU = !%w[0 false].include?(ENV['GPU'].to_s.downcase) && !ENV['GPU'].to_s.empty?

if GPU
  require 'cumo/narray'
  XM = Cumo
else
  require 'numo/narray'
  XM = Numo
end

COLS  = (ENV['COLS'] || 2048).to_i
ROWS  = (ENV['ROWS'] || '256,512,1024').split(',').map(&:to_i)
REPS  = (ENV['REPS'] || 5).to_i
GROUP = ENV['GROUP']

BAND_OK   = (ENV['BAND_OK'] || 50).to_f  # GB/s above which a path counts as bandwidth-bound
FLAT      = (ENV['FLAT']    || 1.3).to_f # us/row counts as flat below this spread

def sync
  XM::CUDA::Runtime.cudaDeviceSynchronize if GPU
end

def clock
  Process.clock_gettime(Process::CLOCK_MONOTONIC)
end

# Take the minimum. A laptop throttles, which lifts the mean but not the floor.
def measure(callable)
  callable.call
  sync
  best = Float::INFINITY
  REPS.times do
    t0 = clock
    callable.call
    sync
    dt = clock - t0
    best = dt if dt < best
  end
  best
end

PROBES = []

# setup returns [callable, work, unit], where unit is 'GB' for bytes moved or
# 'GFLOP' for floating point operations.
def probe(group, name, &setup)
  PROBES << { group: group, name: name, setup: setup }
end

F32 = 4.0

# --- store and copy --------------------------------------------------------

probe('store', 'store contig') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  dst = XM::SFloat.zeros(rows, cols)
  [-> { dst.store(src) }, 2 * rows * cols * F32, 'GB']
end

probe('store', 'store rowslice') do |rows, cols|
  src = XM::SFloat.new(rows * 2, cols).seq
  dst = XM::SFloat.zeros(rows, cols)
  v = src[0...rows, true] # contiguous, as a control
  [-> { dst.store(v) }, 2 * rows * cols * F32, 'GB']
end

probe('store', 'store colslice') do |rows, cols|
  src = XM::SFloat.new(rows, cols * 2).seq
  dst = XM::SFloat.zeros(rows, cols)
  v = src[true, 0...cols] # not contiguous
  [-> { dst.store(v) }, 2 * rows * cols * F32, 'GB']
end

probe('store', 'store transpose') do |rows, cols|
  src = XM::SFloat.new(cols, rows).seq
  dst = XM::SFloat.zeros(rows, cols)
  v = src.transpose
  [-> { dst.store(v) }, 2 * rows * cols * F32, 'GB']
end

probe('store', 'aset colslice') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  dst = XM::SFloat.zeros(rows, cols * 2)
  [-> { dst[true, 0...cols] = src }, 2 * rows * cols * F32, 'GB']
end

probe('store', 'aref index') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  idx = XM::Int32.new(rows).seq
  [-> { src[idx, true] }, 2 * rows * cols * F32, 'GB']
end

probe('store', 'cast f32->f64') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { XM::DFloat.cast(src) }, 3 * rows * cols * F32, 'GB']
end

probe('store', 'cast colslice') do |rows, cols|
  src = XM::SFloat.new(rows, cols * 2).seq
  v = src[true, 0...cols]
  [-> { XM::DFloat.cast(v) }, 3 * rows * cols * F32, 'GB']
end

# --- elementwise -----------------------------------------------------------

probe('elemwise', 'mul scalar contig') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { src * 2.0 }, 2 * rows * cols * F32, 'GB']
end

probe('elemwise', 'mul scalar colslice') do |rows, cols|
  src = XM::SFloat.new(rows, cols * 2).seq
  v = src[true, 0...cols]
  [-> { v * 2.0 }, 2 * rows * cols * F32, 'GB']
end

probe('elemwise', 'add arrays contig') do |rows, cols|
  a = XM::SFloat.new(rows, cols).seq
  b = XM::SFloat.new(rows, cols).seq
  [-> { a + b }, 3 * rows * cols * F32, 'GB']
end

probe('elemwise', 'add broadcast row') do |rows, cols|
  a = XM::SFloat.new(rows, cols).seq
  b = XM::SFloat.new(1, cols).seq
  [-> { a + b }, 2 * rows * cols * F32, 'GB']
end

probe('elemwise', 'add broadcast col') do |rows, cols|
  a = XM::SFloat.new(rows, cols).seq
  b = XM::SFloat.new(rows, 1).seq
  [-> { a + b }, 2 * rows * cols * F32, 'GB']
end

probe('elemwise', 'exp contig') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { XM::NMath.exp(src) }, 2 * rows * cols * F32, 'GB']
end

probe('elemwise', 'exp colslice') do |rows, cols|
  src = XM::SFloat.new(rows, cols * 2).seq
  v = src[true, 0...cols]
  [-> { XM::NMath.exp(v) }, 2 * rows * cols * F32, 'GB']
end

probe('elemwise', 'sqrt colslice') do |rows, cols|
  src = XM::SFloat.new(rows, cols * 2).seq.abs
  v = src[true, 0...cols]
  [-> { XM::NMath.sqrt(v) }, 2 * rows * cols * F32, 'GB']
end

probe('elemwise', 'clip contig') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { src.clip(0.0, 100.0) }, 2 * rows * cols * F32, 'GB']
end

probe('elemwise', 'cmp contig') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { src > 0.5 }, rows * cols * F32, 'GB']
end

probe('elemwise', 'cmp colslice') do |rows, cols|
  src = XM::SFloat.new(rows, cols * 2).seq
  v = src[true, 0...cols]
  [-> { v > 0.5 }, rows * cols * F32, 'GB']
end

# --- reductions ------------------------------------------------------------

probe('reduce', 'sum all contig') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { src.sum }, rows * cols * F32, 'GB']
end

probe('reduce', 'sum axis1 contig') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { src.sum(axis: 1) }, rows * cols * F32, 'GB']
end

probe('reduce', 'sum axis0 contig') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { src.sum(axis: 0) }, rows * cols * F32, 'GB']
end

probe('reduce', 'sum axis1 colslice') do |rows, cols|
  src = XM::SFloat.new(rows, cols * 2).seq
  v = src[true, 0...cols]
  [-> { v.sum(axis: 1) }, rows * cols * F32, 'GB']
end

probe('reduce', 'max axis1 contig') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { src.max(axis: 1) }, rows * cols * F32, 'GB']
end

probe('reduce', 'mean axis1 colslice') do |rows, cols|
  src = XM::SFloat.new(rows, cols * 2).seq
  v = src[true, 0...cols]
  [-> { v.mean(axis: 1) }, rows * cols * F32, 'GB']
end

probe('reduce', 'cumsum contig') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { src.cumsum }, 2 * rows * cols * F32, 'GB']
end

probe('reduce', 'count_true') do |rows, cols|
  bit = (XM::SFloat.new(rows, cols).seq > 0.5)
  [-> { bit.count_true }, rows * cols / 8.0, 'GB']
end

# --- GEMM ------------------------------------------------------------------

probe('gemm', 'dot contig') do |rows, cols|
  a = XM::SFloat.new(rows, cols).seq * 0.001
  b = XM::SFloat.new(cols, rows).seq * 0.001
  [-> { a.dot(b) }, 2.0 * rows * cols * rows, 'GFLOP']
end

probe('gemm', 'dot colslice lhs') do |rows, cols|
  a = XM::SFloat.new(rows, cols * 2).seq * 0.001
  b = XM::SFloat.new(cols, rows).seq * 0.001
  v = a[true, 0...cols]
  [-> { v.dot(b) }, 2.0 * rows * cols * rows, 'GFLOP']
end

probe('gemm', 'dot transposed rhs') do |rows, cols|
  a = XM::SFloat.new(rows, cols).seq * 0.001
  b = XM::SFloat.new(rows, cols).seq * 0.001
  [-> { a.dot(b.transpose) }, 2.0 * rows * cols * rows, 'GFLOP']
end

# --- host transfer and allocation ------------------------------------------

probe('host', 'to_binary (D2H)') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { src.to_binary }, rows * cols * F32, 'GB']
end

probe('host', 'from_binary (H2D)') do |rows, cols|
  bin = Array.new(rows * cols, 1.0).pack('f*')
  [-> { XM::SFloat.from_binary(bin, [rows, cols]) }, rows * cols * F32, 'GB']
end

probe('host', 'scalar read a[0,0]') do |rows, cols|
  src = XM::SFloat.new(rows, cols).seq
  [-> { src[0, 0] }, F32, 'GB']
end

probe('alloc', 'zeros') do |rows, cols|
  [-> { XM::SFloat.zeros(rows, cols) }, rows * cols * F32, 'GB']
end

probe('alloc', 'temp chain a+b+c+d') do |rows, cols|
  a = XM::SFloat.new(rows, cols).seq
  b = XM::SFloat.new(rows, cols).seq
  c = XM::SFloat.new(rows, cols).seq
  d = XM::SFloat.new(rows, cols).seq
  [-> { a + b + c + d }, 7 * rows * cols * F32, 'GB']
end

# --- main ------------------------------------------------------------------

version = begin
  XM::NArray::VERSION
rescue StandardError
  'unknown'
end

puts "ruby      : #{RUBY_VERSION} (#{RUBY_PLATFORM})"
puts "backend   : #{XM} #{version}"
puts "params    : COLS=#{COLS} ROWS=#{ROWS.join(',')} REPS=#{REPS}"

tiny = XM::SFloat.new(1).seq
launch = measure(-> { tiny * 2.0 })
puts format('baseline  : one operation = %.2f us (Ruby, launch and synchronize)', launch * 1e6)
puts
puts format('  %-22s %12s  %-22s %s', 'probe', 'throughput', 'us/row', 'verdict')
puts "  #{'-' * 76}"

current_group = nil
PROBES.each do |p|
  next if GROUP && p[:group] != GROUP

  if p[:group] != current_group
    current_group = p[:group]
    puts "  [#{current_group}]"
  end

  begin
    GC.start
    samples = ROWS.map do |rows|
      call, work, unit = p[:setup].call(rows, COLS)
      t = measure(call)
      { rows: rows, time: t, work: work, unit: unit }
    end
  rescue StandardError, NotImplementedError => e
    msg = e.message.to_s.split("\n").first.to_s[0, 46]
    puts format('  %-22s %12s  %-22s %s', p[:name], '-', '-', "ERROR #{e.class}: #{msg}")
    next
  end

  last = samples.last
  rate = last[:work] / last[:time] / 1e9
  per_row = samples.map { |s| s[:time] / s[:rows] * 1e6 }
  spread = per_row.max / [per_row.min, 1e-12].max

  verdict =
    if last[:unit] == 'GFLOP'
      ''
    elsif rate >= BAND_OK
      'ok (bandwidth)'
    elsif spread < FLAT
      format('LAUNCH-BOUND (~%.2f us/row)', per_row.sum / per_row.size)
    else
      '?'
    end

  puts format('  %-22s %8.2f %-3s  %-22s %s',
              p[:name], rate, "#{last[:unit]}/s",
              per_row.map { |v| format('%.2f', v) }.join(' '), verdict)
end

puts
puts '  A row whose us/row barely moves is the one to suspect of launching'
puts '  once per row. Compare it against the baseline above: a path at roughly'
puts '  one launch per row lands near that figure.'
puts '  GEMM rows are reported in GFLOP/s and carry no verdict.'
