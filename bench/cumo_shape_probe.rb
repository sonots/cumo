#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# Cumo shape sweep
#
# cumo_probe.rb runs every kernel family at one shape and reads each member
# against the fastest of its group. What that cannot see is a family whose
# speed depends on the shape it is handed: a reduction along a short row, an
# operand that is broadcast or strided, a copy that transposes. This sweep
# holds the bytes fixed, varies the shape and the layout, and reads every case
# against the same bytes moved down the best path there is for them: the flat
# reduction, the add of two contiguous arrays, the contiguous copy.
#
#   GPU=1 ruby cumo_shape_probe.rb                     # the default sweep
#   GPU=1 GROUP=reduce ruby cumo_shape_probe.rb        # reduce, elementwise or copy
#   GPU=1 ALL=1 ruby cumo_shape_probe.rb               # every row, not just the flagged
#   GPU=1 FLAG=3 ruby cumo_shape_probe.rb              # how far off the reference is called out
#   GPU=1 DTYPE=DFloat ELEMENTS=67108864 ruby cumo_shape_probe.rb
#   GPU=1 JSON=before.json ruby cumo_shape_probe.rb    # keep this run's rows ...
#   GPU=1 COMPARE=before.json ruby cumo_shape_probe.rb # ... and read a later build against them
#   ruby cumo_shape_probe.rb                           # the same sweep on Numo
#
# Every case queues LAUNCHES calls and synchronizes once, and the run opens
# with a few seconds of load. On a laptop the memory clock climbs in steps,
# and a best-of-N that synchronizes between calls can settle on a step below
# the top and read a third low without anything being wrong.
#
# The sweep runs in one process. The cases work in place or reduce, so the
# only results allocated are the reductions', a thousandth of their input.
#
# The default 4M elements of SFloat is 16 MB, inside the L2 of this class of
# GPU, which is where the activations of a model live. ELEMENTS=67108864 moves
# the sweep to DRAM, where a cost paid per element hides behind the bandwidth.
# ---------------------------------------------------------------------------

require 'json'

GPU = !%w[0 false].include?(ENV['GPU'].to_s.downcase) && !ENV['GPU'].to_s.empty?

if GPU
  require 'cumo/narray'
  XM = Cumo
else
  require 'numo/narray'
  XM = Numo
end

GROUP    = ENV['GROUP']
FLAG     = (ENV['FLAG'] || 2.0).to_f
SHOW_ALL = !ENV['ALL'].to_s.empty?
LAUNCHES = (ENV['LAUNCHES'] || 10).to_i
ROUNDS   = (ENV['ROUNDS'] || 4).to_i
ELEMENTS = (ENV['ELEMENTS'] || (1 << 22)).to_i
DTYPE    = XM.const_get(ENV['DTYPE'] || 'SFloat')
JSON_OUT = ENV['JSON']
COMPARE  = ENV['COMPARE']

def sync
  XM::CUDA::Runtime.cudaDeviceSynchronize if GPU
end

def now
  Process.clock_gettime(Process::CLOCK_MONOTONIC)
end

def elmsz
  DTYPE::ELEMENT_BYTE_SIZE
end

def warm(seconds)
  a = DTYPE.new(1 << 22).fill(0)
  t = now
  a.inplace + 1 while now - t < seconds
  sync
end

# Seconds for one call, best of ROUNDS batches of LAUNCHES calls.
def time_one
  2.times { yield }
  sync
  best = Float::INFINITY
  ROUNDS.times do
    sync
    t = now
    LAUNCHES.times { yield }
    sync
    best = [best, (now - t) / LAUNCHES].min
  end
  best
end

Row = Struct.new(:group, :name, :shape, :us, :gbs, :ref, keyword_init: true) do
  def key
    "#{group}/#{name}"
  end
end

ROWS = []

# bytes is what the case moves, and the reference row is the one this case is
# read against.
def probe(group, name, shape, bytes, ref: nil, &blk)
  return if GROUP && !group.start_with?(GROUP)

  s = time_one(&blk)
  ROWS << Row.new(group: group, name: name, shape: shape.inspect, us: s * 1e6, gbs: bytes / s / 1e9, ref: ref || name)
end

def rand_array(*shape)
  a = DTYPE.new(*shape)
  DTYPE.name.include?('Int') ? a.rand(100) : a.rand
end

# --- reductions ---------------------------------------------------------------
#
# The same ELEMENTS reduced along the last axis, the first axis and a middle
# axis, over rows of every length, all read against the flat reduction of the
# same array. Each output of every case is a reduction over len elements.
def sweep_reduce
  n = ELEMENTS
  es = elmsz
  a = rand_array(n)
  b = rand_array(n)
  methods = %i[sum max mean argmax mulsum].select { |m| a.respond_to?(m) }
  methods.each do |m|
    bytes = (m == :mulsum ? 2 : 1) * n * es
    probe('reduce', "#{m} flat", [n], bytes) { m == :mulsum ? a.mulsum(b) : a.send(m) }
  end
  [4, 16, 64, 256, 1024, 4096].each do |len|
    next if len > n

    rows = n / len
    r1 = 1 << (Math.log2(rows).to_i / 2)
    r2 = rows / r1
    last = a.reshape(rows, len)
    last_b = b.reshape(rows, len)
    first = a.reshape(len, rows)
    first_b = b.reshape(len, rows)
    mid = a.reshape(r1, len, r2)
    mid_b = b.reshape(r1, len, r2)
    methods.each do |m|
      bytes = (m == :mulsum ? 2 : 1) * n * es
      ref = "#{m} flat"
      call = lambda do |x, y, axis|
        m == :mulsum ? x.mulsum(y, axis: axis) : x.send(m, axis: axis)
      end
      probe('reduce', "#{m} len #{len} last", [rows, len], bytes, ref: ref) { call.call(last, last_b, 1) }
      probe('reduce', "#{m} len #{len} first", [len, rows], bytes, ref: ref) { call.call(first, first_b, 0) }
      probe('reduce', "#{m} len #{len} middle", [r1, len, r2], bytes, ref: ref) { call.call(mid, mid_b, 1) }
    end
  end
end

# --- elementwise --------------------------------------------------------------
#
# One [R, C] array added to in place, its operand contiguous, a scalar, a
# broadcast row or column, a slice, a stepped view or a transpose, plus the
# same broadcast through three and four dimensions. The reference is the add
# of two contiguous arrays, which the loop walks as one run.
def sweep_elementwise
  n = ELEMENTS
  es = elmsz
  r = 1024
  c = n / r
  a = rand_array(r, c)
  b = rand_array(r, c)
  row = rand_array(c)
  col = rand_array(r, 1)
  probe('elementwise', 'a + b contiguous', [r, c], 3 * n * es) { a.inplace + b }
  probe('elementwise', 'a * scalar', [r, c], 2 * n * es, ref: 'a + b contiguous') { a.inplace * 2 }
  probe('elementwise', 'a + row broadcast', [r, c], 2 * n * es, ref: 'a + b contiguous') { a.inplace + row }
  probe('elementwise', 'a - column broadcast', [r, c], 2 * n * es, ref: 'a + b contiguous') { a.inplace - col }
  a3 = a.reshape(16, r / 16, c)
  b3 = rand_array(1, r / 16, 1)
  probe('elementwise', 'a + broadcast 3d', [16, r / 16, c], 2 * n * es, ref: 'a + b contiguous') { a3.inplace + b3 }
  a4 = a.reshape(4, 4, r / 16, c)
  b4 = rand_array(1, 4, 1, c)
  probe('elementwise', 'a + broadcast 4d', [4, 4, r / 16, c], 2 * n * es, ref: 'a + b contiguous') { a4.inplace + b4 }
  half = 0...(c / 2)
  probe('elementwise', 'a + b column slice', [r, c / 2], 3 * n * es / 2, ref: 'a + b contiguous') { a[true, half].inplace + b[true, half] }
  every_other = (0..-1) % 2
  probe('elementwise', 'a * scalar stepped rows', [r / 2, c], n * es, ref: 'a + b contiguous') { a[every_other, true].inplace * 2 }
  probe('elementwise', 'a + b transposed views', [c, r], 3 * n * es, ref: 'a + b contiguous') { a.transpose.inplace + b.transpose }
  probe('elementwise', 'a.gt(b) to Bit', [r, c], 2 * n * es, ref: 'a + b contiguous') { a.gt(b) }
  probe('elementwise', 'a.gt(row) to Bit', [r, c], n * es, ref: 'a + b contiguous') { a.gt(row) }
end

# --- copies -------------------------------------------------------------------
#
# A store from a view of every kind into contiguous memory, read against the
# contiguous copy of the same bytes.
def sweep_copy
  n = ELEMENTS
  es = elmsz
  r = 1024
  c = n / r
  a = rand_array(r, c)
  dst = DTYPE.new(r, c).fill(0)
  dst_t = DTYPE.new(c, r).fill(0)
  probe('copy', 'contiguous', [r, c], 2 * n * es) { dst.store(a) }
  half = 0...(c / 2)
  probe('copy', 'column slice', [r, c / 2], n * es, ref: 'contiguous') { dst[true, half].store(a[true, half]) }
  every_other = (0..-1) % 2
  probe('copy', 'stepped rows', [r / 2, c], n * es, ref: 'contiguous') { dst[every_other, true].store(a[every_other, true]) }
  probe('copy', 'reversed rows', [r, c], 2 * n * es, ref: 'contiguous') { dst.store(a.reverse(0)) }
  probe('copy', 'transposed', [c, r], 2 * n * es, ref: 'contiguous') { dst_t.store(a.transpose) }
  idx = XM::Int32.new(r / 2).seq * 2
  probe('copy', 'index-backed rows', [r / 2, c], n * es, ref: 'contiguous') { dst[0...(r / 2), true].store(a[idx, true]) }
end

# --- run ----------------------------------------------------------------------

version = GPU ? Cumo::NArray::VERSION : Numo::NArray::VERSION
puts "backend   : #{XM} #{version}"
puts "dtype     : #{DTYPE}, #{ELEMENTS} elements (#{(ELEMENTS * elmsz / 1e6).round(1)} MB), #{LAUNCHES} launches per sync, best of #{ROUNDS}"
warm(3)
sweep_reduce
sweep_elementwise
sweep_copy

refs = ROWS.to_h { |row| [row.key, row] }
flagged = []
puts
puts format('  %-11s %-26s %-18s %9s %8s %8s', 'group', 'case', 'shape', 'us', 'GB/s', 'vs ref')
ROWS.each do |row|
  ref = refs["#{row.group}/#{row.ref}"]
  ratio = ref ? ref.gbs / row.gbs : 1.0
  slow = ratio > FLAG
  flagged << row if slow
  next unless slow || SHOW_ALL || row.ref == row.name

  puts format('  %-11s %-26s %-18s %9.1f %8.1f %7.1fx%s', row.group, row.name, row.shape, row.us, row.gbs, ratio, slow ? '  <--' : '')
end
puts
if flagged.empty?
  puts format('  nothing was more than %.1fx off its reference', FLAG)
else
  puts format('  %d of %d cases more than %.1fx off their reference', flagged.size, ROWS.size, FLAG)
end

if COMPARE
  before = JSON.parse(File.read(COMPARE))
  puts
  puts "  against #{File.basename(COMPARE)} (#{before['meta']['backend']} #{before['meta']['version']}, saved #{before['meta']['saved']})"
  puts format('  %-11s %-26s %9s %9s %8s', 'group', 'case', 'before us', 'after us', 'speedup')
  ROWS.each do |row|
    was = before['rows'][row.key]
    next unless was

    speedup = was['us'] / row.us
    next unless SHOW_ALL || speedup > 1.25 || speedup < 0.8

    puts format('  %-11s %-26s %9.1f %9.1f %7.2fx', row.group, row.name, was['us'], row.us, speedup)
  end
end

if JSON_OUT
  out = { 'meta' => { 'backend' => XM.to_s, 'version' => version, 'dtype' => DTYPE.to_s, 'elements' => ELEMENTS,
                      'saved' => Time.now.strftime('%Y-%m-%d %H:%M') },
          'rows' => ROWS.to_h { |row| [row.key, { 'shape' => row.shape, 'us' => row.us, 'gbs' => row.gbs }] } }
  File.write(JSON_OUT, JSON.pretty_generate(out))
  puts "  wrote #{ROWS.size} rows to #{JSON_OUT}"
end
