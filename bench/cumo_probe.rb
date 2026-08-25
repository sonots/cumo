#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# Cumo kernel sweep
#
# Runs every kernel family cumo builds and reports the ones that are slow for
# their kind. There is no absolute bar: within one group, one dtype and one
# layout, the fastest member is the reference and the others are read against
# it. A bar in GB/s cannot work here, since a Bit operand moves a thirty-second
# of what a float one does and a reduction writes nothing at all.
#
#   GPU=1 ruby cumo_probe.rb                      # the default sweep
#   GPU=1 GROUP=bit ruby cumo_probe.rb            # one group
#   GPU=1 DTYPES=all LAYOUTS=all ruby cumo_probe.rb
#   GPU=1 ALL=1 ruby cumo_probe.rb                # every row, not just the flagged
#   GPU=1 COLS=4096 ROWS=1024 REPS=7 ruby cumo_probe.rb
#   ruby cumo_probe.rb                            # the same sweep on Numo
#
# Every row is also run on a quarter of the rows, which says whether what makes
# it slow is paid per element or once per launch or per row.
#
# Each case runs in a process of its own, which is what the wall clock goes on.
# Measuring allocating cases in sequence does not work in cumo: the pool hands a
# block back only once Ruby has collected the result, and a collection hands
# device memory back to the driver, so what a case reads depends on where it
# sits in the run. Six unary cases in one process put abs at 649 GB/s and copy
# at 94, and reordering them swaps the two.
#
# Coverage is against the kernels the build contains, not against this file's
# list. The build has 1425 kernel launchers over 224 families; ask it what it
# has and ask a run what it fired:
#
#   nm --defined-only ../lib/cumo.so | grep -oP '\S+kernel_launch\S*' | sort -u
#   nsys profile -t cuda ruby cumo_probe.rb
#   nsys stats --report cuda_gpu_kern_sum <rep>
#
# Two things the build has no launcher for and this sweep cannot reach: RObject,
# whose loops stay on the host by design, and the cuDNN entry points, which run
# the library's own kernels. bincount builds one launcher per weight dtype and
# only one of the four is exercised.
# ---------------------------------------------------------------------------

GPU = !%w[0 false].include?(ENV['GPU'].to_s.downcase) && !ENV['GPU'].to_s.empty?

if GPU
  require 'cumo/narray'
  XM = Cumo
else
  require 'numo/narray'
  XM = Numo
end

COLS    = (ENV['COLS'] || 2048).to_i
ROWS    = (ENV['ROWS'] || 512).to_i
REPS    = (ENV['REPS'] || 5).to_i
GROUP   = ENV['GROUP']
FLAG    = (ENV['FLAG'] || 5.0).to_f   # times its group's reference before a row is called out
SHOW_ALL = !ENV['ALL'].to_s.empty?

DTYPE_SETS = {
  'default' => %w[SFloat Int32 Bit],
  'all' => %w[SFloat DFloat Int32 Int64 UInt8 DComplex Bit]
}.freeze
DTYPES = (ENV['DTYPES'] == 'all' ? DTYPE_SETS['all'] : (ENV['DTYPES']&.split(',') || DTYPE_SETS['default']))

LAYOUT_SETS = {
  'default' => %i[contig colslice],
  'all' => %i[contig colslice transpose step2 index]
}.freeze
LAYOUTS = (ENV['LAYOUTS'] == 'all' ? LAYOUT_SETS['all'] : (ENV['LAYOUTS']&.split(',')&.map(&:to_sym) || LAYOUT_SETS['default']))

def sync
  XM::CUDA::Runtime.cudaDeviceSynchronize if GPU
end

def clock
  Process.clock_gettime(Process::CLOCK_MONOTONIC)
end

# Take the minimum, and report how far the reps spread while doing it. A laptop
# throttles, which lifts the mean but not the floor, and the pool occasionally
# grows mid-run, which shows up as a single rep two orders out.
#
# The spread is what says whether the floor is the kernel or a lucky draw. A
# case that allocates its own result reads bimodally: a rep that finds a block
# of the right size in the pool runs at memory bandwidth, and one that has to
# grow the pool pays for fresh managed pages an order or two more. DFloat.cast
# of a 4MB operand gave 723, 18, 17, 586, 641 microseconds in one run of five,
# and its floor works out at 745 GB/s, above what the card can do. Reporting the
# median instead is worse, not better -- it makes every allocating case read at
# its pool luck rather than its kernel, and took a sweep from 39 flags to 109,
# including a plain add at 36x. So keep the floor and refuse to let a row that
# spread like that be the reference its neighbours are read against.
#
# Deliberately no collection anywhere near this. A collection hands the pool's
# blocks back to its own free list rather than to the driver -- total_bytes does
# not fall across GC.start -- but it costs more than the reuse it buys: one
# between the warm-up and the reps took a plain store from 11.9 microseconds to
# 18.3, and two took it to 154.6. What the pool needs is warming, not emptying
# -- see warm_pool below.
def measure(reps = REPS)
  3.times { yield }
  sync
  ts = Array.new(reps) do
    t0 = clock
    yield
    sync
    clock - t0
  end
  [ts.min, ts.max / [ts.min, 1e-15].max]
end

# Leaves a handful of results of the sizes the sweep produces in flight, so the
# pool has touched blocks to hand out and no case pays for first-touch pages
# that its neighbours got for free. Without this the first case of a cell reads
# an order slower than the rest of its group and flags for nothing.
def warm_pool(ctx)
  16.times { ctx.a.copy }
  sync
rescue StandardError
  nil
end

# --- operands --------------------------------------------------------------

# Every layout holds the same number of elements, so a row can be read against
# the same row in another layout as well as against its own group.
# Bit has no seq of its own, so its operand is built from an integer one. The
# pattern is deliberately not all ones: a mask that selects everything makes
# the compaction's output as long as its input and hides nothing, but it also
# never exercises the case where a warp writes fewer bits than it holds.
def base(klass, rows, cols)
  if klass == XM::Bit
    XM::UInt8.new(rows, cols).seq(1) % 3 > 0
  else
    klass.new(rows, cols).seq(1)
  end
end

def operand(klass, rows, cols, layout)
  case layout
  when :contig    then base(klass, rows, cols)
  when :colslice  then base(klass, rows, cols * 2)[true, 0...cols]
  when :transpose then base(klass, cols, rows).transpose
  when :step2     then base(klass, rows, cols * 2)[true, (0...(cols * 2)).step(2)]
  when :index     then base(klass, rows, cols)[true, (0...cols).to_a.rotate(7)]
  else raise ArgumentError, "unknown layout #{layout}"
  end
end

Ctx = Struct.new(:klass, :dtype, :layout, :rows, :cols, :n, :elmsz, :a, :b, :scalar, :mask, :idx, :sq,
                 :unit, :dst, :sc, :usc, :wide, :int, :tdst, :fdst, :bdst)

# A case that allocates its own result does not measure a kernel in cumo, it
# measures whether the pool happened to hold a block of the size it wanted --
# which is an order or two either way and swamps everything else in its group.
# So the elementwise cases write in place, into sc rather than into a, and the
# stores write into a destination made here. Measured that way every one of
# them lands at 11 to 13 microseconds, where allocating forms of the same
# expressions read anywhere between 20 and 800.
#
# Letting sc drift as the reps run it up to infinity costs nothing: exp of it
# stays at 11 microseconds on the device and 720 on the host once every element
# is Inf, so there is nothing to restore between reps.
def context(dtype, layout, rows = ROWS)
  klass = XM.const_get(dtype)
  cols = COLS
  n = rows * cols
  a = operand(klass, rows, cols, layout)
  b = operand(klass, rows, cols, layout)
  sc = operand(klass, rows, cols, layout)
  elmsz = dtype == 'Bit' ? 0.125 : klass.new(1).byte_size.to_f
  scalar = case dtype
           when 'Bit' then 1
           when 'DComplex', 'SComplex' then Complex(2, 1)
           when /Int|Byte/ then 3
           else 2.0
           end
  mask = dtype == 'Bit' ? a : (XM::UInt8.new(rows, cols).seq(1) % 3) > 0
  idx = XM::Int32.new(rows).seq(0)
  side = Math.sqrt(n).to_i
  sq = klass == XM::Bit ? nil : klass.new(side, side).seq(1)
  # the domain most of NMath wants, built once: a case that makes its own input
  # times that input too, which for a cheap kernel is most of what it reports
  unit = klass == XM::Bit ? nil : (a.abs % 1.0 rescue nil)
  usc = unit&.copy
  # The destinations the store group writes into, allocated here rather than in
  # the timed body. fill(0) is what makes the pages exist: new() alone does not
  # allocate until something touches the array.
  dst = klass.new(rows, cols).fill(dtype == 'Bit' ? 1 : 0)
  bdst = XM::Bit.new(rows, cols).fill(1)
  wide = klass == XM::Bit ? nil : (klass == XM::DFloat ? XM::SFloat : XM::DFloat).new(rows, cols).fill(0)
  int = (klass == XM::Int64 ? XM::Int32 : XM::Int64).new(rows, cols).fill(0)
  tdst = klass.new(cols, rows).fill(dtype == 'Bit' ? 1 : 0)
  fdst = klass.new(n).fill(dtype == 'Bit' ? 1 : 0)
  Ctx.new(klass, dtype, layout, rows, cols, n, elmsz, a, b, scalar, mask, idx, sq,
          unit, dst, sc, usc, wide, int, tdst, fdst, bdst)
end

# --- the case table --------------------------------------------------------

# work returns the bytes a healthy kernel has to move, given the context. It is
# per group rather than per case, so every member of a group is measured on the
# same footing and the ratio between them means something.
WORK = {
  unary: ->(c) { 2 * c.n * c.elmsz },
  binary: ->(c) { 3 * c.n * c.elmsz },
  scalar: ->(c) { 2 * c.n * c.elmsz },
  math: ->(c) { 2 * c.n * c.elmsz },
  cmp: ->(c) { c.n * c.elmsz + c.n / 8.0 },
  bit: ->(c) { 3 * c.n / 8.0 },
  reduce: ->(c) { c.n * c.elmsz },
  axis: ->(c) { c.n * c.elmsz },
  scan: ->(c) { 2 * c.n * c.elmsz },
  compact: ->(c) { c.n / 8.0 + c.n * 8.0 },
  index: ->(c) { 2 * c.n * c.elmsz },
  store: ->(c) { 2 * c.n * c.elmsz },
  sort: ->(c) { 2 * c.n * c.elmsz },
  alloc: ->(c) { c.n * c.elmsz },
  host: ->(c) { c.n * c.elmsz }
}.freeze

CASES = []

# dt names which dtypes a case applies to: :float, :int, :complex, :real, :bit,
# :num (anything but Bit), or :any.
#
# peer: false marks a case that has to allocate its result and has no in-place
# form -- two outputs, or a subscript that builds its own array. Its number is
# still printed, but it is neither flagged nor allowed to be a reference, since
# what separates it from its group is the allocation and not the kernel.
def op(group, name, dt = :num, work: nil, peer: true, &body)
  CASES << { group: group, name: name, dt: dt, work: work, peer: peer, body: body }
end

def applies?(dt, dtype)
  case dt
  when :any then true
  when :num then dtype != 'Bit'
  when :bit then dtype == 'Bit'
  when :float then dtype.start_with?('SFloat', 'DFloat')
  when :real then %w[SFloat DFloat Int32 Int64 UInt8].include?(dtype)
  when :int then dtype.start_with?('Int', 'UInt')
  when :complex then dtype.end_with?('Complex')
  when :signed then !dtype.start_with?('UInt') && dtype != 'Bit'
  else false
  end
end

# elementwise, one operand
op(:unary, 'abs') { |c| c.sc.inplace.abs }
op(:unary, 'sign', :real) { |c| c.sc.inplace.sign }
op(:unary, 'square') { |c| c.sc.inplace.square }
# 1/x truncates to zero on the integer types, so a second pass over the scratch
# would divide by it. Those keep the allocating form, and are read for coverage
# rather than against their peers.
op(:unary, 'reciprocal', :float) { |c| c.sc.inplace.reciprocal }
op(:unary, 'reciprocal', :complex) { |c| c.sc.inplace.reciprocal }
op(:unary, 'reciprocal', :int, peer: false) { |c| c.a.reciprocal }
op(:unary, '-@') { |c| -c.sc.inplace }
op(:unary, 'floor', :float) { |c| c.sc.inplace.floor }
op(:unary, 'ceil', :float) { |c| c.sc.inplace.ceil }
op(:unary, 'round', :float) { |c| c.sc.inplace.round }
op(:unary, 'rint', :float) { |c| c.sc.inplace.rint }
op(:unary, 'trunc', :float) { |c| c.sc.inplace.trunc }
op(:unary, 'conj', :complex) { |c| c.sc.inplace.conj }
op(:unary, 'real', :complex) { |c| c.a.real }
op(:unary, 'imag', :complex) { |c| c.a.imag }
op(:unary, 'copy') { |c| c.dst.store(c.a) }

# elementwise, two operands
op(:binary, 'add') { |c| c.sc.inplace + c.b }
op(:binary, 'sub') { |c| c.sc.inplace - c.b }
op(:binary, 'mul') { |c| c.sc.inplace * c.b }
op(:binary, 'div') { |c| c.sc.inplace / c.b }
op(:binary, 'mod', :real) { |c| c.sc.inplace % c.b }
op(:binary, 'pow') { |c| c.sc.inplace**c.b }
op(:binary, 'copysign', :float) { |c| c.sc.inplace.copysign(c.b) }
op(:binary, 'left_shift', :int) { |c| c.sc.inplace << 1 }
op(:binary, 'right_shift', :int) { |c| c.sc.inplace >> 1 }
op(:binary, 'clip') { |c| c.sc.inplace.clip(c.b, c.b) }

# the scalar path, which is a kernel of its own for every one of these
op(:scalar, 'add_s') { |c| c.sc.inplace + c.scalar }
op(:scalar, 'sub_s') { |c| c.sc.inplace - c.scalar }
op(:scalar, 'mul_s') { |c| c.sc.inplace * c.scalar }
op(:scalar, 'div_s') { |c| c.sc.inplace / c.scalar }
op(:scalar, 'mod_s', :real) { |c| c.sc.inplace % c.scalar }
op(:scalar, 'pow_s') { |c| c.sc.inplace**c.scalar }
op(:scalar, 'pow_int_s', :float) { |c| c.sc.inplace**3 }
op(:scalar, 'clip_s') { |c| c.sc.inplace.clip(0, 100) }
op(:scalar, 'clip_min_s') { |c| c.sc.inplace.clip(0, nil) }
op(:scalar, 'clip_max_s') { |c| c.sc.inplace.clip(nil, 100) }
op(:scalar, 'scalar_left') { |c| c.scalar - c.sc.inplace }

# NMath
%w[sqrt exp exp2 exp10 expm1 log log2 log10 log1p sin cos tan asin acos atan
   sinh cosh tanh asinh acosh atanh cbrt erf erfc sinc].each do |f|
  op(:math, f, :float) { |c| XM::NMath.send(f, c.usc.inplace) }
end
op(:math, 'atan2', :float) { |c| XM::NMath.atan2(c.sc.inplace, c.b) }
# The Ruby Float makes NMath dispatch to DFloat::Math, so this one allocates a
# double array and runs the double kernel however it is written. That is the
# finding, not an artefact of the harness -- see SFloat::Math.atan2 for the
# same call at a seventh of the time.
op(:math, 'atan2_s', :float) { |c| XM::NMath.atan2(c.sc.inplace, 1.0) }
op(:math, 'hypot', :float) { |c| XM::NMath.hypot(c.sc.inplace, c.b) }
op(:math, 'ldexp', :float) { |c| XM::NMath.ldexp(c.sc.inplace, c.idx[0]) }
op(:math, 'frexp', :float, peer: false) { |c| XM::NMath.frexp(c.a) }
op(:math, 'modf', :float, peer: false) { |c| c.a.modf }

# comparisons, whose output is a Bit array a thirty-second the size, so what it
# costs to allocate does not carry the measurement the way a float one does
op(:cmp, 'gt') { |c| c.a > c.scalar }
op(:cmp, 'ge') { |c| c.a >= c.scalar }
op(:cmp, 'lt') { |c| c.a < c.scalar }
op(:cmp, 'le') { |c| c.a <= c.scalar }
op(:cmp, 'eq') { |c| c.a.eq(c.scalar) }
op(:cmp, 'ne') { |c| c.a.ne(c.scalar) }
op(:cmp, 'gt_arr') { |c| c.a > c.b }
op(:cmp, 'eq_arr') { |c| c.a.eq(c.b) }
op(:cmp, 'nearly_eq', :float) { |c| c.a.nearly_eq(c.b) }
op(:cmp, 'isnan', :float) { |c| c.a.isnan }
op(:cmp, 'isinf', :float) { |c| c.a.isinf }
op(:cmp, 'isfinite', :float) { |c| c.a.isfinite }
op(:cmp, 'isposinf', :float) { |c| c.a.isposinf }
op(:cmp, 'isneginf', :float) { |c| c.a.isneginf }
op(:cmp, 'signbit', :float) { |c| c.a.signbit }
op(:cmp, 'cast_to_bit', :real) { |c| c.bdst.store(c.a) }

# Bit's own operations
op(:bit, 'and', :bit) { |c| c.a & c.b }
op(:bit, 'or', :bit) { |c| c.a | c.b }
op(:bit, 'xor', :bit) { |c| c.a ^ c.b }
op(:bit, 'not', :bit) { |c| ~c.a }
op(:bit, 'and_s', :bit) { |c| c.a & 1 }
op(:bit, 'eq', :bit) { |c| c.a.eq(c.b) }
op(:bit, 'store', :bit) { |c| c.dst.store(c.a) }
op(:bit, 'fill', :bit) { |c| c.dst.fill(1) }

# reductions over the whole array
op(:reduce, 'sum') { |c| c.a.sum }
op(:reduce, 'sum_nan', :float) { |c| c.a.sum(nan: true) }
op(:reduce, 'prod') { |c| c.a.prod }
op(:reduce, 'mean', :float) { |c| c.a.mean }
op(:reduce, 'mean_nan', :float) { |c| c.a.mean(nan: true) }
op(:reduce, 'stddev', :float) { |c| c.a.stddev }
op(:reduce, 'var', :float) { |c| c.a.var }
op(:reduce, 'rms', :float) { |c| c.a.rms }
op(:reduce, 'min', :real) { |c| c.a.min }
op(:reduce, 'max', :real) { |c| c.a.max }
op(:reduce, 'max_nan', :float) { |c| c.a.max(nan: true) }
op(:reduce, 'minmax', :real) { |c| c.a.minmax }
op(:reduce, 'ptp', :real) { |c| c.a.ptp }
op(:reduce, 'max_index', :real) { |c| c.a.max_index }
op(:reduce, 'min_index', :real) { |c| c.a.min_index }
op(:reduce, 'argmax', :real) { |c| c.a.argmax }
op(:reduce, 'argmin', :real) { |c| c.a.argmin }
op(:reduce, 'mulsum') { |c| c.a.mulsum(c.b) }
op(:reduce, 'median', :real) { |c| c.a.median }
op(:reduce, 'count_true', :bit) { |c| c.a.count_true }
op(:reduce, 'count_false', :bit) { |c| c.a.count_false }
op(:reduce, 'all?', :bit) { |c| c.a.all? }
op(:reduce, 'any?', :bit) { |c| c.a.any? }

# reductions along one axis, where the addressing differs from the whole array
op(:axis, 'sum axis0') { |c| c.a.sum(axis: 0) }
op(:axis, 'sum axis1') { |c| c.a.sum(axis: 1) }
op(:axis, 'max axis1', :real) { |c| c.a.max(axis: 1) }
op(:axis, 'mean axis1', :float) { |c| c.a.mean(axis: 1) }
op(:axis, 'stddev axis1', :float) { |c| c.a.stddev(axis: 1) }
op(:axis, 'max_index axis1', :real) { |c| c.a.max_index(axis: 1) }
op(:axis, 'mulsum axis1') { |c| c.a.mulsum(c.b, axis: 1) }
op(:axis, 'count_true axis1', :bit) { |c| c.a.count_true(axis: 1) }
op(:axis, 'all? axis1', :bit) { |c| c.a.all?(axis: 1) }

# scans
op(:scan, 'cumsum') { |c| c.sc.inplace.cumsum }
op(:scan, 'cumprod') { |c| c.sc.inplace.cumprod }
op(:scan, 'cumsum_nan', :float) { |c| c.sc.inplace.cumsum(nan: true) }

# compaction, whose output is an index array
op(:compact, 'where', :bit) { |c| c.a.where }
op(:compact, 'where2', :bit) { |c| c.a.where2 }
op(:compact, 'mask') { |c| c.a[c.mask] }

# Subscripting, where the group's own work is wrong for three of the five: a
# subscript that returns a view writes one index array and never reads the
# array it indexes, and a scalar assignment to half the rows writes half the
# array and reads nothing. Left at 2*n*elmsz the scalar assignment reports
# what the card cannot do and becomes the reference the rest are read against.
op(:index, 'aref index', :num, peer: false, work: ->(c) { c.rows * 8 }) { |c| c.a[c.idx, true] }
op(:index, 'aset scalar', work: ->(c) { (c.rows / 2) * c.cols * c.elmsz }) { |c| c.dst[0...(c.rows / 2), true] = c.scalar }
op(:index, 'aset array') { |c| c.dst[c.idx, true] = c.b }
op(:index, 'at', :num, peer: false, work: ->(c) { c.rows * 8 }) { |c| c.a.at(c.idx, c.idx) }
op(:index, 'diagonal', :num, work: ->(c) { 2 * Math.sqrt(c.n) * c.elmsz }) { |c| c.sq.diagonal.copy }

# store and cast, each into a destination the context already allocated
op(:store, 'store same', :any) { |c| c.dst.store(c.a) }
op(:store, 'cast wider', :num, work: ->(c) { 3 * c.n * c.elmsz }) { |c| c.wide.store(c.a) }
op(:store, 'cast int', :real) { |c| c.int.store(c.a) }
op(:store, 'cast from bit', :bit, work: ->(c) { c.n * (0.125 + 8) }) { |c| c.int.store(c.a) }
op(:store, 'fill', :num, work: ->(c) { c.n * c.elmsz }) { |c| c.dst.fill(c.scalar) }
op(:store, 'seq', :num, work: ->(c) { c.n * c.elmsz }) { |c| c.dst.seq(1) }
op(:store, 'transpose copy', :num) { |c| c.tdst.store(c.a.transpose) }
op(:store, 'reshape copy', :num) { |c| c.fdst.store(c.a.flatten) }

# sorting
op(:sort, 'sort', :real) { |c| c.a.sort }
op(:sort, 'sort_index', :real) { |c| c.a.sort_index }

# allocation and host transfer, which have no peer to be read against
op(:alloc, 'zeros', :num) { |c| c.klass.zeros(c.rows, c.cols) }
op(:alloc, 'chain a+b+a+b', :num) { |c| c.a + c.b + c.a + c.b }
op(:host, 'to_binary', :num) { |c| c.a.to_binary }
op(:host, 'scalar read', :num) { |c| c.a[0, 0] }

# --- run -------------------------------------------------------------------

# One case to a process. Measuring allocating cases in sequence does not work in
# cumo: the pool hands a block back only once Ruby has collected the result, and
# a collection hands device memory back to the driver, so what a case reads
# depends on where it sits in the run. Six unary cases in one process put abs at
# 649 GB/s and copy at 94, and reordering them swaps the two. Half a second of
# process start is the price of a number that means something.

CHILD = ENV['CUMO_PROBE_CASE']

def case_rows(group, name, dtype, layout)
  ctx = context(dtype, layout)
  warm_pool(ctx)
  out = []
  CASES.each do |kase|
    next unless kase[:group].to_s == group && kase[:name] == name
    next unless applies?(kase[:dt], dtype)

    begin
      # The same case on a quarter of the rows says which way it is slow. A cost
      # paid per element takes about four times as long on four times the work;
      # one paid per launch or per row hardly moves. Comparing the times rather
      # than the times per element is what keeps the smaller run's own launch
      # overhead from answering the question.
      t, spread = measure { kase[:body].call(ctx) }
      quarter = context(dtype, layout, [ROWS / 4, 1].max)
      tq, = measure(3) { kase[:body].call(quarter) }
      work = (kase[:work] || WORK[kase[:group]]).call(ctx)
      shape = t / [tq, 1e-15].max > 2.0 ? 'per element' : 'per launch or per row'
      out << ['ROW', group, kase[:name], dtype, layout, t, work, shape, spread,
              kase[:peer] ? 1 : 0].join("\t")
    rescue StandardError, NotImplementedError => e
      out << ['SKIP', group, kase[:name], dtype, layout,
              "#{e.class}: #{e.message.to_s.split("\n").first.to_s[0, 40]}"].join("\t")
    end
  end
  out
end

if CHILD
  group, name, dtype, layout = CHILD.split('|')
  begin
    tiny = XM.const_get(dtype == 'Bit' ? 'SFloat' : dtype).new(1).seq(1)
    puts ['LAUNCH', measure(9) { tiny * 2 }.first].join("\t")
    puts case_rows(group, name, dtype, layout.to_sym)
  rescue StandardError => e
    puts ['CELL', group, name, dtype, layout, "#{e.class}: #{e.message.to_s.split("\n").first.to_s[0, 40]}"].join("\t")
  end
  exit
end

version = begin
  XM::NArray::VERSION
rescue StandardError
  'unknown'
end

wanted = CASES.reject { |k| GROUP && k[:group].to_s != GROUP }
cells = wanted.flat_map do |k|
  DTYPES.flat_map do |dtype|
    next [] unless applies?(k[:dt], dtype)

    LAYOUTS.map { |layout| [k[:group].to_s, k[:name], dtype, layout] }
  end
end

puts "ruby      : #{RUBY_VERSION} (#{RUBY_PLATFORM})"
puts "backend   : #{XM} #{version}"
puts "params    : #{ROWS}x#{COLS} REPS=#{REPS} FLAG=#{FLAG}x"
puts "dtypes    : #{DTYPES.join(' ')}"
puts "layouts   : #{LAYOUTS.join(' ')}"
puts "cases     : #{cells.size} processes, one per case and dtype and layout"
puts

require 'open3'
includes = $LOAD_PATH.reject { |d| d.start_with?(RbConfig::CONFIG['rubylibdir'], RbConfig::CONFIG['sitedir'], RbConfig::CONFIG['vendordir']) }
              .flat_map { |d| ['-I', d] }
child_env = ENV.to_h.merge('COLS' => COLS.to_s, 'ROWS' => ROWS.to_s, 'REPS' => REPS.to_s)

rows = []
skipped = Hash.new { |h, k| h[k] = [] }
launches = []

cells.each_with_index do |(group, name, dtype, layout), i|
  env = child_env.merge('CUMO_PROBE_CASE' => [group, name, dtype, layout].join('|'))
  out, = Open3.capture2(env, RbConfig.ruby, *includes, $PROGRAM_NAME)
  out.each_line do |line|
    f = line.chomp.split("\t")
    case f.first
    when 'LAUNCH' then launches << f[1].to_f
    when 'ROW'
      t = f[5].to_f
      rows << { group: f[1].to_sym, name: f[2], dtype: f[3], layout: f[4].to_sym,
                time: t, rate: f[6].to_f / t / 1e9, shape: f[7], spread: f[8].to_f,
                peer: f[9] != '0' }
    when 'SKIP' then skipped[f[5]] << "#{f[1]}/#{f[2]}/#{f[3]}"
    when 'CELL' then skipped[f[5]] << "#{f[1]}/#{f[2]}/#{f[3]} raised before it could be measured"
    end
  end
  print '.'
  $stdout.flush if (i % 40).zero?
end
puts
puts

launch = launches.empty? ? 0.0 : launches.sort[launches.size / 2]
puts format('launch    : %.2f us for one operation on one element', launch * 1e6)
puts

# Within one group, one dtype and one layout, the fastest member is what the
# hardware can do for that kind of work, so the others are read against it.
#
# A row whose reps spread this far did not measure a kernel, it measured whether
# the pool happened to hold a block of the size it wanted, so its floor is not
# what the hardware can do and it cannot stand as the reference. It still gets a
# ratio of its own -- it is only barred from setting one.
STEADY = 10.0

rows.group_by { |r| [r[:group], r[:dtype], r[:layout]] }.each_value do |cell|
  steady = cell.select { |r| r[:peer] && r[:spread] < STEADY }
  ref = (steady.empty? ? cell : steady).max_by { |r| r[:rate] }
  cell.each { |r| r[:ref] = ref[:rate]; r[:ratio] = ref[:rate] / r[:rate] }
end

NO_PEER = %i[alloc host compact scan sort].freeze
flagged = rows.reject { |r| NO_PEER.include?(r[:group]) }
              .select { |r| r[:peer] }
              .reject { |r| r[:time] < launch * 10 }
              .select { |r| r[:ratio] >= FLAG }
              .sort_by { |r| -r[:ratio] }

puts format('  %-7s %-17s %-8s %-9s %10s %8s  %s', 'group', 'case', 'dtype', 'layout', 'GB/s', 'vs peer', 'shape')
puts "  #{'-' * 84}"
(SHOW_ALL ? rows.sort_by { |r| [r[:group].to_s, r[:name], r[:dtype], r[:layout].to_s] } : flagged).each do |r|
  puts format('  %-7s %-17s %-8s %-9s %10.2f %7.1fx  %s',
              r[:group], r[:name], r[:dtype], r[:layout], r[:rate], r[:ratio], r[:shape])
end
puts format('  (nothing was more than %.1fx off its peers)', FLAG) if flagged.empty? && !SHOW_ALL
puts

puts format('  %d rows over %d cases, %d dtypes, %d layouts', rows.size, CASES.size, DTYPES.size, LAYOUTS.size)
unless skipped.empty?
  puts '  not measured:'
  skipped.sort_by { |_, v| -v.size }.each do |why, what|
    puts format('    %-46s %3d  %s', why, what.size, what.first(3).join(' '))
  end
end
puts
puts '  A row is read against the fastest member of its own group, dtype and'
puts '  layout, never against a fixed number of GB/s: a Bit operand moves a'
puts '  thirty-second of what a float one does, and a reduction writes nothing.'
puts '  alloc, host, compact, scan and sort have no peer to be read against and'
puts '  are never flagged, nor is a row within ten launches of the floor above;'
puts '  ALL=1 prints them with the rest, along with the cases that have to'
puts '  allocate their result and so cannot be compared with in-place peers.'
puts format('  A row whose own reps spread more than %.0fx is read like any other but', STEADY)
puts '  cannot be the reference, since its floor is pool luck rather than what'
puts '  the hardware can do.'
