#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# 損益分岐点スイープ — 演算種別ごとに Numo と Cumo が入れ替わる要素数を求める
#
#   ruby crossover_bench.rb                     # 両方を 1 プロセスで測る
#   ONLY=cumo JSON=cumo.json ruby crossover_bench.rb
#   ONLY=numo JSON=numo.json ruby crossover_bench.rb
#   MERGE=numo.json,cumo.json ruby crossover_bench.rb   # 別プロセスの結果を合成
#
#   SIZES=1024,4096,16384 TIME_CAP=0.5 ruby crossover_bench.rb
#
# 「大きい配列で何倍速いか」ではなく「小さい配列でいつ負けるか」を測る。
# Cumo の 1 演算あたりの固定費(起動 + Ruby + ndloop)を回収できる要素数は
# 演算種別で違うはずで、その分岐点がそのまま利用者への指針になる。
#
# 各点を 2 通りで測る:
#   連続  BATCH 回まとめて流して最後に 1 度だけ同期する。続けて演算を流す
#         使い方の 1 演算あたりのコストで、分岐点はこちらで判定する。
#   同期  1 演算ごとに同期する。結果をすぐ読み戻す使い方の値。
# Cumo は同期を挟むと次の起動を重ねられないので、同じ演算でも 2〜4 倍違う。
#
# 計測は best-of-N。さらに開始前に GPU を 1 秒空回しして、
# ラップトップのクロック立ち上がりを計測から外す。
#
# Numo 側の実行時間は同一プロセスの履歴で数倍ぶれる(確保が glibc の mmap
# 閾値 128 KB を超えると毎回 page fault が入る)ので、分岐点は桁として読む。
# 気になるときは ONLY と MERGE でプロセスを分けること。
# ---------------------------------------------------------------------------

require 'json'

ONLY      = ENV['ONLY']
JSON_OUT  = ENV['JSON']
MERGE     = ENV['MERGE']
TIME_CAP  = (ENV['TIME_CAP'] || 1.0).to_f  # 1 回がこれを超えたらそのサイズで打ち切り
WARMUP    = (ENV['WARMUP']   || 1.0).to_f  # GPU クロックを上げるための空回し秒数
MIN_REPS  = 3
MAX_REPS  = 30
MIN_TOTAL = 0.15                            # 各点にかける時間の目安
MAX_BATCH = (ENV['BATCH'] || 64).to_i       # 連続実行でまとめる演算数の上限
# まとめている間の出力は生かしておけないので(プールが再利用できなくなる)
# 1 回ごとに解放する。そのぶん確保と解放の代金は連続の値にも含まれる。
BATCH_BYTES = (ENV['BATCH_BYTES'] || (64 << 20)).to_i

SIZES = (ENV['SIZES'] || '256,1024,4096,16384,65536,262144,1048576,4194304,16777216,67108864')
        .split(',').map(&:to_i)

def clock
  Process.clock_gettime(Process::CLOCK_MONOTONIC)
end

# --- バックエンドの読み込み ------------------------------------------------
# Numo と Cumo は名前空間が別なので同一プロセスに同居できる。

BACKENDS = {}

unless ONLY == 'cumo'
  begin
    require 'numo/narray'
    BACKENDS[:numo] = { mod: Numo, sync: -> {} }
  rescue LoadError => e
    warn "numo-narray を読み込めません: #{e.message}"
  end
end

unless ONLY == 'numo'
  begin
    require 'cumo/narray'
    BACKENDS[:cumo] = { mod: Cumo, sync: -> { Cumo::CUDA::Runtime.cudaDeviceSynchronize } }
  rescue LoadError => e
    warn "cumo を読み込めません: #{e.message}"
  end
end

# --- 計測 ------------------------------------------------------------------

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

# 出力を残したままにするとプールが再利用できず確保が毎回新品になるので、
# どちらの測り方でも同じように解放する。2 つの列の差を同期だけにするため。
def run_once(callable, frees)
  r = callable.call
  r.free if frees && r.respond_to?(:free)
end

# 1 演算ごとに同期する。結果をすぐ読み戻す使い方の値。
def measure_sync(callable, sync, frees)
  run_once(callable, frees)
  sync.call
  best_of do
    run_once(callable, frees)
    sync.call
  end
end

# batch 回まとめて流し、最後に 1 度だけ同期する。1 演算あたりに直して返す。
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

# まとめる回数は、生かしておく出力の合計が BATCH_BYTES を超えない範囲で決める。
def batch_for(bytes)
  return MAX_BATCH if bytes <= 0

  (BATCH_BYTES / bytes).clamp(1, MAX_BATCH)
end

# --- 演算の定義 ------------------------------------------------------------
# dims が 2 のものは一辺 sqrt(n) の正方形にする。

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
    # store は書き込み先そのものを返すので、解放してはいけない
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

# --- 実行 ------------------------------------------------------------------

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
  puts "sizes     : #{SIZES.first} .. #{SIZES.last} (#{SIZES.size} 点)"
  puts "params    : TIME_CAP=#{TIME_CAP}s WARMUP=#{WARMUP}s"
  puts

  # GPU のクロックを上げてから測る
  if BACKENDS[:cumo] && WARMUP.positive?
    a = Cumo::SFloat.new(4_000_000).seq
    t0 = clock
    (a * 1.0001).free while clock - t0 < WARMUP
    BACKENDS[:cumo][:sync].call
    puts format('  ウォームアップ %.1f s 完了', WARMUP)
    puts
  end

  # 最初に測る演算はプロセス 1 回きりの初期化を抱えるので、全部を一度捨て打ちする
  BACKENDS.each do |key, be|
    OPS.each do |op|
      call = op[:setup].call(be[:mod], SIZES.first)
      3.times { run_once(call, op.fetch(:frees, true)) }
      be[:sync].call
    rescue StandardError => e
      warn format('  %s / %s の捨て打ち: %s', key, op[:name], e.class)
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
        break if t_sync > TIME_CAP # これ以上大きくすると時間がかかりすぎる
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

# --- 報告 ------------------------------------------------------------------

def fmt_time(t)
  return '        -' unless t

  if t < 1.0e-3
    format('%7.2f us', t * 1e6)
  else
    format('%7.2f ms', t * 1e3)
  end
end

# 比が 1 をまたぐ区間を対数線形で内挿する
def crossover(points)
  bracket = points.each_cons(2).find { |(_, r1), (_, r2)| r1 < 1.0 && r2 >= 1.0 }
  return nil unless bracket

  (n1, r1), (n2, r2) = bracket
  f = Math.log(r1) / (Math.log(r1) - Math.log(r2))
  Math.exp(Math.log(n1) + (f * (Math.log(n2) - Math.log(n1))))
end

# 各点は [1 演算ごとに同期した時間, 連続で流したときの 1 演算あたりの時間]
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
  puts format('  %12s %11s %11s %10s %13s', '要素数', 'numo', 'cumo', 'numo/cumo', 'cumo(同期毎)')

  ratios = []
  marked = false
  SIZES.each do |n|
    tn = pipe(numo, n)
    tc = pipe(cumo, n)
    ratio = (tn && tc) ? tn / tc : nil
    ratios << [n, ratio] if ratio
    mark = if ratio && ratio >= 1.0 && !marked
             marked = true
             '  <== ここから Cumo が速い'
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

puts '== まとめ'
puts format('  %-16s %14s %12s %12s %10s',
            '演算', '分岐点(要素数)', 'cumo 固定費', '同期込み', '最大倍率')
summary.each do |name, cross, fixed, fixed_sync, best|
  puts format('  %-16s %14s %12s %12s %10s',
              name,
              cross ? format('~%d', cross.round) : '-',
              fixed ? format('%.2f us', fixed * 1e6) : '-',
              fixed_sync ? format('%.2f us', fixed_sync * 1e6) : '-',
              best ? format('%.1fx', best) : '-')
end

puts
puts '  cumo の列と分岐点は「連続で流したとき」の値です。'
puts '  「cumo 固定費」は最小サイズでの 1 演算あたり = 起動 + Ruby + ndloop の下限。'
puts '  「同期込み」は 1 演算ごとに完了を待った場合で、その差は同期そのものの'
puts '  代金ではなく、次の起動を前の演算に重ねられない分です。'
puts '  分岐点はこの固定費を CPU 側の計算時間が追い越す点なので、'
puts '  固定費が下がるか、演算あたりの仕事が増えるほど左に動きます。'
puts '  分岐点が出ない演算は、測った範囲では常に一方が速いということです。'
puts
if results[:numo].key?('dot (GEMM)') && !defined?(Numo::Linalg)
  puts '  注意: numo-linalg が無いので dot は Numo 内蔵の素朴な実装が相手です。'
  puts '  BLAS を入れると dot の分岐点は大きく右に動きます。'
end
puts '  注意: Numo 側は同一プロセスの履歴で数倍ぶれます(確保が glibc の mmap'
puts '  閾値 128 KB を超えると毎回 page fault が入るため)。分岐点は桁として'
puts '  読み、詰めるときは ONLY と MERGE でプロセスを分けてください。'
