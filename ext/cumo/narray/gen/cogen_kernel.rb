#! /usr/bin/env ruby
# frozen_string_literal: true

thisdir = File.dirname(__FILE__)
libpath = File.absolute_path(File.dirname(__FILE__)) + "/../../../../lib"
$LOAD_PATH.unshift libpath

require_relative "narray_def"
require_relative "kernel_parts"

$line_number = false
$kernel_part = "main"

while true
  if ARGV[0] == "-l"
    $line_number = true
    ARGV.shift
  elsif ARGV[0] == "-p"
    ARGV.shift
    $kernel_part = ARGV.shift
  elsif ARGV[0] == "-o"
    ARGV.shift
    $output = ARGV.shift
    require "fileutils"
    FileUtils.rm_f($output)
  else
    break
  end
end

if ARGV.size != 1
  puts "usage:\n  ruby #{$0} [-l] [-p part] [-o output] type_file"
  exit 1
end

type_file, = ARGV
type_name = File.basename(type_file, ".rb")

erb_dir = ["tmpl"]
erb_dir.unshift("tmpl_bit") if (type_name == "bit")
erb_dir.map! { |d| File.join(thisdir, d) }

REDUCE_VARIANT_TEMPLATES = %w[accum accum_index accum_arg].freeze
REDUCE_TEMPLATES = %w[cum layer_norm rms_norm softmax bit_count bit_reduce bit_stat].freeze

abort "unknown kernel part: #{$kernel_part}" unless KERNEL_PARTS.include?($kernel_part)

class ErbPP
  # A template that launches reductions is emitted into every part that holds
  # one of its forms, and each launcher picks its own part.
  def kernel_parts
    case get(:erb_base)
    when "accum_binary" then %w[reduce reduce_nan reduce_from]
    when *REDUCE_VARIANT_TEMPLATES then %w[reduce reduce_nan]
    when *REDUCE_TEMPLATES then %w[reduce]
    else %w[main]
    end
  end

  def kernel_part?(nan: false, from: nil)
    $kernel_part == (from ? "reduce_from" : nan ? "reduce_nan" : "reduce")
  end

  def indexer_dims(narrow = false)
    dims = (0..opt_indexer_ndim).to_a << ""
    dims += opt_indexer_narrow_dims.map { |d| "#{d}n" } if narrow
    dims
  end

  def indexer_switch(kernel, args, narrow: nil)
    launch = "<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(#{args}); break;"
    cases = (0..opt_indexer_ndim).map do |d|
      if narrow && opt_indexer_narrow_dims.include?(d)
        "case #{d}: if (cumo_na_indexer_is_narrow(indexer, narrow_, #{narrow.size})) { #{kernel}_dim#{d}n#{launch} } #{kernel}_dim#{d}#{launch}"
      else
        "case #{d}: #{kernel}_dim#{d}#{launch}"
      end
    end
    switch = "switch (indexer->ndim) { #{cases.join(' ')} default: #{kernel}_dim#{launch} }"
    narrow ? "{ const cumo_na_iarray_t* const narrow_[] = {#{narrow.join(', ')}}; #{switch} }" : switch
  end
end

class DefModule
  def method_code
    @children.select { |c| c.kernel_parts.include?($kernel_part) }.map { |c| c.result }.join("\n")
  end
end

code = DefLib.new do
  set line_number: $line_number
  set erb_dir: erb_dir
  set erb_suffix: "_kernel.cu"
  set ns_var: "mCumo"

  set file_name: $output || ""
  set type_name: type_name
  set lib_name: "cumo_" + type_name

  indexer_h = File.read(File.expand_path("../../../include/cumo/indexer.h", __FILE__))
  set opt_indexer_ndim: indexer_h.match(/CUMO_NA_INDEXER_OPTIMIZED_NDIM (\d+)/)[1].to_i
  set opt_indexer_narrow_dims: indexer_h.scan(/^CUMO_NA_IARRAY_AT_NARROW\((\d+)\)/).flatten.map(&:to_i).sort

  def_class do
    extend NArrayMethod
    extend NArrayType
    eval File.read(type_file), binding, type_file
    eval File.read(File.join(thisdir, "spec.rb")), binding, "spec.rb"
  end
end.result

if $output
  open($output, "w").write(code)
else
  $stdout.write(code)
end
