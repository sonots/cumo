#! /usr/bin/env ruby
# frozen_string_literal: true

thisdir = File.dirname(__FILE__)
libpath = File.absolute_path(File.dirname(__FILE__)) + "/../../../../lib"
$LOAD_PATH.unshift libpath

require_relative "narray_def"

$line_number = false

while true
  if ARGV[0] == "-l"
    $line_number = true
    ARGV.shift
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
  puts "usage:\n  ruby #{$0} [-l] erb_base [type_file]"
  exit 1
end

type_file, = ARGV
type_name = File.basename(type_file, ".rb")

erb_dir = ["tmpl"]
erb_dir.unshift("tmpl_bit") if (type_name == "bit")
erb_dir.map! { |d| File.join(thisdir, d) }

class ErbPP
  def indexer_dims(narrow = false)
    dims = (0..opt_indexer_ndim).to_a << ""
    dims += (2..opt_indexer_narrow_ndim).map { |d| "#{d}n" } if narrow
    dims
  end

  def indexer_switch(kernel, args, narrow: nil)
    launch = "<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(#{args}); break;"
    cases = (0..opt_indexer_ndim).map do |d|
      if narrow && (2..opt_indexer_narrow_ndim).cover?(d)
        "case #{d}: if (cumo_na_indexer_is_narrow(indexer, #{narrow.size}, #{narrow.join(', ')})) { #{kernel}_dim#{d}n#{launch} } #{kernel}_dim#{d}#{launch}"
      else
        "case #{d}: #{kernel}_dim#{d}#{launch}"
      end
    end
    "switch (indexer->ndim) { #{cases.join(' ')} default: #{kernel}_dim#{launch} }"
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
  set opt_indexer_narrow_ndim: indexer_h.match(/CUMO_NA_INDEXER_NARROW_NDIM (\d+)/)[1].to_i

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
