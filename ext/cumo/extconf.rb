# frozen_string_literal: true

require 'rbconfig.rb'
require 'fileutils'
require "erb"
require_relative '../../3rd_party/mkmf-cu/lib/mkmf-cu'

def d(file)
  File.join(__dir__, file)
end

def have_numo_narray!
  version_path = d("../../numo-narray-version")
  version = File.read(version_path).strip
  gem_spec = Gem::Specification.find_by_name("numo-narray", version)

  $INCFLAGS += " -I#{gem_spec.gem_dir}/ext/numo/narray"
  if !have_header("numo/narray.h")
    puts "
    Header numo/narray.h was not found. Give pathname as follows:
    % ruby extconf.rb --with-narray-include=narray_h_dir"
    exit(1)
  end

  if RUBY_PLATFORM =~ /cygwin|mingw/
    $LDFLAGS += " -L#{gem_spec.gem_dir}/ext/numo/narray"
    unless have_library("narray", "nary_new")
      puts "libnarray.a not found"
      exit(1)
    end
  end
end

def create_depend
  message "creating depend\n"
  File.open(d("depend"), "w") do |depend|
    depend_erb_path = d("depend.erb")
    File.open(depend_erb_path, "r") do |depend_erb|
      erb = ERB.new(depend_erb.read)
      erb.filename = depend_erb_path
      depend.print(erb.result)
    end
  end
end

rm_f d('include/cumo/extconf.h')

MakeMakefileCuda.install!(cxx: true)

if ENV['DEBUG']
  $CFLAGS << " -g -O0 -Wall"
end
$CXXFLAGS << " -std=c++14"
#$CFLAGS=" $(cflags) -O3 -m64 -msse2 -funroll-loops"
#$CFLAGS=" $(cflags) -O3"
$INCFLAGS = "-I$(srcdir)/include -I$(srcdir)/narray -I$(srcdir)/cuda #{$INCFLAGS}"

$INSTALLFILES = Dir.glob(%w[include/cumo/*.h include/cumo/types/*.h include/cumo/cuda/*.h]).map { |x| [x, '$(archdir)'] }
$INSTALLFILES << ['include/cumo/extconf.h', '$(archdir)']
if /cygwin|mingw/ =~ RUBY_PLATFORM
  $INSTALLFILES << ['libcumo.a', '$(archdir)']
end

srcs = %w(
cumo
narray/narray
narray/array
narray/step
narray/index
narray/index_kernel
narray/ndloop
narray/ndloop_kernel
narray/data
narray/data_kernel
narray/types/bit
narray/types/int8
narray/types/int16
narray/types/int32
narray/types/int64
narray/types/uint8
narray/types/uint16
narray/types/uint32
narray/types/uint64
narray/types/sfloat
narray/types/dfloat
narray/types/scomplex
narray/types/dcomplex
narray/types/robject
narray/types/bit_kernel
narray/types/int8_kernel
narray/types/int16_kernel
narray/types/int32_kernel
narray/types/int64_kernel
narray/types/uint8_kernel
narray/types/uint16_kernel
narray/types/uint32_kernel
narray/types/uint64_kernel
narray/types/sfloat_kernel
narray/types/dfloat_kernel
narray/types/scomplex_kernel
narray/types/dcomplex_kernel
narray/types/robject_kernel
narray/math
narray/SFMT
narray/struct
narray/rand
cuda/cublas
cuda/driver
cuda/memory_pool
cuda/memory_pool_impl
cuda/runtime
cuda/nvrtc
cuda/cudnn
cuda/cudnn_impl
)

$objs = srcs.map { |src| "#{src}.o" }

dir_config("narray")

have_numo_narray!

if have_header("dlfcn.h")
  exit(1) unless have_library("dl")
  exit(1) unless have_func("dlopen")
elsif have_header("windows.h")
  exit(1) unless have_func("LoadLibrary")
end

if have_header("stdbool.h")
  stdbool = "stdbool.h"
else
  stdbool = nil
end

if have_header("stdint.h")
  stdint = "stdint.h"
elsif have_header("sys/types.h")
  stdint = "sys/types.h"
else
  stdint = nil
end

have_type("bool", stdbool)
unless have_type("u_int8_t", stdint)
  have_type("uint8_t", stdint)
end
unless have_type("u_int16_t", stdint)
  have_type("uint16_t", stdint)
end
have_type("int32_t", stdint)
unless have_type("u_int32_t", stdint)
  have_type("uint32_t", stdint)
end
have_type("int64_t", stdint)
unless have_type("u_int64_t", stdint)
  have_type("uint64_t", stdint)
end
have_func("exp10")
have_func("rb_arithmetic_sequence_extract")
have_func("RTYPEDDATA_GET_DATA")

have_var("rb_cComplex")
have_func("rb_thread_call_without_gvl")

create_header d('include/cumo/extconf.h')
$extconf_h = nil # nvcc does not support #include RUBY_EXTCONF_H

# Create *.o directories
FileUtils.mkdir_p('narray')
FileUtils.mkdir_p('cuda')

create_depend

HEADER_DIRS = (ENV['CPATH'] || '').split(File::PATH_SEPARATOR)
LIB_DIRS = (ENV['LIBRARY_PATH'] || '').split(File::PATH_SEPARATOR)
dir_config('cumo', HEADER_DIRS, LIB_DIRS)

have_library('cuda')
have_library('cudart')
have_library('nvrtc')
have_library('cublas')
# have_library('cusolver')
# have_library('curand')
if have_library('cudnn') # TODO(sonots): cuDNN version check
  $CFLAGS << " -DCUDNN_FOUND"
  $CXXFLAGS << " -DCUDNN_FOUND"
end

have_library('stdc++')

create_makefile('cumo')

begin
  require 'extconf_compile_commands_json'
  ExtconfCompileCommandsJson.generate!
rescue LoadError
end
