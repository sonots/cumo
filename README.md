**This project is still under development. This project received grant from [Ruby Association Grant 2017](http://www.ruby.or.jp/ja/news/20171206).**

# Cumo

Cumo (pronounced like "koomo") is CUDA-aware numerical library whose interface is highly compatible with [Ruby Numo](https://github.com/ruby-numo).
This library provides the benefit of speedup using GPU by replacing Numo with only a small piece of codes.


## Requirements

* Ruby 2.0 or later
* NVIDIA GPU Compute Capability 6.0 (Pascal) or later
* CUDA 9.0 or later

## Preparation

Install CUDA and setup environment variables as follows:

```bash
export CUDA_PATH="/usr/local/cuda"
export CPATH="$CUDA_PATH/include:$CPATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/lib:$LD_LIBRARY_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/lib:$LIBRARY_PATH"
```

## Installation

Add a following line to your Gemfile:

```ruby
gem 'cumo'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install cumo

## How To Use

### Quick start

An example:

```ruby
[1] pry(main)> require "cumo/narray"
=> true
[2] pry(main)> a = Cumo::DFloat.new(3,5).seq
=> Cumo::DFloat#shape=[3,5]
[[0, 1, 2, 3, 4],
 [5, 6, 7, 8, 9],
 [10, 11, 12, 13, 14]]
[3] pry(main)> a.shape
=> [3, 5]
[4] pry(main)> a.ndim
=> 2
[5] pry(main)> a.class
=> Cumo::DFloat
[6] pry(main)> a.size
=> 15
```

### How to switch from Numo to Cumo

Basically, following command should make it work with Cumo.

```
find . -type f | xargs sed -i -e 's/Numo/Cumo/g' -e 's/numo/cumo/g'
```

If you want to switch Numo and Cumo dynamically, following snippets should work:

```ruby
if gpu
  require 'cumo/narray'
  Xumo = Cumo
else
  require 'numo/narray'
  Xumo = Numo
end

a = Xumo::DFloat.new(3,5).seq
```

### Incompatibility With Numo

Following methods behave incompatibly with Numo as default for performance.

* `extract`
* `[]`
* `count_true`
* `count_false`

Numo returns a Ruby numeric object for 0-dimensional NArray, but Cumo returns the 0-dimensional NArray instead of a Ruby numeric object.
This is to avoid synchnoziation between CPU and GPU for performance.

You may set `CUMO_COMPATIBLE_MODE=ON` environment variable to force Cumo NArray behave compatibly with Numo NArray.

You may enable or disable `compatible_mode` as:

```
require 'cumo'
Cumo.enable_compatible_mode # enable
Cumo.compattible_mode_enabled? #=> true
Cumo.disable_compatible_mode # disable
Cumo.compattible_mode_enabled? #=> false
```

You can also use following methods which behaves as Numo NArray's methods. Behaviors of these methods do not depend on `compatible_mode`.

* `extract_cpu`
* `aref_cpu(*idx)`
* `count_true_cpu`
* `count_false_cpu`

### Select a GPU device ID

Set `CUDA_VISIBLE_DEVICES=id` environment variable, or

```
require 'cumo'
Cumo::CUDA::Runtime.cudaSetDevice(id)
```

where `id` is an integer.

### Disable GPU Memory Pool

Set `CUMO_MEMORY_POOL=OFF` environment variable , or

```
require 'cumo'
Cumo::CUDA::MemoryPool.disable
```

if you found GPU memory pool implementation has a bug.

## Documentation

See https://github.com/ruby-numo/numo-narray#documentation and replace Numo to Cumo.

## Development

Install ruby dependencies:

```
bundle install --path vendor/bundle
```

Compile:

```
bundle exec rake compile
```

Run tests:

```
bundle exec rake test
```

Generate docs:

```
bundle exec rake docs
```

## Advanced Tips on Development

### ccache

[ccache](https://ccache.samba.org/) would be useful to speedup compilation time.
Install ccache and setup as:


```bash
export PATH="$HOME/opt/ccache/bin:$PATH"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/gcc"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/g++"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/nvcc"
```

## Run tests only a specific line

`--location` option is available as:

```
bundle exec ruby test/narray_test.rb --location 121
```

### Compile and run tests only a specific type

`DTYPE` environment variable is available as:

```
bundle exec DTYPE=dfloat rake compile
```

```
bundle exec DTYPE=dfloat ruby test/narray_test.rb
```

### Run tests with gdb

Compile with debug option:

```
bundle exec DEBUG=1 rake compile
```

Run tests with gdb:

```
bundle exec gdb -x run.gdb --args ruby test/narray_test.rb
```

You may put a breakpoint by calling `cumo_debug_breakpoint()` at C source codes.

### Run program always synchronizing CPU and GPU

```
bundle exec CUDA_LAUNCH_BLOCKING=1
```

### Source code organizations

* `*_kernel.{h,cuh,cu}` files are for device (CUDA kernels).
    * .cu files are compiled via nvcc.
    * .cu files define C wrapper functions to launch CUDA kernels to enable to be called from .c files.
    * Technically, it is not possible to use CRuby API such as `VALUE` in .cu files.
        * CRuby API is not callable from CUDA kernel because they do not have `__device__` modifier.
        * nvcc does not support `#include RUBY_EXTCONF_H`, so can not include `ruby.h`.
    * (RULE) It is allowed to use C++14 codes in .cu files.
* Rest of `*.{h,c}` files are for host (CPU).
    * Call C wrapper functions defined in .cu files.
    * It can use CRuby API.
    * (RULE) It is not allowed to use C++ codes in host files.

Ruby's `mkmf` (or `extconf.rb`) does not support to specify 3rd compiler such as NVCC for another files of extensions `.cu`.
Therefore, cumo specify a wrapper command `bin/mkmf-cu-nvcc` as a compiler and changes its behavor depending on extensions of files to compile.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/sonots/cumo.

## License

* [LICENSE.txt](./LICENSE.txt)
* [3rd_party/LICENSE.txt](./3rd_party/LICENSE.txt)

## Related Materials

* [Fast Numerical Computing and Deep Learning in Ruby with Cumo](https://speakerdeck.com/sonots/fast-numerical-computing-and-deep-learning-in-ruby-with-cumo) - Presentation Slide at [RubyKaigi 2018](https://rubykaigi.org/2018/presentations/sonots.html#may31)
