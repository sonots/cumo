# Cumo

Cumo (pronounced "koomo") is a CUDA-aware, GPU-optimized numerical library that offers a significant performance boost over [Ruby Numo](https://github.com/ruby-numo), while (mostly) maintaining drop-in compatibility.

<img src="https://raw.githubusercontent.com/sonots/cumo-logo/master/logo_transparent.png" alt="cumo logo" title="cumo logo" width="50%">

## Requirements

* Ruby 3.0 or later
* NVIDIA GPU Compute Capability 3.5 (Kepler) or later
* CUDA 11.0 or later

## Preparation

Install CUDA and set your environment variables as follows:

```bash
export CUDA_PATH="/usr/local/cuda"
export CPATH="$CUDA_PATH/include:$CPATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/lib:$LD_LIBRARY_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/lib:$LIBRARY_PATH"
```

To use cuDNN features, install cuDNN and set your environment variables as follows:

```
export CUDNN_ROOT_DIR=/path/to/cudnn
export CPATH=$CUDNN_ROOT_DIR/include:$CPATH
export LD_LIBRARY_PATH=$CUDNN_ROOT_DIR/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDNN_ROOT_DIR/lib64:$LIBRARY_PATH
```

FYI: I use [cudnnenv](https://github.com/unnonouno/cudnnenv) to install cudnn under my home directory like `export CUDNN_ROOT_DIR=/home/sonots/.cudnn/active/cuda`.

## Installation

Add the following line to your Gemfile:

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

### Switching from Numo to Cumo

The following find-and-replace should just work:

```
find . -type f | xargs sed -i -e 's/Numo/Cumo/g' -e 's/numo/cumo/g'
```

If you want to dynamically switch between Numo and Cumo, something like the following will work:

```ruby
if gpu
  require 'cumo/narray'
  xm = Cumo
else
  require 'numo/narray'
  xm = Numo
end

a = xm::DFloat.new(3,5).seq
```

### Incompatibility With Numo

The following methods behave incompatibly with Numo by default for performance reasons:

* `extract`
* `[]`
* `count_true`
* `count_false`

Numo returns a Ruby numeric object for 0-dimensional NArray, while Cumo returns the 0-dimensional NArray instead of a Ruby numeric object.
Cumo differs in this way to avoid synchronization and minimize CPU ⇄ GPU data transfer.

Set the `CUMO_COMPATIBLE_MODE` environment variable to `ON` to force Numo NArray compatibility (for worse performance).

You may enable or disable `compatible_mode` as:

```
require 'cumo'
Cumo.enable_compatible_mode # enable
Cumo.compatible_mode_enabled? #=> true
Cumo.disable_compatible_mode # disable
Cumo.compatible_mode_enabled? #=> false
```

You can also use the following methods which behave like Numo's NArray methods. The behavior of these methods does not depend on `compatible_mode`.

* `extract_cpu`
* `aref_cpu(*idx)`
* `count_true_cpu`
* `count_false_cpu`

### Select a GPU device ID

Set the `CUDA_VISIBLE_DEVICES=id` environment variable, or

```
require 'cumo'
Cumo::CUDA::Runtime.cudaSetDevice(id)
```

where `id` is an integer.

### Disable GPU Memory Pool

GPU memory pool is enabled by default. To disable it, set `CUMO_MEMORY_POOL=OFF`, or:

```
require 'cumo'
Cumo::CUDA::MemoryPool.disable
```

## Documentation

See https://github.com/ruby-numo/numo-narray#documentation, replacing Numo with Cumo.

## Contributions

This project is under active development. See [issues](https://github.com/sonots/cumo/issues) for future works.

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

## Advanced Development Tips

### ccache

[ccache](https://ccache.samba.org/) would be useful to speedup compilation time.
Install ccache and configure with:


```bash
export PATH="$HOME/opt/ccache/bin:$PATH"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/gcc"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/g++"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/nvcc"
```

### Build in parallel

Set `MAKEFLAGS` to specify `make` command options. You can build in parallel as:

```
bundle exec env MAKEFLAG=-j8 rake compile
```

### Specify nvcc --generate-code options

```
bundle exec env CUMO_NVCC_GENERATE_CODE=arch=compute_60,code=sm_60 rake compile
```

This is useful even on development because it makes it possible to skip JIT compilation of PTX to cubin during runtime.

### Run tests with gdb

Compile with debugging enabled:

```
bundle exec DEBUG=1 rake compile
```

Run tests with gdb:

```
bundle exec gdb -x run.gdb --args ruby test/narray_test.rb
```

You may put a breakpoint by calling `cumo_debug_breakpoint()` at C source codes.

### Run tests only a specific line
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

### Run program always synchronizing CPU and GPU

```
bundle exec CUDA_LAUNCH_BLOCKING=1
```

### Show GPU synchronization warnings

Cumo shows warnings if CPU and GPU synchronization occurs if:

```
export CUMO_SHOW_WARNING=ON
```

By default, Cumo shows warnings that occurred at the same place only once.
To show all, multiple warnings, set:

```
export CUMO_SHOW_WARNING=ON
export CUMO_SHOW_WARNING_ONCE=OFF
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/sonots/cumo.

## License

* [LICENSE.txt](./LICENSE.txt)
* [3rd_party/LICENSE.txt](./3rd_party/LICENSE.txt)

## Related Materials

* [Fast Numerical Computing and Deep Learning in Ruby with Cumo](https://speakerdeck.com/sonots/fast-numerical-computing-and-deep-learning-in-ruby-with-cumo) - Presentation Slide at [RubyKaigi 2018](https://rubykaigi.org/2018/presentations/sonots.html#may31)
