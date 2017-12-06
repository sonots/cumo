# Cumo

Cumo is CUDA-aware numerical library whose interface is highly compatible with [Ruby Numo](https://github.com/ruby-numo).
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

```ruby
gem 'cumo'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install cumo

## Quick start

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

### Advanced Tips

[ccache](https://ccache.samba.org/) would be useful to speedup compilation time.
Install ccache and setup as:


```bash
export PATH="$HOME/opt/ccache/bin:$PATH"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/gcc"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/g++"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/nvcc"
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/sonots/cumo.

## License

[LICENSE.txt](./LICENSE.txt)
