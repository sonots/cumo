# Cumo

Cumo is CUDA-aware numerical library whose interface is highly compatible with [Ruby Numo](https://github.com/ruby-numo).
This library provides the benefit of speedup using GPU by replacing Numo with only a small piece of codes.


## Requirements

* Ruby 2.4 or later
* NVIDIA GPU Compute Capability 6.0 (Pascal) or later
* CUDA 9.0 or later

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

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/sonots/cumo.

## License

[LICENSE.txt](./LICENSE.txt)
