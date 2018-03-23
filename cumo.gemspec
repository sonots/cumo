# coding: utf-8
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)
require "cumo/version"

Gem::Specification.new do |spec|
  spec.name          = "cumo"
  spec.version       = Cumo::VERSION
  spec.authors       = ["Naotoshi Seo"]
  spec.email         = ["sonots@gmail.com"]

  spec.summary       = %q{Cumo is CUDA aware numerical library whose interface is highly compatible with Ruby Numo.}
  spec.description   = %q{Cumo is CUDA aware numerical library whose interface is highly compatible with Ruby Numo.}
  spec.homepage      = "https://github.com/sonots/cumo"
  spec.license       = "BSD 3-clause"

  spec.files         = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.test_files    = `git ls-files -- {test,spec,features}/*`.split("\n")
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.extensions    = ["ext/cumo/extconf.rb"]

  spec.add_runtime_dependency "numo-narray", ">= 0.9.1.1"

  spec.add_development_dependency "bundler", "~> 1.15"
  spec.add_development_dependency "rake", "~> 10.0"
end
