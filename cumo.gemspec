# coding: utf-8
# frozen_string_literal: true
lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

cumo_version = File.read(File.join(__dir__, "ext/cumo/include/cumo.h")).match(/CUMO_VERSION "([^"]+)"/)[1]
numo_narray_version = File.read(File.join(__dir__, "numo-narray-version")).strip

Gem::Specification.new do |spec|
  spec.name          = "cumo"
  spec.version       = cumo_version
  spec.authors       = ["Naotoshi Seo"]
  spec.email         = ["sonots@gmail.com"]

  spec.summary       = %q{Cumo is CUDA aware numerical library whose interface is highly compatible with Ruby Numo}
  spec.description   = %q{Cumo is CUDA aware numerical library whose interface is highly compatible with Ruby Numo.}
  spec.homepage      = "https://github.com/sonots/cumo"
  spec.license       = "BSD-3-Clause"

  spec.required_ruby_version = ">= 3.0.0"

  spec.files = `git ls-files -z`.split("\x0").reject do |f|
    f.match(%r{^(test|spec|features)/})
  end
  spec.test_files    = `git ls-files -- {test,spec,features}/*`.split("\n")
  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.extensions    = ["ext/cumo/extconf.rb"]

  spec.add_dependency "numo-narray", numo_narray_version
end
