# frozen_string_literal: true

lib = File.expand_path("../lib", __FILE__)
$LOAD_PATH.unshift(lib) unless $LOAD_PATH.include?(lib)

cumo_version = File.read(File.join(__dir__, "lib/cumo/version.rb"))[/VERSION = "([^"]+)"/, 1]

Gem::Specification.new do |spec|
  spec.name          = "cumo"
  spec.version       = cumo_version
  spec.authors       = ["Naotoshi Seo"]
  spec.email         = ["sonots@gmail.com"]

  spec.summary       = %q{Cumo is CUDA aware numerical library whose interface is highly compatible with Ruby Numo}
  spec.description   = %q{Cumo is CUDA aware numerical library whose interface is highly compatible with Ruby Numo.}
  spec.homepage      = "https://github.com/sonots/cumo"
  spec.license       = "MIT"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["documentation_uri"] = "https://rubydoc.info/gems/cumo/"
  spec.metadata["changelog_uri"] = "https://github.com/sonots/cumo/blob/master/CHANGELOG.md"
  spec.metadata["bug_tracker_uri"] = "https://github.com/sonots/cumo/issues"
  spec.metadata["rubygems_mfa_required"] = "true"

  spec.required_ruby_version = ">= 3.0.0"

  spec.files = `git ls-files -z`.split("\x0").reject do |f|
    f.start_with?(*%w[.github/ test/ examples/ bin/ bench/ docker- Dockerfile Gemfile .gitignore .rubocop])
  end

  revision = begin
    IO.popen(%W[git -C #{__dir__} describe --tags --match v[0-9]* --dirty --always], err: File::NULL, &:read).strip
  rescue SystemCallError
    ""
  end
  spec.metadata["cumo_revision"] = revision unless revision.empty?

  spec.bindir        = "exe"
  spec.executables   = spec.files.grep(%r{^exe/}) { |f| File.basename(f) }
  spec.require_paths = ["lib"]
  spec.extensions    = ["ext/cumo/extconf.rb"]
end
