# frozen_string_literal: true

require "bundler/gem_tasks"
require "rake/testtask"
require "rubygems/package"

Rake::Task["release"].clear
desc "Tag the version, build the gem at the tag, and push the branch, the tag and the gem"
task :release, [:remote] do |_, args|
  git = ->(*argv) { IO.popen(["git", *argv], err: File::NULL, &:read).strip }
  version = File.read(File.join(__dir__, "lib/cumo/version.rb"))[/VERSION = "([^"]+)"/, 1]
  tag = "v#{version}"
  gem_file = "pkg/cumo-#{version}.gem"
  remote = args[:remote] || "origin"
  raise "There are files that need to be committed first." unless system("git", "diff", "--quiet") && system("git", "diff-index", "--quiet", "--cached", "HEAD")

  tagged = git.call("rev-parse", "-q", "--verify", "#{tag}^{commit}")
  raise "#{tag} is on #{tagged[0, 8]}, not on HEAD." unless tagged.empty? || tagged == git.call("rev-parse", "HEAD")

  sh "git", "tag", "-m", "Version #{version}", tag if tagged.empty?
  begin
    mkdir_p "pkg"
    sh "gem", "build", "cumo.gemspec", "--output", gem_file
    revision = Gem::Package.new(gem_file).spec.metadata["cumo_revision"]
    raise "#{gem_file} records #{revision.inspect} as cumo_revision, not #{tag.inspect}." unless revision == tag
  rescue StandardError
    sh "git", "tag", "-d", tag if tagged.empty?
    raise
  end
  sh "git", "push", remote, "HEAD"
  sh "git", "push", remote, "refs/tags/#{tag}"
  sh "gem", "push", gem_file unless ENV["gem_push"] == "no"
end

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
end

require "rake/extensiontask"
ext = Rake::ExtensionTask.new("cumo")
# The Makefile lists an object for each kernel part, so a new part needs extconf.rb to run again.
file "#{ext.tmp_dir}/#{ext.platform}/#{ext.name}/#{RUBY_VERSION}/Makefile" => "ext/cumo/narray/gen/kernel_parts.rb"

desc 'Build the C++ memory pool test harness without running it (running it needs a GPU)'
task :build_ctest do
  sh 'cd ext/cumo && ruby extconf.rb && make build-ctest'
end

task :ctest => :build_ctest do
  sh 'cd ext/cumo && make run-ctest'
end

# Both need a GPU, so CI runs neither. Tie them together so the one place
# they do run does not skip half of them.
task :test => :ctest

task :docs do
  dir = "ext/cumo"
  srcs = %w[array.c data.c index.c math.c narray.c rand.c struct.c].map { |s| File.join(dir, "narray", s) }
  srcs += %w[cublas.c driver.c nvrtc.c runtime.c memory_pool.cpp].map { |s| File.join(dir, "cuda", s) }
  srcs << File.join(dir, "narray", "types/*.c")
  srcs << "lib/cumo/narray/extra.rb"
  sh "cd ext/cumo; ruby extconf.rb; make src"
  sh "rm -rf docs .yardoc; yard doc -o docs -m markdown -r README.md #{srcs.join(' ')}"
end
task :doc => :docs

task :gdb do
  sh "gdb -x run.gdb --args ruby -I. ./test.rb"
end

task :default => [:compile, :test]

desc 'Open an irb session preloaded with the gem library'
task :console do
  sh 'irb -rubygems -I lib'
end
task :c => :console
