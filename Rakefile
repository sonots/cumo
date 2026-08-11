# frozen_string_literal: true

require "bundler/gem_tasks"
require "rake/testtask"

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
end

require "rake/extensiontask"
Rake::ExtensionTask.new("cumo")

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
