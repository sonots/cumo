require "bundler/gem_tasks"
require "rake/testtask"

Rake::TestTask.new(:test) do |t|
  t.libs << "test"
  t.libs << "lib"
  t.test_files = FileList["test/**/*_test.rb"]
end

task :compile do
  sh 'cd ext/cumo && ruby extconf.rb && make && make build-ctest'
end

task :ctest do
  sh 'cd ext/cumo && ruby extconf.rb && make run-ctest'
end

task :clean do
  sh 'cd ext/cumo && make clean'
end

task :docs do
  dir = "ext/cumo/narray"
  src = %w[array.c data.c index.c math.c narray.c rand.c struct.c].
    map{|s| File.join(dir,s)} +
    [File.join(dir,"types/*.c"), "lib/cumo/narray/extra.rb"]
  sh "cd ext/cumo; ruby extconf.rb; make src"
  sh "rm -rf docs .yardoc; yard doc -o docs -m markdown -r README.md #{src.join(' ')}"
end
task :doc => :docs

task :gdb do
  sh "gdb -x run.gdb --args ruby -I. ./test.rb"
end

task :default => [:clobber, :compile, :test]

desc 'Open an irb session preloaded with the gem library'
task :console do
  sh 'irb -rubygems -I lib'
end
task :c => :console
