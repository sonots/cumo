begin
  major, minor, _ = RUBY_VERSION.split(/\./)
  require "#{major}.#{minor}/cumo/narray.so"
rescue LoadError
  require "cumo/narray.so"
end

require "cumo/narray/extra"
