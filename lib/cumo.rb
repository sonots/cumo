begin
  require_relative 'cumo.so'
rescue LoadError
  require_relative File.join(__dir__, '../ext/cumo/cumo')
end
require_relative 'cumo/cuda'
require_relative 'cumo/narray/extra'
