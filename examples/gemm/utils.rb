# frozen_string_literal: true

require 'cumo/narray'

def read_code(code_filename, params)
  code = File.read(code_filename)
  params.each do |k, v|
    code = "#define #{k} #{v}\n" + code
  end
  code
end

def benchmark(func, args, n_run)
  times = []
  n_run.times do
    start = Cumo::CUDA::Event.new
    stop = Cumo::CUDA::Event.new
    start.record
    func.call(*args)
    stop.record
    stop.synchronize
    times << Cumo::CUDA.get_elapsed_time(start, stop) # milliseconds
  end
  times
end
