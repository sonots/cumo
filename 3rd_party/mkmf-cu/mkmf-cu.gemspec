Gem::Specification.new do |s|
  s.name        = 'mkmf-cu'
  s.version     = '0.1.2'
  s.date        = '2016-03-26'
  s.summary     = "Write Ruby extension in C/C++ with NVIDIA CUDA."
  s.description =
  "Write Ruby extension in C/C++ with NVIDIA CUDA. A simple wrapper command for nvcc and a monkey patch for mkmf."
  s.authors     = ["Takashi Tamura"]
  s.email       = ''
  s.files       = ["lib/mkmf-cu.rb", "lib/mkmf-cu/opt.rb", "LICENSE", "README.md"]
  s.homepage    = 'https://github.com/ruby-accel/mkmf-cu'
  s.license     = 'MIT'
  s.executables << "mkmf-cu-nvcc"

  s.add_development_dependency "rake", "~> 10.0"
end
