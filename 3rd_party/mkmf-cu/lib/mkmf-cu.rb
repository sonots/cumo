require "mkmf"

BIN_PATH = File.join(File.dirname(__dir__), 'bin', 'mkmf-cu-nvcc')

MKMF_CU = true
MakeMakefile::CONFIG["CC"]  = "#{BIN_PATH} --mkmf-cu-ext=c"
MakeMakefile::CONFIG["CXX"] = "#{BIN_PATH} --mkmf-cu-ext=cxx"
MakeMakefile::C_EXT << "cu"
MakeMakefile::SRC_EXT << "cu"

def treat_cu_as_cxx
  MakeMakefile::C_EXT.delete("cu")
  MakeMakefile::CXX_EXT << "cu"
end

def treat_cu_as_cc
  MakeMakefile::CXX_EXT.delete("cu")
  MakeMakefile::C_EXT << "cu"
end

def use_default_cc_compiler
  MakeMakefile::CONFIG["CC"] = RbConfig::CONFIG["CC"]
  MakeMakefile::C_EXT.delete("cu")
end

def use_default_cxx_compiler
  MakeMakefile::CONFIG["CXX"] = RbConfig::CONFIG["CXX"]
  MakeMakefile::CXX_EXT.delete("cu")
end
