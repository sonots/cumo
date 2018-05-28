require "mkmf"

module MakeMakefileCuda
  BIN_PATH = File.join(File.dirname(__dir__), 'bin', 'mkmf-cu-nvcc')

  class << self
    # @params [cxx] Treat .cu files as C++ files
    def install!(cxx: false)
      MakeMakefile::CONFIG["CC"]  = "#{BIN_PATH} --mkmf-cu-ext=c"
      MakeMakefile::CONFIG["CXX"] = "#{BIN_PATH} --mkmf-cu-ext=cxx"
      if cxx
        MakeMakefile::CXX_EXT << "cu"
      else
        MakeMakefile::C_EXT << "cu"
      end
      MakeMakefile::SRC_EXT << "cu"
      @installed = true
    end

    def installed?
      !!@installed
    end

    def uninstall!
      MakeMakefile::CONFIG["CC"] = RbConfig::CONFIG["CC"]
      MakeMakefile::CONFIG["CXX"] = RbConfig::CONFIG["CXX"]
      MakeMakefile::C_EXT.delete("cu")
      MakeMakefile::CXX_EXT.delete("cu")
      @installed = false
    end
  end
end
