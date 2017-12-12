require "test/unit"
require "mkmf-cu/opt"
require "mkmf-cu"

class TestMkmfCuOpt < Test::Unit::TestCase

  def setup
    @opt, @opt_h = build_optparser()
  end

  def test_opt
    argv = ["-pipe", "-Os", "-x", "cu", "-O2", "-Wall", "-arch", "x86_64"]
    parse_ill_short(argv, @opt_h)
    assert_equal(["-Os", "-x", "cu", "-O2", "-Wall", "--arch", "x86_64"],
                 argv)
  end

  def test_Gg
    argv = ["-g", "-G"]
    parse_ill_short(argv, @opt_h)
    @opt.parse(argv)
    assert_equal({"-g" => [""], "-G" => [""]}, @opt_h)
  end

  def test_compiler_option
    @opt_h.merge!({"-shared"=>[""], "-pipe"=>[""]})
    assert_equal(" --compiler-options -pipe", compiler_option(@opt_h))
  end

  def test_Wl
    argv = ["-Wl,-headerpad_max_install_names", "-fstack-protector", "-L/opt/local/lib",
            "-Wl,-undefined,dynamic_lookup", "-Wl,-multiply_defined,suppress"]
    parse_ill_short_with_arg(argv, @opt_h)
    assert_equal(["--Wl=-headerpad_max_install_names", "-fstack-protector", "-L/opt/local/lib",
                  "--Wl=-undefined,dynamic_lookup", "--Wl=-multiply_defined,suppress"],
                 argv)
  end

  def test_linker_option
    @opt_h.merge!({"-Wl"=>["-a", "-b"]})
    assert_equal(" --linker-options -a --linker-options -b",
                 linker_option(@opt_h))
  end

  def test_std
    argv = ["-std=c++11"]
    parse_ill_short_with_arg(argv, @opt_h)
    @opt.parse(argv)
    assert_equal({"-std" => ["c++11"]}, @opt_h)
  end

  def test_compiler_bin
    h = Hash.new{|h, k| h[k] = [] }.merge({"-shared"=>[""], "-pipe"=>[""], "--mkmf-cu-ext"=>["c"]})
    assert_equal(" --compiler-bindir " + RbConfig::CONFIG["CC"],
                 compiler_bin(h))
  end

  def test_mkmf_cu
    assert MakeMakefile::C_EXT.include?("cu")
    assert MakeMakefile::SRC_EXT.include?("cu")

    treat_cu_as_cxx()
    assert !MakeMakefile::C_EXT.include?("cu")
    assert MakeMakefile::CXX_EXT.include?("cu")
  end

end
