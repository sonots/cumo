require "optparse"
require "rbconfig"

def build_optparser
  opt = OptionParser.new
  opt_h = Hash.new{|h, k| h[k] = [] }

  opt.on("--arch arg") {|v| opt_h["-arch"] << v }
  opt.on("--std arg") {|v| opt_h["-std"] << v }
  opt.on("--stdlib arg") {|v| opt_h["-stdlib"] << v }

  opt.on("--Wl arg") {|v| opt_h["-Wl"] << v }

  opt.on('--profile') {|v| opt_h["-pg"] << "" }
  opt.on('-g') {|v| opt_h["-g"] << "" }
  opt.on('-G', "--device-debug") {|v| opt_h["-G"] << "" }

  opt.on('-I path') {|v| opt_h["-I"] << v }
  opt.on('-D flag') {|v| opt_h["-D"] << v }
  opt.on('-W flag') {|v| opt_h["-W"] << v }
  opt.on('-o output') {|v| opt_h["-o"] << v }
  opt.on('-c file') {|v| opt_h["-c"] << v }
  opt.on('-f flag') {|v| opt_h["-f"] << v }
  opt.on('-l file') {|v| opt_h["-l"] << v }
  opt.on('-L path') {|v| opt_h["-L"] << v }
  opt.on('-x pat', "--x pat") {|v| opt_h["-x"] << v }
  opt.on('-O num'){|v| opt_h["-O"] << v if /[0-9]/ =~ v }
  opt.on('--mkmf-cu-ext ext'){|v| opt_h["--mkmf-cu-ext"] << v}

  return [opt, opt_h]
end

def parse_ill_short(argv, opt_h)
  ["-shared", "-rdynamic", "-dynamic", "-bundle",  "-pipe", "-pg"].each{|opt|
    if ind = argv.find_index(opt)
      opt_h[opt] << ""
      argv.delete_at(ind)
    end
  }
  ["-arch", "-std", "-stdlib"].each{|opt|
    if ind = argv.find_index(opt)
      argv[ind] = "-" + opt
    end
  }
end

def parse_ill_short_with_arg(argv, opt_h)  
  [/\A(\-stdlib)=(.*)/, /\A(\-std)=(.*)/, /\A(\-Wl),(.*)/].each{|reg|
    argv.each{|e|
      if reg =~ e
        e[0..-1] = "-" + $1 + '=' + $2
      end
    }
  }
end

def compiler_option(opt_h)
  ret = ""
  ["-f", "-W", "-pipe"].each{|op|
    opt_h[op].each{|e|
      ret << " --compiler-options " + "#{op}#{e}"
    }
  }
  ["-stdlib", "-std"].each{|op|
    opt_h[op].each{|e|
      ret << " --compiler-options " + "#{op}=#{e}"
    }
  }
  return ret
end

def linker_option(opt_h)
  ret = " -shared "
  ["-dynamic", "-bundle"].each{|op|
    opt_h[op].each{|e|
      ret << " --linker-options " + op
    }
  }
  opt_h["-Wl"].each{|e|
    ret << " --linker-options " + e
  }
  return ret
end

def compiler_bin(opt_h)
  if opt_h["--mkmf-cu-ext"][0] == "c"
    " --compiler-bindir " + RbConfig::CONFIG["CC"]
  elsif opt_h["--mkmf-cu-ext"][0] == "cxx"
    " --compiler-bindir " + RbConfig::CONFIG["CXX"]
  end
end

def generate_compiling_command_line(opt_h)
  s = ""
  # options nvcc can uderstatnd
  ["-std", "-pg", "-g", "-G", "-x", "-I", "-D", "-o", "-c", "-O"].each{|op|
    opt_h[op].each{|e|
      case op
      when "-o", "-c", "-x", "-std"
        s << " #{op} #{e}"
      else
        s << " #{op}#{e}"
      end
    }
  }
  s << compiler_option(opt_h)
  s << compiler_bin(opt_h)
  return s
end

def generate_linking_command_line(argv, opt_h)
  s = ""
  ["-L", "-l", "-o", "-c", "-O"].each{|op|
    opt_h[op].each{|e|
      case op
      when "-o", "-c"
        s << " #{op} #{e}"
        s << " " + argv.join(" ") + " " if op == "-o"
      else
        s << " #{op}#{e}"
      end
    }
  }
  s << compiler_option(opt_h)
  s << linker_option(opt_h)
  s << compiler_bin(opt_h)
  return s
end
