require "fileutils"
require "optparse"
require "rbconfig"

out = "tmp/yard/types"
if $0 == __FILE__
  OptionParser.new do |opt|
    opt.on("-o", "--output DIR") { |v| out = v }
  end.parse!
end

FileUtils.mkdir_p(out)
Dir[File.join(__dir__, "def", "*.rb")].sort.each do |def_rb|
  dst = File.join(out, File.basename(def_rb, ".rb") + ".c")
  system(RbConfig.ruby, File.join(__dir__, "cogen.rb"), "-l", "-o", dst, def_rb, exception: true)
end
