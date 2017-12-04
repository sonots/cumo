$LOAD_PATH.unshift File.expand_path("../../lib", __FILE__)
require "cumo"

require "test/unit"
require File.join(__dir__, "../ext/numo/narray/narray")
require File.join(__dir__, "../lib/numo/narray/extra")
