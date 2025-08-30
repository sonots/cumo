$LOAD_PATH.unshift File.expand_path("../../lib", __FILE__)
require "cumo"

require "pry"
require "test/unit"

class CumoTestBase < Test::Unit::TestCase
  FLOAT_TYPES = [
    Cumo::DFloat,
    Cumo::DComplex,
  ]

  TYPES = [
    *FLOAT_TYPES,
    Cumo::SFloat,
    Cumo::SComplex,
    Cumo::Int64,
    Cumo::Int32,
    Cumo::Int16,
    Cumo::Int8,
    Cumo::UInt64,
    Cumo::UInt32,
    Cumo::UInt16,
    Cumo::UInt8,

    ## TODO: RObject causes failures on assertion
    # Cumo::RObject,
  ]
end
