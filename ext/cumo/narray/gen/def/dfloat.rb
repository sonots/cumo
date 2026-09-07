# frozen_string_literal: true

set name:                "dfloat"
set type_name:           "dfloat"
set full_class_name:     "Cumo::DFloat"
set class_name:          "DFloat"
set class_alias:         "Float64"
set class_var:           "cT"
set ctype:               "double"

set has_math:            true
set is_bit:              false
set is_int:              false
set is_unsigned:         false
set is_float:            true
set is_complex:          false
set is_object:           false
set is_real:             true
set is_comparable:       true
set is_double_precision: true
set is_half:             false
set need_align:          true

set cudnn_dtype:         "CUDNN_DATA_DOUBLE"
set cudnn_compute_dtype: "CUDNN_DATA_DOUBLE"
set cudnn_scalar_t:      "dtype"
set cudnn_param_class:   "cT"
set cudnn_math_type:     "CUDNN_DEFAULT_MATH"

upcast_rb "Integer"
upcast_rb "Float"
upcast_rb "Complex", "DComplex"

upcast "RObject",  "RObject"
upcast "DComplex", "DComplex"
upcast "SComplex", "DComplex"
upcast "DFloat",   "DFloat"
upcast "SFloat",   "DFloat"
upcast "HFloat",   "DFloat"
upcast "Int64",    "DFloat"
upcast "Int32",    "DFloat"
upcast "Int16",    "DFloat"
upcast "Int8",     "DFloat"
upcast "UInt64",   "DFloat"
upcast "UInt32",   "DFloat"
upcast "UInt16",   "DFloat"
upcast "UInt8",    "DFloat"
