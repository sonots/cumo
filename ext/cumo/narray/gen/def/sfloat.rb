# frozen_string_literal: true

set name:                "sfloat"
set type_name:           "sfloat"
set full_class_name:     "Cumo::SFloat"
set class_name:          "SFloat"
set class_alias:         "Float32"
set class_var:           "cT"
set ctype:               "float"

set has_math:            true
set is_bit:              false
set is_int:              false
set is_unsigned:         false
set is_float:            true
set is_complex:          false
set is_object:           false
set is_real:             true
set is_comparable:       true
set is_double_precision: false
set is_half:             false
set need_align:          true

set cudnn_dtype:         "CUDNN_DATA_FLOAT"
set cudnn_compute_dtype: "CUDNN_DATA_FLOAT"
set cudnn_scalar_t:      "dtype"
set cudnn_param_class:   "cT"
set cudnn_math_type:     "CUDNN_DEFAULT_MATH"

upcast_rb "Integer"
upcast_rb "Float"
upcast_rb "Complex", "SComplex"

upcast "RObject",  "RObject"
upcast "DComplex", "DComplex"
upcast "SComplex", "SComplex"
upcast "DFloat",   "DFloat"
upcast "SFloat",   "SFloat"
upcast "HFloat",   "SFloat"
upcast "Int64",    "SFloat"
upcast "Int32",    "SFloat"
upcast "Int16",    "SFloat"
upcast "Int8",     "SFloat"
upcast "UInt64",   "SFloat"
upcast "UInt32",   "SFloat"
upcast "UInt16",   "SFloat"
upcast "UInt8",    "SFloat"
