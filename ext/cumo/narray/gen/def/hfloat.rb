# frozen_string_literal: true

set name:                "hfloat"
set type_name:           "hfloat"
set full_class_name:     "Cumo::HFloat"
set class_name:          "HFloat"
set class_alias:         "Float16"
set class_var:           "cT"
set ctype:               "cumo_half"

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
set is_half:             true
set need_align:          true

set cudnn_dtype:         "CUDNN_DATA_HALF"
set cudnn_compute_dtype: "CUDNN_DATA_FLOAT"
set cudnn_scalar_t:      "float"
set cudnn_param_class:   "cumo_cSFloat"
set cudnn_math_type:     "CUDNN_TENSOR_OP_MATH"

upcast_rb "Integer"
upcast_rb "Float"
upcast_rb "Complex", "SComplex"

upcast "RObject",  "RObject"
upcast "DComplex", "DComplex"
upcast "SComplex", "SComplex"
upcast "DFloat",   "DFloat"
upcast "SFloat",   "SFloat"
upcast "HFloat",   "HFloat"
upcast "Int64",    "HFloat"
upcast "Int32",    "HFloat"
upcast "Int16",    "HFloat"
upcast "Int8",     "HFloat"
upcast "UInt64",   "HFloat"
upcast "UInt32",   "HFloat"
upcast "UInt16",   "HFloat"
upcast "UInt8",    "HFloat"
