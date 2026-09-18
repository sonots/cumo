# frozen_string_literal: true

set name:                "uint64"
set type_name:           "uint64"
set full_class_name:     "Cumo::UInt64"
set class_name:          "UInt64"
set class_var:           "cT"
set ctype:               "u_int64_t"

set has_math:            false
set is_bit:              false
set is_int:              true
set is_unsigned:         true
set is_float:            false
set is_complex:          false
set is_object:           false
set is_real:             true
set is_comparable:       true
set is_double_precision: false
set acc_type:            ""
set acc_class:           "cT"
set to_acc:              ""
set from_acc:            ""
set acc_zero:            "m_zero"
set acc_one:             "m_one"
set cublas_dtype:        ""
set cublas_prefix:        ""
set cutype:              ""
set step_down:           ""
set cudnn_dtype:         ""
set need_align:          true

upcast_rb "Integer"
upcast_rb "Float", "DFloat"
upcast_rb "Complex", "DComplex"

upcast "RObject",  "RObject"
upcast "DComplex", "DComplex"
upcast "SComplex", "SComplex"
upcast "DFloat", "DFloat"
upcast "SFloat", "SFloat"
upcast "HFloat",   "HFloat"
upcast "BFloat",   "BFloat"
upcast "Int64",  "Int64"
upcast "Int32",  "Int64"
upcast "Int16",  "Int64"
upcast "Int8",   "Int64"
upcast "UInt64"
upcast "UInt32"
upcast "UInt16"
upcast "UInt8"
