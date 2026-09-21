# frozen_string_literal: true

set name:                "bfloat"
set type_name:           "bfloat"
set full_class_name:     "Cumo::BFloat"
set class_name:          "BFloat"
set class_alias:         "BFloat16"
set class_var:           "cT"
set ctype:               "cumo_bfloat"

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
set acc_type:            "float"
set acc_class:           "cumo_cSFloat"
set to_acc:              "cumo_bfloat2float"
set from_acc:            "cumo_float2bfloat"
set acc_zero:            "0.0f"
set acc_one:             "1.0f"
set cublas_dtype:        "CUDA_R_16BF"
set step_down:           "cumo_f16_step_down"
set need_align:          true

set cudnn_dtype:         "CUDNN_DATA_BFLOAT16"
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
upcast "HFloat",   "SFloat"
upcast "BFloat",   "BFloat"
upcast "Int64",    "BFloat"
upcast "Int32",    "BFloat"
upcast "Int16",    "BFloat"
upcast "Int8",     "BFloat"
upcast "UInt64",   "BFloat"
upcast "UInt32",   "BFloat"
upcast "UInt16",   "BFloat"
upcast "UInt8",    "BFloat"
