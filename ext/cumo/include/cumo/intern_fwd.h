#ifndef CUMO_INTERN_NARRAY_H
#define CUMO_INTERN_NARRAY_H

/* Add cumo_ prefix to avoid C symbol collisions with Numo without modifying C implementations */

#define na_debug_flag na_debug_flag
extern int na_debug_flag;

#define mCumo rb_mCumo
extern VALUE rb_mCumo;
#define cNArray cumo_cNArray
extern VALUE cumo_cNArray;
#define na_eCastError na_eCastError
extern VALUE na_eCastError;
#define na_eShapeError na_eShapeError
extern VALUE na_eShapeError;
#define na_eOperationError na_eOperationError
extern VALUE na_eOperationError;
#define na_eDimensionError na_eDimensionError
extern VALUE na_eDimensionError;
#define na_eValueError na_eValueError
extern VALUE na_eValueError;
#define na_data_type na_data_type
extern const rb_data_type_t na_data_type;

#define na_cStep na_cStep
extern VALUE na_cStep;

#define sym_reduce cumo_sym_reduce
extern VALUE cumo_sym_reduce;
#define sym_option cumo_sym_option
extern VALUE cumo_sym_option;
#define sym_loop_opt cumo_sym_loop_opt
extern VALUE cumo_sym_loop_opt;
#define sym_init cumo_sym_init
extern VALUE cumo_sym_init;

#endif /* CUMO_INTERN_NARRAY_H */
