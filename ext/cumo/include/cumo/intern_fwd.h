#ifndef CUMO_INTERN_NARRAY_H
#define CUMO_INTERN_NARRAY_H

/* Add cumo_ prefix to avoid C symbol collisions with Numo without modifying C implementations */

#define mCumo rb_mCumo
extern VALUE rb_mCumo;
#define cNArray cumo_cNArray
extern VALUE cumo_cNArray;
#define nary_eCastError cumo_nary_eCastError
extern VALUE cumo_nary_eCastError;
#define nary_eShapeError cumo_nary_eShapeError
extern VALUE cumo_nary_eShapeError;
#define nary_eOperationError cumo_nary_eOperationError
extern VALUE cumo_nary_eOperationError;
#define nary_eDimensionError cumo_nary_eDimensionError
extern VALUE cumo_nary_eDimensionError;
#define nary_eValueError cumo_nary_eValueError
extern VALUE cumo_nary_eValueError;
#define na_data_type cumo_na_data_type
extern const rb_data_type_t cumo_na_data_type;

#define na_cStep cumo_na_cStep
extern VALUE cumo_na_cStep;

#define sym_reduce cumo_sym_reduce
extern VALUE cumo_sym_reduce;
#define sym_option cumo_sym_option
extern VALUE cumo_sym_option;
#define sym_loop_opt cumo_sym_loop_opt
extern VALUE cumo_sym_loop_opt;
#define sym_init cumo_sym_init
extern VALUE cumo_sym_init;

#endif /* CUMO_INTERN_NARRAY_H */
