#include "cumo.h"

VALUE rb_mCumo;

void
Init_cumo(void)
{
  rb_mCumo = rb_define_module("Cumo");
}
