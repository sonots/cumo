#ifndef CUMO_H
#define CUMO_H

#include "cumo/narray.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

#define CUMO_VERSION "0.1.1"
#define CUMO_VERSION_CODE 11

bool cumo_compatible_mode_enabled_p();
bool cumo_show_warning_enabled_p();
bool cumo_warning_once_enabled_p();

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_H */
