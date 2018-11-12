#ifndef CUMO_H
#define CUMO_H

#include "cumo/narray.h"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

#define CUMO_VERSION "0.2.1"
#define CUMO_VERSION_CODE 21

bool cumo_compatible_mode_enabled_p();
bool cumo_show_warning_enabled_p();
bool cumo_warning_once_enabled_p();

#define CUMO_SHOW_WARNING_ONCE( c_str ) \
    { \
        if (cumo_show_warning_enabled_p()) { \
            if (cumo_warning_once_enabled_p()) { \
                static bool show_warning = true; \
                if (show_warning) { \
                    fprintf(stderr, (c_str)); \
                    show_warning = false; \
                } \
            } else { \
                fprintf(stderr, (c_str)); \
            } \
        } \
    }

#define CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE( func_name, type_name ) \
    CUMO_SHOW_WARNING_ONCE("Warning: FIXME: Method \"" func_name "\" for dtype \"" type_name "\" synchronizes with CPU.\n")

#define CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE( func_name, type_name ) \
    CUMO_SHOW_WARNING_ONCE("Warning: Method \"" func_name "\" for dtype \"" type_name "\" synchronizes with CPU.\n")

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#endif /* ifndef CUMO_H */
