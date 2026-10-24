#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#include "cumo/sort_kernel.h"

// sort_index and median sort with the same kernels, so their entry points are
// made here too.
CUMO_DEF_SORT(<%=type_name%>, dtype, <%=is_float%>)

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif
