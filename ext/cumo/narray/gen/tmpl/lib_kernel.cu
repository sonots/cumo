#include "cumo/narray_kernel.h"
#include <<%="cumo/types/#{type_name}_kernel.h"%>>
#include "cumo/cuda/cumo_thrust.hpp"

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

<% children.each do |c|%>
<%= "#{c.result}\n\n" %>
<% end %>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
