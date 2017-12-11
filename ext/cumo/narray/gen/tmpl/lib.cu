#include "cumo/narray_kernel.h"
#include <<%="cumo/types/#{type_name}_kernel.h"%>>

<% children.each do |c|%>
<%= "#{c.result}\n\n" %>
<% end %>
