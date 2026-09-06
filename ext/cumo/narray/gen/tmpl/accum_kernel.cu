<% unless defined?($cumo_narray_gen_tmpl_accum_kernel_included) %>
<% $cumo_narray_gen_tmpl_accum_kernel_included = 1 %>

<% if type_name.include?('int') || is_half %>
<%# half takes the shared ones only: a count and a running mean of its own width
   cannot carry a moment, so var and its neighbours wait for a wider accumulator. %>
<%= load_erb("real_accum").result(binding) %>
<% elsif type_name.include?('float') %>
<%= load_erb("float_accum").result(binding) %>
<% elsif type_name.include?('complex') %>
<%= load_erb("complex_accum").result(binding) %>
<% end %>

<% end %>
