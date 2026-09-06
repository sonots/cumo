<% unless defined?($cumo_narray_gen_tmpl_accum_kernel_included) %>
<% $cumo_narray_gen_tmpl_accum_kernel_included = 1 %>

<% if type_name.include?('int') %>
<%= load_erb("real_accum").result(binding) %>
<% elsif type_name.include?('float') %>
<%= load_erb("float_accum").result(binding) %>
<% elsif type_name.include?('complex') %>
<%= load_erb("complex_accum").result(binding) %>
<% end %>

<% end %>
