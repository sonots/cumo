<% unless defined?($cumo_narray_gen_tmpl_accum_kernel_included) %>
<% $cumo_narray_gen_tmpl_accum_kernel_included = 1 %>

<% if type_name.include?('int') %>
<% f = File.join(File.dirname(__FILE__), 'real_accum_kernel.cu'); ERB.new(File.read(f)).tap {|erb| erb.filename = f }.result(binding) %>
<% elsif type_name.include?('float') %>
<%= f = File.join(File.dirname(__FILE__), 'float_accum_kernel.cu'); ERB.new(File.read(f)).tap {|erb| erb.filename = f }.result(binding) %>
<% elsif type_name.include?('complex') %>
<%= f = File.join(File.dirname(__FILE__), 'complex_accum_kernel.cu'); ERB.new(File.read(f)).tap {|erb| erb.filename = f }.result(binding) %>
<% end %>

<% end %>
