<% children.select { |c| c.kernel_parts.include?($kernel_part) }.each do |c|%>
<%= c.result %>
<% end %>
