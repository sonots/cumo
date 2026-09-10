    /*
      Document-class: <%= full_class_name %>
      <%= description %>
    */
    cT = rb_define_class_under(<%=ns_var%>, "<%=class_name%>", cNArray);

  <% for x in class_alias %>
    // alias of <%=class_name%>
    rb_define_const(<%=ns_var%>, "<%=x%>", <%=type_var%>);
  <% end %>

    <% @children.each do |m| %>
    <%= m.init_def %><% end %>
    rb_define_singleton_method(cT, "[]", <%=find("cast").c_func%>, -2);
