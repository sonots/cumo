<%
  # A doc tool shows the C expression of a constant as its value, so the table
  # the entries below build is written out as the value of UPCAST. to_h keeps
  # the position of the first of a repeated key and the value of the last, as
  # rb_hash_aset does. The whole value has to stay on the first line of the
  # comment, which is as far as rdoc looks for it, and the backslashes keep
  # the colons of a class name out of the "value: description" split.
  doc_pairs = ([["Array", "Cumo::#{class_name}"]] + upcast_doc).to_h
  doc_table = "{" + doc_pairs.map { |k, v| "#{k}=>#{v}" }.join(", ").gsub("::") { "\\:\\:" } + "}"
%>
    hCast = rb_hash_new();
    /* <%= doc_table %>:
       Upcasting rules of <%=class_name%>, a Hash from the class of the other
       operand to the class both are cast to before an operation. Two arrays of
       the same class need no entry, and an NArray class that is not a key here
       is looked up in its own UPCAST instead. */
    rb_define_const(cT, "UPCAST", hCast);
    rb_hash_aset(hCast, rb_cArray,   cT);
    <% for x in upcast %>
    <%= x %><% end %>
    rb_obj_freeze(hCast);
