<%
  # A doc tool shows the C expression of a constant as its value, so the table
  # the entries below build is written out as the value of UPCAST. Backslashes
  # keep the colons of a class name out of the "value: description" split.
  doc_class = lambda do |v|
    case v
    when "cT" then "Cumo\\:\\:#{class_name}"
    when /\Acumo_c(\w+)\z/ then "Cumo\\:\\:#{$1}"
    when "rb_cInteger", "rb_cFixnum", "rb_cBignum" then "Integer"
    when /\Arb_c(\w+)\z/ then $1
    else raise "init_upcast.c: #{v} is not a class variable it knows"
    end
  end
  doc_pairs = { "Array" => doc_class["cT"] }
  upcast.each do |stmt|
    next unless (m = /\Arb_hash_aset\(hCast,\s*(\w+),\s*(\w+)\);\z/.match(stmt.strip))
    doc_pairs[doc_class[m[1]]] ||= doc_class[m[2]]
  end
  # The expansion below has to end inside the comment. erbln.rb writes a #line
  # directive where an expansion runs over its template line, and one between
  # the comment and rb_define_const would leave UPCAST undocumented.
  doc_table = "{\n" + doc_pairs.map { |k, v| "         #{k}=>#{v}" }.join(",\n")
%>
    hCast = rb_hash_new();
    /* <%= doc_table %>
       }: Upcasting rules of <%=class_name%>, a Hash from the class of the other operand to the class both are cast to before an operation. */
    rb_define_const(cT, "UPCAST", hCast);
    rb_hash_aset(hCast, rb_cArray,   cT);
    <% for x in upcast %>
    <%= x %><% end %>
    rb_obj_freeze(hCast);
