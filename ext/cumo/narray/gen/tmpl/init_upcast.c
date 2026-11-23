    hCast = rb_hash_new();
    /* Upcasting rules of <%=class_name%>, a Hash from the class of the other operand to the class both are cast to before an operation. */
    rb_define_const(cT, "UPCAST", hCast);
    rb_hash_aset(hCast, rb_cArray,   cT);
    <% for x in upcast %>
    <%= x %><% end %>
    rb_obj_freeze(hCast);
