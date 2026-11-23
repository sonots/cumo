    hCast = rb_hash_new();
    /* The class of the result of an operation between <%=class_name%> and an operand of each key's class. */
    rb_define_const(cT, "UPCAST", hCast);
    rb_hash_aset(hCast, rb_cArray,   cT);
    <% for x in upcast %>
    <%= x %><% end %>
    rb_obj_freeze(hCast);
