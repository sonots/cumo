/*
  <%= file_name %>
  Ruby/Cumo::GSL - GSL wrapper for Ruby/Cumo::NArray

  created on: 2017-03-11
  Copyright (C) 2017 Masahiro Tanaka
  Copyright (C) 2018 Naotoshi Seo
*/

#include <ruby.h>
#include <assert.h>
#include "cumo.h"
#include "cumo/narray.h"
#include "cumo/template.h"
#include "SFMT.h"
#include "cumo/cuda/memory_pool.h"
#include "cumo/cuda/runtime.h"
<% unless type_name == 'robject' %>
#include "cumo/indexer.h"
<% end %>

#define m_map(x) m_num_to_data(rb_yield(m_data_to_num(x)))

<% cumo_id_decl.each do |x| %>
<%= x %>
<% end %>

#include <<%="cumo/types/#{type_name}.h"%>>

VALUE cT;
extern VALUE cRT;

<% children.each do |c|%>
<%= c.result+"\n\n" %>
<% end %>

void
Init_<%=lib_name%>(void)
{
    VALUE <%=ns_var%>;

    <%=ns_var%> = rb_define_module("Cumo");

    <% cumo_id_assign.each do |x| %>
    <%= x %><% end %>

<% children.each do |c| %>
<%= c.init_def %>
<% end %>
}

// cumo.c runs this after the last Init_cumo_*. Any class defined after this
// one is still 0 during Init, and its entry would land under the key false.
void
Init_<%=lib_name%>_upcast(void)
{
    VALUE hCast;

<% children.each do |c| %><%= c.upcast_def %><% end %>
}
