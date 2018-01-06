set breakpoint pending on
b nary_debug_breakpoint
set $_exitcode = -999
run
if $_exitcode != -999
  quit
end
