set breakpoint pending on
b cumo_debug_breakpoint
set $_exitcode = -999
run
if $_exitcode != -999
  quit
end
