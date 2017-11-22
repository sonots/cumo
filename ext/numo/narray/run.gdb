set breakpoint pending on
b nary_debug_breakpoint
# handle SIGINT nostop
# handle SIGPIPE nostop
# b rb_longjmp
# source ./breakpoints.gdb
# source ./.gdbinit
set $_exitcode = -999
run
if $_exitcode != -999
  quit
end
