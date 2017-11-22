
SHELL = /bin/sh

# V=0 quiet, V=1 verbose.  other values don't work.
V = 0
Q1 = $(V:1=)
Q = $(Q1:0=@)
ECHO1 = $(V:1=@ :)
ECHO = $(ECHO1:0=@ echo)
NULLCMD = :

#### Start of system configuration section. ####

srcdir = .
topdir = /home/sonots/.rbenv/versions/2.4.1/include/ruby-2.4.0
hdrdir = $(topdir)
arch_hdrdir = /home/sonots/.rbenv/versions/2.4.1/include/ruby-2.4.0/x86_64-linux
PATH_SEPARATOR = :
VPATH = $(srcdir):$(arch_hdrdir)/ruby:$(hdrdir)/ruby
prefix = $(DESTDIR)/home/sonots/.rbenv/versions/2.4.1
rubysitearchprefix = $(rubylibprefix)/$(sitearch)
rubyarchprefix = $(rubylibprefix)/$(arch)
rubylibprefix = $(libdir)/$(RUBY_BASE_NAME)
exec_prefix = $(prefix)
vendorarchhdrdir = $(vendorhdrdir)/$(sitearch)
sitearchhdrdir = $(sitehdrdir)/$(sitearch)
rubyarchhdrdir = $(rubyhdrdir)/$(arch)
vendorhdrdir = $(rubyhdrdir)/vendor_ruby
sitehdrdir = $(rubyhdrdir)/site_ruby
rubyhdrdir = $(includedir)/$(RUBY_VERSION_NAME)
vendorarchdir = $(vendorlibdir)/$(sitearch)
vendorlibdir = $(vendordir)/$(ruby_version)
vendordir = $(rubylibprefix)/vendor_ruby
sitearchdir = $(sitelibdir)/$(sitearch)
sitelibdir = $(sitedir)/$(ruby_version)
sitedir = $(rubylibprefix)/site_ruby
rubyarchdir = $(rubylibdir)/$(arch)
rubylibdir = $(rubylibprefix)/$(ruby_version)
sitearchincludedir = $(includedir)/$(sitearch)
archincludedir = $(includedir)/$(arch)
sitearchlibdir = $(libdir)/$(sitearch)
archlibdir = $(libdir)/$(arch)
ridir = $(datarootdir)/$(RI_BASE_NAME)
mandir = $(datarootdir)/man
localedir = $(datarootdir)/locale
libdir = $(exec_prefix)/lib
psdir = $(docdir)
pdfdir = $(docdir)
dvidir = $(docdir)
htmldir = $(docdir)
infodir = $(datarootdir)/info
docdir = $(datarootdir)/doc/$(PACKAGE)
oldincludedir = $(DESTDIR)/usr/include
includedir = $(prefix)/include
localstatedir = $(prefix)/var
sharedstatedir = $(prefix)/com
sysconfdir = $(prefix)/etc
datadir = $(datarootdir)
datarootdir = $(prefix)/share
libexecdir = $(exec_prefix)/libexec
sbindir = $(exec_prefix)/sbin
bindir = $(exec_prefix)/bin
archdir = $(rubyarchdir)


CC = gcc
CXX = g++
LIBRUBY = $(LIBRUBY_A)
LIBRUBY_A = lib$(RUBY_SO_NAME)-static.a
LIBRUBYARG_SHARED = -Wl,-R$(libdir) -L$(libdir) 
LIBRUBYARG_STATIC = -Wl,-R$(libdir) -L$(libdir) -l$(RUBY_SO_NAME)-static
empty =
OUTFLAG = -o $(empty)
COUTFLAG = -o $(empty)
CSRCFLAG = $(empty)

RUBY_EXTCONF_H = numo/extconf.h
cflags   = $(optflags) $(debugflags) $(warnflags)
cxxflags = $(optflags) $(debugflags) $(warnflags)
optflags = -O3 -fno-fast-math
debugflags = -ggdb3
warnflags = -Wall -Wextra -Wno-unused-parameter -Wno-parentheses -Wno-long-long -Wno-missing-field-initializers -Wno-tautological-compare -Wno-parentheses-equality -Wno-constant-logical-operand -Wno-self-assign -Wunused-variable -Wimplicit-int -Wpointer-arith -Wwrite-strings -Wdeclaration-after-statement -Wimplicit-function-declaration -Wdeprecated-declarations -Wno-packed-bitfield-compat -Wsuggest-attribute=noreturn -Wsuggest-attribute=format -Wno-maybe-uninitialized
CCDLFLAGS = -fPIC
CFLAGS   = $(CCDLFLAGS) -I /home/sonots/nccl/build/include -I /home/sonots/.cudnn/active/cuda/include  $(ARCH_FLAG)
INCFLAGS = -I. -Itypes -I$(arch_hdrdir) -I$(hdrdir)/ruby/backward -I$(hdrdir) -I$(srcdir)
DEFS     = 
CPPFLAGS = -DRUBY_EXTCONF_H=\"$(RUBY_EXTCONF_H)\" -I/home/sonots/.rbenv/versions/2.4.1/include  $(DEFS) $(cppflags)
CXXFLAGS = $(CCDLFLAGS) $(cxxflags) $(ARCH_FLAG)
ldflags  = -L. -L/home/sonots/.rbenv/versions/2.4.1/lib -L /home/sonots/nccl/build/lib -I /home/sonots/.cudnn/active/cuda/include -fstack-protector -rdynamic -Wl,-export-dynamic
dldflags = -L/home/sonots/.rbenv/versions/2.4.1/lib -L /home/sonots/nccl/build/lib -I /home/sonots/.cudnn/active/cuda/include -Wl,--compress-debug-sections=zlib 
ARCH_FLAG = 
DLDFLAGS = $(ldflags) $(dldflags) $(ARCH_FLAG)
LDSHARED = $(CC) -shared
LDSHAREDXX = $(CXX) -shared
AR = ar
EXEEXT = 

RUBY_INSTALL_NAME = $(RUBY_BASE_NAME)
RUBY_SO_NAME = ruby
RUBYW_INSTALL_NAME = 
RUBY_VERSION_NAME = $(RUBY_BASE_NAME)-$(ruby_version)
RUBYW_BASE_NAME = rubyw
RUBY_BASE_NAME = ruby

arch = x86_64-linux
sitearch = $(arch)
ruby_version = 2.4.0
ruby = $(bindir)/$(RUBY_BASE_NAME)
RUBY = $(ruby)
ruby_headers = $(hdrdir)/ruby.h $(hdrdir)/ruby/backward.h $(hdrdir)/ruby/ruby.h $(hdrdir)/ruby/defines.h $(hdrdir)/ruby/missing.h $(hdrdir)/ruby/intern.h $(hdrdir)/ruby/st.h $(hdrdir)/ruby/subst.h $(arch_hdrdir)/ruby/config.h $(RUBY_EXTCONF_H)

RM = rm -f
RM_RF = $(RUBY) -run -e rm -- -rf
RMDIRS = rmdir --ignore-fail-on-non-empty -p
MAKEDIRS = /bin/mkdir -p
INSTALL = /usr/bin/install -c
INSTALL_PROG = $(INSTALL) -m 0755
INSTALL_DATA = $(INSTALL) -m 644
COPY = cp
TOUCH = exit >

#### End of system configuration section. ####

preload = 
libpath = . $(libdir)
LIBPATH =  -L. -L$(libdir) -Wl,-R$(libdir)
DEFFILE = 

CLEANFILES = mkmf.log
DISTCLEANFILES = 
DISTCLEANDIRS = 

extout = 
extout_prefix = 
target_prefix = /numo
LOCAL_LIBS = 
LIBS =   -lpthread -ldl -lcrypt -lm   -lc
ORIG_SRCS = SFMT.c array.c data.c index.c kwargs.c math.c narray.c ndloop.c rand.c step.c struct.c
SRCS = $(ORIG_SRCS) narray.c array.c step.c index.c ndloop.c data.c bit.c int8.c int16.c int32.c int64.c uint8.c uint16.c uint32.c uint64.c sfloat.c dfloat.c scomplex.c dcomplex.c robject.c math.c SFMT.c struct.c rand.c
OBJS = narray.o array.o step.o index.o ndloop.o data.o types/bit.o types/int8.o types/int16.o types/int32.o types/int64.o types/uint8.o types/uint16.o types/uint32.o types/uint64.o types/sfloat.o types/dfloat.o types/scomplex.o types/dcomplex.o types/robject.o math.o SFMT.o struct.o rand.o
HDRS = $(srcdir)/SFMT.h $(srcdir)/SFMT-params19937.h $(srcdir)/SFMT-params.h
LOCAL_HDRS = 
TARGET = narray
TARGET_NAME = narray
TARGET_ENTRY = Init_$(TARGET_NAME)
DLLIB = $(TARGET).so
EXTSTATIC = 
STATIC_LIB = 

TIMESTAMP_DIR = .
BINDIR        = $(bindir)
RUBYCOMMONDIR = $(sitedir)$(target_prefix)
RUBYLIBDIR    = $(sitelibdir)$(target_prefix)
RUBYARCHDIR   = $(sitearchdir)$(target_prefix)
HDRDIR        = $(rubyhdrdir)/ruby$(target_prefix)
ARCHHDRDIR    = $(rubyhdrdir)/$(arch)/ruby$(target_prefix)
TARGET_SO_DIR =
TARGET_SO     = $(TARGET_SO_DIR)$(DLLIB)
CLEANLIBS     = $(TARGET_SO) 
CLEANOBJS     = *.o  *.bak

all:    $(DLLIB)
static: $(STATIC_LIB) install-rb
.PHONY: all install static install-so install-rb
.PHONY: clean clean-so clean-static clean-rb

clean-static::
clean-rb-default::
clean-rb::
clean-so::
clean: clean-so clean-static clean-rb-default clean-rb
		-$(Q)$(RM) $(CLEANLIBS) $(CLEANOBJS) $(CLEANFILES) .*.time

distclean-rb-default::
distclean-rb::
distclean-so::
distclean-static::
distclean: clean distclean-so distclean-static distclean-rb-default distclean-rb
		-$(Q)$(RM) Makefile $(RUBY_EXTCONF_H) conftest.* mkmf.log
		-$(Q)$(RM) core ruby$(EXEEXT) *~ $(DISTCLEANFILES)
		-$(Q)$(RMDIRS) $(DISTCLEANDIRS) 2> /dev/null || true

realclean: distclean
install: install-so install-rb

install-so: $(DLLIB) $(TIMESTAMP_DIR)/.sitearchdir.-.numo.time
	$(INSTALL_PROG) $(DLLIB) $(RUBYARCHDIR)
clean-static::
	-$(Q)$(RM) $(STATIC_LIB)
install-rb: pre-install-rb do-install-rb install-rb-default
install-rb-default: pre-install-rb-default do-install-rb-default
pre-install-rb: Makefile
pre-install-rb-default: Makefile
do-install-rb:
do-install-rb-default:
pre-install-rb-default:
	@$(NULLCMD)
pre-install-rb: $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.time
do-install-rb: $(RUBYARCHDIR)/numo/compat.h
$(RUBYARCHDIR)/numo/compat.h: $(srcdir)/numo/compat.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/compat.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/narray.h
$(RUBYARCHDIR)/numo/narray.h: $(srcdir)/numo/narray.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/narray.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/ndloop.h
$(RUBYARCHDIR)/numo/ndloop.h: $(srcdir)/numo/ndloop.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/ndloop.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/intern.h
$(RUBYARCHDIR)/numo/intern.h: $(srcdir)/numo/intern.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/intern.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/template.h
$(RUBYARCHDIR)/numo/template.h: $(srcdir)/numo/template.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/template.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/extconf.h
$(RUBYARCHDIR)/numo/extconf.h: $(srcdir)/numo/extconf.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/extconf.h $(@D)
pre-install-rb: $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
do-install-rb: $(RUBYARCHDIR)/numo/types/complex_macro.h
$(RUBYARCHDIR)/numo/types/complex_macro.h: $(srcdir)/numo/types/complex_macro.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/complex_macro.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/int_macro.h
$(RUBYARCHDIR)/numo/types/int_macro.h: $(srcdir)/numo/types/int_macro.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/int_macro.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/uint16.h
$(RUBYARCHDIR)/numo/types/uint16.h: $(srcdir)/numo/types/uint16.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/uint16.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/uint32.h
$(RUBYARCHDIR)/numo/types/uint32.h: $(srcdir)/numo/types/uint32.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/uint32.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/bit.h
$(RUBYARCHDIR)/numo/types/bit.h: $(srcdir)/numo/types/bit.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/bit.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/float_def.h
$(RUBYARCHDIR)/numo/types/float_def.h: $(srcdir)/numo/types/float_def.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/float_def.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/dfloat.h
$(RUBYARCHDIR)/numo/types/dfloat.h: $(srcdir)/numo/types/dfloat.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/dfloat.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/robject.h
$(RUBYARCHDIR)/numo/types/robject.h: $(srcdir)/numo/types/robject.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/robject.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/uint_macro.h
$(RUBYARCHDIR)/numo/types/uint_macro.h: $(srcdir)/numo/types/uint_macro.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/uint_macro.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/sfloat.h
$(RUBYARCHDIR)/numo/types/sfloat.h: $(srcdir)/numo/types/sfloat.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/sfloat.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/int32.h
$(RUBYARCHDIR)/numo/types/int32.h: $(srcdir)/numo/types/int32.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/int32.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/complex.h
$(RUBYARCHDIR)/numo/types/complex.h: $(srcdir)/numo/types/complex.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/complex.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/int64.h
$(RUBYARCHDIR)/numo/types/int64.h: $(srcdir)/numo/types/int64.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/int64.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/uint64.h
$(RUBYARCHDIR)/numo/types/uint64.h: $(srcdir)/numo/types/uint64.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/uint64.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/real_accum.h
$(RUBYARCHDIR)/numo/types/real_accum.h: $(srcdir)/numo/types/real_accum.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/real_accum.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/dcomplex.h
$(RUBYARCHDIR)/numo/types/dcomplex.h: $(srcdir)/numo/types/dcomplex.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/dcomplex.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/float_macro.h
$(RUBYARCHDIR)/numo/types/float_macro.h: $(srcdir)/numo/types/float_macro.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/float_macro.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/int8.h
$(RUBYARCHDIR)/numo/types/int8.h: $(srcdir)/numo/types/int8.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/int8.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/scomplex.h
$(RUBYARCHDIR)/numo/types/scomplex.h: $(srcdir)/numo/types/scomplex.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/scomplex.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/robj_macro.h
$(RUBYARCHDIR)/numo/types/robj_macro.h: $(srcdir)/numo/types/robj_macro.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/robj_macro.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/uint8.h
$(RUBYARCHDIR)/numo/types/uint8.h: $(srcdir)/numo/types/uint8.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/uint8.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/xint_macro.h
$(RUBYARCHDIR)/numo/types/xint_macro.h: $(srcdir)/numo/types/xint_macro.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/xint_macro.h $(@D)
do-install-rb: $(RUBYARCHDIR)/numo/types/int16.h
$(RUBYARCHDIR)/numo/types/int16.h: $(srcdir)/numo/types/int16.h $(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time
	$(Q) $(INSTALL_DATA) $(srcdir)/numo/types/int16.h $(@D)
pre-install-rb:
	$(Q1:0=@$(MAKE) -q do-install-rb || )$(ECHO1:0=echo) installing narray libraries
$(TIMESTAMP_DIR)/.sitearchdir.-.numo.time:
	$(Q) $(MAKEDIRS) $(@D) $(RUBYARCHDIR)
	$(Q) $(TOUCH) $@
$(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.time:
	$(Q) $(MAKEDIRS) $(@D) $(RUBYARCHDIR)/numo
	$(Q) $(TOUCH) $@
$(TIMESTAMP_DIR)/.sitearchdir.-.numo.-.numo.-.types.time:
	$(Q) $(MAKEDIRS) $(@D) $(RUBYARCHDIR)/numo/types
	$(Q) $(TOUCH) $@

site-install: site-install-so site-install-rb
site-install-so: install-so
site-install-rb: install-rb

.SUFFIXES: .c .m .cc .mm .cxx .cpp .o .S

.cc.o:
	$(ECHO) compiling $(<)
	$(Q) $(CXX) $(INCFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(COUTFLAG)$@ -c $(CSRCFLAG)$<

.cc.S:
	$(ECHO) translating $(<)
	$(Q) $(CXX) $(INCFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(COUTFLAG)$@ -S $(CSRCFLAG)$<

.mm.o:
	$(ECHO) compiling $(<)
	$(Q) $(CXX) $(INCFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(COUTFLAG)$@ -c $(CSRCFLAG)$<

.mm.S:
	$(ECHO) translating $(<)
	$(Q) $(CXX) $(INCFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(COUTFLAG)$@ -S $(CSRCFLAG)$<

.cxx.o:
	$(ECHO) compiling $(<)
	$(Q) $(CXX) $(INCFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(COUTFLAG)$@ -c $(CSRCFLAG)$<

.cxx.S:
	$(ECHO) translating $(<)
	$(Q) $(CXX) $(INCFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(COUTFLAG)$@ -S $(CSRCFLAG)$<

.cpp.o:
	$(ECHO) compiling $(<)
	$(Q) $(CXX) $(INCFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(COUTFLAG)$@ -c $(CSRCFLAG)$<

.cpp.S:
	$(ECHO) translating $(<)
	$(Q) $(CXX) $(INCFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(COUTFLAG)$@ -S $(CSRCFLAG)$<

.c.o:
	$(ECHO) compiling $(<)
	$(Q) $(CC) $(INCFLAGS) $(CPPFLAGS) $(CFLAGS) $(COUTFLAG)$@ -c $(CSRCFLAG)$<

.c.S:
	$(ECHO) translating $(<)
	$(Q) $(CC) $(INCFLAGS) $(CPPFLAGS) $(CFLAGS) $(COUTFLAG)$@ -S $(CSRCFLAG)$<

.m.o:
	$(ECHO) compiling $(<)
	$(Q) $(CC) $(INCFLAGS) $(CPPFLAGS) $(CFLAGS) $(COUTFLAG)$@ -c $(CSRCFLAG)$<

.m.S:
	$(ECHO) translating $(<)
	$(Q) $(CC) $(INCFLAGS) $(CPPFLAGS) $(CFLAGS) $(COUTFLAG)$@ -S $(CSRCFLAG)$<

$(TARGET_SO): $(OBJS) Makefile
	$(ECHO) linking shared-object numo/$(DLLIB)
	-$(Q)$(RM) $(@)
	$(Q) $(LDSHARED) -o $@ $(OBJS) $(LIBPATH) $(DLDFLAGS) $(LOCAL_LIBS) $(LIBS)



###
$(OBJS): $(RUBY_EXTCONF_H)

TAGSRC = \
 ../../ruby/include/ruby/*.h \
 ../../ruby/*.c \
 *.h \
 types/*.h \
 *.c \
 types/*.c

tags : TAGS
TAGS : $(TAGSRC)
	etags $(TAGSRC)

doc :
	yard doc *.c types/*.c

C_TMPL = gen/tmpl/lib.c gen/tmpl/seq.c gen/tmpl/fill.c gen/tmpl/powint.c gen/tmpl/new_dim0.c gen/tmpl/coerce_cast.c gen/tmpl/eye.c gen/tmpl/format_to_a.c gen/tmpl/set2.c gen/tmpl/poly.c gen/tmpl/rand_norm.c gen/tmpl/store_numeric.c gen/tmpl/extract.c gen/tmpl/class.c gen/tmpl/map_with_index.c gen/tmpl/unary.c gen/tmpl/binary_s.c gen/tmpl/cond_unary.c gen/tmpl/unary_ret2.c gen/tmpl/minmax.c gen/tmpl/store_array.c gen/tmpl/qsort.c gen/tmpl/module.c gen/tmpl/format.c gen/tmpl/aset.c gen/tmpl/alloc_func.c gen/tmpl/unary2.c gen/tmpl/each.c gen/tmpl/accum_index.c gen/tmpl/pow.c gen/tmpl/accum.c gen/tmpl/store.c gen/tmpl/cast_array.c gen/tmpl/store_bit.c gen/tmpl/binary2.c gen/tmpl/cum.c gen/tmpl/rand.c gen/tmpl/inspect.c gen/tmpl/unary_s.c gen/tmpl/to_a.c gen/tmpl/sort_index.c gen/tmpl/logseq.c gen/tmpl/init_module.c gen/tmpl/binary.c gen/tmpl/sort.c gen/tmpl/frexp.c gen/tmpl/clip.c gen/tmpl/median.c gen/tmpl/extract_data.c gen/tmpl/each_with_index.c gen/tmpl/allocate.c gen/tmpl/aref.c gen/tmpl/store_from.c gen/tmpl/cond_binary.c gen/tmpl/accum_binary.c gen/tmpl/init_class.c gen/tmpl/cast.c gen/tmpl/bincount.c gen/tmpl_bit/where.c gen/tmpl_bit/fill.c gen/tmpl_bit/format_to_a.c gen/tmpl_bit/extract.c gen/tmpl_bit/unary.c gen/tmpl_bit/where2.c gen/tmpl_bit/store_array.c gen/tmpl_bit/format.c gen/tmpl_bit/aset.c gen/tmpl_bit/each.c gen/tmpl_bit/bit_count.c gen/tmpl_bit/store_bit.c gen/tmpl_bit/none_p.c gen/tmpl_bit/mask.c gen/tmpl_bit/inspect.c gen/tmpl_bit/to_a.c gen/tmpl_bit/binary.c gen/tmpl_bit/bit_reduce.c gen/tmpl_bit/each_with_index.c gen/tmpl_bit/allocate.c gen/tmpl_bit/aref.c gen/tmpl_bit/store_from.c

COGEN = gen/cogen.rb
DEPENDS = $(C_TMPL) gen/*.rb


types/uint32.c: gen/def/uint32.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/uint32.rb

types/sfloat.c: gen/def/sfloat.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/sfloat.rb

types/int64.c: gen/def/int64.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/int64.rb

types/dcomplex.c: gen/def/dcomplex.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/dcomplex.rb

types/uint8.c: gen/def/uint8.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/uint8.rb

types/int32.c: gen/def/int32.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/int32.rb

types/int16.c: gen/def/int16.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/int16.rb

types/bit.c: gen/def/bit.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/bit.rb

types/uint64.c: gen/def/uint64.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/uint64.rb

types/int8.c: gen/def/int8.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/int8.rb

types/scomplex.c: gen/def/scomplex.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/scomplex.rb

types/robject.c: gen/def/robject.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/robject.rb

types/uint16.c: gen/def/uint16.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/uint16.rb

types/dfloat.c: gen/def/dfloat.rb $(DEPENDS)
	$(MAKEDIRS) $(@D) types
	ruby $(COGEN) -l -o $@ gen/def/dfloat.rb


src : types/uint32.c types/sfloat.c types/int64.c types/dcomplex.c types/uint8.c types/int32.c types/int16.c types/bit.c types/uint64.c types/int8.c types/scomplex.c types/robject.c types/uint16.c types/dfloat.c

CLEANOBJS = *.o */*.o *.bak types/*.c
