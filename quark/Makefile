###
#
# @file Makefile
#
#  MAGMA is a software package provided by Univ. of Tennessee,
#  Univ. of California Berkeley and Univ. of Colorado Denver
#
# @version 2.5.0
# @author Asim YarKhan
# @date 2010-11-15
#
###

MAGMA_DIR ?= ..
include ../Makefile.internal

TRACE_SHARED_LIB_DESTINATION_DIRECTORY ?= $(MAGMA_DIR)/lib

all: libquark.a trace_shared_libs
lib: libquark.a trace_shared_libs

# Decide whether to enable EZTrace tracing, and to build the shared tracing library.
# Note, for binaries EZTrace should be in the include and library paths ( -leztrace -lfxt )
# EZTrace tracing is enabled by setting QUARK_TRACE=1 in a makefile (or make.inc). For example,
# QUARK_TRACE=1
ifeq ($(QUARK_TRACE), 1)
EZT_DIR ?= /usr
CFLAGS += -DTRACE_QUARK -I${EZT_DIR}/include
# LDFLAGS += -L${EZT_DIR}/lib -leztrace -lfxt
QUARK_TRACE_SO_LIB=$(TRACE_SHARED_LIB_DESTINATION_DIRECTORY)/libeztrace-convert-quark.so
QUARK_TRACE_OBJS=eztrace_convert_quark.$(o_ext) 
$(QUARK_TRACE_SO_LIB): eztrace_convert_quark.$(o_ext)
	$(CC) -shared $(CFLAGS) $(INC) $^ -o $@
eztrace_convert_quark.$(o_ext): eztrace_convert_quark.c quark_trace.h
	$(CC) -fPIC $(CFLAGS) $(INC) -c $< -o $@
endif

trace_shared_libs: $(QUARK_TRACE_SO_LIB)

icl_hash.$(o_ext): icl_hash.c icl_hash.h
icl_list.$(o_ext): icl_list.c icl_list.h
quark.$(o_ext): quark.c quark.h bsd_queue.h bsd_tree.h quark_trace.h
quarkos.$(o_ext): quarkos.c quarkos-hwloc.c

libquark.a: icl_hash.$(o_ext) icl_list.$(o_ext) quarkos.$(o_ext) quark.$(o_ext) $(QUARK_TRACE_OBJS)
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@

.c.$(o_ext):
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

clean:
	rm -f *.$(o_ext) *~

cleanall: clean
	rm -f *.a

.PHONY: clean
