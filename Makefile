#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = .
include ./Makefile.internal
-include Makefile.local

.PHONY: lib

all: lib test

lib: libmagma libmagmablas

# libmagmablas is not a true dependency, but adding it forces parallel make
# to do one directory at a time, making output less confusing.
libmagma: libmagmablas
	@echo ======================================== src
	( cd src            && $(MAKE) )
	@echo ======================================== control
	( cd control        && $(MAKE) )
	@echo ======================================== interface
	( cd interface_cuda && $(MAKE) )

libmagmablas:
	@echo ======================================== magmablas
	( cd magmablas      && $(MAKE) )

libquark:
	@echo ======================================== quark
	( cd quark          && $(MAKE) )

lapacktest:
	@echo ======================================== lapacktest
	( cd testing/matgen && $(MAKE) )
	( cd testing/lin    && $(MAKE) )

test: lib
	@echo ======================================== test
	( cd testing        && $(MAKE) )

clean:
	( cd include        && $(MAKE) clean )
	( cd control        && $(MAKE) clean )
	( cd src            && $(MAKE) clean )
	( cd interface_cuda && $(MAKE) clean )
	( cd testing        && $(MAKE) clean )
	( cd testing/lin    && $(MAKE) clean )
	( cd magmablas      && $(MAKE) clean ) 
#	( cd quark          && $(MAKE) clean )
	-rm -f $(LIBMAGMA) $(LIBMAGMABLAS)

cleanall:
	( cd include        && $(MAKE) cleanall )
	( cd control        && $(MAKE) cleanall )
	( cd src            && $(MAKE) cleanall )
	( cd interface_cuda && $(MAKE) cleanall )
	( cd testing        && $(MAKE) cleanall )
	( cd testing/lin    && $(MAKE) cleanall )
	( cd magmablas      && $(MAKE) cleanall ) 
	( cd lib            && rm -f *.a )
#	( cd quark          && $(MAKE) cleanall )
	$(MAKE) cleanall2

# cleanall2 is a dummy rule to run cleanmkgen at the *end* of make cleanall, so
# .Makefile.gen files aren't deleted and immediately re-created. see Makefile.gen
cleanall2:
	@echo

dir:
	mkdir -p $(prefix)
	mkdir -p $(prefix)/include
	mkdir -p $(prefix)/lib
	mkdir -p $(prefix)/lib/pkgconfig

install: lib dir
#       MAGMA
	cp $(MAGMA_DIR)/include/*.h  $(prefix)/include
	cp $(LIBMAGMA)               $(prefix)/lib
	cp $(LIBMAGMABLAS)           $(prefix)/lib
#       QUARK
#	cp $(QUARKDIR)/include/quark.h             $(prefix)/include
#	cp $(QUARKDIR)/include/quark_unpack_args.h $(prefix)/include
#	cp $(QUARKDIR)/include/icl_hash.h          $(prefix)/include
#	cp $(QUARKDIR)/include/icl_list.h          $(prefix)/include
#	cp $(QUARKDIR)/lib/libquark.a              $(prefix)/lib
#       pkgconfig
	cat $(MAGMA_DIR)/lib/pkgconfig/magma.pc | \
	    sed -e s:\__PREFIX:"$(prefix)":     | \
	    sed -e s:\__LIBEXT:"$(LIBEXT)":       \
	    > $(prefix)/lib/pkgconfig/magma.pc

# ========================================
# This is a crude manner of creating shared libraries.
# First create objects (with -fPIC) and static .a libraries,
# then assume all objects in these directories go into the shared libraries.
# Better solution would be to use non-recursive make, so make knows all the
# objects in each subdirectory, or use libtool, or put rules for, e.g., the
# control directory in src/Makefile (as done in src/CMakeLists.txt)
LIBMAGMA_SO     = $(LIBMAGMA:.a=.so)
LIBMAGMABLAS_SO = $(LIBMAGMABLAS:.a=.so)

shared: lib
	$(MAKE) $(LIBMAGMA_SO) $(LIBMAGMABLAS_SO)

$(LIBMAGMABLAS_SO): interface_cuda/*.o magmablas/*.cu_o
	@echo ======================================== libmagmablas.so
	$(CC) $(LDOPTS) -shared -o $(LIBMAGMABLAS_SO) \
	interface_cuda/*.o magmablas/*.cu_o \
	$(LIBDIR) \
	$(LIB)

$(LIBMAGMA_SO): src/*.o control/*.o $(LIBMAGMABLAS_SO)
	@echo ======================================== libmagma.so
	$(CC) $(LDOPTS) -shared -o $(LIBMAGMA_SO) \
	src/*.o control/*.o \
	-L lib -lmagmablas \
	$(LIBDIR) \
	$(LIB)
