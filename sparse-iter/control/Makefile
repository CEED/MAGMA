#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ../..
include ../../Makefile.internal

ZSRC := \
        matrix_zio.cpp \

-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)	\
					mmio.cpp

ALLOBJ := $(ALLSRC:.cpp=.o)
ALLOBJ := $(ALLOBJ:.f90=.o)
ALLOBJ := $(ALLOBJ:.F90=.o)

.PHONY: all clean cleanall

all: $(LIBMAGMA_SPARSE)

$(LIBMAGMA_SPARSE): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@
	cp *.mod ../include

clean:
	rm -f *.cu_o *.o *~ *.linkinfo *.mod ../include/*.mod sizeptr

cleanall: clean
	rm -f $(LIBMAGMA_SPARSE)


