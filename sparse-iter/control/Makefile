#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ../..
include ../../Makefile.internal

INC += -I../include

ZSRC := \
        matrix_zio.cpp \
				z_transpose_csr.cpp \
				magma_z_transfer.cpp \
        magma_z_mconverter.cpp \
        magma_z_init.cpp \
        magma_z_vio.cpp \
        magma_z_free.cpp \
        magma_z_matrixchar.cpp \
 

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

clean:
	rm -f *.cu_o *.o *~ *.linkinfo *.mod ../include/*.mod sizeptr

cleanall: clean
	rm -f $(LIBMAGMA_SPARSE)


