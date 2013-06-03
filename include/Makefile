#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

ZHDR =  magma_z.h	\
	magma_zc.h	\
	magmablas_z.h	\
	magmablas_zc.h	\
	magma_zlapack.h	\
	magma_zbulge.h	\
	magma_zbulgeinc.h	\
	magma_zgehrd_m.h	\

-include Makefile.local
-include Makefile.src

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR)

.PHONY: all clean cleanall

all: $(ALLHDR)

clean:
	rm -f *~ *.mod

cleanall: clean

