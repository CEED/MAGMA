#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date
#//////////////////////////////////////////////////////////////////////////////

DIR = include
MAGMA_DIR = ..
include $(MAGMA_DIR)/Makefile.internal

ZHDR := \
	magma_z.h		\
	magma_zc.h		\
	magmablas_z.h		\
	magmablas_zc.h		\
	magmablas_z_q.h		\
	magmablas_zc_q.h	\
	magma_zlapack.h		\
	magma_zbulge.h		\
	magma_zbulgeinc.h	\
	magma_zgehrd_m.h	\
	magma_zbatched.h	\

-include Makefile.local
-include Makefile.src

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR)

.PHONY: all clean cleanall

.DEFAULT_GOAL := all
all: $(ALLHDR)

clean:
	rm -f *~ *.mod

cleanall: clean

