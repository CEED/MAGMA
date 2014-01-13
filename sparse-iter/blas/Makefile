#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/blas
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

ZHDR  = 

# alphabetic order by base name (ignoring precision)
ZSRC := \
	clag2z_sparse.cu	\
	magma_z_blaswrapper.cpp	\
	magma_z_mpksetup.cu	\
	magma_clag2z.cpp	\
	magma_zlag2c.cpp	\
	zgecsrmv.cu		\
	zgeellmv.cu		\
	zgeelltmv.cu		\
	zjacobisetup.cu		\
	zlag2c_sparse.cu	\
	zmdot.cu		\
	zmergebicgstab.cu	\
	zmergebicgstab2.cu	\
	zmergeidr.cu		\
	zmgecsrmv.cu		\
	zmgeellmv.cu		\
	zmgeelltmv.cu		\
	zpipelinedgmrescopyscale.cu	\
	zpipelinedgs.cu		\
	zbcsrswp.cu		\
	zbcsrtrsv.cu		\

-include Makefile.local
-include Makefile.src

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR)

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)

ALLOBJ := $(ALLSRC:.cu=.cu_o) 
ALLOBJ := $(ALLOBJ:.cpp=.o)

.PHONY: all clean cleanall

all: $(LIBMAGMA_SPARSE) 

$(LIBMAGMA_SPARSE): $(ALLHDR) $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.cu_o *.o *~ *.linkinfo

cleanall: clean
	rm -f $(LIBMAGMA_SPARSE)
