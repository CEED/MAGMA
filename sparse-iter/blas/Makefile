#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ../..
include ../../Makefile.internal

ZHDR  = 

# alphabetic order by base name (ignoring precision)
ZSRC := \
	magma_z_blaswrapper.cpp \
	magma_z_mpk.cpp      \
	zgecsrmv.cu		\
	zmgecsrmv.cu		\
	zgeellmv.cu		\
	zmgeellmv.cu		\
	zgeelltmv.cu            \
	zmgeelltmv.cu            \
	zmpkgeelltmv.cu            \
    zlag2c_sparse.cu        \
    clag2z_sparse.cu        \
    zjacobisetup.cu         \
	magma_zlag2c.cpp        \
	magma_clag2z.cpp        \
	zp1.cu                  \
	#zp1_b.cu                  \


# multi-GPU
ZSRC += \


# Card-specific versions that need precisions generated must be above in ZSRC,
# so they are all put into Makefile.src in the release. The code is #ifdef'd
# for the correct card.

-include Makefile.local
-include Makefile.src

FERMISRC = \

TESLASRC = \

ALLHDR := $(ZHDR) $(CHDR) $(DHDR) $(SHDR) \

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC) \

ifeq (${GPU_TARGET}, Tesla)
    ALLSRC += $(TESLASRC)
else ifeq (${GPU_TARGET}, Fermi)
    ALLSRC += $(FERMISRC)
else ifeq (${GPU_TARGET}, Kepler)
    ALLSRC += $(FERMISRC)
endif

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
