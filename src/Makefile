#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

ZSRC  = magma_zf77.cpp
CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

#ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLSRC= $(SSRC)	\
	spotrf.o \
        spotrf_gpu.o \
        sgeqrf.o \
        sgeqrf_gpu.o \
        sgeqrf_gpu-v2.o \
        sgeqrs_gpu.o \
        slarfb.o \
        sgetrf.o \
        sgetrf_gpu.o \
        sgetrs_gpu.o \
        sgehrd.o \
        slahru.o \
        slahr2.o \
        sgeqlf.o \
        sgelqf.o \
        \
        dsgetrs_gpu.o \
        dsgesv_gpu.o \
        dsgeqrsv_gpu.o \
        dsposv_gpu.o \
        \
        auxiliary.o \
        \
        sgemm_gpu.o \
        spotrs_gpu.o \
        xerbla.o \
        \
        ssytrd.o \
        slatrd.o \
        sgebrd.o \
        slabrd.o \
        sorgqr.o \
        sormtr.o \
        sormqr.o \
        sormqr_gpu.o \

ALLOBJ = $(ALLSRC:.cpp=.o)

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.o *~ 

cleanall: clean
	rm -f $(LIBMAGMA)

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

.PHONY: clean cleanall
