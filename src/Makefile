#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 0.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      June 2009
#//////////////////////////////////////////////////////////////////////////////

include ../make.inc

    ALLOBJ = spotrf.o \
             spotrf-v2.o \
             spotrf_gpu.o \
             sgeqrf.o \
             sgeqrf_gpu.o \
             sgeqrf_gpu-v2.o \
             sormqr_gpu-v2.o \
             slarfb.o \
             sgetrf.o \
             sgetrf_gpu-v2.o \
             sgetrf_gpu.o \
             sgehrd.o \
             slahru.o \
             slahr2.o \
             sgeqlf.o \
             sgelqf-v2.o \
             sgelqf.o \
             sdgetrs_gpu.o \
             \
             dpotrf.o \
             dpotrf_gpu.o \
             dgeqrf.o \
             dgeqrf_gpu.o \
             dlarfb.o \
             dgetrf.o \
             dgetrf_gpu.o \
             dgehrd.o \
             dlahru.o \
             dlahr2.o \
             \
             cpotrf.o \
             cgeqrf.o \
             clarfb.o \
             cgetrf.o \
             \
             zpotrf.o \
             zgeqrf.o \
             zlarfb.o \
             zgetrf.o \
             \
             auxiliary.o \
             \
             magma_dsgesv.o \
             magma_dsposv.o \
             magma_dgemm.o \
             magma_sgemm.o \
             magma_spotrs_gpu.o \
             magma_dpotrs_gpu.o \

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.o *~ *.a

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@
