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
             zgetrf.o \
             \
             auxiliary.o \
             \
             magma_dsgesv.o \
             \

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.o *~ *.a

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@
