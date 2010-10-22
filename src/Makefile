#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

include ../make.inc

    ALLOBJ = spotrf.o \
             spotrf-v2.o \
             spotrf-v3.o \
             spotrf_gpu.o \
             sgeqrf.o \
             sgeqrf-v2.o \
             sgeqrf_gpu.o \
             sgeqrf_gpu-v2.o \
             sgeqrs_gpu.o \
             slarfb.o \
             sgetrf.o \
             sgetrf_gpu-v2.o \
             sgetrf_gpu.o \
             sgetrs_gpu.o \
             sgehrd.o \
             slahru.o \
             slahr2.o \
             sgeqlf.o \
             sgelqf-v2.o \
             sgelqf.o \
             sdgetrs_gpu.o \
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
             sgessm.o \
             ststrf.o \

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.o *~ $(LIBMAGMA)

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@
