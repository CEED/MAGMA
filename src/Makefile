#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 0.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      June 2009
#//////////////////////////////////////////////////////////////////////////////

include ../make.inc

    ALLOBJ = spotrf.o \
             spotrf_gpu.o \
             sgeqrf.o \
             sgeqrf_gpu.o \
             slarfb.o \
             sgetrf.o \
             sgetrf_gpu.o \
             sgehrd.o \
             slahru.o \
             slahr2.o \
             sgeqlf.o \
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
             \
             auxiliary.o \

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.o *~ *.a

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@
