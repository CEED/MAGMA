#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

ZSRC  = magma_zf77.cpp	\
	zpotrf.cpp	\
        zpotrf_gpu.cpp	\
        zgeqrf.cpp	\
        zgeqrf_gpu.cpp	\
        zgeqrf_gpu-v2.cpp \
        zgeqrs_gpu.cpp	\
        zlarfb.cpp	\
        zgetrf.cpp	\
        zgetrf_gpu.cpp 	\
        zgetrs_gpu.cpp 	\
        zgehrd.cpp 	\
        zlahru.cpp 	\
        zlahr2.cpp 	\
        zgeqlf.cpp 	\
        zgelqf.cpp 	\
        zgemm_gpu.cpp 	\
        zpotrs_gpu.cpp 	\
        \
        zsytrd.cpp 	\
        zlatrd.cpp 	\
        zgebrd.cpp 	\
        zlabrd.cpp 	\
        zungqr.cpp 	\
        zunmtr.cpp 	\
        zunmqr.cpp 	\
        zunmqr_gpu.cpp

CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

#ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLSRC= $(SSRC)	\
        dsgetrs_gpu.o \
        dsgesv_gpu.o \
        dsgeqrsv_gpu.o \
        dsposv_gpu.o \
        \
        auxiliary.o \
        \
        xerbla.o \

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
