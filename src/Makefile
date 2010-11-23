#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

ZSRC  = zpotrf.cpp	\
        zpotrf_gpu.cpp	\
        zgeqrf.cpp	\
        zgeqrf-v2.cpp   \
        zgeqrf-v3.cpp   \
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
        zpotrs_gpu.cpp 	\
        \
        zhetrd.cpp 	\
        zlatrd.cpp 	\
        zgebrd.cpp 	\
        zlabrd.cpp 	\
        zungqr.cpp 	\
        zunmtr.cpp 	\
        zunmqr.cpp 	\
        zunmqr_gpu.cpp	\
	\
        zcgetrs_gpu.cpp \
        zcgesv_gpu.cpp 	\
        zcgeqrsv_gpu.cpp\
        zcposv_gpu.cpp

ifeq ($(PGIFORTRAN), 0)
ZSRC := $(ZSRC) magma_zf77.cpp
else
ZSRC := $(ZSRC) magma_zf77pgi.cpp
endif

CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)	\
        auxiliary.cpp 	\
        xerbla.cpp 	\
        sorgqr_gpu.cpp	\
	get_nb.cpp

ALLOBJ = $(ALLSRC:.cpp=.o)

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

clean:
	rm -f *.cu_o *.o *~ *.linkinfo

cleanall: clean
	rm -f *.a
	rm -f $(LIBMAGMA)
