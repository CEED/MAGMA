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
        zpotrs_gpu.cpp	\
        zposv_gpu.cpp	\
        zgetrf.cpp	\
        zgetrf_gpu.cpp 	\
        zgetrs_gpu.cpp 	\
        zgesv_gpu.cpp 	\
	zgeqrf_gpu.cpp  \
	zgeqrf2_gpu.cpp	\
	zgeqrs_gpu.cpp	\
	zgels_gpu.cpp	\
	zlarfb_gpu.cpp	\
        zunmqr_gpu.cpp	\
        zunghr.cpp      \
	\
	zcposv_gpu.cpp	\
	zcgesv_gpu.cpp	\
	zcgetrs_gpu.cpp	\
	\
	zgeqrf.cpp	\
        zgeqrf_mc.cpp	\
        zgetrf_mc.cpp	\
        zpotrf_mc.cpp	\
        zgeqrf-v2.cpp   \
        zgeqrf-v3.cpp   \
        zgehrd.cpp 	\
        zgehrd2.cpp     \
        zlahru.cpp 	\
        zlahr2.cpp 	\
        zgeqlf.cpp 	\
        zgelqf.cpp 	\
        \
        zhetrd.cpp 	\
        zlatrd.cpp 	\
        zgebrd.cpp 	\
        zlabrd_gpu.cpp	\
        zungqr_gpu.cpp	\
        zungqr.cpp      \
        zunmtr.cpp 	\
        zunmqr.cpp 	\
        \
        zcgeqrsv_gpu.cpp\
        \
        zgeev.cpp       \
        zheevd.cpp      \
        zgesvd.cpp      \

ifeq ($(PGIFORTRAN), 0)
ZSRC := $(ZSRC) 		\
	magma_zf77.cpp		\
	magma_zfortran.f90	\
	magma.f90
FFLAGS:= $(OPTS)
else
ZSRC := $(ZSRC) 		\
	magma_zf77pgi.cpp
FFLAGS:= $(OPTS) -J../include
endif

CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)	\
        init.cpp        \
        auxiliary.cpp 	\
        xerbla.cpp 	\
	get_nb.cpp

ALLOBJ  = $(ALLSRC:.cpp=.o)
ALLOBJ := $(ALLOBJ:.f90=.o)

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

%.o : %.f90 
	$(FORT) $(FFLAGS) $(INC) -c $< -o $@

clean:
	rm -f *.cu_o *.o *~ *.linkinfo

cleanall: clean
	rm -f *.a
	rm -f $(LIBMAGMA)

# Extra dependencies
magma.o: magma_zfortran.o magma_cfortran.o magma_dfortran.o magma_sfortran.o 