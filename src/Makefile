#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

INC += -I$(MAGMA_DIR)/control

ZSRC  = zpotrf.cpp	\
        zpotrf_gpu.cpp	\
        zpotrs_gpu.cpp	\
        zposv_gpu.cpp	\
	\
        zgetrf.cpp	\
        zgetrf_gpu.cpp 	\
        zgetrs_gpu.cpp 	\
        zgesv_gpu.cpp 	\
	\
	zgeqrf.cpp	\
	zgeqlf.cpp	\
	zgelqf.cpp	\
        zungqr.cpp	\
	zgeqrf_gpu.cpp  \
	zgeqrf2_gpu.cpp	\
	zgeqrs_gpu.cpp	\
	zgels_gpu.cpp	\
	zlarfb_gpu.cpp	\
        zunmqr_gpu.cpp	\
        zungqr_gpu.cpp	\
	\
	zgebrd.cpp 	\
	zlabrd_gpu.cpp	\
	zgehrd.cpp 	\
        zgehrd2.cpp     \
        zlahru.cpp 	\
        zlahr2.cpp 	\
        zunghr.cpp      \
        zhetrd.cpp 	\
        zlatrd.cpp 	\
        zunmtr.cpp 	\
        zunmqr.cpp 	\
        \
        zgeev.cpp       \
        dgeev.cpp       \
        zheevd.cpp      \
        dsyevd.cpp      \
        zgesvd.cpp	\
	dgesvd.cpp	\
	\
	zcposv_gpu.cpp	\
	zcgesv_gpu.cpp	\
	zcgetrs_gpu.cpp	\
	zcgeqrsv_gpu.cpp
	\
	zgessm_gpu.cpp	\
	zssssm_gpu.cpp
#	zgetfl_gpu.cpp	\
#	ztstrf_gpu.cpp	\

CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLOBJ  = $(ALLSRC:.cpp=.o)

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

clean:
	rm -f *.o *~ *.linkinfo

cleanall: clean
	rm -f *.a
	rm -f $(LIBMAGMA)
