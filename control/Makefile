#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

ifeq ($(PGIFORTRAN), 0)
ZSRC := $(ZSRC) 		\
	magma_zf77.cpp		\
	magma_param.f90		\
	magma_zfortran.f90	\
	magma.f90
else
ZSRC := $(ZSRC) 		\
	magma_zf77pgi.cpp
endif

CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)	\
        auxiliary.cpp 	\
        xerbla.cpp 	\
	get_nb.cpp

ALLOBJ  = $(ALLSRC:.cpp=.o)
ALLOBJ := $(ALLOBJ:.f90=.o)

all: $(LIBMAGMA)

$(LIBMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@
	cp *.mod ../include

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

%.o : %.f90 
	$(FORT) $(FOPTS) $(INC) -c $< -o $@

clean:
	rm -f *.cu_o *.o *~ *.linkinfo *.mod

cleanall: clean
	rm -f *.a
	rm -f $(LIBMAGMA)

# Extra dependencies
magma.o: magma_zfortran.o magma_cfortran.o magma_dfortran.o magma_sfortran.o 
