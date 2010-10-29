#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

ZSRC  = testing_zgesv_gpu.cpp	\
	testing_zpotrf.cpp	\
        testing_zpotrf_gpu.cpp 	\

CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

#ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLSRC= $(SSRC)	\
        testing_sgeqrf.cpp \
        testing_sgeqrf_gpu.cpp \
        testing_sgeqrs_gpu.cpp \
        testing_sgesv_gpu.cpp \
        testing_sgetrf.cpp \
        testing_sgetrf_gpu.cpp \
        testing_sgeqlf.cpp \
        testing_sgelqf.cpp \
        \
        testing_sgehrd.cpp \
        testing_ssytrd.cpp \
        testing_sgebrd.cpp

ALLOBJ = $(ALLSRC:.cpp=.o)
EXE = $(ALLOBJ:.o=)

all: $(EXE)

$(EXE): $(ALLOBJ)

clean:
	rm -f $(OBJ) $(EXE) *.o *~

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

.o:
	$(FORT) $(LDOPTS) $< -o $@ lin/liblapacktest.a -L../lib \
          -lcuda -lmagma -lmagmablas $(LIBDIR) $(LIB) get_nb.cpp
