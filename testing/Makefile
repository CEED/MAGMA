#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

ZSRC  = testing_zgetrf.cpp 	\
        testing_zgetrf_gpu.cpp 	\
	testing_zgesv_gpu.cpp	\
	testing_zpotrf.cpp	\
        testing_zpotrf_gpu.cpp 	\
        testing_zgeqrf.cpp 	\
        testing_zgeqrf_gpu.cpp 	\
        testing_zgeqrs_gpu.cpp 	\
        testing_zgeqlf.cpp 	\
        testing_zgelqf.cpp 	\
        \
        testing_zgehrd.cpp 	\
        testing_zhetrd.cpp 	\
        testing_zgebrd.cpp      \
        \
        testing_zgemm.cpp

#	testing_zcgesv_gpu.cpp	\
#	testing_zcgeqrsv_gpu.cpp\
#	testing_zcposv_gpu.cpp

OBJ = get_nb.o

ifeq ($(FORT), pgfortran)
	OBJ  := $(OBJ) fortran_thunking.o
	ZSRCF = testing_zgetrf_gpu_f.cuf
else
	OBJ  := $(OBJ) fortran.o
	ZSRCF = testing_zgetrf_gpu_f.f
endif

ZSRC += $(ZSRCF)
CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLOBJF = $(ALLSRC:.f=.o)
ALLOBJF:= $(ALLOBJF:.cuf=.o)
ALLOBJ = $(ALLOBJF:.cpp=.o)
EXE = $(ALLOBJ:.o=)

all: $(EXE)

$(EXE): $(LIBMAGMA) $(LIBMAGMABLAS)

clean:
	rm -f $(OBJ) *.o *~

cleanall: clean
	rm -f $(EXE)

ifeq ($(FORT), pgfortran)
testing_%.o : testing_%.cuf
	$(FORT) $(OPTS) $(INC) -c $< -o $@ 
else
testing_%.o : testing_%.f
	$(FORT) $(OPTS) $(INC) -c $< -o $@ 
endif

testing_%.o : testing_%.cpp
	$(CC) $(OPTS) $(INC) -c $< -o $@

%.o: %.cpp
	$(CC) $(OPTS) $(INC) -c $< -o $@

fortran.o : $(CUDADIR)/src/fortran.c
	$(CC) $(OPTS) $(INC) -c $< -o $@

fortran_thunking.o : $(CUDADIR)/src/fortran_thunking.c
	$(CC) $(OPTS) $(INC) -c $< -o $@

testing_%_f : testing_%_f.o $(OBJ)
	$(FORT) $(OPTS) $(LDOPTS) $< $(OBJ) -o $@ lin/liblapacktest.a -L../lib \
          -lcuda -lmagma -lmagmablas $(LIBDIR) $(LIB) 

testing_% : testing_%.o $(OBJ)
	$(FORT) $(OPTS) $(LDOPTS) $(NOMAIN) $< $(OBJ) -o $@ lin/liblapacktest.a -L../lib \
          -lcuda -lmagma -lmagmablas $(LIBDIR) $(LIB) 

