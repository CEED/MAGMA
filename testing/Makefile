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
        testing_zsytrd.cpp 	\
        testing_zgebrd.cpp	\
	\
	testing_zcgesv_gpu.cpp	\
	testing_zcgeqrsv_gpu.cpp\
	testing_zcposv_gpu.cpp
#        testing_zgetrf_f.f


CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

#ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLSRC= $(SSRC)

ALLOBJF= $(ALLSRC:.f=.o)
ALLOBJ = $(ALLOBJF:.cpp=.o)
EXE = $(ALLOBJ:.o=)

all: $(EXE)

$(EXE): $(ALLOBJ)

clean:
	rm -f $(OBJ) $(EXE) *.o *~

cleanall: clean
	rm -f $(EXE)

testing_%.o : testing_%.f
	$(FORT) $(OPTS) $(INC) -c $< -o $@ 

testing_%.o : testing_%.cpp
	$(CC) $(OPTS) $(INC) -c $< -o $@

.o:
	$(FORT) $(LDOPTS) $< -o $@ lin/liblapacktest.a -L../lib \
          -lcuda -lmagma -lmagmablas $(LIBDIR) $(LIB) get_nb.cpp
