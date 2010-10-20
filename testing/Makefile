#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

include ../make.inc

    SRC = testing_spotrf.cpp \
          testing_spotrf_gpu.cpp \
          testing_sgeqrf.cpp \
          testing_sgeqrf_gpu.cpp \
          testing_sgeqrs_gpu.cpp \
          testing_sgesv_gpu.cpp \
          testing_sgetrf.cpp \
          testing_sgetrf_gpu-v2.cpp \
          testing_sgetrf_gpu.cpp \
          testing_sgeqlf.cpp \
          testing_sgelqf.cpp \
          \
          testing_sgehrd.cpp \
          testing_ssytrd.cpp \
          testing_sgebrd.cpp \


    OBJ = $(SRC:.cpp=.o)
    EXE = $(OBJ:.o=)

all: $(EXE)

$(EXE): $(OBJ)

clean:
	rm -f $(OBJ) $(EXE) *.o *~

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

.o:
	$(FORT) $(LDOPTS) $< -o $@ lin/liblapacktest.a -L../lib \
          -lcuda -lmagma -lmagmablas $(LIBDIR) $(LIB) get_nb.cpp
