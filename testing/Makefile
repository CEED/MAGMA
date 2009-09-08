#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 0.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      June 2009
#//////////////////////////////////////////////////////////////////////////////

include ../make.inc

    SRC = testing_spotrf.cpp \
          testing_spotrf_gpu.cpp \
          testing_sgeqrf.cpp \
          testing_sgeqrf_gpu.cpp \
          testing_sgetrf.cpp \
          testing_sgetrf_gpu.cpp \
          testing_sgehrd.cpp \
          testing_sgeqlf.cpp \
          \
          testing_dpotrf.cpp \
          testing_dpotrf_gpu.cpp \
          testing_dgeqrf.cpp \
          testing_dgeqrf_gpu.cpp \
          testing_dgetrf.cpp \
          testing_dgetrf_gpu.cpp \
          testing_dgehrd.cpp \
          \
          testing_cpotrf.cpp \

    OBJ = $(SRC:.cpp=.o)
    EXE = $(OBJ:.o=)

all: $(EXE)

$(EXE): $(OBJ)

clean:
	rm -f $(OBJ) $(EXE) *~

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

.o:
	$(CC) $(LDOPTS) $< -o $@ -L../lib \
          -lcuda -lmagma -lmagmablas $(LIBDIR) $(LIB) get_nb.cpp
