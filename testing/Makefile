#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 0.2) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2009
#//////////////////////////////////////////////////////////////////////////////

include ../make.inc

    SRC = testing_spotrf.cpp \
          testing_spotrf_gpu.cpp \
          testing_sgeqrf.cpp \
          testing_sgeqrf_gpu.cpp \
          testing_sgeqrf_gpu-v2.cpp \
          testing_sgeqrs_gpu.cpp \
          testing_sgetrf.cpp \
          testing_sgetrf_gpu-v2.cpp \
          testing_sgetrf_gpu.cpp \
          testing_sgehrd.cpp \
          testing_sgeqlf.cpp \
          testing_sgelqf.cpp \
          \
          testing_dpotrf.cpp \
          testing_dpotrf_gpu.cpp \
          testing_dgeqrf.cpp \
          testing_dgeqrf_gpu.cpp \
          testing_dgeqrs_gpu.cpp \
          testing_dgetrf.cpp \
          testing_dgetrf_gpu.cpp \
          testing_dgehrd.cpp \
          testing_dgeqlf.cpp \
          testing_dgelqf.cpp \
          \
          testing_cpotrf.cpp \
          testing_cgetrf.cpp \
          testing_cgeqrf.cpp \
          testing_cpotrf_gpu.cpp \
          testing_cgetrf_gpu.cpp \
          testing_cgeqrf_gpu.cpp \
          \
          testing_zpotrf.cpp \
          testing_zgetrf.cpp \
          testing_zgeqrf.cpp \
          testing_zpotrf_gpu.cpp \
          testing_zgetrf_gpu.cpp \
          testing_zgeqrf_gpu.cpp \
          \
          testing_dsgeqrsv_gpu.cpp \
          testing_dsgesv_gpu.cpp \
          testing_dgesv_gpu.cpp \
          testing_sgesv_gpu.cpp \
          testing_dsposv_gpu.cpp \
          testing_dgemm.cpp \
          testing_sgemm.cpp \
          testing_dsymv.cpp \
          testing_ssymv.cpp \
          \

    OBJ = $(SRC:.cpp=.o)
    EXE = $(OBJ:.o=)

all: $(EXE)

$(EXE): $(OBJ)

clean:
	rm -f $(OBJ) $(EXE) *~

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

.o:
	$(FORT) $(LDOPTS) $< -o $@ lin/liblapacktest.a -L../lib \
          -lcuda -lmagma -lmagmablas $(LIBDIR) $(LIB) get_nb.cpp
