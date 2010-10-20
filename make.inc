#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of	Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

	CC        = gcc
	NVCC      = nvcc
	FORT      = gfortran

	ARCH      = ar
	ARCHFLAGS = cr
	RANLIB    = ranlib

	OPTS      = -O3 -DADD_ -g
	NVOPTS    = --compiler-options -fno-strict-aliasing  -arch sm_13 \
                    -DUNIX -O3
	LDOPTS    = -fPIC

	LIB       = -lmkl_em64t -lguide -lpthread -lcublas -lcudart -lm 

	CUDADIR   = /usr/local/cuda-2.3

	LIBDIR    = -L/home/tomov/intel/mkl/10.0.1.014/lib/em64t \
                    -L$(CUDADIR)/lib64
	INC       = -I../include -I$(CUDADIR)/include
	
	LIBMAGMA     = ../lib/libmagma.a
	LIBMAGMABLAS = ../lib/libmagmablas.a
