#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of	Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

#
# GPU_TARGET specifies for which GPU you want to compile MAGMA
#      0: Tesla Family
#      1: Fermi Family
#
GPU_TARGET = 0

MAKE      = make -j 4
CC        = gcc
NVCC      = nvcc
FORT      = gfortran


ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib
OPTS      = -O0 -DADD_ -g3 -Wall
NVOPTS    = -g --compiler-options -fno-strict-aliasing  -DUNIX 
LDOPTS    = -fPIC
LIB       = -Wl,--start-group -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -Wl,--end-group -lpthread -lcublas -lcudart -lm 
CUDADIR   = /mnt/scratch/cuda-3.1
LIBDIR    = -L$(MKLROOT) \
            -L$(CUDADIR)/lib64
INC       = -I../include -I$(CUDADIR)/include

LIBMAGMA     = ../lib/libmagma.a
LIBMAGMABLAS = ../lib/libmagmablas.a
