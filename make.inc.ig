#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 0.2) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of	Colorado, Denver
#      November 2009
#//////////////////////////////////////////////////////////////////////////////

# setenv LD_LIBRARY_PATH /mnt/scratch/cuda/lib64/:/mnt/scratch/sw/intel/C-11.0.083/lib/intel64/
# setenv PATH ${PATH}:/mnt/scratch/cuda/bin/
# setenv LD_LIBRARY_PATH /mnt/scratch/cuda-2.3/lib64/:/mnt/scratch/sw/intel/C-11.0.083/lib/intel64/
# setenv PATH ${PATH}:/mnt/scratch/cuda-2.3/bin/
# setenv LD_LIBRARY_PATH /mnt/scratch/cuda-3.1/cuda/lib64/:/mnt/scratch/sw/intel/C-11.0.083/lib/intel64/
# setenv PATH ${PATH}:/mnt/scratch/cuda-3.1/cuda/bin/
# numactl --interleave=0-7 ./testing_sgetrf

#
# GPU_TARGET specifies for which GPU you want to compile MAGMA
#      0: Tesla Family
#      1: Fermi Family
#
GPU_TARGET = 1

CC        = gcc
NVCC      = nvcc
FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib

OPTS      = -O3 -DADD_
NVOPTS    = --compiler-options -fno-strict-aliasing -DUNIX -O3 -DADD_
LDOPTS    = -fPIC -L/mnt/scratch/sw/intel/11.1.069/lib/intel64 -L/mnt/scratch/sw/intel/11.1.069/mkl/lib/em64t

LIB       = -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lguide -lpthread -lcublas -lm -fopenmp

#CUDADIR   = /mnt/scratch/cuda/
#CUDADIR   = /mnt/scratch/cuda-2.3
#CUDADIR   = /mnt/scratch/cuda-3.0/cuda/
#CUDADIR   = /mnt/scratch/cuda-3.1/cuda
CUDADIR   = /mnt/scratch/cuda-3.2/

LIBDIR    = -L/mnt/scratch/sw/intel/11.1.069/lib/intel64 \
            -L$(CUDADIR)/lib64

INC       = -I../include -I$(CUDADIR)/include

LIBMAGMA     = ../lib/libmagma.a
LIBMAGMABLAS = ../lib/libmagmablas.a
