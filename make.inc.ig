#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/scratch/cuda/lib64
#export PATH=$PATH:/mnt/scratch/cuda/bin
#
#/mnt/scratch/sw/intel/2011.2.137/mkl/bin/mklvars.sh intel64
#or
#source /home/mfaverge/mklvars.sh intel64

#
# Old config from Stan.
#
# setenv LD_LIBRARY_PATH /mnt/scratch/cuda/lib64/:/mnt/scratch/sw/intel/C-11.0.083/lib/intel64/
# setenv PATH ${PATH}:/mnt/scratch/cuda/bin/
# setenv LD_LIBRARY_PATH /mnt/scratch/cuda-2.3/lib64/:/mnt/scratch/sw/intel/C-11.0.083/lib/intel64/
# setenv PATH ${PATH}:/mnt/scratch/cuda-2.3/bin/
# setenv LD_LIBRARY_PATH /mnt/scratch/cuda-3.1/cuda/lib64/:/mnt/scratch/sw/intel/C-11.0.083/lib/intel64/
# setenv PATH ${PATH}:/mnt/scratch/cuda-3.1/cuda/bin/
# numactl --interleave=0-7 ./testing_sgetrf

#
# GPU_TARGET specifies for which GPU you want to compile MAGMA:
#     "Tesla" (NVIDIA compute capability 1.x cards)
#     "Fermi" (NVIDIA compute capability 2.x cards)
# See http://developer.nvidia.com/cuda-gpus

GPU_TARGET = Fermi

CC        = gcc
NVCC      = nvcc
FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib

OPTS      = -O3 -DADD_
FOPTS     = -O3 -DADD_ -x f95-cpp-input
NVOPTS    = --compiler-options -fno-strict-aliasing -DUNIX -O3 -DADD_
LDOPTS    = -fPIC

# Sequential version
#LIB_EXP   = -lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lpthread -lcublas -lm

# Multi-threaded version
LIB       = -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lcublas -lm -fopenmp

CUDADIR   = /mnt/scratch/cuda/

LIBDIR    = -L${MKLROOT}/lib/intel64 \
            -L$(CUDADIR)/lib64

INC       = -I$(CUDADIR)/include

#LIBMAGMA     = $(MAGMA_DIR)/lib/magma.a
#LIBMAGMABLAS = $(MAGMA_DIR)/lib/magmablas.a
