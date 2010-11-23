#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 0.2) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of	Colorado, Denver
#      November 2009
#
#      Contributed by: Allan Menezes (Ontario, Canada)
#//////////////////////////////////////////////////////////////////////////////

GPU_TARGET = 1

MAKE      = colormake
CC        = gcc
NVCC      = nvcc
FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib

OPTS      = -O0 -g3 -DADD_ -Wall
NOMAIN    = -Mnomain
NVOPTS    = --compiler-options -fno-strict-aliasing -DUNIX -O3
LDOPTS    = -fPIC

LIB       = -lgoto2  -lpthread -lcublas -lcudart -lm 

CUDADIR   = /opt/cuda

LIBDIR    = -L/opt/gotoblas2 -L${CUDADIR}/lib64 -L/usr/lib/nvidia-current
INC       = -I../include -I${CUDADIR}/include

LIBMAGMA     = ../lib/libmagma.a
LIBMAGMABLAS = ../lib/libmagmablas.a
