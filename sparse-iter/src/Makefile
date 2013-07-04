#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ../..
include ../../Makefile.internal

INC += -I$(MAGMA_DIR)/control

# ------------------------------------------------------------
# Loosely, the order is alphabetic in each section, but mixed precision,
# driver, and multi-GPU routines are separated where it made sense.
# Double precision "d" files that generate "s" files must also be listed
# in ZSRC, despite the name.

# Krylov space linear solvers
ZSRC += \
	zcg.cpp			\
	zgmres.cpp		\
	zbicgstab.cpp	\
                    \
	zpcg.cpp		\

# Krylov space eigen-solvers
ZSRC += \
	#zcgeig.cpp		\
	#zlobpcg.cpp		\
	#zlancsoz.cpp		\

# Mixed-precision example solvers
ZSRC += \
	#zcfgmres.cpp		\

# QR and least squares, GPU interface
ZSRC += \
	#zungqr_lapack.cpp	\
	#zungqr_cgs.cpp		\
	#zungqr_mgs.cpp		\
	#zungqr_chol.cpp		\
	#zungqr_iter.cpp		\



# Wrappers, tools etc
ZSRC += \
	magma_z_precond_wrapper.cpp \


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLSRC_F = \

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLOBJ := $(ALLSRC:.cpp=.o) $(ALLSRC_F:.f=.o)

all: $(LIBMAGMA_SPARSE)

$(LIBMAGMA_SPARSE): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.o *~ *.linkinfo

cleanall: clean
	rm -f *.a
	rm -f $(LIBMAGMA_SPARSE)
