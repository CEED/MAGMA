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
	zbicg.cpp               \  #BiCG - preconditioned in other files?
        zcg.cpp                 \  # CG - preconditioned in other file?
	zgmres.cpp		\  # GMRES

# Krylov space eign-solvers
ZSRC += \
        zcgeig.cpp              \  # CG based eigensolver for part of the spectrum
                                   # There are variations for middle of spectrum - like
                                   # folded spectrum, etc., where to put those
        zlobpcg.cpp             \  # LOBPCG - similar to CG
        zlancsoz.cpp            \

# Mixed-precision example solvers
ZSRC += \
	zcfgmres.cpp		\  # Flexible GMRES example of inner - outer loop
                                   # The "preconditioner" is some iterative process in
                                   # in single; there could be variations of this

# QR and least squares, GPU interface
ZSRC += \
        zungqr_lapack.cpp       \  # From a set of vectors to get orthogonal vectors; 
                                   # could be LAPACK QR + zungqr
	zungqr_cgs.cpp		\  # or clasical Gram-Schmidt
	zungqr_mgs.cpp          \  # modified
	zungqr_chol.cpp         \  # Cholesky based
	zungqr_iter.cpp         \  # Iterative for cases where Cholesky fails
                                   # What else can we put?
        \

# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLSRC_F = \

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLOBJ := $(ALLSRC:.cpp=.o) $(ALLSRC_F:.f=.o)

all: $(LIBItMAGMA)

$(LIBItMAGMA): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.o *~ *.linkinfo

cleanall: clean
	rm -f *.a
	rm -f $(LIBItMAGMA)
