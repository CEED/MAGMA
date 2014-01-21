#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

DIR = sparse-iter/testing
MAGMA_DIR = ../..
include $(MAGMA_DIR)/Makefile.internal

# ----------
# utility functions
ZSRC += \
	testing_z_matrix_io.cpp	\
	testing_z_matrix_compressor.cpp	\
	testing_z_mv.cpp	\

# ----------
# Krylov space linear solvers
ZSRC += \
	testing_zcg.cpp		\
	run_zcg.cpp		    \
	testing_zbicgstab.cpp	\
	run_zbicgstab.cpp	\
	testing_ziterref.cpp		\
	run_ziterref.cpp	\
	testing_zgmres.cpp	\
	run_zgmres.cpp	    \
	testing_zjacobi.cpp	\
	run_zjacobi.cpp	    \
#	testing_zp1gmres.cpp	\
#	testing_zbicgstab.cpp	\
#	testing_zir.cpp		\
#	testing_zcir.cpp	\
#	testing_zcpir.cpp	\


# ----------
# Sparse direct solvers
ZSRC += \
	testing_zbcsrlu.cpp	\
	run_zbcsrlu.cpp	\
# ----------
# Mixed-precision solvers
ZSRC += \
#	testing_zpcg.cpp	\
#	testing_zpbicgstab.cpp	\
#	testing_zcpbicgstab.cpp	\
#	testing_zpgmres.cpp	\
#	testing_zgmres_pipe.cpp	\
#	testing_zcpgmres.cpp	\
#	testing_zdot.cpp	\
#	testing_zkernel.cpp	\
#	testing_z_csrmv.cpp	\
#	testing_z_mpk_4.cpp	\
#	testing_z_mpk_5.cpp	\

# debugging
ZSRC += \
#	testing_zdebug.cpp	\
#	testing_zcdebug.cpp	\


# ------------------------------------------------------------
-include Makefile.local
-include Makefile.src

ALLSRC := $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLOBJ := $(ALLSRC:.cpp=.o)
EXE    := $(ALLOBJ:.o=)

LIBSRC  := testing_zutil.o testing_cutil.o testing_dutil.o testing_sutil.o testing_util.o
LIBOBJ  := $(LIBSRC:.cpp=.o)
LIBEXE  := $(LIBOBJ:.o=)

# strip out fake exe files generated from LIBSRC
EXE := $(filter-out $(LIBEXE), $(EXE))

.PHONY: all lib clean cleanall

all: $(EXE)

echo:
	@echo EXE $(EXE)
	@echo ALLOBJ $(ALLOBJ)
	@echo ALLSRC $(ALLSRC)
	@echo ZSRC $(ZSRC)

lib:
	cd .. && $(MAKE) lib

# depend on header
$(ALLOBJ): testings.h


$(EXE): $(LIBMAGMA_SPARSE) testing_util.o

clean:
	rm -f *.o *~ 

cleanexe:
	rm -f $(EXE)

cleanall: clean cleanexe

testing_% : testing_%.o
	$(CC) $(LDOPTS) $(NOMAIN) $< -o $@ \
	testing_util.o -L$(MAGMA_DIR)/lib -lmagma_sparse -lmagma \
	$(LIBDIR) \
	$(LIB)

run_% : run_%.o
	$(CC) $(LDOPTS) $(NOMAIN) $< -o $@ \
	testing_util.o -L$(MAGMA_DIR)/lib -lmagma_sparse -lmagma \
	$(LIBDIR) \
	$(LIB)

# keep intermediate object files, to speed up re-compiling when library changes
.SECONDARY: $(ALLOBJ)
