#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2011
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ../..
include ../../Makefile.internal

# ----------
# Krylov space linear solvers
ZSRC += \
	testing_z_matrix_io.cpp     \
    testing_z_mv.cpp            \
	testing_zcg.cpp	            \
	testing_zgmres.cpp	        \
	testing_zbicgstab.cpp       \


# ----------
# Krylov space eigen-solvers 
ZSRC += \

# ----------
# Mixed-precision solvers
ZSRC += \

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

cleanall: clean
	rm -f $(EXE)

testing_% : testing_%.o
	$(CC) $(LDOPTS) $(NOMAIN) $< -o $@ \
	testing_util.o -L$(MAGMA_DIR)/lib -lmagma_sparse -lmagma \
	$(LIBDIR) \
	$(LIB) 

# keep intermediate object files, to speed up re-compiling when library changes
.SECONDARY: $(ALLOBJ)
