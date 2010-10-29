#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      November 2010
#//////////////////////////////////////////////////////////////////////////////
include make.inc

all: lib test

lib: libmagma libmagmablas

clean: cleanall

libmagma:
	( cd src && $(MAKE) )

libmagmablas:
	( cd magmablas && $(MAKE) )

test:
	( cd testing/lin && $(MAKE) )
	( cd testing     && $(MAKE) )

clean:
	( cd src         && $(MAKE) clean )
	( cd testing     && $(MAKE) clean )
	( cd testing/lin && $(MAKE) clean )
	( cd magmablas   && $(MAKE) clean ) 

cleanall:
	( cd src         && $(MAKE) cleanall )
	( cd testing     && $(MAKE) cleanall )
	( cd testing/lin && $(MAKE) cleanall )
	( cd magmablas   && $(MAKE) cleanall ) 
	( cd lib && rm -f *.a )

include ./Makefile.gen

