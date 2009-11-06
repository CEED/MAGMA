#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 0.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      June 2009
#//////////////////////////////////////////////////////////////////////////////
include make.inc

all: lib test

lib: libmagma libmagmablas

clean: cleanall

libmagma:
	( cd src; $(MAKE) )

libmagmablas:
	( cd magmablas; $(MAKE) )

test:
	( cd testing/lin; $(MAKE) )
	( cd testing; $(MAKE) )

cleanall:
	( cd src; $(MAKE) clean )
	( cd magmablas; $(MAKE) clean ) 
	( cd testing; $(MAKE) clean )
	( cd testing/lin; $(MAKE) clean )
	( cd lib; rm -f *.a )
