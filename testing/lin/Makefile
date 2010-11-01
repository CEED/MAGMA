#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 0.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      June 2009
#//////////////////////////////////////////////////////////////////////////////

include ../../make.inc

    SRC = cqrt02.f  \
          zqrt02.f  \
          shst01.f  \
          sorghr.f  \
          sqrt02.f  \
          slabad.f  \
          sort01.f  \
          \
          dhst01.f  \
          dorghr.f  \
          dqrt02.f  \
          dlabad.f  \
          dort01.f  \
          \
          ssyt21.f \
          slarfy.f \
          dsyt21.f \
          dlarfy.f \
          \
          sorgbr.f \
          sbdt01.f \
          dbdt01.f \
          \
          sstt21.f \
          dstt21.f \

    OBJ = $(SRC:.f=.o)
    EXE = $(OBJ:.o=)

all: liblapacktest.a

liblapacktest.a: $(OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(OBJ)
	$(RANLIB) $@

clean:
	rm -f $(OBJ) liblapacktest.a *~

cleanall: clean
	rm -f liblapacktest.a

.f.o:
	$(FORT) $(OPTS) $(INC) -c $< -o $@

