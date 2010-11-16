#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 0.1) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      June 2009
#//////////////////////////////////////////////////////////////////////////////

include ../../make.inc

SRC = 	zbdt01.f  \
	zhet21.f  \
	zhst01.f  \
	zunt01.f  \
	zqrt02.f  \
	\
	cbdt01.f  \
	chet21.f  \
	chst01.f  \
	cunt01.f  \
	cqrt02.f  \
	\
	dbdt01.f  \
	dsyt21.f  \
	dhst01.f  \
	dort01.f  \
	dqrt02.f  \
	\
	sbdt01.f  \
	ssyt21.f  \
	shst01.f  \
	sort01.f  \
	sqrt02.f  \
	\
	sorghr.f  \
        slabad.f  \
        \
        dorghr.f  \
        dlabad.f  \
        \
        slarfy.f \
        dlarfy.f \
        \
        sorgbr.f \
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

