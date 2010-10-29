#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee
#      Univ. of California Berkeley
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

    include ../make.inc

    ALLSRC = sinplace_transpose.cu \
             stranspose.cu \
             stranspose-v2.cu \
             spermute.cu \
             spermute-v2.cu \
             sdlaswp.cu \
             \
             sauxiliary.cu \
             \
             dlag2s.cu \
             slag2d.cu \
             dgemv_MLU.cu \

    ALLOBJ = $(ALLSRC:.cu=.cu_o) 

all: $(LIBMAGMABLAS)

$(LIBMAGMABLAS): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.cu_o *~ *.a *.linkinfo 

cleanall: clean
	rm -f ../lib/libmagmablas.a

%.cu_o: %.cu
	$(NVCC) $(NVOPTS) -gencode arch=compute_13,code=compute_13 -gencode arch=compute_10,code=compute_10 $(INC) -c $< -o $@
