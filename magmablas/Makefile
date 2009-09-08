#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 0.1) --
#      Univ. of Tennessee
#      Univ. of California Berkeley
#      June 2009
#//////////////////////////////////////////////////////////////////////////////

    include ../make.inc

    ALLSRC = sinplace_transpose.cu \
             stranspose.cu \
             spermute.cu \
             \
             sauxiliary.cu \
             dauxiliary.cu \
             \
             dinplace_transpose.cu \
             dtranspose.cu \
             dpermute.cu \
             \
             ctrsm.cu \
             csyrk.cu \
             cherk.cu \
             \
             sgemv.cu \
             dgemv.cu \

    ALLOBJ = $(ALLSRC:.cu=.cu_o) 

all: $(LIBMAGMABLAS)

$(LIBMAGMABLAS): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.cu_o *~ *.a *.linkinfo ../lib/libmagmablas.a

%.cu_o: %.cu
	$(NVCC) $(NVOPTS) -gencode arch=compute_13,code=compute_13 -gencode arch=compute_10,code=compute_10 $(INC) -c $< -o $@
