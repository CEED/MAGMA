#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 1.0) --
#      Univ. of Tennessee
#      Univ. of California Berkeley
#      November 2010
#//////////////////////////////////////////////////////////////////////////////

MAGMA_DIR = ..
include ../Makefile.internal

ZSRC  = zauxiliary.cu	\
	zpermute.cu	\
	zpermute-v2.cu	\
	zclaswp.cu 	\
	ztranspose.cu 	\
	ztranspose-v2.cu\
	zinplace_transpose.cu

CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

#ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC)
ALLSRC= $(SSRC)		\
        dlag2s.cu \
        slag2d.cu \
        dgemv_MLU.cu \
        \
        sgemv_tesla.cu \
        dgemv_tesla.cu \
        gemv32_tesla.cu

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
