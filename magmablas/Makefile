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
ALLSRC= $(DSRC) $(SSRC)		\
        dlag2s.cu \
        slag2d.cu \
        dgemv_MLU.cu \
        \
        sgemv_tesla.cu \
        dgemv_tesla.cu \
        gemv32_tesla.cu \
        \
        sgemm_tesla.cu \
        sgemm_tesla_a_0.cu \
        sgemm_tesla_ab_0.cu \
        sgemm_tesla_N_N_64_16_16_16_4.cu \
        sgemm_tesla_N_N_64_16_16_16_4_special.cu \
        sgemm_tesla_N_T_64_16_4_16_4.cu \
        sgemm_tesla_T_N_32_32_8_8_8.cu \
        sgemm_tesla_T_T_64_16_16_16_4.cu \
        \
        dgemm_tesla.cu \
        dgemm_tesla_a_0.cu \
        dgemm_tesla_ab_0.cu \
        dgemm_tesla_N_N_64_16_16_16_4.cu \
        dgemm_tesla_N_N_64_16_16_16_4_special.cu \
        dgemm_tesla_N_T_64_16_4_16_4.cu \
        dgemm_tesla_T_N_32_32_8_8_8.cu \
        dgemm_tesla_T_T_64_16_16_16_4.cu \
        dgemm_tesla_T_T_64_16_16_16_4_v2.cu \
        \

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
