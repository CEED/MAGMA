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
	zswap.cu	\
	zswapblk.cu	\
	zlacpy.cu	\
	ztranspose.cu 	\
	ztranspose-v2.cu\
	zinplace_transpose.cu	\
	\
	zclaswp.cu 	\
        zcaxpycp.cu     \
	zlag2c.cu       \
	clag2z.cu       \

ifeq (${GPU_TARGET}, 1)
ZSRC += zlat2c.cu	\
	zlange.cu       \
	zlanhe.cu
endif

CSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p c)
DSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p "d ds")
SSRC  = $(shell $(PYTHON) $(CODEGEN) -f "$(ZSRC)" -o -p s)

FERMISRC =	sgemm_fermi.cu \
	        dgemm_fermi.cu \
		ssymv_fermi.cu \
	        dsymv_fermi.cu \
		zsymv_fermi.cu \
		csymv_fermi.cu \
		chemv_fermi.cu \
	        zhemv_fermi.cu \
                \
                strsm_tesla.cu \
                dtrsm_tesla.cu \

TESLASRC =      sgemv_tesla.cu 	\
	        dgemv_tesla.cu 	\
	        gemv32_tesla.cu \
		\
                strsm_tesla.cu \
                dtrsm_tesla.cu \
                \
	        ssymv_tesla.cu 	\
	        dsymv_tesla.cu 	\
	        chemv_tesla.cu 	\
		    csymv_tesla.cu \
	        \
	        sgemm_tesla.cu 					\
		sgemm_tesla_a_0.cu 				\
		sgemm_tesla_ab_0.cu 				\
		sgemm_tesla_N_N_64_16_16_16_4.cu 		\
		sgemm_tesla_N_N_64_16_16_16_4_special.cu 	\
		sgemm_tesla_N_T_64_16_4_16_4.cu 		\
		sgemm_tesla_T_N_32_32_8_8_8.cu 			\
		sgemm_tesla_T_T_64_16_16_16_4.cu 		\
		\
		dgemm_tesla.cu 					\
		dgemm_tesla_a_0.cu 				\
		dgemm_tesla_ab_0.cu 				\
		dgemm_tesla_N_N_64_16_16_16_4.cu 		\
		dgemm_tesla_N_N_64_16_16_16_4_special.cu 	\
		dgemm_tesla_N_T_64_16_4_16_4.cu 		\
		dgemm_tesla_T_N_32_32_8_8_8.cu 			\
		dgemm_tesla_T_T_64_16_16_16_4.cu 		\
		dgemm_tesla_T_T_64_16_16_16_4_v2.cu

# 		zhemv_tesla.cu

ALLSRC= $(ZSRC) $(CSRC) $(DSRC) $(SSRC) \
        dgemv_MLU.cu

ifeq (${GPU_TARGET}, 0)
ALLSRC += $(TESLASRC)
GPUOPT = ${TESLAOPT}
endif

ifeq (${GPU_TARGET}, 1)
ALLSRC += $(FERMISRC)
GPUOPT = ${FERMIOPT}
endif

ALLOBJ = $(ALLSRC:.cu=.cu_o) 

all: $(LIBMAGMABLAS)

$(LIBMAGMABLAS): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

.cpp.o:
	$(CC) $(OPTS) $(INC) -c $< -o $@

%.cu_o: %.cu
	$(NVCC) $(NVOPTS) $(GPUOPT) $(INC) -c $< -o $@


clean:
	rm -f *.cu_o *.o *~ *.linkinfo

cleanall: clean
	rm -f *.a
	rm -f $(LIBMAGMABLAS)

