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
             sdlaswp.cu \
             \
             sauxiliary.cu \
             dauxiliary.cu \
             \
             dinplace_transpose.cu \
             dtranspose.cu \
             dpermute.cu \
             \
             cinplace_transpose.cu \
             ctranspose.cu \
             cpermute.cu \
             \
             zinplace_transpose.cu \
             ztranspose.cu \
             zpermute.cu \
             ztrsm.cu \
             ztrmm.cu \
             zherk.cu \
             \
             ctrsm.cu \
             ctrmm.cu \
             csyrk.cu \
             cherk.cu \
             \
             sgemv.cu \
             ourdtrsm.cu \
             ourstrsm.cu \
             dgemv.cu \
             gemv32.cu \
             \
             magma_dlacpy.cu \
             magma_dgemv_MLU.cu \
             magma_dlag2s.cu \
             magma_dlange.cu \
             magma_dlansy.cu \
             magma_dlat2s.cu \
             magma_dsymv.cu \
             magma_ssymv.cu \
             magma_sdaxpycp.cu \
             magma_slag2d.cu \
             \
             dgemm_kernel_a_0.cu \
             dgemm_kernel_N_N_64_16_16_16_4_special.cu \
             dgemm_kernel_T_N_32_32_8_8_8.cu \
             dgemm_kernel_T_T_64_16_16_16_4_v2.cu \
             dgemm_kernel_ab_0.cu \
             dgemm_kernel_N_N_64_16_16_16_4.cu \
             dgemm_kernel_N_T_64_16_4_16_4.cu \
             dgemm_kernel_T_T_64_16_16_16_4.cu \
             \
             sgemm_kernel_a_0.cu \
             sgemm_kernel_N_N_64_16_16_16_4_special.cu \
             sgemm_kernel_T_N_32_32_8_8_8.cu \
             sgemm_kernel_T_T_64_16_16_16_4_v2.cu \
             sgemm_kernel_ab_0.cu \
             sgemm_kernel_N_N_64_16_16_16_4.cu \
             sgemm_kernel_N_T_64_16_4_16_4.cu \
             sgemm_kernel_T_T_64_16_16_16_4.cu \

    ALLOBJ = $(ALLSRC:.cu=.cu_o) 

all: $(LIBMAGMABLAS)

$(LIBMAGMABLAS): $(ALLOBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(ALLOBJ)
	$(RANLIB) $@

clean:
	rm -f *.cu_o *~ *.a *.linkinfo ../lib/libmagmablas.a

%.cu_o: %.cu
	$(NVCC) $(NVOPTS) -gencode arch=compute_13,code=compute_13 -gencode arch=compute_10,code=compute_10 $(INC) -c $< -o $@
