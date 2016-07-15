/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal s

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
*/

#include "magma_internal.h"
#define PRECISION_s
#include "gemm_template_kernel_vbatched.cuh"

#include "gemm_config/sgemm_param_nn.h"
#include "gemm_config/sgemm_param_nt.h"
#include "gemm_config/sgemm_param_tn.h"
#include "gemm_config/sgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

#ifdef __cplusplus
extern "C" 
#endif
void 
magmablas_sgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t* lddb,
    float beta,
    float **dC_array, magma_int_t* lddc, 
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, 
    magma_int_t roffA, magma_int_t coffA, magma_int_t roffB, magma_int_t coffB, magma_int_t roffC, magma_int_t coffC, 
    magma_int_t spec_m, magma_int_t spec_n, magma_int_t spec_k, 
    magma_int_t batchCount, magma_queue_t queue )
{
    if(max_m <=0 || max_n <= 0 || max_k <= 0) return;
    
    magma_int_t shape = 0;
    if      (transA == MagmaNoTrans   && transB == MagmaNoTrans)   { shape = 0; } // nn
    else if (transA == MagmaNoTrans   && transB == MagmaTrans)     { shape = 1; } // nt
    else if (transA == MagmaNoTrans   && transB == MagmaConjTrans) { shape = 2; } // nc
    else if (transA == MagmaTrans     && transB == MagmaNoTrans)   { shape = 3; } // tn
    else if (transA == MagmaTrans     && transB == MagmaTrans)     { shape = 4; } // tt
    else if (transA == MagmaTrans     && transB == MagmaConjTrans) { shape = 5; } // tc
    else if (transA == MagmaConjTrans && transB == MagmaNoTrans)   { shape = 6; } // cn
    else if (transA == MagmaConjTrans && transB == MagmaTrans)     { shape = 7; } // ct
    else if (transA == MagmaConjTrans && transB == MagmaConjTrans) { shape = 8; } // cc
    
    switch(shape)
    {
        case 0: // nn
            {
                if(max_k < 64)
                {
                    if(max_k==8 && max_n==24)
                    gemm_template_vbatched_nn<float, version(NN,512), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                    else if (max_n<32)
                    gemm_template_vbatched_nn<float, version(NN,510), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                    else
                    gemm_template_vbatched_nn<float, version(NN,504), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                }
                else
                {
                    gemm_template_vbatched_nn<float, version(NN,518), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                }
            }
            break;
        case 1: // nt
            {
                gemm_template_vbatched_nt<float, version(NT,734), 0, 0>
                (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
            }
            break;
        case 2: // nc
            {
                gemm_template_vbatched_nt<float, version(NT,734), 0, 1>
                (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
            }
            break;
        case 3: // tn
            {
                if(max_k < 64)
                {
                    gemm_template_vbatched_tn<float, version(TN,654), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                }
                else
                {
                    gemm_template_vbatched_tn<float, version(TN,666), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                }
            }
            break;
        case 6: // cn
            {
                if(max_k < 64)
                {
                    gemm_template_vbatched_tn<float, version(TN,654), 1, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                }
                else
                {
                    gemm_template_vbatched_tn<float, version(TN,666), 1, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                }
            }
            break;
        case 4: // tt
            {
                if(max_k < 128)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<float, version(TT,275), 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<float, version(TT,312), 0, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<float, version(TT,312), 0, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                }
            }
            break;
        case 5: // tc
            {
                if(max_k < 128)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<float, version(TT,275), 0, 1>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<float, version(TT,312), 0, 1>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<float, version(TT,312), 0, 1>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                }
            }
            break;
        case 7: // ct
            {
                if(max_k < 128)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<float, version(TT,275), 1, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<float, version(TT,312), 1, 0>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<float, version(TT,312), 1, 0>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                }
            }
            break;
        case 8: // cc
            {
                if(max_k < 128)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<float, version(TT,275), 1, 1>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<float, version(TT,312), 1, 1>
                        (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<float, version(TT,312), 1, 1>
                    (m, n, k, dA_array, ldda, dB_array, lddb, dC_array, lddc, alpha, beta, max_m, max_n, roffA, coffA, roffB, coffB, roffC, coffC, spec_m, spec_n, spec_k, batchCount, queue);
                }
            }
            break;
        default:; // propose something
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
