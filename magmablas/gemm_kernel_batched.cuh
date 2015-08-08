/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
       @author Azzam Haidar

       See [zcds]gemm_fermi.cu for description of related files.
*/

////////////////////////////////////////////////////////////////////////////////////////////////////
extern "C" static __global__
void batched_gemm_kernel_name(precision)(
    int M, int N, int K,
    FloatingPoint_t const * const * Aarray, int LDA,
    FloatingPoint_t const * const * Barray, int LDB,
    FloatingPoint_t**       Carray, int LDC,
    FloatingPoint_t alpha, FloatingPoint_t beta,
    int offsetA, int offsetB )
{
    //if ( blockIdx.y > blockIdx.x ) return; //for lower blkx > blky do not have to compute
    int batchid = blockIdx.z;
    #ifdef TEXTURE_1D
    int matrixA_size = gridDim.z > 1 ?  Aarray[1] - Aarray[0] : 0;
    int matrixB_size = gridDim.z > 1 ?  Aarray[1] - Aarray[0] : 0;
    offsetA += batchid*matrixA_size;
    offsetB += batchid*matrixB_size;
    #endif
    devfunc_name(precision)( M, N, K, Aarray[batchid], LDA, Barray[batchid], LDB, Carray[batchid], LDC, alpha, beta, offsetA, offsetB );
}
