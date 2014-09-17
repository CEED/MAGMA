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
extern "C" __global__
void kernel_name(precision) ## _batched(
    int M, int N, int K,
    const FloatingPoint_t** Aarray, int LDA,
    const FloatingPoint_t** Barray, int LDB,
    FloatingPoint_t**       Carray, int LDC,
    FloatingPoint_t alpha, FloatingPoint_t beta,
    int offsetA, int offsetB )
{
    k = blockIdx.z;
    devfunc_name(precision)( M, N, K, Aarray[k], LDA, Barray[k], LDB, Carray[k], LDC, alpha, beta, offsetA, offsetB );
}
