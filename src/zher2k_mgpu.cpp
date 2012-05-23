/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Mark Gates
*/
#include "common_magma.h"

extern "C"
void magma_zher2k_mgpu_mark1(
    char uplo, char trans, magma_int_t n, magma_int_t k,
    cuDoubleComplex alpha, cuDoubleComplex *dA[], magma_int_t lda,
                           cuDoubleComplex *dB[], magma_int_t ldb,
    double beta,           cuDoubleComplex *dC[], magma_int_t ldc,  magma_int_t offset,
    magma_int_t ngpu, magma_int_t nb, cudaStream_t streams[][10], magma_int_t nstream )
{
    #define dA(dev, i, j) (dA[dev] + (i) + (j)*lda)
    #define dB(dev, i, j) (dB[dev] + (i) + (j)*ldb)
    #define dC(dev, i, j) (dC[dev] + (i) + (j)*ldc)
    
    cuDoubleComplex b = MAGMA_Z_MAKE( beta, 0. );
    
    int cdev;
    cudaGetDevice( &cdev );
    
    // loop over all blocks
    // Faster to have two loops: first does A*B', second does B*A'
    for( int i = 0; i < n; i += nb ) {
        int ib     = min( nb, n-i );      // block size
        int ioff   = i + offset;          // start global index in parent matrix
        int iblock = (ioff / nb) / ngpu;  // local block id
        int dev    = (ioff / nb) % ngpu;
        int di     = iblock*nb;           // local index in parent matrix
        
        cudaSetDevice( dev );
        int s = iblock % nstream;
        magmablasSetKernelStream( streams[ dev ][ s ] );
        
        // C[i:n,i] += A[i:n,0] * B[i,0]'
        //printf( "zgemm  n=%4d, ib=%4d, k=%4d, i=%4d\n", n-i, ib, k, i );
        magma_zgemm( MagmaNoTrans, MagmaConjTrans, n-i, ib, k,
                     alpha, dA(dev,i,0), lda,
                            dB(dev,i,0), ldb,
                     b,     dC(dev,ioff,di), ldc );
    }
    for( int i = 0; i < n; i += nb ) {
        int ib     = min( nb, n-i );      // block size
        int ioff   = i + offset;          // start global index in parent matrix
        int iblock = (ioff / nb) / ngpu;  // local block id
        int dev    = (ioff / nb) % ngpu;
        int di     = iblock*nb;           // local index in parent matrix
        
        cudaSetDevice( dev );
        int s = iblock % nstream;
        magmablasSetKernelStream( streams[ dev ][ s ] );
        
        // C[i:n,i] += B[i:n,0] * A[i,0]'
        //printf( "zgemm  n=%4d, ib=%4d, k=%4d, i=%4d\n", n-i, ib, k, i );
        magma_zgemm( MagmaNoTrans, MagmaConjTrans, n-i, ib, k,
                     alpha, dB(dev,i,0), ldb,
                            dA(dev,i,0), lda,
                     b,     dC(dev,ioff,di), ldc );
    }
    
    cudaSetDevice( cdev );
}
