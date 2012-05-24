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
#include <assert.h>

extern "C" void
magmablas_zsymmetrize( char uplo, int m, cuDoubleComplex *A, int lda );

extern "C"
void magma_zhemm_mgpu(
    char side, char uplo, magma_int_t m, magma_int_t n,
    cuDoubleComplex alpha, cuDoubleComplex *dA[], magma_int_t lda,  magma_int_t offset,
                           cuDoubleComplex *dB[], magma_int_t ldb,
    cuDoubleComplex beta,  cuDoubleComplex *C,    magma_int_t ldc,
    magma_int_t ngpu, magma_int_t nb, cudaStream_t streams[][10], magma_int_t nstream )
{
    #define dA(dev, i, j) (dA[dev] + (i) + (j)*lda)
    #define dB(dev, i, j) (dB[dev] + (i) + (j)*ldb)
    #define dC(dev, i, j) (dC[dev] + (i) + (j)*ldc)
    #define C(i, j) (C + (i) + (j)*ldc)
    
    assert( lda >= m );
    assert( ldb >= m );
    assert( ldc >= m );
    
    cuDoubleComplex c_one = MAGMA_Z_ONE;
    int ione = 1;
    
    cuDoubleComplex *dC[MagmaMaxGPUs], *Ctmp;
    
    int cdev;
    cudaGetDevice( &cdev );
    
    // allocate and zero out result
    for( int d = 0; d < ngpu; ++d ) {
        magma_setdevice( d );
        if ( magma_zmalloc( &dC[d], ldc*n ) != MAGMA_SUCCESS ) {
            exit(1);
        }
        cudaMemset( dC[d], 0, ldc*n*sizeof(cuDoubleComplex) );
    }
    
    // loop over all blocks
    // Faster to have two loops: first does A*B', second does B*A'
    for( int i = 0; i < m; i += nb ) {
        int ib     = min( nb, m-i );      // block size
        int ioff   = i + offset;          // start global index in parent matrix
        int iblock = (ioff / nb) / ngpu;  // local block id
        int dev    = (ioff / nb) % ngpu;
        int di     = iblock*nb;           // local index in parent matrix
        
        cudaSetDevice( dev );
        int s = iblock % nstream;
        magmablasSetKernelStream( streams[ dev ][ s ] );
        
        // make diagonal block symmetric
        magmablas_zsymmetrize( MagmaLower, ib, dA(dev,ioff,di), lda );
        
        // C[i,0] += A[i:m,i]' * B[i:m,0]
        //printf( "ib %d, n %d, m-i %d\n", ib, n, m-i );
        magma_zgemm( MagmaConjTrans, MagmaNoTrans, ib, n, m-i,
                     alpha, dA(dev,ioff,di), lda,
                            dB(dev,i,0),     ldb,
                     c_one, dC(dev,i,0),     ldc );
        
        // C[i+1:n,0] += A[i+1:n,i] * B[i,0]
        //printf( "m-i-ib %d, n %d, ib %d\n", m-i-ib, n, ib );
        magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i-ib, n, ib,
                     alpha, dA(dev,ioff+ib,di), lda,
                            dB(dev,i,0),        ldb,
                     c_one, dC(dev,i+ib,0),     ldc );
    }
    
    // meanwhile on CPU, scale C := beta*C
    for( int j = 0; j < n; ++j ) {
        blasf77_zscal( &m, &beta, C(0,j), &ione );
    }
    
    // wait and reduce results
    int size = ldc*n;
    Ctmp = (cuDoubleComplex*) malloc( size*sizeof(cuDoubleComplex) );
    for( int d = 0; d < ngpu; ++d ) {
        magma_setdevice( d );
        magma_zgetmatrix( m, n, dC[d], ldc, Ctmp, ldc );
        blasf77_zaxpy( &size, &c_one, Ctmp, &ione, C, &ione );
        magma_free( dC[d] );
    }
    free( Ctmp );
    
    cudaSetDevice( cdev );
}
