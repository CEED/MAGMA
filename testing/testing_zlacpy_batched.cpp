/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cblas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

extern "C" void
magmablas_zlacpy_batched(
    char uplo, magma_int_t m, magma_int_t n,
    const cuDoubleComplex * const *Aarray, magma_int_t lda,
    cuDoubleComplex              **Barray, magma_int_t ldb,
    magma_int_t batchCount );

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zlacpy
*/
#define PRECISION_z

int main( int argc, char** argv) 
{
    TESTING_CUDA_INIT();

    //cuDoubleComplex c_zero = MAGMA_Z_ZERO;
    //cuDoubleComplex c_one  = MAGMA_Z_ONE;
    
    cuDoubleComplex *hA, *hB, *hR, *dA, *dB;
    cuDoubleComplex **hAarray, **hBarray, **dAarray, **dBarray;

    //magma_int_t ione     = 1;
    //magma_int_t ISEED[4] = {0, 0, 0, 1};
    
    magma_int_t offsets[] = { 64, 256 };
    magma_int_t size[] = { 0, 1, 32, 63, 64, 65, 255, 256 };
    int ntest   = sizeof(size)    / sizeof(magma_int_t);
    int noffset = sizeof(offsets) / sizeof(magma_int_t);
    
    magma_int_t count = 3;
    magma_int_t n   = 512*3;  // max offset * count
    magma_int_t lda = n;
    
    TESTING_MALLOC   ( hA, cuDoubleComplex, lda*n );
    TESTING_MALLOC   ( hB, cuDoubleComplex, lda*n );
    TESTING_MALLOC   ( hR, cuDoubleComplex, lda*n );
    TESTING_DEVALLOC ( dA, cuDoubleComplex, lda*n );
    TESTING_DEVALLOC ( dB, cuDoubleComplex, lda*n );
    
    TESTING_MALLOC   ( hAarray, cuDoubleComplex*, count );
    TESTING_MALLOC   ( hBarray, cuDoubleComplex*, count );
    TESTING_DEVALLOC ( dAarray, cuDoubleComplex*, count );
    TESTING_DEVALLOC ( dBarray, cuDoubleComplex*, count );
    
    // initialize matrices; entries are (i.j) for A and (800 + i.j) for B.
    double nf = n;
    for( int i = 0; i < n; ++i ) {
        for( int j = 0; j < n; ++j ) {
            hA[i + j*lda] = MAGMA_Z_MAKE( i + j/nf,       0. );
            hB[i + j*lda] = MAGMA_Z_MAKE( i + j/nf + 800, 0. );
        }
    }
    
    printf( "\nNote: ranges use Python notation,\n"
            "i.e., A[i:j] is A[ i, i+1, ..., j-1 ], excluding A[j].\n\n" );
    for( int ii = 0; ii < ntest; ++ii ) {
    for( int jj = 0; jj < ntest; ++jj ) {
    for( int kk = 0; kk < noffset; ++kk ) {
        int mb = size[ii];
        int nb = size[jj];
        int offset = offsets[kk];
        if ( mb > offset || nb > offset ) {
            //printf( "skipping mb=%d, nb=%d, offset=%d because mb > offset or nb > offset\n", mb, nb, offset );
            continue;
        }
        
        magma_zsetmatrix( n, n, hA, lda, dA, lda );
        magma_zsetmatrix( n, n, hB, lda, dB, lda );
        
        // setup pointers
        for( int i = 0; i < count; ++i ) {
            hAarray[i] = &dA[i*offset + i*offset*lda];
            hBarray[i] = &dB[i*offset + i*offset*lda];
        }
        magma_setvector( count, sizeof(cuDoubleComplex*), hAarray, 1, dAarray, 1 );
        magma_setvector( count, sizeof(cuDoubleComplex*), hBarray, 1, dBarray, 1 );
        
        // run kernel
        magmablas_zlacpy_batched( 'F', mb, nb, dAarray, lda, dBarray, lda, count );
        
        // verify result
        int bad_copies = 0;
        int overwrites = 0;
        magma_zgetmatrix( n, n, dB, lda, hR, lda );
                
        for( int j = 0; j < n; ++j ) {
            for( int i = 0; i < n; ++i ) {
                if (    (i / offset) < count
                     && (i / offset) == (j / offset)
                     && (i % offset) < mb
                     && (j % offset) < nb )
                {
                    if ( ! MAGMA_Z_EQUAL( hR[i + j*lda], hA[i + j*lda] )) {
                        bad_copies += 1;
                        printf( "Copy failed at B[%2d,%2d], expected %9.4f, got %9.4f\n",
                                i, j, MAGMA_Z_REAL( hA[i + j*lda] ),
                                      MAGMA_Z_REAL( hR[i + j*lda] ));
                    }
                }
                else {
                    if ( ! MAGMA_Z_EQUAL( hR[i + j*lda], hB[i + j*lda] )) {
                        overwrites += 1;
                        printf( "Overwrote at B[%2d,%2d], expected %9.4f, got %9.4f\n",
                                i, j, MAGMA_Z_REAL( hB[i + j*lda] ),
                                      MAGMA_Z_REAL( hR[i + j*lda] ));
                    }
                }
            }
        }
        printf( "B(i*%-3d:i*%-3d + %3d, i*%-3d:i*%-3d + %3d) = A(...) for i=0,...,%d ",
                offset, offset, mb,
                offset, offset, nb, count-1 );
        if ( bad_copies > 0 || overwrites > 0 ) {
            printf( "failed, %d bad copies, %d overwrites\n", bad_copies, overwrites );
        }
        else {
            printf( "passed\n" );
        }
    }}}
    
    TESTING_FREE( hA );
    TESTING_FREE( hB );
    TESTING_FREE( hR );
    TESTING_DEVFREE( dA );
    TESTING_DEVFREE( dB );
    
    TESTING_FREE( hAarray );
    TESTING_FREE( hBarray );
    TESTING_DEVFREE( dAarray );
    TESTING_DEVFREE( dBarray );
    
    /* Shutdown */
    TESTING_CUDA_FINALIZE();
    return EXIT_SUCCESS;
}
