/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
       @author Mark Gates
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


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgeadd
*/
#define PRECISION_z

int main( int argc, char** argv) 
{
    TESTING_CUDA_INIT();

    //cuDoubleComplex c_zero = MAGMA_Z_ZERO;
    //cuDoubleComplex c_one  = MAGMA_Z_ONE;
    
    cuDoubleComplex *hA, *hB, *hR, *dA, *dB;
    cuDoubleComplex **hAarray, **hBarray, **dAarray, **dBarray;
    cuDoubleComplex alpha = MAGMA_Z_MAKE( 3.1415, 2.718 );
    cuDoubleComplex Rij;
    
    //magma_int_t ione     = 1;
    //magma_int_t ISEED[4] = {0, 0, 0, 1};
    
    magma_int_t offsets[] = { 64, 256 };
    magma_int_t size[]    = { 0, 1, 32, 63, 64, 65, 255, 256 };
    int ntest   = sizeof(size)    / sizeof(magma_int_t);
    int noffset = sizeof(offsets) / sizeof(magma_int_t);
    
    magma_int_t count = 3;
    magma_int_t n   = 256*3;  // max offset * count
    magma_int_t lda = n;
    
    double tol = 2*lapackf77_slamch( "E" );
    double diff;
    
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
    
    /* Check parameters. magma_xerbla calls lapack's xerbla to print out error. */
    //magmablas_zgeadd_batched( -1,  n, alpha, dAarray, lda, dBarray, lda, count );
    //magmablas_zgeadd_batched(  n, -1, alpha, dAarray, lda, dBarray, lda, count );
    //magmablas_zgeadd_batched(  n,  n, alpha, dAarray, n-1, dBarray, lda, count );
    //magmablas_zgeadd_batched(  n,  n, alpha, dAarray, lda, dBarray, n-1, count );
    //magmablas_zgeadd_batched(  n,  n, alpha, dAarray, lda, dBarray, lda, -1    );
    
    printf( "\nNote: ranges use Python notation, i.e., A[i:j] is A[ i, i+1, ..., j-1 ], excluding A[j].\n\n" );
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
        
        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        magma_zsetmatrix( n, n, hA, lda, dA, lda );
        magma_zsetmatrix( n, n, hB, lda, dB, lda );
        
        // setup pointers
        for( int i = 0; i < count; ++i ) {
            hAarray[i] = &dA[i*offset + i*offset*lda];
            hBarray[i] = &dB[i*offset + i*offset*lda];
        }
        magma_setvector( count, sizeof(cuDoubleComplex*), hAarray, 1, dAarray, 1 );
        magma_setvector( count, sizeof(cuDoubleComplex*), hBarray, 1, dBarray, 1 );
        
        magmablas_zgeadd_batched( mb, nb, alpha, dAarray, lda, dBarray, lda, count );
        
        /* ====================================================================
           Check result
           LAPACK doesn't implement zgeadd, sadly.
           =================================================================== */
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
                    // Rij = alpha*Aij + Bij
                    Rij = MAGMA_Z_ADD( MAGMA_Z_MUL( alpha, hA[i + j*lda] ), hB[i + j*lda] );
                    diff = MAGMA_Z_ABS( MAGMA_Z_SUB( hR[i + j*lda], Rij ));
                    if ( diff > tol*MAGMA_Z_ABS( Rij ) ) {
                        bad_copies += 1;
                        printf( "Add failed at B[%2d,%2d], expected %9.4f, got %9.4f, diff %.2e, tol %.2e\n",
                                i, j, MAGMA_Z_REAL( Rij ),
                                      MAGMA_Z_REAL( hR[i + j*lda] ), diff, tol*MAGMA_Z_ABS( Rij ) );
                    }
                }
                else {
                    // Rij = Bij (no change)
                    if ( ! MAGMA_Z_EQUAL( hR[i + j*lda], hB[i + j*lda] )) {
                        overwrites += 1;
                        printf( "Overwrote at B[%2d,%2d], expected %9.4f, got %9.4f\n",
                                i, j, MAGMA_Z_REAL( hB[i + j*lda] ),
                                      MAGMA_Z_REAL( hR[i + j*lda] ));
                    }
                }
            }
        }
        printf( "B(i*%-3d:i*%-3d + %3d, i*%-3d:i*%-3d + %3d) = alpha*A(...) + B(...), for i=0,...,%d ",
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
