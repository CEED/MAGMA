// ------------------------------------------------------------
// Solve A * X = B, where A and X are stored in CPU host memory.
// This uses LAPACK, doing the entire computation on the CPU.
void lapack_interface( int n, int nrhs )
{
    magmaDoubleComplex *A, *X;
    magma_int_t *ipiv;
    int lda  = n;
    int ldx  = lda;
    int info = 0;
    int ione = 1;
    
    magma_zmalloc_cpu( &A, lda*n );
    magma_zmalloc_cpu( &X, ldx*nrhs );
    magma_imalloc_cpu( &ipiv, n );
    if ( A == NULL || X == NULL || ipiv == NULL ) {
        fprintf( stderr, "malloc failed\n" );
        goto cleanup;
    }
    
    zfill_matrix( n, n, A, lda );
    zfill_rhs( n, nrhs, X, ldx );
    
    lapackf77_zgesv( &n, &ione, A, &lda, ipiv, X, &lda, &info );
    if ( info != 0 ) {
        fprintf( stderr, "lapackf77_zgesv failed with info=%d\n", info );
    }
    
    // use result in X
    
cleanup:
    magma_free_cpu( A );
    magma_free_cpu( X );
    magma_free_cpu( ipiv );
}


    printf( "using LAPACK\n" );
    lapack_interface( n, nrhs );

