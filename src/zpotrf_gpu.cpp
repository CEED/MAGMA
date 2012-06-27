/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"

// === Define what BLAS to use ============================================
#define PRECISION_z
#if (defined(PRECISION_s) || defined(PRECISION_d)) 
  #define magma_zgemm magmablas_zgemm
  #define magma_ztrsm magmablas_ztrsm
#endif

#if (GPUSHMEM >= 200)
  #if (defined(PRECISION_s))
     #undef  magma_sgemm
     #define magma_sgemm magmablas_sgemm_fermi80
  #endif
#endif
// === End defining what BLAS to use =======================================

#define dA(i, j)  (dA + (j)*ldda + (i))

extern "C" magma_int_t
magma_zpotrf_gpu(char uplo, magma_int_t n, 
                 cuDoubleComplex *dA, magma_int_t ldda, magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose   
    =======   
    ZPOTRF computes the Cholesky factorization of a complex Hermitian   
    positive definite matrix dA.   

    The factorization has the form   
       dA = U**H * U,  if UPLO = 'U', or   
       dA = L  * L**H,  if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   
    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of dA is stored;   
            = 'L':  Lower triangle of dA is stored.   

    N       (input) INTEGER   
            The order of the matrix dA.  N >= 0.   

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)   
            On entry, the Hermitian matrix dA.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of dA contains the upper   
            triangular part of the matrix dA, and the strictly lower   
            triangular part of dA is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of dA contains the lower   
            triangular part of the matrix dA, and the strictly upper   
            triangular part of dA is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization dA = U**H * U or dA = L * L**H.   

    LDDA     (input) INTEGER   
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            dividable by 16.

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   
    =====================================================================   */


    magma_int_t     j, jb, nb;
    char            uplo_[2] = {uplo, 0};
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex *work;
    double          d_one     =  1.0;
    double          d_neg_one = -1.0;
    int upper = lapackf77_lsame(uplo_, "U");

    *info = 0;
    if ( (! upper) && (! lapackf77_lsame(uplo_, "L")) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    nb = magma_get_zpotrf_nb(n);

    if (MAGMA_SUCCESS != magma_zmalloc_pinned( &work, nb*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    cudaStream_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );

    if ((nb <= 1) || (nb >= n)) {
        /*  Use unblocked code. */
        magma_zgetmatrix( n, n, dA, ldda, work, n );
        lapackf77_zpotrf(uplo_, &n, work, &n, info);
        magma_zsetmatrix( n, n, work, n, dA, ldda );
    } else {

        /* Use blocked code. */
        if (upper) {
            
            /* Compute the Cholesky factorization A = U'*U. */
            for (j=0; j<n; j+=nb) {
                
                /* Update and factorize the current diagonal block and test   
                   for non-positive-definiteness. Computing MIN */
                jb = min(nb, (n-j));
                
                magma_zherk(MagmaUpper, MagmaConjTrans, jb, j, 
                            d_neg_one, dA(0, j), ldda, 
                            d_one,     dA(j, j), ldda);

                magma_zgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, stream[1] );
                
                if ( (j+jb) < n) {
                    /* Compute the current block row. */
                    magma_zgemm(MagmaConjTrans, MagmaNoTrans, 
                                jb, (n-j-jb), j,
                                c_neg_one, dA(0, j   ), ldda, 
                                           dA(0, j+jb), ldda,
                                c_one,     dA(j, j+jb), ldda);
                }
                
                magma_queue_sync( stream[1] );

                lapackf77_zpotrf(MagmaUpperStr, &jb, work, &jb, info);
                magma_zsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, stream[0] );
                if (*info != 0) {
                  *info = *info + j;
                  break;
                }

                if ( (j+jb) < n)
                    magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, 
                                 jb, (n-j-jb),
                                 c_one, dA(j, j   ), ldda, 
                                        dA(j, j+jb), ldda);
            }
        } else {
            //=========================================================
            // Compute the Cholesky factorization A = L*L'.
            for (j=0; j<n; j+=nb) {

                //  Update and factorize the current diagonal block and test   
                //  for non-positive-definiteness. Computing MIN 
                jb = min(nb, (n-j));

                magma_zherk(MagmaLower, MagmaNoTrans, jb, j,
                            d_neg_one, dA(j, 0), ldda, 
                            d_one,     dA(j, j), ldda);
                
                magma_zgetmatrix_async( jb, jb,
                                        dA(j, j), ldda,
                                        work,     jb, stream[1] );
                
                if ( (j+jb) < n) {
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans, 
                                 (n-j-jb), jb, j,
                                 c_neg_one, dA(j+jb, 0), ldda, 
                                            dA(j,    0), ldda,
                                 c_one,     dA(j+jb, j), ldda);
                }

                magma_queue_sync( stream[1] );
                lapackf77_zpotrf(MagmaLowerStr, &jb, work, &jb, info);
                magma_zsetmatrix_async( jb, jb,
                                        work,     jb,
                                        dA(j, j), ldda, stream[0] );
                if (*info != 0) {
                  *info = *info + j;
                  break;
                }
                
                if ( (j+jb) < n)
                    magma_ztrsm(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                                (n-j-jb), jb, 
                                c_one, dA(j,    j), ldda, 
                                       dA(j+jb, j), ldda);
            }

        }
    }

    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
    magma_free_pinned( work );

    return *info;
} /* magma_zpotrf_gpu */
