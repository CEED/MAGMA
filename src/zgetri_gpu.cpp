/*
    -- MAGMA (version 1.0) --
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
  #define cublasZgemm magmablas_zgemm
  //#define cublasZtrsm magmablas_ztrsm  // doesn't work? Nov 2011
#endif

#if (GPUSHMEM >= 200)
#if (defined(PRECISION_s))
     #undef  cublasSgemm
     #define cublasSgemm magmablas_sgemm_fermi80
  #endif
#endif
// === End defining what BLAS to use ======================================

extern "C" magma_int_t
magma_zgetri_gpu( magma_int_t n, cuDoubleComplex *dA, magma_int_t lda,
                  magma_int_t* ipiv, magma_int_t *info )
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

        ZGETRI computes the inverse of a matrix using the LU factorization
        computed by ZGETRF. This method inverts U and then computes inv(A) by
        solving the system inv(A)*L = inv(U) for inv(A).

    Arguments
    =========

        N       (input) INTEGER
                The order of the matrix A.  N >= 0.

        dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDA,N)
                On entry, the factors L and U from the factorization
                A = P*L*U as computed by ZGETRF_GPU.
                On exit, if INFO = 0, the inverse of the original matrix A.

        LDA     (input) INTEGER
                The leading dimension of the array A.  LDA >= max(1,N).

        IPIV    (input) INTEGER array, dimension (N)
                The pivot indices from ZGETRF; for 1<=i<=N, row i of the
                matrix was interchanged with row IPIV(i).

        INFO    (output) INTEGER
                = 0:  successful exit
                < 0:  if INFO = -i, the i-th argument had an illegal value
                > 0:  if INFO = i, U(i,i) is exactly zero; the matrix is
                      singular and its cannot be computed.

  ===================================================================== */

    /* Local variables */
    magma_int_t ret;
    magma_int_t nb;
    cuDoubleComplex c_one = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    
    *info = 0;
    if (n < 0)
        *info = -1;
    else if (lda < max(1,n))
        *info = -3;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }

    /* Quick return if possible */
    if ( n == 0 )
        return MAGMA_SUCCESS;
    
    nb = 32;
    cuDoubleComplex* dL;
    magma_int_t ldl = lda;
    if ( cudaSuccess != cudaMalloc( &dL, ldl*nb*sizeof(cuDoubleComplex))) {
        fprintf( stderr, "device memory allocation error in %s\n", __func__ );
        return MAGMA_ERR_CUBLASALLOC;
    }
    
    //printf( "before trtri, A=" );
    //zprint_gpu( n, n, dA, lda );

    /* Invert the triangular factor U */
    ret = magma_ztrtri_gpu( MagmaUpper, MagmaNonUnit, n, dA, lda, info );
    if ( *info != 0 )
        return ret;
    
    //printf( "after trtri, A=" );
    //zprint_gpu( n, n, dA, lda );

    magma_int_t jmax = ((n-1) / nb)*nb;
    for( int j = jmax; j >= 0; j -= nb ) {
        int jb = min( nb, n-j );
        //printf( "j %d, jb %d, nb %d\n", j, jb, nb );
        
        // copy current block column of L to work space, then replace with zeros in A.
        magmablas_zlacpy( MagmaUpperLower, n-j, jb,
                          &dA[j + j*lda], lda,
                          &dL[j        ], ldl );
        magmablas_zlaset( MagmaLower, n-j, jb, &dA[j + j*lda], lda );
        
        //printf( "after copy & zero, A=" );
        //zprint_gpu( n, n, dA, lda );
        //
        //printf( "and L=" );
        //zprint_gpu( n, jb, dL, ldl );
        
        // compute current block column of Ainv
        // Ainv(:, j:j+jb-1)
        //   = ( U(:, j:j+jb-1) - Ainv(:, j+jb:n) L(j+jb:n, j:j+jb-1) )
        //   * L(j:j+jb-1, j:j+jb-1)^{-1}
        if ( j+jb < n ) {
            cublasZgemm( MagmaNoTrans, MagmaNoTrans, n, jb, n-j-jb,
                         c_neg_one, &dA[(j+jb)*lda], lda,
                                    &dL[ j+jb     ], ldl,
                         c_one,     &dA[     j*lda], lda );
        }
        cublasZtrsm( MagmaRight, MagmaLower, MagmaNoTrans, MagmaUnit,
                     n, jb, c_one,
                     &dL[j    ], ldl,
                     &dA[j*lda], lda );
    }
    
    //printf( "before pivoting, A=" );
    //zprint_gpu( n, n, dA, lda );

    // Apply column interchanges
    // TODO replace with magmablas_zswapblk?
    for( int j = n-2; j >= 0; --j ) {
        int jp = ipiv[j] - 1;
        if ( jp != j ) {
            magmablas_zswap( n, &dA[ j*lda ], 1, &dA[ jp*lda ], 1 );
        }
    }
    
    //printf( "after pivoting, A=" );
    //zprint_gpu( n, n, dA, lda );

    cublasFree( dL );

    return MAGMA_SUCCESS;
}
