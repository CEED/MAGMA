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
// === End defining what BLAS to use ======================================

// ========================================================================
// definition of a non-GPU-resident interface to a single GPU
//extern "C" magma_int_t 
//magma_zpotrf_ooc(char uplo, magma_int_t n, 
//                cuDoubleComplex *a, magma_int_t lda, magma_int_t *info);

// definition of a non-GPU-resident interface to multiple GPUs
extern "C" magma_int_t
magma_zpotrf2_ooc(magma_int_t num_gpus, char uplo, magma_int_t n,
                  cuDoubleComplex *a, magma_int_t lda, magma_int_t *info);
// ========================================================================

#define A(i, j)  (a   +(j)*lda  + (i))
#define dA(i, j) (work+(j)*ldda + (i))

extern "C" magma_int_t 
magma_zpotrf(char uplo, magma_int_t n, 
             cuDoubleComplex *a, magma_int_t lda, magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose   
    =======   

    ZPOTRF computes the Cholesky factorization of a complex Hermitian   
    positive definite matrix A. This version does not require work
    space on the GPU passed as input. GPU memory is allocated in the
    routine.

    The factorization has the form   
       A = U**H * U,  if UPLO = 'U', or   
       A = L  * L**H, if UPLO = 'L',   
    where U is an upper triangular matrix and L is lower triangular.   

    This is the block version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, if INFO = 0, the factor U or L from the Cholesky   
            factorization A = U**H * U or A = L * L**H.   

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_host.

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value 
                  or another error occured, such as memory allocation failed.
            > 0:  if INFO = i, the leading minor of order i is not   
                  positive definite, and the factorization could not be   
                  completed.   

    =====================================================================    */


    /* Local variables */
    char uplo_[2] = {uplo, 0};
    magma_int_t        ldda, nb;
    static magma_int_t j, jb;
    cuDoubleComplex    c_one     = MAGMA_Z_ONE;
    cuDoubleComplex    c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex   *work;
    double             d_one     =  1.0;
    double             d_neg_one = -1.0;
    long int           upper = lapackf77_lsame(uplo_, "U");

    *info = 0;
    if ((! upper) && (! lapackf77_lsame(uplo_, "L"))) {
      *info = -1;
    } else if (n < 0) {
      *info = -2;
    } else if (lda < max(1,n)) {
      *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return */
    if ( n == 0 )
      return *info;

    char * num_gpus_char = getenv("MAGMA_NUM_GPUS");
    magma_int_t num_gpus = 1;

    if( num_gpus_char != NULL ) {
      num_gpus = atoi(num_gpus_char);
    }
    if( num_gpus > 1 ) {
      /* call multiple-GPU interface  */
      return magma_zpotrf2_ooc(num_gpus, uplo, n, a, lda, info);
    }

    ldda = ((n+31)/32)*32;
    
    if (MAGMA_SUCCESS != magma_zmalloc( &work, (n)*ldda )) {
        /* alloc failed so call the non-GPU-resident version */
        return magma_zpotrf2_ooc(num_gpus, uplo, n, a, lda, info);
        //return magma_zpotrf_ooc( uplo, n, a, lda, info);
    }

    static cudaStream_t stream[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );

    nb = magma_get_zpotrf_nb(n);

    if (nb <= 1 || nb >= n) {
        lapackf77_zpotrf(uplo_, &n, a, &lda, info);
    } else {


        /* Use hybrid blocked code. */
        if (upper) {
            /* Compute the Cholesky factorization A = U'*U. */
            for (j=0; j<n; j += nb) {
                /* Update and factorize the current diagonal block and test   
                   for non-positive-definiteness. Computing MIN */
                jb = min(nb, (n-j));
                magma_zsetmatrix( jb, (n-j), A(j, j), lda, dA(j, j), ldda );
                
                magma_zherk(MagmaUpper, MagmaConjTrans, jb, j, 
                            d_neg_one, dA(0, j), ldda, 
                            d_one,     dA(j, j), ldda);

                magma_zgetmatrix_async( (j+jb), jb,
                                        dA(0, j), ldda,
                                        A(0, j),  lda, stream[1] );
                
                if ( (j+jb) < n) {
                    magma_zgemm(MagmaConjTrans, MagmaNoTrans, 
                                jb, (n-j-jb), j,
                                c_neg_one, dA(0, j   ), ldda, 
                                           dA(0, j+jb), ldda,
                                c_one,     dA(j, j+jb), ldda);
                }
             
                magma_queue_sync( stream[1] );
                lapackf77_zpotrf(MagmaUpperStr, &jb, A(j, j), &lda, info);
                if (*info != 0) {
                  *info = *info + j;
                  break;
                }
                magma_zsetmatrix_async( jb, jb,
                                        A(j, j),  lda,
                                        dA(j, j), ldda, stream[0] );
                
                if ( (j+jb) < n )
                  magma_ztrsm(MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit, 
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
                magma_zsetmatrix( (n-j), jb, A(j, j), lda, dA(j, j), ldda );

                magma_zherk(MagmaLower, MagmaNoTrans, jb, j,
                            d_neg_one, dA(j, 0), ldda, 
                            d_one,     dA(j, j), ldda);
                /*
                magma_zgetmatrix_async( jb, j+jb,
                                        dA(j,0), ldda,
                                        A(j, 0), lda, stream[1] );
                */
                magma_zgetmatrix_async( jb, jb,
                                        dA(j,j), ldda,
                                        A(j,j),  lda, stream[1] );
                magma_zgetmatrix_async( jb, j,
                                        dA(j, 0), ldda,
                                        A(j, 0),  lda, stream[0] );

                if ( (j+jb) < n) {
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans, 
                                 (n-j-jb), jb, j,
                                 c_neg_one, dA(j+jb, 0), ldda, 
                                            dA(j,    0), ldda,
                                 c_one,     dA(j+jb, j), ldda);
                }
                
                magma_queue_sync( stream[1] );
                lapackf77_zpotrf(MagmaLowerStr, &jb, A(j, j), &lda, info);
                if (*info != 0){
                    *info = *info + j;
                    break;
                }
                magma_zsetmatrix_async( jb, jb,
                                        A(j, j),  lda,
                                        dA(j, j), ldda, stream[0] );
                
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

    magma_free( work );
    
    return *info;
} /* magma_zpotrf */

