/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "trace.h"
#define PRECISION_z

/* ================================================================================ */
#ifdef __cplusplus
extern "C" {
#endif
magma_int_t 
magma_get_zhetrf_nb( magma_int_t m );

#define lapackf77_zhetf2   FORTRAN_NAME( zhetf2, ZHETF2 )
void
lapackf77_zhetf2( char*, int*, magmaDoubleComplex*, int*, int*, int* );
#ifdef __cplusplus
}
#endif


extern "C" magma_int_t
magma_zlahef_gpu(magma_uplo_t uplo, magma_int_t n, magma_int_t nb, magma_int_t *kb,
                 magmaDoubleComplex *hA, magma_int_t lda, 
                 magmaDoubleComplex *dA, magma_int_t ldda, magma_int_t *ipiv,
                 magmaDoubleComplex *dW, magma_int_t lddw, 
                 magma_queue_t stream[], magma_event_t event[], magma_int_t *info);
/* ================================================================================ */

/**
    Purpose
    =======
 
    ZHETRF computes the factorization of a complex Hermitian matrix A
    using the Bunch-Kaufman diagonal pivoting method.  The form of the
    factorization is
 
     A = U*D*U**H  or  A = L*D*L**H
 
    where U (or L) is a product of permutation and unit upper (lower)
    triangular matrices, and D is Hermitian and block diagonal with
    1-by-1 and 2-by-2 diagonal blocks.

    This is the blocked version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    UPLO    CHARACTER*1
      -     = 'U':  Upper triangle of A is stored;
      -     = 'L':  Lower triangle of A is stored.
 
    @param[in] 
    N       INTEGER
            The order of the matrix A.  N >= 0.
  
    @param[in,out]
    A       COMPLEX*16 array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, the block diagonal matrix D and the multipliers used
            to obtain the factor U or L (see below for further details).
 
    @param[in]
    LDA     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).
 
    @param[out]
    IPIV    INTEGER array, dimension (N)
            Details of the interchanges and the block structure of D.
            If IPIV(k) > 0, then rows and columns k and IPIV(k) were
            interchanged and D(k,k) is a 1-by-1 diagonal block.
            If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0, then rows and
            columns k-1 and -IPIV(k) were interchanged and D(k-1:k,k-1:k)
            is a 2-by-2 diagonal block.  If UPLO = 'L' and IPIV(k) =
            IPIV(k+1) < 0, then rows and columns k+1 and -IPIV(k) were
            interchanged and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.

    @param[out]
    INFO    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, D(i,i) is exactly zero.  The factorization
                  has been completed, but the block diagonal matrix D is
                  exactly singular, and division by zero will occur if it
                  is used to solve a system of equations.

    Further Details
    ===============
    If UPLO = 'U', then A = U*D*U', where
    U = P(n)*U(n)* ... *P(k)U(k)* ...,
    i.e., U is a product of terms P(k)*U(k), where k decreases from n to
    1 in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
    and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
    defined by IPIV(k), and U(k) is a unit upper triangular matrix, such
    that if the diagonal block D(k) is of order s (s = 1 or 2), then
 
               (   I    v    0   )   k-s
       U(k) =  (   0    I    0   )   s
               (   0    0    I   )   n-k
                  k-s   s   n-k
 
    If s = 1, D(k) overwrites A(k,k), and v overwrites A(1:k-1,k).
    If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k),
    and A(k,k), and v overwrites A(1:k-2,k-1:k).
  
    If UPLO = 'L', then A = L*D*L', where
       L = P(1)*L(1)* ... *P(k)*L(k)* ...,
    i.e., L is a product of terms P(k)*L(k), where k increases from 1 to
    n in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1
    and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as
    defined by IPIV(k), and L(k) is a unit lower triangular matrix, such
    that if the diagonal block D(k) is of order s (s = 1 or 2), then
  
               (   I    0     0   )  k-1
       L(k) =  (   0    I     0   )  s
               (   0    v     I   )  n-k-s+1
                  k-1   s  n-k-s+1
  
    If s = 1, D(k) overwrites A(k,k), and v overwrites A(k+1:n,k).
    If s = 2, the lower triangle of D(k) overwrites A(k,k), A(k+1,k),
    and A(k+1,k+1), and v overwrites A(k+2:n,k:k+1).
 
    @ingroup magma_zhetrf_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zhetrf_gpu(
    magma_uplo_t uplo, magma_int_t n, 
    magmaDoubleComplex *A, magma_int_t lda, 
    magma_int_t *ipiv, magma_int_t *info)
{
    #define  A(i, j) ( A + (j)*lda  + (i))
    #define dA(i, j) (dA + (j)*ldda + (i))

    /* .. Local Scalars .. */
    magma_int_t            upper;
    magma_int_t            nb = magma_get_zhetrf_nb(n);
    magma_int_t            iinfo = 0, nk, kb, lwkopt;

    /* .. Executable Statements .. */
    /* Test the input parameters. */
    *info = 0;
    upper = (uplo == MagmaUpper);
    if( !upper && uplo != MagmaLower ) {
         *info = -1;
    } else if ( n < 0 ) {
         *info = -2;
    } else if ( lda < max( 1, n ) ) {
         *info = -4;
    }
    if( *info != 0 ) {
         magma_xerbla( __func__, -(*info) );
         return *info;
    }

    magma_int_t ldda = 32*((n+31)/32);
    magmaDoubleComplex *dA, *dW;
    if ((MAGMA_SUCCESS != magma_zmalloc( &dA, n*ldda  )) ||
          (MAGMA_SUCCESS != magma_zmalloc( &dW, (1+nb)*ldda ))) {
          *info = MAGMA_ERR_DEVICE_ALLOC;
          return *info;
    }
    magma_queue_t stream[2];
    magma_event_t event[2];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_event_create( &event[0] );
    magma_event_create( &event[1] );
    trace_init( 1, 1, 2, (CUstream_st**)stream );

    trace_gpu_start( 0, 0, "set", "setA" );
    magma_zsetmatrix_async( n, n, A(0,0), lda, dA(0,0), ldda, stream[0] );
    trace_gpu_end( 0, 0 );

    if( upper ) {

          /* Factorize A as U*D*U' using the upper triangle of A

            K is the main loop index, decreasing from N to 1 in steps of
            KB, where KB is the number of columns factorized by ZLAHEF;
             KB is either NB or NB-1, or K for the last block */

/*
           K = N;
   10    CONTINUE
*
*        If K < 1, exit from loop
*
         IF( K.LT.1 )
     $      GO TO 40
*
         IF( K.GT.NB ) THEN
*
*           Factorize columns k-kb+1:k of A and use blocked code to
*           update columns 1:k-kb
*
            CALL ZLAHEF( UPLO, K, NB, KB, A, LDA, IPIV, WORK, N, IINFO )
         ELSE
*
*           Use unblocked code to factorize columns 1:k of A
*
            CALL ZHETF2( UPLO, K, A, LDA, IPIV, IINFO )
            KB = K
         END IF
*
*        Set INFO on the first occurrence of a zero pivot
*
         IF( INFO.EQ.0 .AND. IINFO.GT.0 )
     $      INFO = IINFO
*
*        Decrease K and return to the start of the main loop
*
         K = K - KB
         GO TO 10
*
*/
      } else {

        /* Factorize A as L*D*L' using the lower triangle of A

           K is the main loop index, increasing from 1 to N in steps of
           KB, where KB is the number of columns factorized by ZLAHEF;
           KB is either NB or NB-1, or N-K+1 for the last block */

         for (int k = 0; k < n; k+=kb ) {
             nk = n-k;
             kb = min(nb, n - k);
             if ( k < n-nb ) {
                 /* Factorize columns k:k+kb-1 of A and use blocked code to
                    update columns k+kb:n */
                 magma_zlahef_gpu( MagmaLower, nk, nb, &kb, A( k, k ), lda, dA( k, k ), ldda, 
                                   &ipiv[k], dW, ldda, stream, event, &iinfo );

             } else {
                 /* Use unblocked code to factorize columns k:n of A */
                 magma_queue_sync( stream[0] );
                 magma_zgetmatrix( nk,nk, dA(k,k),ldda, A(k,k),lda );
                 lapackf77_zhetf2( "L", &nk, A( k, k ), &lda, &ipiv[k], &iinfo );
             }
             /* Set INFO on the first occurrence of a zero pivot */
             if ( *info == 0 && iinfo > 0 ) *info = iinfo + k;
             /* Adjust IPIV */
             for (int j = k; j < k + kb; j ++) {
                 if( ipiv[j] > 0 ) {
                     ipiv[j] = ipiv[j] + k;
                 } else {
                     ipiv[j] = ipiv[j] - k;
                 }
             }
          }
      }

      trace_finalize( "zhetrf.svg","trace.css" );
      magma_queue_sync( stream[0] );
      magma_queue_sync( stream[1] );
      magmablasSetKernelStream( NULL );
      magma_event_destroy( event[0] );
      magma_event_destroy( event[1] );
      magma_queue_destroy( stream[0] );
      magma_queue_destroy( stream[1] );
      magma_free( dA );
      magma_free( dW );
      return *info;
      /* End of ZHETRF */
}
