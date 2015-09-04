/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/

#include "common_magma.h"
#include "commonblas_z.h"

#define PRECISION_z

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512


/* --------------------------------------------------------------------------- */
/**
    Purpose
    -------
    ZLAQPS computes a step of QR factorization with column pivoting
    of a complex M-by-N matrix A by using Blas-3.  It tries to factorize
    NB columns from A starting from the row OFFSET+1, and updates all
    of the matrix with Blas-3 xGEMM.

    In some cases, due to catastrophic cancellations, it cannot
    factorize NB columns.  Hence, the actual number of factorized
    columns is returned in KB.

    Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. N >= 0

    @param[in]
    offset  INTEGER
            The number of rows of A that have been factorized in
            previous steps.

    @param[in]
    nb      INTEGER
            The number of columns to factorize.

    @param[out]
    kb      INTEGER
            The number of columns actually factorized.

    @param[in,out]
    dA      COMPLEX*16 array, dimension (LDDA,N)
            On entry, the M-by-N matrix A.
            On exit, block A(OFFSET+1:M,1:KB) is the triangular
            factor obtained and block A(1:OFFSET,1:N) has been
            accordingly pivoted, but no factorized.
            The rest of the matrix, block A(OFFSET+1:M,KB+1:N) has
            been updated.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. LDDA >= max(1,M).

    @param[in,out]
    jpvt    INTEGER array, dimension (N)
            JPVT(I) = K <==> Column K of the full matrix A has been
            permuted into position I in AP.

    @param[out]
    dtau    COMPLEX*16 array, dimension (KB)
            The scalar factors of the elementary reflectors.

    @param[in,out]
    dvn1    DOUBLE PRECISION array, dimension (N)
            The vector with the partial column norms.

    @param[in,out]
    dvn2    DOUBLE PRECISION array, dimension (N)
            The vector with the exact column norms.

    @param[in,out]
    dauxv   COMPLEX*16 array, dimension (NB)
            Auxiliar vector.

    @param[in,out]
    dF      COMPLEX*16 array, dimension (LDDF,NB)
            Matrix F**H = L * Y**H * A.

    @param[in]
    lddf    INTEGER
            The leading dimension of the array F. LDDF >= max(1,N).

    @ingroup magma_zgeqp3_aux
    ********************************************************************/
extern "C" magma_int_t
magma_zlaqps2_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaDoubleComplex_ptr dtau, 
    magmaDouble_ptr dvn1, magmaDouble_ptr dvn2,
    magmaDoubleComplex_ptr dauxv,
    magmaDoubleComplex_ptr dF,  magma_int_t lddf)
{
#define dA(i_, j_) (dA + (i_) + (j_)*(ldda))
#define dF(i_, j_) (dF + (i_) + (j_)*(lddf))

    magmaDoubleComplex c_zero    = MAGMA_Z_MAKE( 0.,0.);
    magmaDoubleComplex c_one     = MAGMA_Z_MAKE( 1.,0.);
    magmaDoubleComplex c_neg_one = MAGMA_Z_MAKE(-1.,0.);
    magma_int_t ione = 1;
    
    magma_int_t i__1, i__2;
    
    magma_int_t k, rk;
    magmaDoubleComplex tauk;
    magma_int_t pvt, itemp;
    double tol3z;

    magmaDoubleComplex_ptr dAkk = dauxv;
    dauxv += nb;

    double lsticc, *lsticcs;
    magma_dmalloc( &lsticcs, 1+256*(n+255)/256 );

    tol3z = magma_dsqrt( lapackf77_dlamch("Epsilon"));

    lsticc = 0;
    k = 0;
    while( k < nb && lsticc == 0 ) {
        rk = offset + k;

        /* Determine ith pivot column and swap if necessary */
        pvt = k - 1 + magma_idamax( n-k, &dvn1[k], ione );

        if (pvt != k) {
            magmablas_zswap( k+1, dF(pvt,0), lddf, dF(k,0), lddf);

            itemp     = jpvt[pvt];
            jpvt[pvt] = jpvt[k];
            jpvt[k]   = itemp;
            magma_dswap( 2, &dvn1[pvt], n+offset, &dvn1[k], n+offset );

            magmablas_zswap( m, dA(0,pvt), ione, dA(0, k), ione );
        }

        /* Apply previous Householder reflectors to column K:
           A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)'.
           Optimization: multiply with beta=0; wait for vector and subtract */
        if (k > 0) {
            magmablas_zgemv_conjv( m-rk, k,
                                   c_neg_one, dA(rk, 0), ldda,
                                              dF(k,  0), lddf,
                                   c_one,     dA(rk, k), ione );
        }

        /*  Generate elementary reflector H(k). */
        magma_zlarfg_gpu(m-rk, dA(rk, k), dA(rk + 1, k), &dtau[k], &dvn1[k], &dAkk[k]);
        magma_zsetvector( 1, &c_one,   1, dA(rk, k), 1 );

        /* Compute Kth column of F:
           Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) on the GPU */
        if (k < n-1 || k > 0 ) magma_zgetvector( 1, &dtau[k], 1, &tauk, 1 );
        if (k < n-1) {
            magma_zgemv( MagmaConjTrans, m-rk, n-k-1,
                     tauk,   dA( rk,  k+1 ), ldda,
                             dA( rk,  k   ), 1,
                     c_zero, dF( k+1, k   ), 1 );
        }

        /* Incremental updating of F:
           F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K). 
           F(1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K)
                    := tau(K)(A(RK:M,K+1:N)' - F(1:N,1:K-1)*A(RK:M,1:K-1)') A(RK:M,K)  
           so, F is (updated A)*V */
        if (k > 0) {
            /*z__1 = MAGMA_Z_NEGATE( tauk );
            magma_zgemv( MagmaConjTrans, m-rk, k,
                         z__1,   dA(rk, 0), ldda,
                                 dA(rk, k), ione,
                         c_zero, dauxv, ione ); */

            magma_zgemv_kernel3<<< k, BLOCK_SIZE, 0, magma_stream >>>(m-rk, dA(rk, 0), ldda,
                                                                      dA(rk, k), dauxv, dtau+k);

            /* I think we only need stricly lower-triangular part */
            magma_zgemv( MagmaNoTrans, n-k-1, k,
                         c_one, dF(k+1,0), lddf,
                                dauxv,     ione,
                         c_one, dF(k+1,k), ione );
        }

       /* Update the current row of A:
           A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)'.               */
        if (k < n-1) {
            i__1 = n - k - 1;
            i__2 = k + 1;
            /* left-looking update of rows,                     *
             * since F=A**H v with original A, so no right-looking */
            magma_zgemm( MagmaNoTrans, MagmaConjTrans, ione, i__1, i__2,
                         c_neg_one, dA(rk, 0  ), ldda,
                                    dF(k+1,0  ), lddf,
                         c_one,     dA(rk, k+1), ldda ); 
        }

        /* Update partial column norms. */
        if (rk < min(m, n+offset)-1) {
            magmablas_dznrm2_row_check_adjust(n-k-1, tol3z, &dvn1[k+1], 
                                              &dvn2[k+1], dA(rk,k+1), ldda, lsticcs); 
            
            magma_dgetvector( 1, &lsticcs[0], 1, &lsticc, 1 );
        }

        //*dA(rk, k) = Akk;
        //magma_zsetvector( 1, &Akk, 1, dA(rk, k), 1 );
        //magmablas_zlacpy(MagmaUpperLower, 1, 1, dAkk, 1, dA(rk, k), 1);

        ++k;
    }
    // restore the diagonals
    magma_zcopymatrix( 1, k, dAkk, 1, dA(offset, 0), ldda+1 );

    // leave k as the last column done
    --k;
    *kb = k + 1;
    rk = offset + *kb - 1;

    /* Apply the block reflector to the rest of the matrix:
       A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) - 
                                  A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)'  */
    if (*kb < min(n, m - offset)) {
        i__1 = m - rk - 1;
        i__2 = n - *kb;

        magma_zgemm( MagmaNoTrans, MagmaConjTrans, i__1, i__2, *kb,
                     c_neg_one, dA(rk+1, 0  ), ldda,
                                dF(*kb,  0  ), lddf,
                     c_one,     dA(rk+1, *kb), ldda );
    }

    /* Recomputation of difficult columns. */
    if ( lsticc > 0 ) {
        // printf( " -- recompute dnorms --\n" );
        magmablas_dznrm2_check(m-rk-1, n-*kb, dA(rk+1,*kb), ldda,
                               &dvn1[*kb], lsticcs);
        magma_dcopymatrix( n-*kb, 1, &dvn1[*kb], n, &dvn2[*kb], n );
    }
    magma_free(lsticcs);
    
    return MAGMA_SUCCESS;
} /* magma_zlaqps */
