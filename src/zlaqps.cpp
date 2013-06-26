/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/

#include "common_magma.h"
#include <cblas.h>

#define PRECISION_z

extern "C" magma_int_t
magma_zlaqps(magma_int_t m, magma_int_t n, magma_int_t offset,
             magma_int_t nb, magma_int_t *kb,
             magmaDoubleComplex *A,  magma_int_t lda,
             magmaDoubleComplex *dA, magma_int_t ldda,
             magma_int_t *jpvt, magmaDoubleComplex *tau, double *vn1, double *vn2,
             magmaDoubleComplex *auxv,
             magmaDoubleComplex *F,  magma_int_t ldf,
             magmaDoubleComplex *dF, magma_int_t lddf)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    ZLAQPS computes a step of QR factorization with column pivoting
    of a complex M-by-N matrix A by using Blas-3.  It tries to factorize
    NB columns from A starting from the row OFFSET+1, and updates all
    of the matrix with Blas-3 xGEMM.

    In some cases, due to catastrophic cancellations, it cannot
    factorize NB columns.  Hence, the actual number of factorized
    columns is returned in KB.

    Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A. N >= 0

    OFFSET  (input) INTEGER
            The number of rows of A that have been factorized in
            previous steps.

    NB      (input) INTEGER
            The number of columns to factorize.

    KB      (output) INTEGER
            The number of columns actually factorized.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, block A(OFFSET+1:M,1:KB) is the triangular
            factor obtained and block A(1:OFFSET,1:N) has been
            accordingly pivoted, but no factorized.
            The rest of the matrix, block A(OFFSET+1:M,KB+1:N) has
            been updated.

    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(1,M).

    JPVT    (input/output) INTEGER array, dimension (N)
            JPVT(I) = K <==> Column K of the full matrix A has been
            permuted into position I in AP.

    TAU     (output) COMPLEX_16 array, dimension (KB)
            The scalar factors of the elementary reflectors.

    VN1     (input/output) DOUBLE PRECISION array, dimension (N)
            The vector with the partial column norms.

    VN2     (input/output) DOUBLE PRECISION array, dimension (N)
            The vector with the exact column norms.

    AUXV    (input/output) COMPLEX_16 array, dimension (NB)
            Auxiliar vector.

    F       (input/output) COMPLEX_16 array, dimension (LDF,NB)
            Matrix F' = L*Y'*A.

    LDF     (input) INTEGER
            The leading dimension of the array F. LDF >= max(1,N).

    =====================================================================    */
    
#define  A(i, j) (A  + (i) + (j)*(lda ))
#define dA(i, j) (dA + (i) + (j)*(ldda))
#define  F(i, j) (F  + (i) + (j)*(ldf ))
#define dF(i, j) (dF + (i) + (j)*(lddf))

    magmaDoubleComplex c_zero    = MAGMA_Z_MAKE( 0.,0.);
    magmaDoubleComplex c_one     = MAGMA_Z_MAKE( 1.,0.);
    magmaDoubleComplex c_neg_one = MAGMA_Z_MAKE(-1.,0.);
    magma_int_t ione = 1;
    
    magma_int_t i__1, i__2;
    double d__1;
    magmaDoubleComplex z__1;
    
    magma_int_t j, k, rk;
    magmaDoubleComplex Akk;
    magma_int_t pvt;
    double temp, temp2, tol3z;
    magma_int_t itemp;

    magma_int_t lsticc;
    magma_int_t lastrk;

    lastrk = min( m, n + offset );
    tol3z = magma_dsqrt( lapackf77_dlamch("Epsilon"));

    magma_queue_t stream;
    magma_queue_create( &stream );

    lsticc = 0;
    k = 0;
    while( k < nb && lsticc == 0 ) {
        rk = offset + k;
        
        /* Determine ith pivot column and swap if necessary */
        // Fortran: pvt, k, idamax are all 1-based; subtract 1 from k.
        // C:       pvt, k, idamax are all 0-based; don't subtract 1.
        pvt = k + cblas_idamax( n-k, &vn1[k], ione );
        
        if (pvt != k) {

            if (pvt >= nb) {
                /* 1. Start copy from GPU                           */
                magma_zgetmatrix_async( m - offset - nb, 1,
                                        dA(offset + nb, pvt), ldda,
                                        A (offset + nb, pvt), lda, stream );
            }

            /* F gets swapped so F must be sent at the end to GPU   */
            i__1 = k;
            blasf77_zswap( &i__1, F(pvt,0), &ldf, F(k,0), &ldf );
            itemp     = jpvt[pvt];
            jpvt[pvt] = jpvt[k];
            jpvt[k]   = itemp;
            vn1[pvt] = vn1[k];
            vn2[pvt] = vn2[k];

            if (pvt < nb){
                /* no need of transfer if pivot is within the panel */
                blasf77_zswap( &m, A(0, pvt), &ione, A(0, k), &ione );
            }
            else {
                /* 1. Finish copy from GPU                          */
                magma_queue_sync( stream );

                /* 2. Swap as usual on CPU                          */
                blasf77_zswap(&m, A(0, pvt), &ione, A(0, k), &ione);

                /* 3. Restore the GPU                               */
                magma_zsetmatrix_async( m - offset - nb, 1,
                                        A (offset + nb, pvt), lda,
                                        dA(offset + nb, pvt), ldda, stream);
            }
        }

        /* Apply previous Householder reflectors to column K:
           A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)'.
           Optimization: multiply with beta=0; wait for vector and subtract */
        if (k > 0) {
            #if defined(PRECISION_c) || defined(PRECISION_z)
            for (j = 0; j < k; ++j){
                *F(k,j) = MAGMA_Z_CNJG( *F(k,j) );
            }
            #endif

            i__1 = m - rk;
            i__2 = k;
            blasf77_zgemv( MagmaNoTransStr, &i__1, &i__2,
                           &c_neg_one, A(rk, 0), &lda,
                                       F(k,  0), &ldf,
                           &c_one,     A(rk, k), &ione );

            #if defined(PRECISION_c) || defined(PRECISION_z)
            for (j = 0; j < k; ++j) {
                *F(k,j) = MAGMA_Z_CNJG( *F(k,j) );
            }
            #endif
        }
        
        /*  Generate elementary reflector H(k). */
        if (rk < m-1) {
            i__1 = m - rk;
            lapackf77_zlarfg( &i__1, A(rk, k), A(rk + 1, k), &ione, &tau[k] );
        } else {
            lapackf77_zlarfg( &ione, A(rk, k), A(rk, k), &ione, &tau[k] );
        }
        
        Akk = *A(rk, k);
        *A(rk, k) = c_one;

        /* Compute Kth column of F:
           Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) on the GPU */
        if (k < n-1) {
            i__1 = m - rk;
            i__2 = n - k - 1;
        
            /* Send the vector to the GPU */
            magma_zsetmatrix( i__1, 1, A(rk, k), lda, dA(rk,k), ldda );
        
            /* Multiply on GPU */
            // was CALL ZGEMV( 'Conjugate transpose', M-RK+1, N-K,
            //                 TAU( K ), A( RK,  K+1 ), LDA,
            //                           A( RK,  K   ), 1,
            //                 CZERO,    F( K+1, K   ), 1 )
            magma_int_t i__3 = nb-k-1;
            magma_int_t i__4 = i__2 - i__3;
            magma_int_t i__5 = nb-k;
            magma_zgemv( MagmaConjTrans, i__1 - i__5, i__2 - i__3,
                         tau[k], dA(rk +i__5, k+1+i__3), ldda,
                                 dA(rk +i__5, k       ), ione,
                         c_zero, dF(k+1+i__3, k       ), ione );
            
            magma_zgetmatrix_async( i__2-i__3, 1,
                                    dF(k + 1 +i__3, k), i__2,
                                    F (k + 1 +i__3, k), i__2, stream );
            
            blasf77_zgemv( MagmaConjTransStr, &i__1, &i__3,
                           &tau[k], A(rk,  k+1), &lda,
                                    A(rk,  k  ), &ione,
                           &c_zero, F(k+1, k  ), &ione );
            
            magma_queue_sync( stream );
            blasf77_zgemv( MagmaConjTransStr, &i__5, &i__4,
                           &tau[k], A(rk, k+1+i__3), &lda,
                                    A(rk, k       ), &ione,
                           &c_one,  F(k+1+i__3, k ), &ione );
        }
        
        /* Padding F(1:K,K) with zeros. */
        for (j = 0; j < k; ++j) {
            *F(j, k) = c_zero;
        }
        
        /* Incremental updating of F:
           F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K). */
        if (k > 0) {
            i__1 = m - rk;
            i__2 = k;
            z__1 = MAGMA_Z_NEGATE( tau[k] );
            blasf77_zgemv( MagmaConjTransStr, &i__1, &i__2,
                           &z__1,   A(rk, 0), &lda,
                                    A(rk, k), &ione,
                           &c_zero, auxv, &ione );
            
            i__1 = k;
            blasf77_zgemv( MagmaNoTransStr, &n, &i__1,
                           &c_one, F(0,0), &ldf,
                                   auxv,   &ione,
                           &c_one, F(0,k), &ione );
        }
        
        /* Optimization: On the last iteration start sending F back to the GPU */
        
        /* Update the current row of A:
           A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)'.               */
        if (k < n-1) {
            i__1 = n - k - 1;
            i__2 = k + 1;
            blasf77_zgemm( MagmaNoTransStr, MagmaConjTransStr, &ione, &i__1, &i__2,
                           &c_neg_one, A(rk, 0  ), &lda,
                                       F(k+1,0  ), &ldf,
                           &c_one,     A(rk, k+1), &lda );
        }
        
        /* Update partial column norms. */
        if (rk < lastrk) {
            for (j = k + 1; j < n; ++j) {
                if (vn1[j] != 0.) {
                    /* NOTE: The following 4 lines follow from the analysis in
                       Lapack Working Note 176. */
                    temp = MAGMA_Z_ABS( *A(rk,j) ) / vn1[j];
                    temp = max( 0., ((1. + temp) * (1. - temp)) );
        
                    d__1 = vn1[j] / vn2[j];
                    temp2 = temp * (d__1 * d__1);
        
                    if (temp2 <= tol3z) {
                        vn2[j] = (double) lsticc;
                        lsticc = j;
                    } else {
                        vn1[j] *= magma_dsqrt(temp);
                    }
                }
            }
        }
        
        *A(rk, k) = Akk;
        
        ++k;
    }
    // leave k as the last column done
    --k;
    *kb = k + 1;
    rk = offset + *kb - 1;

    /* Apply the block reflector to the rest of the matrix:
       A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) - A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)'  */
    if (*kb < min(n, m - offset)) {
        i__1 = m - rk - 1;
        i__2 = n - *kb;
        
        /* Send F to the GPU */
        magma_zsetmatrix( i__2, *kb,
                          F (*kb, 0), ldf,
                          dF(*kb, 0), i__2 );

        magma_zgemm( MagmaNoTrans, MagmaConjTrans, i__1, i__2, *kb,
                     c_neg_one, dA(rk+1, 0  ), ldda,
                                dF(*kb,  0  ), i__2,
                     c_one,     dA(rk+1, *kb), ldda );
    }
    
    /* Recomputation of difficult columns. */
    while( lsticc > 0 ) {
        itemp = (magma_int_t)(vn2[lsticc] >= 0. ? floor(vn2[lsticc] + .5) : -floor(.5 - vn2[lsticc]));
        i__1 = m - rk - 1;
        if (lsticc <= nb)
            vn1[lsticc] = cblas_dznrm2(i__1, A(rk + 1, lsticc), ione);
        else {
            /* Where is the data, CPU or GPU ? */
            double r1, r2;
            
            r1 = cblas_dznrm2(nb-k, A(rk + 1, lsticc), ione);
            r2 = magma_dznrm2(m-offset-nb, dA(offset + nb + 1, lsticc), ione);
            
            //vn1[lsticc] = magma_dznrm2(i__1, dA(rk + 1, lsticc), ione);
            vn1[lsticc] = magma_dsqrt(r1*r1+r2*r2);
        }
        
        /* NOTE: The computation of VN1( LSTICC ) relies on the fact that
           SNRM2 does not fail on vectors with norm below the value of SQRT(DLAMCH('S')) */
        vn2[lsticc] = vn1[lsticc];
        lsticc = itemp;
    }
    
    magma_queue_destroy( stream );

    return MAGMA_SUCCESS;
} /* magma_zlaqps */
