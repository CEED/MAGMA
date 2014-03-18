/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

*/
#include "common_magma.h"

#define BWDMAX 1.0
#define ITERMAX 30

/**
    Purpose
    =======
    ZCPOSV computes the solution to a complex system of linear equations
        A * X = B,
    where A is an N-by-N Hermitian positive definite matrix and X and B
    are N-by-NRHS matrices.

    ZCPOSV first attempts to factorize the matrix in complex SINGLE PRECISION
    and use this factorization within an iterative refinement procedure
    to produce a solution with complex DOUBLE PRECISION norm-wise backward error
    quality (see below). If the approach fails the method switches to a
    complex DOUBLE PRECISION factorization and solve.

    The iterative refinement is not going to be a winning strategy if
    the ratio complex SINGLE PRECISION performance over complex DOUBLE PRECISION
    performance is too small. A reasonable strategy should take the
    number of right-hand sides and the size of the matrix into account.
    This might be done with a call to ILAENV in the future. Up to now, we
    always try iterative refinement.

    The iterative refinement process is stopped if
        ITER > ITERMAX
    or for all the RHS we have:
        RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX
    where
        o ITER is the number of the current iteration in the iterative
          refinement process
        o RNRM is the infinity-norm of the residual
        o XNRM is the infinity-norm of the solution
        o ANRM is the infinity-operator-norm of the matrix A
        o EPS is the machine epsilon returned by DLAMCH('Epsilon')
    The value ITERMAX and BWDMAX are fixed to 30 and 1.0D+00 respectively.

    Arguments
    =========
    @param[in]
    uplo    CHARACTER
      -     = 'U':  Upper triangle of A is stored;
      -     = 'L':  Lower triangle of A is stored.

    @param[in]
    n       INTEGER
            The number of linear equations, i.e., the order of the
            matrix A.  N >= 0.

    @param[in]
    nrhs    INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
            On exit, if iterative refinement has been successfully used
            (INFO.EQ.0 and ITER.GE.0, see description below), then A is
            unchanged, if double factorization has been used
            (INFO.EQ.0 and ITER.LT.0, see description below), then the
            array dA contains the factor U or L from the Cholesky
            factorization A = U**T*U or A = L*L**T.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).

    @param[in]
    dB      COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            The N-by-NRHS right hand side matrix B.

    @param[in]
    lddb    INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,N).

    @param[out]
    dX      COMPLEX_16 array on the GPU, dimension (LDDX,NRHS)
            If INFO = 0, the N-by-NRHS solution matrix X.

    @param[in]
    lddx    INTEGER
            The leading dimension of the array dX.  LDDX >= max(1,N).

    @param
    dworkd  (workspace) COMPLEX_16 array on the GPU, dimension (N*NRHS)
            This array is used to hold the residual vectors.

    @param
    dworks  (workspace) COMPLEX array on the GPU, dimension (N*(N+NRHS))
            This array is used to store the complex single precision matrix
            and the right-hand sides or solutions in single precision.

    @param[out]
    iter    INTEGER
            < 0: iterative refinement has failed, double precision
                 factorization has been performed
                 -1 : the routine fell back to full precision for
                      implementation- or machine-specific reasons
                 -2 : narrowing the precision induced an overflow,
                      the routine fell back to full precision
                 -3 : failure of SPOTRF
                 -31: stop the iterative refinement after the 30th iteration
            > 0: iterative refinement has been successfully used.
                 Returns the number of iterations

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i of (DOUBLE
                  PRECISION) A is not positive definite, so the
                  factorization could not be completed, and the solution
                  has not been computed.

    @ingroup magma_zposv_driver
    ********************************************************************/
extern "C" magma_int_t
magma_zcposv_gpu(magma_uplo_t uplo, magma_int_t n, magma_int_t nrhs,
                 magmaDoubleComplex *dA, magma_int_t ldda,
                 magmaDoubleComplex *dB, magma_int_t lddb,
                 magmaDoubleComplex *dX, magma_int_t lddx,
                 magmaDoubleComplex *dworkd, magmaFloatComplex *dworks,
                 magma_int_t *iter, magma_int_t *info)
{
    #define dB(i,j)     (dB + (i) + (j)*lddb)
    #define dX(i,j)     (dX + (i) + (j)*lddx)
    #define dR(i,j)     (dR + (i) + (j)*lddr)
    #define dSX(i,j)    (dSX + (i) + (j)*lddsx)

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magma_int_t     ione  = 1;
    magmaDoubleComplex *dR;
    magmaFloatComplex  *dSA, *dSX;
    magmaDoubleComplex Xnrmv, Rnrmv;
    double          Anrm, Xnrm, Rnrm, cte, eps;
    magma_int_t     i, j, iiter, lddsa, lddsx, lddr;

    /* Check arguments */
    *iter = 0;
    *info = 0;
    if ( n < 0 )
        *info = -1;
    else if ( nrhs < 0 )
        *info = -2;
    else if ( ldda < max(1,n))
        *info = -4;
    else if ( lddb < max(1,n))
        *info = -7;
    else if ( lddx < max(1,n))
        *info = -9;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if ( n == 0 || nrhs == 0 )
        return *info;

    lddsa = n;
    lddsx = n;
    lddr  = n;
    
    dSA = dworks;
    dSX = dSA + lddsa*n;
    dR  = dworkd;

    eps  = lapackf77_dlamch("Epsilon");
    Anrm = magmablas_zlanhe(MagmaInfNorm, uplo, n, dA, ldda, (double*)dworkd );
    cte  = Anrm * eps * pow((double)n, 0.5) * BWDMAX;

    /*
     * Convert to single precision
     */
    magmablas_zlag2c( n, nrhs, dB, lddb, dSX, lddsx, info );
    if (*info != 0) {
        *iter = -2;
        goto FALLBACK;
    }

    magmablas_zlat2c( uplo, n, dA, ldda, dSA, lddsa, info );
    if (*info != 0) {
        *iter = -2;
        goto FALLBACK;
    }
    
    // factor dSA in single precision
    magma_cpotrf_gpu( uplo, n, dSA, lddsa, info );
    if (*info != 0) {
        *iter = -3;
        goto FALLBACK;
    }
    
    // solve dSA*dSX = dB in single precision
    magma_cpotrs_gpu( uplo, n, nrhs, dSA, lddsa, dSX, lddsx, info );

    // residual dR = dB - dA*dX in double precision
    magmablas_clag2z( n, nrhs, dSX, lddsx, dX, lddx, info );
    magmablas_zlacpy( MagmaUpperLower, n, nrhs, dB, lddb, dR, lddr );
    if ( nrhs == 1 ) {
        magma_zhemv( uplo, n,
                     c_neg_one, dA, ldda,
                                dX, 1,
                     c_one,     dR, 1 );
    }
    else {
        magma_zhemm( MagmaLeft, uplo, n, nrhs,
                     c_neg_one, dA, ldda,
                                dX, lddx,
                     c_one,     dR, lddr );
    }

    // TODO: use MAGMA_Z_ABS( dX(i,j) ) instead of zlange?
    for( j=0; j < nrhs; j++ ) {
        i = magma_izamax( n, dX(0,j), 1) - 1;
        magma_zgetmatrix( 1, 1, dX(i,j), 1, &Xnrmv, 1 );
        Xnrm = lapackf77_zlange( "F", &ione, &ione, &Xnrmv, &ione, NULL );

        i = magma_izamax ( n, dR(0,j), 1 ) - 1;
        magma_zgetmatrix( 1, 1, dR(i,j), 1, &Rnrmv, 1 );
        Rnrm = lapackf77_zlange( "F", &ione, &ione, &Rnrmv, &ione, NULL );

        if ( Rnrm >  Xnrm*cte ) {
            goto REFINEMENT;
        }
    }
    
    *iter = 0;
    return *info;

REFINEMENT:
    for( iiter=1; iiter < ITERMAX; ) {
        *info = 0;
        // convert residual dR to single precision dSX
        magmablas_zlag2c( n, nrhs, dR, lddr, dSX, lddsx, info );
        if (*info != 0) {
            *iter = -2;
            goto FALLBACK;
        }
        // solve dSA*dSX = R in single precision
        magma_cpotrs_gpu( uplo, n, nrhs, dSA, lddsa, dSX, lddsx, info );

        // Add correction and setup residual
        // dX += dSX [including conversion]  --and--
        // dR = dB
        for( j=0; j < nrhs; j++ ) {
            magmablas_zcaxpycp( n, dSX(0,j), dX(0,j), dB(0,j), dR(0,j) );
        }

        // residual dR = dB - dA*dX in double precision
        if ( nrhs == 1 ) {
            magma_zhemv( uplo, n,
                         c_neg_one, dA, ldda,
                                    dX, 1,
                         c_one,     dR, 1 );
        }
        else {
            magma_zhemm( MagmaLeft, uplo, n, nrhs,
                         c_neg_one, dA, ldda,
                                    dX, lddx,
                         c_one,     dR, lddr );
        }

        /*  Check whether the nrhs normwise backward errors satisfy the
         *  stopping criterion. If yes, set ITER=IITER > 0 and return. */
        for( j=0; j < nrhs; j++ ) {
            i = magma_izamax( n, dX(0,j), 1) - 1;
            magma_zgetmatrix( 1, 1, dX(i,j), 1, &Xnrmv, 1 );
            Xnrm = lapackf77_zlange( "F", &ione, &ione, &Xnrmv, &ione, NULL );

            i = magma_izamax ( n, dR(0,j), 1 ) - 1;
            magma_zgetmatrix( 1, 1, dR(i,j), 1, &Rnrmv, 1 );
            Rnrm = lapackf77_zlange( "F", &ione, &ione, &Rnrmv, &ione, NULL );

            if ( Rnrm >  Xnrm*cte ) {
                goto L20;
            }
        }

        /*  If we are here, the nrhs normwise backward errors satisfy
         *  the stopping criterion, we are good to exit. */
        *iter = iiter;
        return *info;
        
      L20:
        iiter++;
    }
    
    /* If we are at this place of the code, this is because we have
     * performed ITER=ITERMAX iterations and never satisified the
     * stopping criterion. Set up the ITER flag accordingly and follow
     * up on double precision routine. */
    *iter = -ITERMAX - 1;

FALLBACK:
    /* Single-precision iterative refinement failed to converge to a
     * satisfactory solution, so we resort to double precision. */
    magma_zpotrf_gpu( uplo, n, dA, ldda, info );
    if (*info == 0) {
        magmablas_zlacpy( MagmaUpperLower, n, nrhs, dB, lddb, dX, lddx );
        magma_zpotrs_gpu( uplo, n, nrhs, dA, ldda, dX, lddx, info );
    }
    
    return *info;
}
