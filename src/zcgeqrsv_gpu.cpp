/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions mixed zc -> ds

*/
#include "common_magma.h"

#define BWDMAX 1.0
#define ITERMAX 30

extern "C" magma_int_t
magma_zcgeqrsv_gpu(magma_int_t M, magma_int_t N, magma_int_t NRHS, 
                   magmaDoubleComplex *dA,  magma_int_t ldda, 
                   magmaDoubleComplex *dB,  magma_int_t lddb, 
                   magmaDoubleComplex *dX,  magma_int_t lddx, 
                   magma_int_t *iter, magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    ZCGEQRSV solves the least squares problem 
       min || A*X - B ||,
    where A is an M-by-N matrix and X and B are M-by-NRHS matrices.

    ZCGEQRSV first attempts to factorize the matrix in SINGLE PRECISION
    and use this factorization within an iterative refinement procedure
    to produce a solution with DOUBLE PRECISION norm-wise backward error
    quality (see below). If the approach fails the method switches to a
    DOUBLE PRECISION factorization and solve.

    The iterative refinement is not going to be a winning strategy if
    the ratio SINGLE PRECISION performance over DOUBLE PRECISION
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

    M       (input) INTEGER   
            The number of rows of the matrix A. M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A. M >= N >= 0.

    NRHS    (input) INTEGER
            The number of right hand sides, i.e., the number of columns
            of the matrix B.  NRHS >= 0.

    dA      (input or input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N coefficient matrix A.
            On exit, if iterative refinement has been successfully used
            (info.EQ.0 and ITER.GE.0, see description below), A is
            unchanged. If double precision factorization has been used
            (info.EQ.0 and ITER.LT.0, see description below), then the
            array dA contains the QR factorization of A as returned by
            function DGEQRF_GPU.

    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).

    dB      (input) COMPLEX_16 array on the GPU, dimension (LDDB,NRHS)
            The M-by-NRHS right hand side matrix B.

    LDDB    (input) INTEGER
            The leading dimension of the array dB.  LDDB >= max(1,M).

    dX      (output) COMPLEX_16 array on the GPU, dimension (LDDX,NRHS)
            If info = 0, the N-by-NRHS solution matrix X.

    LDDX    (input) INTEGER
            The leading dimension of the array dX.  LDDX >= max(1,N).

    WORK    (workspace) COMPLEX_16 array, dimension (N*NRHS)
            This array is used to hold the residual vectors.

    SWORK   (workspace) COMPLEX array, dimension (M*(N+NRHS))
            This array is used to store the single precision matrix and the
            right-hand sides or solutions in single precision.

    ITER    (output) INTEGER
            < 0: iterative refinement has failed, double precision
                 factorization has been performed
                 -1 : the routine fell back to full precision for
                      implementation- or machine-specific reasons
                 -2 : narrowing the precision induced an overflow,
                      the routine fell back to full precision
                 -3 : failure of SGETRF
                 -31: stop the iterative refinement after the 30th iteration
            > 0: iterative refinement has been successfully used.
                 Returns the number of iterations
 
    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if info = -i, the i-th argument had an illegal value

    =====================================================================    */

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magma_int_t     ione  = 1;
    magmaDoubleComplex *dworkd, *hworkd;
    magmaFloatComplex  *dworks, *hworks;
    magmaDoubleComplex *dR, *tau, *dT;
    magmaFloatComplex  *dSA, *dSX, *dST, *stau;
    magmaDoubleComplex Xnrmv, Rnrmv; 
    double          Anrm, Xnrm, Rnrm, cte, eps; 
    magma_int_t     i, j, iiter, nb, lhwork, minmn, size;
    
    /*
      Check The Parameters. 
    */
    *iter = 0 ;
    *info = 0 ;
    if ( N < 0 )
        *info = -1;
    else if(NRHS<0)
        *info = -3;
    else if( ldda < max(1,N))
        *info = -5;
    else if( lddb < max(1,N))
        *info = -7;
    else if( lddx < max(1,N))
        *info = -9;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    if( N == 0 || NRHS == 0 )
        return *info;

    nb   = magma_get_cgeqrf_nb(M);
    minmn= min(M, N);

    /*
     * Allocate temporary buffers
     */
    /* dworks(dSA + dSX + dST) */
    size = ldda*N + N*NRHS + ( 2*minmn + ((N+31)/32)*32 )*nb;
    if (MAGMA_SUCCESS != magma_cmalloc( &dworks, size )) {
        fprintf(stderr, "Allocation of dworks failed (%d)\n", (int) size);
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    dSA = dworks;
    dSX = dSA + ldda*N;
    dST = dSX + N*NRHS;
    
    /* dworkd(dR) = N*NRHS */
    size = N*NRHS;
    if (MAGMA_SUCCESS != magma_zmalloc( &dworkd, size )) {
        magma_free( dworks );
        fprintf(stderr, "Allocation of dworkd failed\n");
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    dR = dworkd;

    /* hworks(stau + workspace for cgeqrs) = min(M,N) + lhworks */
    lhwork = nb*max((M-N+nb+2*(NRHS)), 1);
    lhwork = max(lhwork, N*nb); /* We hope that magma nb is bigger than lapack nb to have enough memory in workspace */
    size = minmn + lhwork;
    magma_cmalloc_cpu( &hworks, size );
    if ( hworks == NULL ) {
        magma_free( dworks );
        magma_free( dworkd );
        fprintf(stderr, "Allocation of hworks failed\n");
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }
    stau = hworks + lhwork;

    eps  = lapackf77_dlamch("Epsilon");
    Anrm = magmablas_zlange('I', M, N, dA, ldda, (double*)dworkd );
    cte  = Anrm * eps *  pow((double)N, 0.5) * BWDMAX ;

    /*
     * Convert to single precision
     */
    magmablas_zlag2c(N, NRHS, dB, lddb, dSX, N, info );
    if( *info != 0 ) {
        *iter = -2; goto L40;
    }

    magmablas_zlag2c(N, N, dA, ldda, dSA, ldda, info );
    if(*info !=0){
        *iter = -2; goto L40;
    }

    // In an ideal version these variables should come from user.
    magma_cgeqrf_gpu(M, N, dSA, ldda, stau, dST, info);
    if( *info != 0 ) {
        *iter = -3; goto L40;
    }

    magma_cgeqrs_gpu(M, N, NRHS, dSA, ldda, stau, dST, dSX, N, hworks, lhwork, info);

    // dX = dSX
    magmablas_clag2z(N, NRHS, dSX, N, dX, lddx, info);

    // dR = dB
    magmablas_zlacpy(MagmaUpperLower, N, NRHS, dB, lddb, dR, N);

    // dR = dB - dA * dX
    if( NRHS == 1 )
        magma_zgemv( MagmaNoTrans, N, N, 
                     c_neg_one, dA, ldda, 
                                dX, 1, 
                     c_one,     dR, 1);
    else
        magma_zgemm( MagmaNoTrans, MagmaNoTrans, N, NRHS, N, 
                     c_neg_one, dA, ldda, 
                                dX, lddx, 
                     c_one,     dR, N );

    for(i=0; i<NRHS; i++){
        j = magma_izamax( N, dX+i*N, 1);
        magma_zgetmatrix( 1, 1, dX+i*N+j-1, 1, &Xnrmv, 1 );
        Xnrm = lapackf77_zlange( "F", &ione, &ione, &Xnrmv, &ione, NULL );
      
        j = magma_izamax ( N, dR+i*N, 1 );
        magma_zgetmatrix( 1, 1, dR+i*N+j-1, 1, &Rnrmv, 1 );
        Rnrm = lapackf77_zlange( "F", &ione, &ione, &Rnrmv, &ione, NULL );
      
        if( Rnrm >  Xnrm *cte ) goto L10;
    }

    *iter = 0;

    /* Free workspaces */
    magma_free( dworks );
    magma_free( dworkd );
    magma_free_cpu( hworks );
    return *info;

  L10:
    for(iiter=1; iiter<ITERMAX; ) {
        *info = 0 ;
        /*  Convert R from double precision to single precision
            and store the result in SX.
            Solve the system SA*SX = SR.
            -- These two Tasks are merged here. */
        // make SWORK = WORK ... residuals... 
        magmablas_zlag2c( N, NRHS, dR, N, dSX, N, info );
        magma_cgeqrs_gpu( M, N, NRHS, dSA, ldda, stau, dST, dSX, N, hworks, lhwork, info);

        if( *info != 0 ){
            *iter = -3; goto L40;
        }

        for(i=0; i<NRHS; i++) {
            magmablas_zcaxpycp( dSX+i*N, dX+i*lddx, N, dB+i*lddb, dR+i*N );
        }

        /* unnecessary may be */
        magmablas_zlacpy(MagmaUpperLower, N, NRHS, dB, lddb, dR, N);
        if( NRHS == 1 )
            magma_zgemv( MagmaNoTrans, N, N, 
                         c_neg_one, dA, ldda,
                                    dX, 1,
                         c_one,     dR, 1);
        else
            magma_zgemm( MagmaNoTrans, MagmaNoTrans, N, NRHS, N, 
                         c_neg_one, dA, ldda,
                                    dX, lddx,
                         c_one,     dR, N);

        /*  Check whether the NRHS normwise backward errors satisfy the
            stopping criterion. If yes, set ITER=IITER>0 and return.     */
        for(i=0;i<NRHS;i++)
        {
            j = magma_izamax( N, dX+i*N, 1);
            magma_zgetmatrix( 1, 1, dX+i*N+j-1, 1, &Xnrmv, 1 );
            Xnrm = lapackf77_zlange( "F", &ione, &ione, &Xnrmv, &ione, NULL );
            
            j = magma_izamax ( N, dR+i*N, 1 );
            magma_zgetmatrix( 1, 1, dR+i*N+j-1, 1, &Rnrmv, 1 );
            Rnrm = lapackf77_zlange( "F", &ione, &ione, &Rnrmv, &ione, NULL );
            
            if( Rnrm >  Xnrm *cte ) goto L20;
        }

        /*  If we are here, the NRHS normwise backward errors satisfy
            the stopping criterion, we are good to exit.                    */
        *iter = iiter ;

        /* Free workspaces */
        magma_free( dworks );
        magma_free( dworkd );
        magma_free_cpu( hworks );
        return *info;
      L20:
        iiter++;
    }

    /* If we are at this place of the code, this is because we have
       performed ITER=ITERMAX iterations and never satisified the
       stopping criterion, set up the ITER flag accordingly and follow
       up on double precision routine.                                    */
    *iter = -ITERMAX - 1 ;

  L40:
    magma_free( dworks );

    /*
     * Allocate temporary buffers
     */
    /* dworkd(dT + tau) = min_mn + min_mn*nb*3 */
    nb   = magma_get_zgeqrf_nb(M);
    size = minmn * (3 * nb + 1);
    if ( size > (N*NRHS) ) {
        magma_free( dworkd );
        if (MAGMA_SUCCESS != magma_zmalloc( &dworkd, size )) {
            fprintf(stderr, "Allocation of dworkd2 failed\n");
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
    }
    tau = dworkd;
    dT  = tau + minmn;

    /* hworks(stau + workspace for cgeqrs) = min(M,N) + lhworks */
    /* re-use hworks memory for hworkd if possible, else re-allocate. */
    if ( (2*lhwork) <= (minmn+lhwork) ) {
        hworkd = (magmaDoubleComplex*) hworks;
    }
    else {
        magma_free_cpu( hworks );
        magma_zmalloc_cpu( &hworkd, lhwork );
        if ( hworkd == NULL ) {
            magma_free( dworkd );
            fprintf(stderr, "Allocation of hworkd2 failed\n");
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
    }

    /* Single-precision iterative refinement failed to converge to a
       satisfactory solution, so we resort to double precision.           */
    magma_zgeqrf_gpu(M, N, dA, ldda, tau, dT, info);
    if ( *info == 0 ) {
        magmablas_zlacpy(MagmaUpperLower, N, NRHS, dB, lddb, dX, lddx);
        magma_zgeqrs_gpu(M, N, NRHS, dA, ldda, tau, dT, dX, lddx, hworkd, lhwork, info);
    }
    
    magma_free( dworkd );
    magma_free_cpu( hworkd );
    return *info;
}

