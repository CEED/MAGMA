/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions mixed zc -> ds

*/
#include <stdio.h>
#include <math.h>
#include <cublas.h>
#include <cuda.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include "magma.h"
#include "magmablas.h"

#define BWDMAX 1.0
#define ITERMAX 30

extern "C" magma_int_t
magma_zcgeqrsv_gpu(magma_int_t M, magma_int_t N, magma_int_t NRHS, 
                   cuDoubleComplex *A, magma_int_t LDA, 
                   cuDoubleComplex *B, magma_int_t LDB, 
                   cuDoubleComplex *X, magma_int_t LDX, 
                   cuDoubleComplex *WORK, cuFloatComplex *SWORK, 
		   magma_int_t *ITER, magma_int_t *INFO,
                   cuFloatComplex *tau, magma_int_t lwork, 
                   cuFloatComplex *h_work, cuFloatComplex *d_work, 
                   cuDoubleComplex *tau_d, magma_int_t lwork_d, 
                   cuDoubleComplex *h_work_d,
		   cuDoubleComplex *d_work_d)
{
/*  -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

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

    A       (input or input/output) DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the M-by-N coefficient matrix A.
            On exit, if iterative refinement has been successfully used
            (INFO.EQ.0 and ITER.GE.0, see description below), A is
            unchanged. If double2 precision factorization has been used
            (INFO.EQ.0 and ITER.LT.0, see description below), then the
            array A contains the QR factorization of A as returned by
            function DGEQRF_GPU.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    B       (input) DOUBLE PRECISION array, dimension (LDB,NRHS)
            The M-by-NRHS right hand side matrix B.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,M).

    X       (output) DOUBLE PRECISION array, dimension (LDX,NRHS)
            If INFO = 0, the N-by-NRHS solution matrix X.

    LDX     (input) INTEGER
            The leading dimension of the array X.  LDX >= max(1,N).

    WORK    (workspace) DOUBLE PRECISION array, dimension (N*NRHS)
            This array is used to hold the residual vectors.

    SWORK   (workspace) REAL array, dimension (M*(N+NRHS))
            This array is used to store the single precision matrix and the
            right-hand sides or solutions in single precision.

    ITER    (output) INTEGER
            < 0: iterative refinement has failed, double2 precision
                 factorization has been performed
                 -1 : the routine fell back to full precision for
                      implementation- or machine-specific reasons
                 -2 : narrowing the precision induced an overflow,
                      the routine fell back to full precision
                 -3 : failure of SGETRF
                 -31: stop the iterative refinement after the 30th
                      iterations
            > 0: iterative refinement has been successfully used.
                 Returns the number of iterations
 
    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value

    TAU     (output) REAL array, dimension (N)
            On exit, TAU(i) contains the scalar factor of the elementary
            reflector H(i), as returned by magma_cgeqrf_gpu.

    LWORK   (input) INTEGER   
            The dimension of the array H_WORK.  LWORK >= (M+N+NB)*NB,   
            where NB can be obtained through magma_get_sgeqrf_nb(M).

    H_WORK  (workspace/output) REAL array, dimension (MAX(1,LWORK))   
            Higher performance is achieved if H_WORK is in pinned memory, e.g.
            allocated using cudaMallocHost.

    D_WORK  (workspace/output)  REAL array on the GPU, dimension 2*N*NB,
            where NB can be obtained through magma_get_sgeqrf_nb(M).
            It starts with NB*NB blocks that store the triangular T 
            matrices, followed by the NB*NB blocks of the diagonal 
            inverses for the R matrix.

    TAU_D   (output) DOUBLE REAL array, dimension (N)
            On exit, if the matrix had to be factored in double2 precision,
            TAU(i) contains the scalar factor of the elementary
            reflector H(i), as returned by magma_zgeqrf_gpu.

    LWORK_D (input) INTEGER   
            The dimension of the array H_WORK_D. LWORK_D >= (M+N+NB)*NB,   
            where NB can be obtained through magma_get_dgeqrf_nb(M).

    H_WORK_D (workspace/output) DOUBLE REAL array, dimension (MAX(1,LWORK_D))
            This memory is unattached if the iterative refinement worked, 
            otherwise it is used as workspace to factor the matrix in
            double2 precision. Higher performance is achieved if H_WORK_D is 
            in pinned memory, e.g. allocated using cudaMallocHost. 

    D_WORK_D (workspace/output) DOUBLE REAL array on the GPU, dimension 2*N*NB,
            where NB can be obtained through magma_get_dgeqrf_nb(M).
            This memory is unattached if the iterative refinement worked, 
            otherwise it is used as workspace to factor the matrix in
            double2 precision. It starts with NB*NB blocks that store the 
            triangular T matrices, followed by the NB*NB blocks of the 
            diagonal inverses for the R matrix.

    =====================================================================    */

  #define max(a,b)       (((a)>(b))?(a):(b))

    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    int             c_ione    = 1;
    cuDoubleComplex XNRMv, RNRMv; 
    double          XNRM, RNRM; 
    double          ANRM , CTE , EPS;
    int             IITER, PTSA;
    int             i, j;

    /*
      Check The Parameters. 
    */
    *ITER = 0 ;
    *INFO = 0 ;
    if ( N <0)
        *INFO = -1;
    else if(NRHS<0)
        *INFO =-3;
    else if(LDA < max(1,N))
        *INFO =-5;
    else if( LDB < max(1,N))
        *INFO =-7;
    else if( LDX < max(1,N))
        *INFO =-9;

    if(*INFO!=0){
        printf("%d %d %d\n", M , N , NRHS);
        magma_xerbla("magma_zcgeqrsv_gpu",INFO) ;
    }

    if( N == 0 || NRHS == 0 )
        return 0;

    EPS  = lapackf77_dlamch("Epsilon");
    ANRM = magmablas_zlange('I', N, N, A, LDA, (double*)WORK );
    CTE  = ANRM * EPS *  pow((double)N,0.5) * BWDMAX ;
    PTSA = N*NRHS;

    magmablas_zlag2c(N, NRHS, B, LDB, SWORK, N, INFO );
    if(*INFO !=0){
        *ITER = -2 ;
        printf("magmablas_zlag2c\n");
        goto L40;
    }
    magmablas_zlag2c(N, N, A, LDA, SWORK+PTSA, N, INFO ); // Merge with DLANGE /
    if(*INFO !=0){
        *ITER = -2 ;
        printf("magmablas_zlag2c\n");
        goto L40;
    }

    // In an ideal version these variables should come from user.
    magma_cgeqrf_gpu(M, N, SWORK+PTSA, N, tau, d_work, INFO);

    if(INFO[0] !=0){
        *ITER = -3 ;
        goto L40;
    }

    // SWORK = B 
    magma_cgeqrs_gpu(M, N, NRHS, SWORK+PTSA, N, tau, d_work, 
		     SWORK, M, h_work, lwork, INFO);

    // SWORK = X in SP 
    magmablas_clag2z(N, NRHS, SWORK, N, X, LDX, INFO);

    // X = X in DP 
    magmablas_zlacpy(MagmaUpperLower, N, NRHS, B , LDB, WORK, N);

    // WORK = B in DP; WORK contains the residual ...
    if( NRHS == 1 )
        cublasZgemv( MagmaNoTrans, N, N, c_neg_one, A, LDA, X, 1, c_one, WORK, 1);
    else
        cublasZgemm( MagmaNoTrans, MagmaNoTrans, N, NRHS, N, 
                     c_neg_one, A,    LDA, 
                                X,    LDX, 
                     c_one,     WORK, N );

    for(i=0; i<NRHS; i++){
        j = cublasIzamax( N, X+i*N, 1) ;
        cublasGetMatrix( 1, 1, sizeof(cuDoubleComplex), X+i*N+j-1, 1, &XNRMv, 1);
        XNRM = lapackf77_zlange( "F", &c_ione, &c_ione, &XNRMv, &c_ione, NULL );
      
        j = cublasIzamax ( N, WORK+i*N, 1 );
        cublasGetMatrix( 1, 1, sizeof(cuDoubleComplex), WORK+i*N+j-1, 1, &RNRMv, 1);
        RNRM = lapackf77_zlange( "F", &c_ione, &c_ione, &RNRMv, &c_ione, NULL );
      
        if( RNRM >  XNRM *CTE ){
            goto L10;
        }
    }

    *ITER =0;
    return 0;

  L10:
    ;

    for(IITER=1; IITER<ITERMAX; ) {
        *INFO = 0 ;
        /*  Convert R (in WORK) from double2 precision to single precision
            and store the result in SX.
            Solve the system SA*SX = SR.
            -- These two Tasks are merged here. */
        // make SWORK = WORK ... residuals... 
        magmablas_zlag2c( N, NRHS, WORK, LDB, SWORK, N, INFO );
        magma_cgeqrs_gpu( M, N, NRHS, SWORK+PTSA, N, tau, d_work,
			  SWORK, M, h_work, lwork, INFO);

        if(INFO[0] !=0) {
            *ITER = -3 ;
            goto L40;
        }
        for(i=0;i<NRHS;i++) {
            magmablas_zcaxpycp( SWORK+i*N, X+i*N, N, N, LDA, B+i*N, WORK+i*N);
        }

        /* unnecessary may be */
        magmablas_zlacpy(MagmaUpperLower, N, NRHS, B, LDB, WORK, N);
        if( NRHS == 1 )
            cublasZgemv( MagmaNoTrans, N, N, c_neg_one, A, LDA, X, 1, c_one, WORK, 1);
        else
            cublasZgemm( MagmaNoTrans, MagmaNoTrans, N, NRHS, N, 
                         c_neg_one, A, LDA, X, LDX, c_one, WORK, N);

        /*  Check whether the NRHS normwise backward errors satisfy the
            stopping criterion. If yes, set ITER=IITER>0 and return.     */
        for(i=0;i<NRHS;i++)
        {
            j = cublasIzamax( N, X+i*N, 1) ;
            cublasGetMatrix( 1, 1, sizeof(cuDoubleComplex), X+i*N+j-1, 1, &XNRMv, 1);
            XNRM = lapackf77_zlange( "F", &c_ione, &c_ione, &XNRMv, &c_ione, NULL );
            
            j = cublasIzamax ( N, WORK+i*N, 1 );
            cublasGetMatrix( 1, 1, sizeof(cuDoubleComplex), WORK+i*N+j-1, 1, &RNRMv, 1 );
            RNRM = lapackf77_zlange( "F", &c_ione, &c_ione, &RNRMv, &c_ione, NULL );
            
            if( RNRM >  XNRM *CTE ){
                goto L20;
            }
        }

        /*  If we are here, the NRHS normwise backward errors satisfy
            the stopping criterion, we are good to exit.                    */
        *ITER = IITER ;
        return 0;
      L20:
        IITER++ ;
    }

    /* If we are at this place of the code, this is because we have
       performed ITER=ITERMAX iterations and never satisified the
       stopping criterion, set up the ITER flag accordingly and follow
       up on double2 precision routine.                                    */
    *ITER = -ITERMAX - 1 ;

  L40:

    /* Single-precision iterative refinement failed to converge to a
       satisfactory solution, so we resort to double2 precision.           */
    magma_zgeqrf_gpu(M, N, A, N, tau_d, d_work_d, INFO);
    if( *INFO != 0 ){
        return 0;
    }
    magmablas_zlacpy(MagmaUpperLower, N, NRHS, B , LDB, X, N);
    magma_zgeqrs_gpu(M, N, NRHS, A, N, tau_d, d_work_d,
                     X, M, h_work_d, lwork_d, INFO);
    return 0;
}

#undef max
