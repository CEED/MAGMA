/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Azzam Haidar
       @author Stan Tomov

       @precisions normal z -> c

*/
#include "common_magma.h"
#include "timer.h"

#define PRECISION_z

/**
    Purpose
    -------
    ZHEGVD computes all the eigenvalues, and optionally, the eigenvectors
    of a complex generalized Hermitian-definite eigenproblem, of the form
    A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
    B are assumed to be Hermitian and B is also positive definite.
    If eigenvectors are desired, it uses a divide and conquer algorithm.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    ---------
    @param[in]
    itype   INTEGER
            Specifies the problem type to be solved:
            = 1:  A*x = (lambda)*B*x
            = 2:  A*B*x = (lambda)*x
            = 3:  B*A*x = (lambda)*x

    @param[in]
    jobz    CHARACTER*1
      -     = 'N':  Compute eigenvalues only;
      -     = 'V':  Compute eigenvalues and eigenvectors.

    @param[in]
    uplo    CHARACTER*1
      -     = 'U':  Upper triangles of A and B are stored;
      -     = 'L':  Lower triangles of A and B are stored.

    @param[in]
    n       INTEGER
            The order of the matrices A and B.  N >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA, N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = 'L',
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
    \n
            On exit, if JOBZ = 'V', then if INFO = 0, A contains the
            matrix Z of eigenvectors.  The eigenvectors are normalized
            as follows:
            if ITYPE = 1 or 2, Z**H*B*Z = I;
            if ITYPE = 3, Z**H*inv(B)*Z = I.
            If JOBZ = 'N', then on exit the upper triangle (if UPLO='U')
            or the lower triangle (if UPLO='L') of A, including the
            diagonal, is destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[in,out]
    B       COMPLEX_16 array, dimension (LDB, N)
            On entry, the Hermitian matrix B.  If UPLO = 'U', the
            leading N-by-N upper triangular part of B contains the
            upper triangular part of the matrix B.  If UPLO = 'L',
            the leading N-by-N lower triangular part of B contains
            the lower triangular part of the matrix B.
    \n
            On exit, if INFO <= N, the part of B containing the matrix is
            overwritten by the triangular factor U or L from the Cholesky
            factorization B = U**H*U or B = L*L**H.

    @param[in]
    ldb     INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    @param[out]
    w       DOUBLE PRECISION array, dimension (N)
            If INFO = 0, the eigenvalues in ascending order.

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The length of the array WORK.
            If N <= 1,                LWORK >= 1.
            If JOBZ  = 'N' and N > 1, LWORK >= N + N*NB.
            If JOBZ  = 'V' and N > 1, LWORK >= max( N + N*NB, 2*N + N**2 ).
            NB can be obtained through magma_get_zhetrd_nb(N).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal sizes of the WORK, RWORK and
            IWORK arrays, returns these values as the first entries of
            the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    @param[out]
    rwork   (workspace) DOUBLE PRECISION array, dimension (LRWORK)
            On exit, if INFO = 0, RWORK[0] returns the optimal LRWORK.

    @param[in]
    lrwork  INTEGER
            The dimension of the array RWORK.
            If N <= 1,                LRWORK >= 1.
            If JOBZ  = 'N' and N > 1, LRWORK >= N.
            If JOBZ  = 'V' and N > 1, LRWORK >= 1 + 5*N + 2*N**2.
    \n
            If LRWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    @param[out]
    iwork   (workspace) INTEGER array, dimension (MAX(1,LIWORK))
            On exit, if INFO = 0, IWORK[0] returns the optimal LIWORK.

    @param[in]
    liwork  INTEGER
            The dimension of the array IWORK.
            If N <= 1,                LIWORK >= 1.
            If JOBZ  = 'N' and N > 1, LIWORK >= 1.
            If JOBZ  = 'V' and N > 1, LIWORK >= 3 + 5*N.
    \n
            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  ZPOTRF or ZHEEVD returned an error code:
               <= N:  if INFO = i and JOBZ = 'N', then the algorithm
                      failed to converge; i off-diagonal elements of an
                      intermediate tridiagonal form did not converge to
                      zero;
                      if INFO = i and JOBZ = 'V', then the algorithm
                      failed to compute an eigenvalue while working on
                      the submatrix lying in rows and columns INFO/(N+1)
                      through mod(INFO,N+1);
               > N:   if INFO = N + i, for 1 <= i <= N, then the leading
                      minor of order i of B is not positive definite.
                      The factorization of B could not be completed and
                      no eigenvalues or eigenvectors were computed.

    Further Details
    ---------------
    Based on contributions by
       Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA

    Modified so that no backsubstitution is performed if ZHEEVD fails to
    converge (NEIG in old code could be greater than N causing out of
    bounds reference to A - reported by Ralf Meyer).  Also corrected the
    description of INFO and the test on ITYPE. Sven, 16 Feb 05.

    @ingroup magma_zhegv_driver
    ********************************************************************/
extern "C" magma_int_t
magma_zhegvd(magma_int_t itype, magma_vec_t jobz, magma_uplo_t uplo, magma_int_t n,
             magmaDoubleComplex *A, magma_int_t lda, magmaDoubleComplex *B, magma_int_t ldb,
             double *w, magmaDoubleComplex *work, magma_int_t lwork,
             double *rwork, magma_int_t lrwork,
             magma_int_t *iwork, magma_int_t liwork, magma_int_t *info)
{
    const char* uplo_ = lapack_uplo_const( uplo );
    const char* jobz_ = lapack_vec_const( jobz );

    magmaDoubleComplex c_one = MAGMA_Z_ONE;

    magmaDoubleComplex *da;
    magmaDoubleComplex *db;
    magma_int_t ldda = n;
    magma_int_t lddb = n;

    magma_int_t lower;
    magma_trans_t trans;
    magma_int_t wantz;
    magma_int_t lquery;

    magma_int_t lwmin;
    magma_int_t liwmin;
    magma_int_t lrwmin;

    magma_queue_t stream;
    magma_queue_create( &stream );

    wantz = (jobz == MagmaVec);
    lower = (uplo == MagmaLower);
    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (! (wantz || (jobz == MagmaNoVec))) {
        *info = -2;
    } else if (! (lower || (uplo == MagmaUpper))) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < max(1,n)) {
        *info = -6;
    } else if (ldb < max(1,n)) {
        *info = -8;
    }

    magma_int_t nb = magma_get_zhetrd_nb( n );
    if ( n <= 1 ) {
        lwmin  = 1;
        lrwmin = 1;
        liwmin = 1;
    }
    else if ( wantz ) {
        lwmin  = max( n + n*nb, 2*n + n*n );
        lrwmin = 1 + 5*n + 2*n*n;
        liwmin = 3 + 5*n;
    }
    else {
        lwmin  = n + n*nb;
        lrwmin = n;
        liwmin = 1;
    }

    double one_eps = 1. + lapackf77_dlamch("Epsilon");
    work[0]  = MAGMA_Z_MAKE( lwmin * one_eps, 0.);  // round up
    rwork[0] = lrwmin * one_eps;
    iwork[0] = liwmin;

    if (lwork < lwmin && ! lquery) {
        *info = -11;
    } else if (lrwork < lrwmin && ! lquery) {
        *info = -13;
    } else if (liwork < liwmin && ! lquery) {
        *info = -15;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (n == 0) {
        return *info;
    }

    /* Check if matrix is very small then just call LAPACK on CPU, no need for GPU */
    if (n <= 128) {
        #ifdef ENABLE_DEBUG
        printf("--------------------------------------------------------------\n");
        printf("  warning matrix too small N=%d NB=%d, calling lapack on CPU  \n", (int) n, (int) nb);
        printf("--------------------------------------------------------------\n");
        #endif
        lapackf77_zhegvd(&itype, jobz_, uplo_,
                         &n, A, &lda, B, &ldb,
                         w, work, &lwork,
                         #if defined(PRECISION_z) || defined(PRECISION_c)
                         rwork, &lrwork,
                         #endif
                         iwork, &liwork, info);
        return *info;
    }

    // TODO fix memory leak
    if (MAGMA_SUCCESS != magma_zmalloc( &da, n*ldda ) ||
        MAGMA_SUCCESS != magma_zmalloc( &db, n*lddb )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    /* Form a Cholesky factorization of B. */
    magma_zsetmatrix( n, n, B, ldb, db, lddb );

    magma_zsetmatrix_async( n, n,
                           A,  lda,
                           da, ldda, stream );

    magma_timer_t time;
    timer_start( time );
    magma_zpotrf_gpu(uplo, n, db, lddb, info);
    if (*info != 0) {
        *info = n + *info;
        return *info;
    }
    timer_stop( time );
    timer_printf( "time zpotrf_gpu = %6.2f\n", time );

    magma_queue_sync( stream );
    magma_zgetmatrix_async( n, n,
                           db, lddb,
                           B,  ldb, stream );

    timer_start( time );
    /* Transform problem to standard eigenvalue problem and solve. */
    magma_zhegst_gpu(itype, uplo, n, da, ldda, db, lddb, info);
    timer_stop( time );
    timer_printf( "time zhegst_gpu = %6.2f\n", time );

    /* simple fix to be able to run bigger size.
     * need to have a dwork here that will be used
     * a db and then passed to  dsyevd.
     * */
    if (n > 5000) {
        magma_queue_sync( stream );
        magma_free( db );
    }

    timer_start( time );
    magma_zheevd_gpu(jobz, uplo, n, da, ldda, w, A, lda,
                     work, lwork, rwork, lrwork, iwork, liwork, info);
    timer_stop( time );
    timer_printf( "time zheevd_gpu = %6.2f\n", time );

    if (wantz && *info == 0) {
        timer_start( time );
        
        /* allocate and copy db back */
        if (n > 5000) {
            if (MAGMA_SUCCESS != magma_zmalloc( &db, n*lddb ) ) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            magma_zsetmatrix( n, n, B, ldb, db, lddb );
        }
        /* Backtransform eigenvectors to the original problem. */
        if (itype == 1 || itype == 2) {
            /* For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
               backtransform eigenvectors: x = inv(L)'*y or inv(U)*y */
            if (lower) {
                trans = MagmaConjTrans;
            } else {
                trans = MagmaNoTrans;
            }

            magma_ztrsm(MagmaLeft, uplo, trans, MagmaNonUnit,
                        n, n, c_one, db, lddb, da, ldda);
        }
        else if (itype == 3) {
            /* For B*A*x=(lambda)*x;
               backtransform eigenvectors: x = L*y or U'*y */
            if (lower) {
                trans = MagmaNoTrans;
            } else {
                trans = MagmaConjTrans;
            }

            magma_ztrmm(MagmaLeft, uplo, trans, MagmaNonUnit,
                        n, n, c_one, db, lddb, da, ldda);
        }

        magma_zgetmatrix( n, n, da, ldda, A, lda );
        
        /* free db */
        if (n > 5000) {
            magma_free( db );
        }
        
        timer_stop( time );
        timer_printf( "time ztrsm/mm + getmatrix = %6.2f\n", time );
    }

    magma_queue_sync( stream );
    magma_queue_destroy( stream );

    work[0]  = MAGMA_Z_MAKE( lwmin * one_eps, 0.);  // round up
    rwork[0] = lrwmin * one_eps;
    iwork[0] = liwmin;

    magma_free( da );
    if (n <= 5000) {
        magma_free( db );
    }

    return *info;
} /* magma_zhegvd */
