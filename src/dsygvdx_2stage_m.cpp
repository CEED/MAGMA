/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Azzam Haidar

       @precisions normal d -> s

*/
#include "common_magma.h"
#include "magma_timer.h"
#include "magma_bulge.h"
#include "magma_dbulge.h"

#define PRECISION_d
#define REAL

/**
    Purpose
    -------
    DSYGVDX_2STAGE computes all the eigenvalues, and optionally, the eigenvectors
    of a complex generalized Hermitian-definite eigenproblem, of the form
    A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
    B are assumed to be Hermitian and B is also positive definite.
    It uses a two-stage algorithm for the tridiagonalization.
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
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    itype   INTEGER
            Specifies the problem type to be solved:
            = 1:  A*x = (lambda)*B*x
            = 2:  A*B*x = (lambda)*x
            = 3:  B*A*x = (lambda)*x

    @param[in]
    range   magma_range_t
      -     = MagmaRangeAll: all eigenvalues will be found.
      -     = MagmaRangeV:   all eigenvalues in the half-open interval (VL,VU]
                   will be found.
      -     = MagmaRangeI:   the IL-th through IU-th eigenvalues will be found.

    @param[in]
    jobz    magma_vec_t
      -     = MagmaNoVec:  Compute eigenvalues only;
      -     = MagmaVec:    Compute eigenvalues and eigenvectors.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangles of A and B are stored;
      -     = MagmaLower:  Lower triangles of A and B are stored.

    @param[in]
    n       INTEGER
            The order of the matrices A and B.  N >= 0.

    @param[in,out]
    A       DOUBLE PRECISION array, dimension (LDA, N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = MagmaLower,
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
    \n
            On exit, if JOBZ = MagmaVec, then if INFO = 0, A contains the
            matrix Z of eigenvectors.  The eigenvectors are normalized
            as follows:
            if ITYPE = 1 or 2, Z**H*B*Z = I;
            if ITYPE = 3, Z**H*inv(B)*Z = I.
            If JOBZ = MagmaNoVec, then on exit the upper triangle (if UPLO=MagmaUpper)
            or the lower triangle (if UPLO=MagmaLower) of A, including the
            diagonal, is destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[in,out]
    B       DOUBLE PRECISION array, dimension (LDB, N)
            On entry, the Hermitian matrix B.  If UPLO = MagmaUpper, the
            leading N-by-N upper triangular part of B contains the
            upper triangular part of the matrix B.  If UPLO = MagmaLower,
            the leading N-by-N lower triangular part of B contains
            the lower triangular part of the matrix B.
    \n
            On exit, if INFO <= N, the part of B containing the matrix is
            overwritten by the triangular factor U or L from the Cholesky
            factorization B = U**H*U or B = L*L**H.

    @param[in]
    ldb     INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    @param[in]
    vl      DOUBLE PRECISION
    @param[in]
    vu      DOUBLE PRECISION
            If RANGE=MagmaRangeV, the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeI.

    @param[in]
    il      INTEGER
    @param[in]
    iu      INTEGER
            If RANGE=MagmaRangeI, the indices (in ascending order) of the
            smallest and largest eigenvalues to be returned.
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
            Not referenced if RANGE = MagmaRangeAll or MagmaRangeV.

    @param[out]
    m       INTEGER
            The total number of eigenvalues found.  0 <= M <= N.
            If RANGE = MagmaRangeAll, M = N, and if RANGE = MagmaRangeI, M = IU-IL+1.

    @param[out]
    w       DOUBLE PRECISION array, dimension (N)
            If INFO = 0, the eigenvalues in ascending order.

    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The length of the array WORK.
            If N <= 1,                      LWORK >= 1.
            If JOBZ = MagmaNoVec and N > 1, LWORK >= LQ2 + 2*N + N*NB.
            If JOBZ = MagmaVec   and N > 1, LWORK >= LQ2 + 1 + 6*N + 2*N**2.
            where LQ2 is the size needed to store the Q2 matrix
            and is returned by magma_bulge_get_lq2.
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal sizes of the WORK, RWORK and
            IWORK arrays, returns these values as the first entries of
            the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    @param[out]
    iwork   (workspace) INTEGER array, dimension (MAX(1,LIWORK))
            On exit, if INFO = 0, IWORK[0] returns the optimal LIWORK.

    @param[in]
    liwork  INTEGER
            The dimension of the array IWORK.
            If N <= 1,                      LIWORK >= 1.
            If JOBZ = MagmaNoVec and N > 1, LIWORK >= 1.
            If JOBZ = MagmaVec   and N > 1, LIWORK >= 3 + 5*N.
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
               <= N:  if INFO = i and JOBZ = MagmaNoVec, then the algorithm
                      failed to converge; i off-diagonal elements of an
                      intermediate tridiagonal form did not converge to
                      zero;
                      if INFO = i and JOBZ = MagmaVec, then the algorithm
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

    @ingroup magma_dsygv_driver
    ********************************************************************/
extern "C" magma_int_t
magma_dsygvdx_2stage_m(
    magma_int_t ngpu,
    magma_int_t itype, magma_vec_t jobz, magma_range_t range, magma_uplo_t uplo, magma_int_t n,
    double *A, magma_int_t lda,
    double *B, magma_int_t ldb,
    double vl, double vu, magma_int_t il, magma_int_t iu,
    magma_int_t *m, double *w,
    double *work, magma_int_t lwork,
    #ifdef COMPLEX
    double *rwork, magma_int_t lrwork,
    #endif
    magma_int_t *iwork, magma_int_t liwork,
    magma_int_t *info)
{
    const char* uplo_  = lapack_uplo_const( uplo  );
    const char* jobz_  = lapack_vec_const( jobz  );

    double d_one = MAGMA_D_ONE;

    magma_int_t lower;
    magma_trans_t trans;
    magma_int_t wantz;
    magma_int_t lquery;
    magma_int_t alleig, valeig, indeig;

    magma_int_t lwmin;
    magma_int_t liwmin;

    /* determine the number of threads */
    magma_int_t parallel_threads = magma_get_parallel_numthreads();

    wantz  = (jobz  == MagmaVec);
    lower  = (uplo  == MagmaLower);
    alleig = (range == MagmaRangeAll);
    valeig = (range == MagmaRangeV);
    indeig = (range == MagmaRangeI);
    lquery = (lwork == -1 || liwork == -1);

    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (! (alleig || valeig || indeig)) {
        *info = -2;
    } else if (! (wantz || (jobz == MagmaNoVec))) {
        *info = -3;
    } else if (! (lower || (uplo == MagmaUpper))) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < max(1,n)) {
        *info = -7;
    } else if (ldb < max(1,n)) {
        *info = -9;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -11;
            }
        } else if (indeig) {
            if (il < 1 || il > max(1,n)) {
                *info = -12;
            } else if (iu < min(n,il) || iu > n) {
                *info = -13;
            }
        }
    }

    magma_int_t nb = magma_get_dbulge_nb(n, parallel_threads);
    magma_int_t lq2 = magma_dbulge_get_lq2(n, parallel_threads);

    if (wantz) {
        lwmin  = lq2 + 1 + 6*n + 2*n*n;
        liwmin = 3 + 5*n;
    } else {
        lwmin  = 2*n + n*nb;
        liwmin = 1;
    }

    // multiply by 1+eps (in Double!) to ensure length gets rounded up,
    // if it cannot be exactly represented in floating point.
    real_Double_t one_eps = 1. + lapackf77_dlamch("Epsilon");
    work[0] = lwmin * one_eps;
    iwork[0] = liwmin;

    if (lwork < lwmin && ! lquery) {
        *info = -17;
    } else if (liwork < liwmin && ! lquery) {
        *info = -19;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info));
        return *info;
    } else if (lquery) {
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
        lapackf77_dsygvd(&itype, jobz_, uplo_,
                         &n, A, &lda, B, &ldb,
                         w, work, &lwork,
                         iwork, &liwork, info);
        *m = n;
        return *info;
    }

    /* Form A Cholesky factorization of B. */
    magma_timer_t time=0;
    timer_start( time );

    magma_dpotrf_m(ngpu, uplo, n, B, ldb, info);
    if (*info != 0) {
        *info = n + *info;
        return *info;
    }

    timer_stop( time );
    timer_printf( "time dpotrf_m = %6.2f\n", time );
    timer_start( time );

    /* Transform problem to standard eigenvalue problem and solve. */
    magma_dsygst_m(ngpu, itype, uplo, n, A, lda, B, ldb, info);

    timer_stop( time );
    timer_printf( "time dsygst_m = %6.2f\n", time );
    timer_start( time );

    magma_dsyevdx_2stage_m(ngpu, jobz, range, uplo, n, A, lda, vl, vu, il, iu, m, w, work, lwork, iwork, liwork, info);

    timer_stop( time );
    timer_printf( "time dsyevdx_2stage_m = %6.2f\n", time );

    if (wantz && *info == 0) {
        timer_start( time );

        /* Backtransform eigenvectors to the original problem. */
        if (itype == 1 || itype == 2) {
            /* For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
               backtransform eigenvectors: x = inv(L)'*y or inv(U)*y */
            if (lower) {
                trans = MagmaTrans;
            } else {
                trans = MagmaNoTrans;
            }

            magma_dtrsm_m(ngpu, MagmaLeft, uplo, trans, MagmaNonUnit, n, *m, d_one, B, ldb, A, lda);
        }
        else if (itype == 3) {
            /* For B*A*x=(lambda)*x;
               backtransform eigenvectors: x = L*y or U'*y */
            if (lower) {
                trans = MagmaNoTrans;
            } else {
                trans = MagmaTrans;
            }

            //magma_dtrmm_m(ngpu, MagmaLeft, uplo, trans, MagmaNonUnit, n, *m, d_one, B, ldb, A, lda);
            printf("--- the multi GPU version is falling back to 1 GPU to perform the last TRMM since there is no TRMM_mgpu --- \n");
            double *dA=NULL, *dB=NULL;
            magma_int_t ldda = n;
            magma_int_t lddb = n;
            
            if (MAGMA_SUCCESS != magma_dmalloc( &dB, n*lddb ) ) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            if (MAGMA_SUCCESS != magma_dmalloc( &dA, n*ldda ) ) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            magma_dsetmatrix( n, n, B, ldb, dB, lddb );
            magma_dsetmatrix( n, n, A, lda, dA, ldda );
            magma_dtrmm(MagmaLeft, uplo, trans, MagmaNonUnit,
                        n, n, d_one, dB, lddb, dA, ldda);
            magma_dgetmatrix( n, n, dA, ldda, A, lda );        }

        timer_stop( time );
        timer_printf( "time dtrsm/mm + getmatrix = %6.2f\n", time );
    }

    work[0] = lwmin * one_eps;
    iwork[0] = liwmin;

    return *info;
} /* magma_dsygvdx_2stage_m */
