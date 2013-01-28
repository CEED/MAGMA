/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Raffaele Solca
       @author Stan Tomov

       @precisions normal d -> s

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_dpotrf_m(magma_int_t num_gpus, char uplo, magma_int_t n,
                  double *a, magma_int_t lda, magma_int_t *info);

extern "C" magma_int_t
magma_dsygst_m(magma_int_t nrgpu, magma_int_t itype, char uplo, magma_int_t n,
               double *a, magma_int_t lda,
               double *b, magma_int_t ldb, magma_int_t *info);

extern "C" magma_int_t
magma_dsyevdx_m(magma_int_t nrgpu, char jobz, char range, char uplo,
                magma_int_t n,
                double *a, magma_int_t lda,
                double vl, double vu, magma_int_t il, magma_int_t iu,
                magma_int_t *m, double *w,
                double *work, magma_int_t lwork,
                magma_int_t *iwork, magma_int_t liwork,
                magma_int_t *info);

extern "C" magma_int_t
magma_dtrsm_m (magma_int_t nrgpu, char side, char uplo, char transa, char diag,
               magma_int_t m, magma_int_t n, double alpha, double *a,
               magma_int_t lda, double *b, magma_int_t ldb);

extern "C" magma_int_t
magma_dsygvdx_m(magma_int_t nrgpu, magma_int_t itype, char jobz, char range, char uplo, magma_int_t n,
                double *a, magma_int_t lda, double *b, magma_int_t ldb,
                double vl, double vu, magma_int_t il, magma_int_t iu,
                magma_int_t *m, double *w, double *work, magma_int_t lwork,
                magma_int_t *iwork, magma_int_t liwork, magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    DSYGVD computes all the eigenvalues, and optionally, the eigenvectors
    of a real generalized symmetric-definite eigenproblem, of the form
    A*x=(lambda)*B*x,  A*Bx=(lambda)*x,  or B*A*x=(lambda)*x.  Here A and
    B are assumed to be symmetric and B is also positive definite.
    If eigenvectors are desired, it uses a divide and conquer algorithm.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    =========
    ITYPE   (input) INTEGER
            Specifies the problem type to be solved:
            = 1:  A*x = (lambda)*B*x
            = 2:  A*B*x = (lambda)*x
            = 3:  B*A*x = (lambda)*x

    RANGE   (input) CHARACTER*1
            = 'A': all eigenvalues will be found.
            = 'V': all eigenvalues in the half-open interval (VL,VU]
                   will be found.
            = 'I': the IL-th through IU-th eigenvalues will be found.

    JOBZ    (input) CHARACTER*1
            = 'N':  Compute eigenvalues only;
            = 'V':  Compute eigenvalues and eigenvectors.

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangles of A and B are stored;
            = 'L':  Lower triangles of A and B are stored.

    N       (input) INTEGER
            The order of the matrices A and B.  N >= 0.

    A       (input/output) COMPLEX*16 array, dimension (LDA, N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = 'L',
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.

            On exit, if JOBZ = 'V', then if INFO = 0, A contains the
            matrix Z of eigenvectors.  The eigenvectors are normalized
            as follows:
            if ITYPE = 1 or 2, Z**T*B*Z = I;
            if ITYPE = 3, Z**T*inv(B)*Z = I.
            If JOBZ = 'N', then on exit the upper triangle (if UPLO='U')
            or the lower triangle (if UPLO='L') of A, including the
            diagonal, is destroyed.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    B       (input/output) COMPLEX*16 array, dimension (LDB, N)
            On entry, the symmetric matrix B.  If UPLO = 'U', the
            leading N-by-N upper triangular part of B contains the
            upper triangular part of the matrix B.  If UPLO = 'L',
            the leading N-by-N lower triangular part of B contains
            the lower triangular part of the matrix B.

            On exit, if INFO <= N, the part of B containing the matrix is
            overwritten by the triangular factor U or L from the Cholesky
            factorization B = U**T*U or B = L*L**T.

    LDB     (input) INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).

    VL      (input) DOUBLE PRECISION
    VU      (input) DOUBLE PRECISION
            If RANGE='V', the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = 'A' or 'I'.

    IL      (input) INTEGER
    IU      (input) INTEGER
            If RANGE='I', the indices (in ascending order) of the
            smallest and largest eigenvalues to be returned.
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
            Not referenced if RANGE = 'A' or 'V'.

    M       (output) INTEGER
            The total number of eigenvalues found.  0 <= M <= N.
            If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.

    W       (output) DOUBLE PRECISION array, dimension (N)
            If INFO = 0, the eigenvalues in ascending order.

    WORK    (workspace/output) COMPLEX*16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The length of the array WORK.
            If N <= 1,                LWORK >= 1.
            If JOBZ  = 'N' and N > 1, LWORK >= N + 1.
            If JOBZ  = 'V' and N > 1, LWORK >= 2*N*nb + N**2.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal sizes of the WORK, RWORK and
            IWORK arrays, returns these values as the first entries of
            the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    RWORK   (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LRWORK))
            On exit, if INFO = 0, RWORK(1) returns the optimal LRWORK.

    LRWORK  (input) INTEGER
            The dimension of the array RWORK.
            If N <= 1,                LRWORK >= 1.
            If JOBZ  = 'N' and N > 1, LRWORK >= N.
            If JOBZ  = 'V' and N > 1, LRWORK >= 1 + 5*N + 2*N**2.

            If LRWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
            On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.

    LIWORK  (input) INTEGER
            The dimension of the array IWORK.
            If N <= 1,                LIWORK >= 1.
            If JOBZ  = 'N' and N > 1, LIWORK >= 1.
            If JOBZ  = 'V' and N > 1, LIWORK >= 3 + 5*N.

            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  DPOTRF or DSYEVD returned an error code:
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
    ===============

    Based on contributions by
       Mark Fahey, Department of Mathematics, Univ. of Kentucky, USA

    Modified so that no backsubstitution is performed if DSYEVD fails to
    converge (NEIG in old code could be greater than N causing out of
    bounds reference to A - reported by Ralf Meyer).  Also corrected the
    description of INFO and the test on ITYPE. Sven, 16 Feb 05.
    =====================================================================  */

    char uplo_[2] = {uplo, 0};
    char jobz_[2] = {jobz, 0};
    char range_[2] = {range, 0};
    
    double c_one = MAGMA_D_ONE;
    
    magma_int_t lower;
    char trans[1];
    magma_int_t wantz;
    magma_int_t lquery;
    magma_int_t alleig, valeig, indeig;
    
    magma_int_t lwmin;
    magma_int_t liwmin;
    
    wantz = lapackf77_lsame(jobz_, MagmaVectorsStr);
    lower = lapackf77_lsame(uplo_, MagmaLowerStr);
    alleig = lapackf77_lsame(range_, "A");
    valeig = lapackf77_lsame(range_, "V");
    indeig = lapackf77_lsame(range_, "I");
    lquery = lwork == -1 || liwork == -1;
    
    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (! (alleig || valeig || indeig)) {
        *info = -2;
    } else if (! (wantz || lapackf77_lsame(jobz_, MagmaNoVectorsStr))) {
        *info = -3;
    } else if (! (lower || lapackf77_lsame(uplo_, MagmaUpperStr))) {
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
    
    magma_int_t nb = magma_get_dsytrd_nb( n );
    if ( n <= 1 ) {
        lwmin  = 1;
        liwmin = 1;
    }
    else if ( wantz ) {
        lwmin  = 1 + 6*n + 2*n*n;
        liwmin = 3 + 5*n;
    }
    else {
        lwmin  = 2*n + n*nb;
        liwmin = 1;
    }
    // multiply by 1+eps to ensure length gets rounded up,
    // if it cannot be exactly represented in floating point.
    work[0]  = lwmin * (1. + lapackf77_dlamch("Epsilon"));
    iwork[0] = liwmin;
    
    if (lwork < lwmin && ! lquery) {
        *info = -17;
    } else if (liwork < liwmin && ! lquery) {
        *info = -19;
    }
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info));
        return *info;
    }
    else if (lquery) {
        return *info;
    }
    
    /*  Quick return if possible */
    if (n == 0) {
        return *info;
    }

//#define ENABLE_TIMER
#ifdef ENABLE_TIMER
        magma_timestr_t start, end;

        start = get_current_time();
#endif

    magma_dpotrf_m(nrgpu, uplo_[0], n, b, ldb, info);
    if (*info != 0) {
        *info = n + *info;
        return *info;
    }

#ifdef ENABLE_TIMER
        end = get_current_time();

        printf("time dpotrf = %6.2f\n", GetTimerValue(start,end)/1000.);

        start = get_current_time();
#endif

    /*  Transform problem to standard eigenvalue problem and solve. */
    magma_dsygst_m(nrgpu, itype, uplo_[0], n, a, lda, b, ldb, info);

#ifdef ENABLE_TIMER
        end = get_current_time();

        printf("time dsygst = %6.2f\n", GetTimerValue(start,end)/1000.);

        start = get_current_time();
#endif

    magma_dsyevdx_m(nrgpu, jobz, range, uplo, n, a, lda, vl, vu, il, iu, m, w, work, lwork, iwork, liwork, info);

#ifdef ENABLE_TIMER
        end = get_current_time();

        printf("time dsyevd = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

    if (wantz && *info == 0)
    {

#ifdef ENABLE_TIMER
        start = get_current_time();
#endif

        /* Backtransform eigenvectors to the original problem. */
        if (itype == 1 || itype == 2)
        {
            /* For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
             backtransform eigenvectors: x = inv(L)'*y or inv(U)*y */
            if (lower) {
                *(unsigned char *)trans = MagmaTrans;
            } else {
                *(unsigned char *)trans = MagmaNoTrans;
            }

            magma_dtrsm_m(nrgpu, MagmaLeft, uplo_[0], *trans, MagmaNonUnit,
                          n, *m, c_one, b, ldb, a, lda);

        }
        else if (itype == 3)
        {
            /*  For B*A*x=(lambda)*x;
             backtransform eigenvectors: x = L*y or U'*y */
            if (lower) {
                *(unsigned char *)trans = MagmaNoTrans;
            } else {
                *(unsigned char *)trans = MagmaTrans;
            }

            //magma_dtrmm(MagmaLeft, uplo_[0], *trans, MagmaNonUnit,
            //            n, n, c_one, db, lddb, da, ldda);
        }

#ifdef ENABLE_TIMER
        end = get_current_time();

        printf("time setmatrices trsm/mm + getmatrices = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif
    }

    work[0]  = lwmin * (1. + lapackf77_dlamch("Epsilon"));  // round up
    iwork[0] = liwmin;


    return *info;
} /* magma_dsygvd_m */
