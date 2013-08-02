/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Stan Tomov
       @author Raffaele Solca
       @author Mark Gates
       @author Azzam Haidar

       @precisions normal d -> s

*/
#include "common_magma.h"


extern "C" magma_int_t
magma_dsyevdx_gpu(char jobz, char range, char uplo,
                  magma_int_t n,
                  double *da, magma_int_t ldda,
                  double vl, double vu, magma_int_t il, magma_int_t iu,
                  magma_int_t *m, double *w,
                  double *wa,  magma_int_t ldwa,
                  double *work, magma_int_t lwork,
                  magma_int_t *iwork, magma_int_t liwork,
                  magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    DSYEVDX computes selected eigenvalues and, optionally, eigenvectors
    of a real symmetric matrix A. Eigenvalues and eigenvectors can
    be selected by specifying either a range of values or a range of
    indices for the desired eigenvalues.
    If eigenvectors are desired, it uses a divide and conquer algorithm.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    =========
    JOBZ    (input) CHARACTER*1
            = 'N':  Compute eigenvalues only;
            = 'V':  Compute eigenvalues and eigenvectors.

    RANGE   (input) CHARACTER*1
            = 'A': all eigenvalues will be found.
            = 'V': all eigenvalues in the half-open interval (VL,VU]
                   will be found.
            = 'I': the IL-th through IU-th eigenvalues will be found.

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    DA      (device input/output) DOUBLE_PRECISION array on the GPU,
            dimension (LDDA, N).
            On entry, the symmetric matrix A.  If UPLO = 'U', the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = 'L',
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
            On exit, if JOBZ = 'V', then if INFO = 0, the first m columns
            of A contains the required
            orthonormal eigenvectors of the matrix A.
            If JOBZ = 'N', then on exit the lower triangle (if UPLO='L')
            or the upper triangle (if UPLO='U') of A, including the
            diagonal, is destroyed.

    LDDA    (input) INTEGER
            The leading dimension of the array DA.  LDDA >= max(1,N).

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
            If INFO = 0, the required m eigenvalues in ascending order.

    WA      (workspace) DOUBLE PRECISION array, dimension (LDWA, N)

    LDWA    (input) INTEGER
            The leading dimension of the array WA.  LDWA >= max(1,N).

    WORK    (workspace/output) DOUBLE_PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    LWORK   (input) INTEGER
            The length of the array WORK.
            If N <= 1,                LWORK >= 1.
            If JOBZ  = 'N' and N > 1, LWORK >= 2*N + N*NB.
            If JOBZ  = 'V' and N > 1, LWORK >= max( 2*N + N*NB, 1 + 6*N + 2*N**2 ).
            NB can be obtained through magma_get_dsytrd_nb(N).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal sizes of the WORK and IWORK
            arrays, returns these values as the first entries of the WORK
            and IWORK arrays, and no error message related to LWORK or
            LIWORK is issued by XERBLA.

    IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
            On exit, if INFO = 0, IWORK[0] returns the optimal LIWORK.

    LIWORK  (input) INTEGER
            The dimension of the array IWORK.
            If N <= 1,                LIWORK >= 1.
            If JOBZ  = 'N' and N > 1, LIWORK >= 1.
            If JOBZ  = 'V' and N > 1, LIWORK >= 3 + 5*N.

            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK and
            IWORK arrays, returns these values as the first entries of
            the WORK and IWORK arrays, and no error message related to
            LWORK or LIWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i and JOBZ = 'N', then the algorithm failed
                  to converge; i off-diagonal elements of an intermediate
                  tridiagonal form did not converge to zero;
                  if INFO = i and JOBZ = 'V', then the algorithm failed
                  to compute an eigenvalue while working on the submatrix
                  lying in rows and columns INFO/(N+1) through
                  mod(INFO,N+1).

    Further Details
    ===============
    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA

    Modified description of INFO. Sven, 16 Feb 05.
    =====================================================================   */

    char uplo_[2] = {uplo, 0};
    char jobz_[2] = {jobz, 0};
    char range_[2] = {range, 0};
    magma_int_t ione = 1;

    double d__1;

    double eps;
    magma_int_t inde;
    double anrm;
    double rmin, rmax;
    double sigma;
    magma_int_t iinfo, lwmin;
    magma_int_t lower;
    magma_int_t wantz;
    magma_int_t indwk2, llwrk2;
    magma_int_t iscale;
    double safmin;
    double bignum;
    magma_int_t indtau;
    magma_int_t indwrk, liwmin;
    magma_int_t llwork;
    double smlnum;
    magma_int_t lquery;
    magma_int_t alleig, valeig, indeig;

    double *dwork;
    magma_int_t lddc = ldda;

    wantz = lapackf77_lsame(jobz_, MagmaVecStr);
    lower = lapackf77_lsame(uplo_, MagmaLowerStr);

    alleig = lapackf77_lsame( range_, "A" );
    valeig = lapackf77_lsame( range_, "V" );
    indeig = lapackf77_lsame( range_, "I" );

    lquery = lwork == -1 || liwork == -1;

    *info = 0;
    if (! (wantz || lapackf77_lsame(jobz_, MagmaNoVecStr))) {
        *info = -1;
    } else if (! (alleig || valeig || indeig)) {
        *info = -2;
    } else if (! (lower || lapackf77_lsame(uplo_, MagmaUpperStr))) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (ldda < max(1,n)) {
        *info = -6;
    } else if (ldwa < max(1,n)) {
        *info = -14;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -8;
            }
        } else if (indeig) {
            if (il < 1 || il > max(1,n)) {
                *info = -9;
            } else if (iu < min(n,il) || iu > n) {
                *info = -10;
            }
        }
    }

    magma_int_t nb = magma_get_dsytrd_nb( n );
    if ( n <= 1 ) {
        lwmin  = 1;
        liwmin = 1;
    }
    else if ( wantz ) {
        lwmin  = max( 2*n + n*nb, 1 + 6*n + 2*n*n );
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

    if ((lwork < lwmin) && !lquery) {
        *info = -16;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -18;
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

    if (n == 1) {
        double tmp;
        magma_dgetvector( 1, da, 1, &tmp, 1 );
        w[0] = tmp;
        if (wantz) {
            tmp = 1.;
            magma_dsetvector( 1, &tmp, 1, da, 1 );
        }
        return *info;
    }

    magma_queue_t stream;
    magma_queue_create( &stream );

    // n*lddc for dsytrd2_gpu
    // n for dlansy
    magma_int_t ldwork = n*lddc;
    if ( wantz ) {
        // need 3n^2/2 for dstedx
        ldwork = max( ldwork, 3*n*(n/2 + 1));
    }
    if (MAGMA_SUCCESS != magma_dmalloc( &dwork, ldwork )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }

    /* Get machine constants. */
    safmin = lapackf77_dlamch("Safe minimum");
    eps    = lapackf77_dlamch("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = magma_dsqrt(smlnum);
    rmax = magma_dsqrt(bignum);

    /* Scale matrix to allowable range, if necessary. */
    anrm = magmablas_dlansy('M', uplo, n, da, ldda, dwork);
    iscale = 0;
    sigma  = 1;
    if (anrm > 0. && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        magmablas_dlascl(uplo, 0, 0, 1., sigma, n, n, da, ldda, info);
    }

    /* Call DSYTRD to reduce symmetric matrix to tridiagonal form. */
    // dsytrd work: e (n) + tau (n) + llwork (n*nb)  ==>  2n + n*nb
    // dstedx work: e (n) + tau (n) + z (n*n) + llwrk2 (1 + 4*n + n^2)  ==>  1 + 6n + 2n^2
    inde   = 0;
    indtau = inde   + n;
    indwrk = indtau + n;
    indwk2 = indwrk + n*n;
    llwork = lwork - indwrk;
    llwrk2 = lwork - indwk2;

//
#ifdef ENABLE_TIMER
    magma_timestr_t start, end;
    start = get_current_time();
#endif

#ifdef FAST_SYMV
    magma_dsytrd2_gpu(uplo, n, da, ldda, w, &work[inde],
                      &work[indtau], wa, ldwa, &work[indwrk], llwork,
                      dwork, n*lddc, &iinfo);
#else
    magma_dsytrd_gpu(uplo, n, da, ldda, w, &work[inde],
                     &work[indtau], wa, ldwa, &work[indwrk], llwork,
                     &iinfo);
#endif

#ifdef ENABLE_TIMER
    end = get_current_time();
    printf("time dsytrd = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

    /* For eigenvalues only, call DSTERF.  For eigenvectors, first call
       DSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
       tridiagonal matrix, then call DORMTR to multiply it to the Householder
       transformations represented as Householder vectors in A. */

    if (! wantz) {
        lapackf77_dsterf(&n, w, &work[inde], info);

        magma_dmove_eig(range, n, w, &il, &iu, vl, vu, m);

    } else {

#ifdef ENABLE_TIMER
        start = get_current_time();
#endif

        magma_dstedx(range, n, vl, vu, il, iu, w, &work[inde],
                     &work[indwrk], n, &work[indwk2],
                     llwrk2, iwork, liwork, dwork, info);

#ifdef ENABLE_TIMER
        end = get_current_time();
        printf("time dstedx = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

#ifdef ENABLE_TIMER
        start = get_current_time();
#endif

        magma_dmove_eig(range, n, w, &il, &iu, vl, vu, m);

        magma_dsetmatrix( n, *m, &work[indwrk + n* (il-1) ], n, dwork, lddc );

        magma_dormtr_gpu(MagmaLeft, uplo, MagmaNoTrans, n, *m, da, ldda, &work[indtau],
                         dwork, lddc, wa, ldwa, &iinfo);

        magma_dcopymatrix( n, *m, dwork, lddc, da, ldda );

#ifdef ENABLE_TIMER
        end = get_current_time();
        printf("time dormtr + copy = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
    if (iscale == 1) {
        d__1 = 1. / sigma;
        blasf77_dscal(&n, &d__1, w, &ione);
    }

    work[0]  = lwmin * (1. + lapackf77_dlamch("Epsilon"));  // round up
    iwork[0] = liwmin;

    magma_queue_destroy( stream );
    magma_free( dwork );

    return *info;
} /* magma_dsyevd_gpu */
