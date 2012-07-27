/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Raffaele Solca
       @author Stan Tomov
       @author Mark Gates

       @precisions normal z -> c

*/
#include "common_magma.h"

// === Define what BLAS to use ============================================

//#define FAST_HEMV

// === End defining what BLAS to use ======================================

extern "C" magma_int_t
magma_zheevd_gpu(char jobz, char uplo,
                 magma_int_t n,
                 cuDoubleComplex *da, magma_int_t ldda,
                 double *w,
                 cuDoubleComplex *wa,  magma_int_t ldwa,
                 cuDoubleComplex *work, magma_int_t lwork,
                 double *rwork, magma_int_t lrwork,
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
    ZHEEVD_GPU computes all eigenvalues and, optionally, eigenvectors of a
    complex Hermitian matrix A.  If eigenvectors are desired, it uses a
    divide and conquer algorithm.

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

    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    DA      (device input/output) COMPLEX_16 array on the GPU,
            dimension (LDDA, N).
            On entry, the Hermitian matrix A.  If UPLO = 'U', the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = 'L',
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
            On exit, if JOBZ = 'V', then if INFO = 0, A contains the
            orthonormal eigenvectors of the matrix A.
            If JOBZ = 'N', then on exit the lower triangle (if UPLO='L')
            or the upper triangle (if UPLO='U') of A, including the
            diagonal, is destroyed.

    LDDA    (input) INTEGER
            The leading dimension of the array DA.  LDDA >= max(1,N).

    W       (output) DOUBLE PRECISION array, dimension (N)
            If INFO = 0, the eigenvalues in ascending order.

    WA      (workspace) COMPLEX_16 array, dimension (LDWA, N)

    LDWA    (input) INTEGER
            The leading dimension of the array WA.  LDWA >= max(1,N).

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.

    LWORK   (input) INTEGER
            The length of the array WORK.
            If N <= 1,                LWORK must be at least 1.
            If JOBZ  = 'N' and N > 1, LWORK must be at least N + N*NB.
            If JOBZ  = 'V' and N > 1, LWORK must be at least 2*N + N**2.
            NB can be obtained through magma_get_zhetrd_nb(N).

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal sizes of the WORK, RWORK and
            IWORK arrays, returns these values as the first entries of
            the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    RWORK   (workspace/output) DOUBLE PRECISION array, dimension (LRWORK)
            On exit, if INFO = 0, RWORK[0] returns the optimal LRWORK.

    LRWORK  (input) INTEGER
            The dimension of the array RWORK.
            If N <= 1,                LRWORK must be at least 1.
            If JOBZ  = 'N' and N > 1, LRWORK must be at least N.
            If JOBZ  = 'V' and N > 1, LRWORK must be at least 1 + 5*N + 2*N**2.

            If LRWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

    IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK))
            On exit, if INFO = 0, IWORK[0] returns the optimal LIWORK.

    LIWORK  (input) INTEGER
            The dimension of the array IWORK.
            If N <= 1,                LIWORK must be at least 1.
            If JOBZ  = 'N' and N > 1, LIWORK must be at least 1.
            If JOBZ  = 'V' and N > 1, LIWORK must be at least 3 + 5*N.

            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal sizes of the WORK, RWORK
            and IWORK arrays, returns these values as the first entries
            of the WORK, RWORK and IWORK arrays, and no error message
            related to LWORK or LRWORK or LIWORK is issued by XERBLA.

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
    magma_int_t ione = 1;

    double d__1;

    double eps;
    magma_int_t inde;
    double anrm;
    magma_int_t imax;
    double rmin, rmax;
    double sigma;
    magma_int_t iinfo, lwmin;
    magma_int_t lower;
    magma_int_t llrwk;
    magma_int_t wantz;
    magma_int_t indwk2, llwrk2;
    magma_int_t iscale;
    double safmin;
    double bignum;
    magma_int_t indtau;
    magma_int_t indrwk, indwrk, liwmin;
    magma_int_t lrwmin, llwork;
    double smlnum;
    magma_int_t lquery;

    double *dwork;
    cuDoubleComplex *dc;
    magma_int_t lddc = ldda;

    wantz = lapackf77_lsame(jobz_, MagmaVectorsStr);
    lower = lapackf77_lsame(uplo_, MagmaLowerStr);
    lquery = lwork == -1 || lrwork == -1 || liwork == -1;

    *info = 0;
    if (! (wantz || lapackf77_lsame(jobz_, MagmaNoVectorsStr))) {
        *info = -1;
    } else if (! (lower || lapackf77_lsame(uplo_, MagmaUpperStr))) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldda < max(1,n)) {
        *info = -5;
    } else if (ldwa < max(1,n)) {
        *info = -8;
    }

    magma_int_t nb = magma_get_zhetrd_nb( n );
    if ( n <= 1 ) {
        lwmin  = 1;
        lrwmin = 1;
        liwmin = 1;
    }
    else if ( wantz ) {
        lwmin  = 2*n + n*n;
        lrwmin = 1 + 5*n + 2*n*n;
        liwmin = 3 + 5*n;
    }
    else {
        lwmin  = n + n*nb;
        lrwmin = n;
        liwmin = 1;
    }
    // multiply by 1+eps to ensure length gets rounded up,
    // if it cannot be exactly represented in floating point.    
    work[0]  = MAGMA_Z_MAKE( lwmin * (1. + dlamch_("Epsilon")), 0.);
    rwork[0] = lrwmin * (1. + dlamch_("Epsilon"));
    iwork[0] = liwmin;

    if ((lwork < lwmin) && !lquery) {
        *info = -10;
    } else if ((lrwork < lrwmin) && ! lquery) {
        *info = -12;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -14;
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
        cuDoubleComplex tmp;
        magma_zgetvector( 1, da, 1, &tmp, 1 );
        w[0] = MAGMA_Z_REAL(tmp);
        if (wantz) {
            tmp = MAGMA_Z_ONE;
            magma_zsetvector( 1, &tmp, 1, da, 1 );
        }
        return *info;
    }

    cudaStream_t stream;
    magma_queue_create( &stream );

    // dc and dwork are never used together, so use one buffer for both;
    // unfortunately they're different types (complex and double).
    // (this works better in dsyevd_gpu where they're both double).
    // n*lddc for zhetrd2_gpu, *2 for complex
    // n for zlanhe
    magma_int_t ldwork = n*lddc*2;
    if ( wantz ) {
        // need 3n^2/2 for zstedx
        ldwork = max( ldwork, 3*n*(n/2 + 1) );
    }
    if (MAGMA_SUCCESS != magma_dmalloc( &dwork, ldwork )) {
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
    }
    dc = (cuDoubleComplex*) dwork;

    /* Get machine constants. */
    safmin = lapackf77_dlamch("Safe minimum");
    eps    = lapackf77_dlamch("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = magma_dsqrt(smlnum);
    rmax = magma_dsqrt(bignum);

    /* Scale matrix to allowable range, if necessary. */
    anrm = magmablas_zlanhe('M', uplo, n, da, ldda, dwork);
    iscale = 0;
    if (anrm > 0. && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        magmablas_zlascl(uplo, 0, 0, 1., sigma, n, n, da, ldda, info);
    }

    /* Call ZHETRD to reduce Hermitian matrix to tridiagonal form. */
    // zhetrd rwork: e (n)
    // zstedx rwork: e (n) + llrwk (1 + 4*N + 2*N**2)  ==>  1 + 5n + 2n^2
    inde   = 0;
    indrwk = inde + n;
    llrwk  = lrwork - indrwk;
    
    // zhetrd work: tau (n) + llwork (n*nb)  ==>  n + n*nb
    // zstedx work: tau (n) + z (n^2)
    // zunmtr work: tau (n) + z (n^2) + llwrk2 (n or n*nb)  ==>  2n + n^2, or n + n*nb + n^2
    indtau = 0;
    indwrk = indtau + n;
    indwk2 = indwrk + n*n;
    llwork = lwork - indwrk;
    llwrk2 = lwork - indwk2;

//#define ENABLE_TIMER
#ifdef ENABLE_TIMER
    magma_timestr_t start, end;
    start = get_current_time();
#endif

#ifdef FAST_HEMV
    magma_zhetrd2_gpu(uplo, n, da, ldda, w, &rwork[inde],
                      &work[indtau], wa, ldwa, &work[indwrk], llwork,
                      dc, n*lddc, &iinfo);
#else
    magma_zhetrd_gpu (uplo, n, da, ldda, w, &rwork[inde],
                      &work[indtau], wa, ldwa, &work[indwrk], llwork, &iinfo);
#endif

#ifdef ENABLE_TIMER
    end = get_current_time();
    printf("time zhetrd_gpu = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

    /* For eigenvalues only, call DSTERF.  For eigenvectors, first call
       ZSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
       tridiagonal matrix, then call ZUNMTR to multiply it to the Householder
       transformations represented as Householder vectors in A. */
    if (! wantz) {
        lapackf77_dsterf(&n, w, &rwork[inde], info);
    } else {

#ifdef ENABLE_TIMER
        start = get_current_time();
#endif

        magma_zstedx('A', n, 0., 0., 0, 0, w, &rwork[inde],
                      &work[indwrk], n, &rwork[indrwk],
                      llrwk, iwork, liwork, dwork, info);

#ifdef ENABLE_TIMER
        end = get_current_time();
        printf("time zstedx = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

        magma_zsetmatrix( n, n, &work[indwrk], n, dc, lddc );

#ifdef ENABLE_TIMER
        start = get_current_time();
#endif

        magma_zunmtr_gpu(MagmaLeft, uplo, MagmaNoTrans, n, n, da, ldda, &work[indtau],
                         dc, lddc, wa, ldwa, &iinfo);

        magma_zcopymatrix( n, n, dc, lddc, da, ldda );
        
#ifdef ENABLE_TIMER
        end = get_current_time();
        printf("time zunmtr_gpu + copy = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        d__1 = 1. / sigma;
        blasf77_dscal(&imax, &d__1, w, &ione);
    }

    work[0]  = MAGMA_Z_MAKE( lwmin * (1. + dlamch_("Epsilon")), 0.);  // round up
    rwork[0] = lrwmin * (1. + dlamch_("Epsilon"));
    iwork[0] = liwmin;

    magma_queue_destroy( stream );
    magma_free( dwork );

    return *info;
} /* magma_zheevd_gpu */
