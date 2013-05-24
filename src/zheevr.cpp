/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Raffaele Solca

       @precisions normal z -> c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zheevr(char jobz, char range, char uplo, magma_int_t n,
             magmaDoubleComplex *a, magma_int_t lda, double vl, double vu,
             magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m,
             double *w, magmaDoubleComplex *z, magma_int_t ldz, magma_int_t *isuppz,
             magmaDoubleComplex *work, magma_int_t lwork,
             double *rwork, magma_int_t lrwork, magma_int_t *iwork,
             magma_int_t liwork, magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    ZHEEVR computes selected eigenvalues and, optionally, eigenvectors
    of a complex Hermitian matrix T.  Eigenvalues and eigenvectors can
    be selected by specifying either a range of values or a range of
    indices for the desired eigenvalues.

    Whenever possible, ZHEEVR calls ZSTEGR to compute the
    eigenspectrum using Relatively Robust Representations.  ZSTEGR
    computes eigenvalues by the dqds algorithm, while orthogonal
    eigenvectors are computed from various "good" L D L^T representations
    (also known as Relatively Robust Representations). Gram-Schmidt
    orthogonalization is avoided as far as possible. More specifically,
    the various steps of the algorithm are as follows. For the i-th
    unreduced block of T,
       (a) Compute T - sigma_i = L_i D_i L_i^T, such that L_i D_i L_i^T
            is a relatively robust representation,
       (b) Compute the eigenvalues, lambda_j, of L_i D_i L_i^T to high
           relative accuracy by the dqds algorithm,
       (c) If there is a cluster of close eigenvalues, "choose" sigma_i
           close to the cluster, and go to step (a),
       (d) Given the approximate eigenvalue lambda_j of L_i D_i L_i^T,
           compute the corresponding eigenvector by forming a
           rank-revealing twisted factorization.
    The desired accuracy of the output can be specified by the input
    parameter ABSTOL.

    For more details, see "A new O(n^2) algorithm for the symmetric
    tridiagonal eigenvalue/eigenvector problem", by Inderjit Dhillon,
    Computer Science Division Technical Report No. UCB//CSD-97-971,
    UC Berkeley, May 1997.


    Note 1 : ZHEEVR calls ZSTEGR when the full spectrum is requested
    on machines which conform to the ieee-754 floating point standard.
    ZHEEVR calls DSTEBZ and ZSTEIN on non-ieee machines and
    when partial spectrum requests are made.

    Normal execution of ZSTEGR may create NaNs and infinities and
    hence may abort due to a floating point exception in environments
    which do not handle NaNs and infinities in the ieee standard default
    manner.

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

    A       (input/output) COMPLEX_16 array, dimension (LDA, N)
            On entry, the Hermitian matrix A.  If UPLO = 'U', the
            leading N-by-N upper triangular part of A contains the
            upper triangular part of the matrix A.  If UPLO = 'L',
            the leading N-by-N lower triangular part of A contains
            the lower triangular part of the matrix A.
            On exit, the lower triangle (if UPLO='L') or the upper
            triangle (if UPLO='U') of A, including the diagonal, is
            destroyed.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

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

    ABSTOL  (input) DOUBLE PRECISION
            The absolute error tolerance for the eigenvalues.
            An approximate eigenvalue is accepted as converged
            when it is determined to lie in an interval [a,b]
            of width less than or equal to

                    ABSTOL + EPS *   max( |a|,|b| ) ,

            where EPS is the machine precision.  If ABSTOL is less than
            or equal to zero, then  EPS*|T|  will be used in its place,
            where |T| is the 1-norm of the tridiagonal matrix obtained
            by reducing A to tridiagonal form.

            See "Computing Small Singular Values of Bidiagonal Matrices
            with Guaranteed High Relative Accuracy," by Demmel and
            Kahan, LAPACK Working Note #3.

            If high relative accuracy is important, set ABSTOL to
            DLAMCH( 'Safe minimum' ).  Doing so will guarantee that
            eigenvalues are computed to high relative accuracy when
            possible in future releases.  The current code does not
            make any guarantees about high relative accuracy, but
            furutre releases will. See J. Barlow and J. Demmel,
            "Computing Accurate Eigensystems of Scaled Diagonally
            Dominant Matrices", LAPACK Working Note #7, for a discussion
            of which matrices define their eigenvalues to high relative
            accuracy.

    M       (output) INTEGER
            The total number of eigenvalues found.  0 <= M <= N.
            If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.

    W       (output) DOUBLE PRECISION array, dimension (N)
            The first M elements contain the selected eigenvalues in
            ascending order.

    Z       (output) COMPLEX_16 array, dimension (LDZ, max(1,M))
            If JOBZ = 'V', then if INFO = 0, the first M columns of Z
            contain the orthonormal eigenvectors of the matrix A
            corresponding to the selected eigenvalues, with the i-th
            column of Z holding the eigenvector associated with W(i).
            If JOBZ = 'N', then Z is not referenced.
            Note: the user must ensure that at least max(1,M) columns are
            supplied in the array Z; if RANGE = 'V', the exact value of M
            is not known in advance and an upper bound must be used.

    LDZ     (input) INTEGER
            The leading dimension of the array Z.  LDZ >= 1, and if
            JOBZ = 'V', LDZ >= max(1,N).

    ISUPPZ  (output) INTEGER ARRAY, dimension ( 2*max(1,M) )
            The support of the eigenvectors in Z, i.e., the indices
            indicating the nonzero elements in Z. The i-th eigenvector
            is nonzero only in elements ISUPPZ( 2*i-1 ) through
            ISUPPZ( 2*i ).
   ********* Implemented only for RANGE = 'A' or 'I' and IU - IL = N - 1

    WORK    (workspace/output) COMPLEX_16 array, dimension (LWORK)
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.

    LWORK   (input) INTEGER
            The length of the array WORK.  LWORK >= max(1,2*N).
            For optimal efficiency, LWORK >= (NB+1)*N,
            where NB is the max of the blocksize for ZHETRD and for
            ZUNMTR as returned by ILAENV.

            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued by XERBLA.

    RWORK   (workspace/output) DOUBLE PRECISION array, dimension (LRWORK)
            On exit, if INFO = 0, RWORK(1) returns the optimal
            (and minimal) LRWORK.

    LRWORK  (input) INTEGER
            The length of the array RWORK.  LRWORK >= max(1,24*N).

            If LRWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the RWORK array, returns
            this value as the first entry of the RWORK array, and no error
            message related to LRWORK is issued by XERBLA.

    IWORK   (workspace/output) INTEGER array, dimension (LIWORK)
            On exit, if INFO = 0, IWORK(1) returns the optimal
            (and minimal) LIWORK.

    LIWORK  (input) INTEGER
            The dimension of the array IWORK.  LIWORK >= max(1,10*N).

            If LIWORK = -1, then a workspace query is assumed; the
            routine only calculates the optimal size of the IWORK array,
            returns this value as the first entry of the IWORK array, and
            no error message related to LIWORK is issued by XERBLA.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  Internal error

    Further Details
    ===============
    Based on contributions by
       Inderjit Dhillon, IBM Almaden, USA
       Osni Marques, LBNL/NERSC, USA
       Ken Stanley, Computer Science Division, University of
         California at Berkeley, USA
    =====================================================================     */
    
    char uplo_[2] = {uplo, 0};
    char jobz_[2] = {jobz, 0};
    char range_[2] = {range, 0};
    
    magma_int_t izero = 0;
    magma_int_t ione = 1;
    float szero = 0.;
    float sone = 1.;
    
    magma_int_t indrd, indre;
    magma_int_t imax;
    magma_int_t lopt, itmp1, indree, indrdd;
    magma_int_t lower, wantz, tryrac;
    magma_int_t i, j, jj, i__1;
    magma_int_t alleig, valeig, indeig;
    magma_int_t iscale, indibl, indifl;
    magma_int_t indiwo, indisp, indtau;
    magma_int_t indrwk, indwk;
    magma_int_t llwork, llrwork, nsplit;
    magma_int_t lquery, ieeeok;
    magma_int_t iinfo;
    magma_int_t lwmin, lrwmin, liwmin;
    double safmin;
    double bignum;
    double smlnum;
    double eps, tmp1;
    double anrm;
    double sigma, d__1;
    double rmin, rmax;
    
    lower = lapackf77_lsame(uplo_, MagmaLowerStr);
    wantz = lapackf77_lsame(jobz_, MagmaVecStr);
    alleig = lapackf77_lsame(range_, "A");
    valeig = lapackf77_lsame(range_, "V");
    indeig = lapackf77_lsame(range_, "I");
    lquery = lwork == -1 || lrwork == -1 || liwork == -1;
    
    *info = 0;
    if (! (wantz || lapackf77_lsame(jobz_, MagmaNoVecStr))) {
        *info = -1;
    } else if (! (alleig || valeig || indeig)) {
        *info = -2;
    } else if (! (lower || lapackf77_lsame(uplo_, MagmaUpperStr))) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < max(1,n)) {
        *info = -6;
    } else if (ldz < 1 || wantz && ldz < n) {
        *info = -15;
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
    
    magma_int_t nb = magma_get_zhetrd_nb(n);
    
    lwmin =  n * (nb + 1);
    lrwmin = 24 * n;
    liwmin = 10 * n;
    
    MAGMA_Z_SET2REAL(work[0],(double)lwmin);
    rwork[0] = lrwmin;
    iwork[0] = liwmin;
    
    if (lwork < lwmin && ! lquery) {
        *info = -18;
    } else if ((lrwork < lrwmin) && ! lquery) {
        *info = -20;
    } else if ((liwork < liwmin) && ! lquery) {
        *info = -22;
    }
    
    if (*info != 0) {
        magma_xerbla(__func__, -(*info));
        return *info;
    } else if (lquery) {
        return *info;
    }
    
    /* Quick return if possible */
    *m = 0;
    if (n == 0) {
        return *info;
    }
    
    if (n == 1) {
        w[0] = MAGMA_Z_REAL(a[0]);
        if (alleig || indeig) {
            *m = 1;
        } else if (valeig) {
            if (vl < w[0] && vu >= w[0]) {
                *m = 1;
            }
        }
        if (wantz) {
            z[0]=MAGMA_Z_ONE;
        }
        return *info;
    }
    
    --w;
    --work;
    --rwork;
    --iwork;
    --isuppz;
    
    /* Get machine constants. */
    safmin = lapackf77_dlamch("Safe minimum");
    eps = lapackf77_dlamch("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = magma_dsqrt(smlnum);
    rmax = magma_dsqrt(bignum);
    
    /* Scale matrix to allowable range, if necessary. */
    anrm = lapackf77_zlanhe("M", uplo_, &n, a, &lda, &rwork[1]);
    iscale = 0;
    if (anrm > 0. && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        d__1 = 1.;
        lapackf77_zlascl(uplo_, &izero, &izero, &d__1, &sigma, &n, &n, a,
                         &lda, info);
        
        if (abstol > 0.) {
            abstol *= sigma;
        }
        if (valeig) {
            vl *= sigma;
            vu *= sigma;
        }
    }
    
    /* Call ZHETRD to reduce Hermitian matrix to tridiagonal form. */
    indtau = 1;
    indwk = indtau + n;
    
    indre = 1;
    indrd = indre + n;
    indree = indrd + n;
    indrdd = indree + n;
    indrwk = indrdd + n;
    llwork = lwork - indwk + 1;
    llrwork = lrwork - indrwk + 1;
    
    indifl = 1;
    indibl = indifl + n;
    indisp = indibl + n;
    indiwo = indisp + n;
    
    magma_zhetrd(uplo, n, a, lda, &rwork[indrd], &rwork[indre], &work[indtau], &work[indwk], llwork, &iinfo);
    
    lopt = n + (magma_int_t)MAGMA_Z_REAL(work[indwk]);
    
    /* If all eigenvalues are desired and ABSTOL is less than or equal to
       zero, then call DSTERF
       or ZUNGTR and ZSTEQR.  If this fails for
       some eigenvalue, then try DSTEBZ. */
    ieeeok = lapackf77_ieeeck( &ione, &szero, &sone);
    
    /* If only the eigenvalues are required call DSTERF for all or DSTEBZ for a part */
    if (! wantz) {
        blasf77_dcopy(&n, &rwork[indrd], &ione, &w[1], &ione);
        i__1 = n - 1;
        if ((alleig || indeig && il == 1 && iu == n)){
            lapackf77_dsterf(&n, &w[1], &rwork[indre], info);
            *m = n;
        } else {
            lapackf77_dstebz(range_, "E", &n, &vl, &vu, &il, &iu, &abstol,
                             &rwork[indrd], &rwork[indre], m,
                             &nsplit, &w[1], &iwork[indibl], &iwork[indisp],
                             &rwork[indrwk], &iwork[indiwo], info);
        }
        
        /* Otherwise call ZSTEMR if infinite and NaN arithmetic is supported */
    }
    else if (ieeeok==1){
        i__1 = n - 1;
        
        blasf77_dcopy(&i__1, &rwork[indre], &ione, &rwork[indree], &ione);
        blasf77_dcopy(&n, &rwork[indrd], &ione, &rwork[indrdd], &ione);
        
        if (abstol < 2*n*eps)
            tryrac=1;
        else
            tryrac=0;
        
        lapackf77_zstemr(jobz_, range_, &n, &rwork[indrdd], &rwork[indree], &vl, &vu, &il,
                         &iu, m, &w[1], z, &ldz, &n, &isuppz[1], &tryrac, &rwork[indrwk],
                         &llrwork, &iwork[1], &liwork, info);
        
        if (*info == 0 && wantz) {
            magma_zunmtr(MagmaLeft, uplo, MagmaNoTrans, n, *m, a, lda, &work[indtau],
                         z, ldz, &work[indwk], llwork, &iinfo);
        }
    }
    
    
    /* Call DSTEBZ and ZSTEIN if infinite and NaN arithmetic is not supported or ZSTEMR didn't converge. */
    if (wantz && (ieeeok ==0 || *info != 0)) {
        *info = 0;
        
        lapackf77_dstebz(range_, "B", &n, &vl, &vu, &il, &iu, &abstol, &rwork[indrd], &rwork[indre], m,
                         &nsplit, &w[1], &iwork[indibl], &iwork[indisp], &rwork[indrwk], &iwork[indiwo], info);
        
        lapackf77_zstein(&n, &rwork[indrd], &rwork[indre], m, &w[1], &iwork[indibl], &iwork[indisp],
                         z, &ldz, &rwork[indrwk], &iwork[indiwo], &iwork[indifl], info);
        
        /* Apply unitary matrix used in reduction to tridiagonal
           form to eigenvectors returned by ZSTEIN. */
        magma_zunmtr(MagmaLeft, uplo, MagmaNoTrans, n, *m, a, lda, &work[indtau],
                     z, ldz, &work[indwk], llwork, &iinfo);
    }
    
    /* If matrix was scaled, then rescale eigenvalues appropriately. */
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        d__1 = 1. / sigma;
        blasf77_dscal(&imax, &d__1, &w[1], &ione);
    }
    
    /* If eigenvalues are not in order, then sort them, along with
       eigenvectors. */
    if (wantz) {
        for (j = 1; j <= *m-1; ++j) {
            i = 0;
            tmp1 = w[j];
            for (jj = j + 1; jj <= *m; ++jj) {
                if (w[jj] < tmp1) {
                    i = jj;
                    tmp1 = w[jj];
                }
            }
            
            if (i != 0) {
                itmp1 = iwork[indibl + i - 1];
                w[i] = w[j];
                iwork[indibl + i - 1] = iwork[indibl + j - 1];
                w[j] = tmp1;
                iwork[indibl + j - 1] = itmp1;
                blasf77_zswap(&n, z + (i-1)*ldz, &ione, z + (j-1)*ldz, &ione);
            }
        }
    }
    
    /* Set WORK(1) to optimal complex workspace size. */
    work[1] = MAGMA_Z_MAKE((double) lopt, 0.);
    rwork[1] = (double) lrwmin;
    iwork[1] = liwmin;
    
    return *info;
} /* magma_zheevr */
