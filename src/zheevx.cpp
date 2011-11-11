/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
 
       @author Raffaele Solca

       @precisions normal z -> c
       
 */
#include "common_magma.h"

extern "C" magma_int_t 
magma_zheevx(char jobz, char range, char uplo, magma_int_t n, 
             cuDoubleComplex *a, magma_int_t lda, double vl, double vu, 
             magma_int_t il, magma_int_t iu, double abstol, magma_int_t *m, 
             double *w, cuDoubleComplex *z, magma_int_t ldz, cuDoubleComplex *work, magma_int_t lwork,
             double *rwork, magma_int_t *iwork, magma_int_t *ifail, magma_int_t *info)
{
/*  -- MAGMA (version 1.0) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    November 2010
   
    Purpose   
    =======   
    ZHEEVX computes selected eigenvalues and, optionally, eigenvectors   
    of a complex Hermitian matrix A.  Eigenvalues and eigenvectors can   
    be selected by specifying either a range of values or a range of   
    indices for the desired eigenvalues.   

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

    A       (input/output) COMPLEX*16 array, dimension (LDA, N)   
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

            Eigenvalues will be computed most accurately when ABSTOL is   
            set to twice the underflow threshold 2*DLAMCH('S'), not zero.   
            If this routine returns with INFO>0, indicating that some   
            eigenvectors did not converge, try setting ABSTOL to   
            2*DLAMCH('S').   

            See "Computing Small Singular Values of Bidiagonal Matrices   
            with Guaranteed High Relative Accuracy," by Demmel and   
            Kahan, LAPACK Working Note #3.   

    M       (output) INTEGER   
            The total number of eigenvalues found.  0 <= M <= N.   
            If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.   

    W       (output) DOUBLE PRECISION array, dimension (N)   
            On normal exit, the first M elements contain the selected   
            eigenvalues in ascending order.   

    Z       (output) COMPLEX*16 array, dimension (LDZ, max(1,M))   
            If JOBZ = 'V', then if INFO = 0, the first M columns of Z   
            contain the orthonormal eigenvectors of the matrix A   
            corresponding to the selected eigenvalues, with the i-th   
            column of Z holding the eigenvector associated with W(i).   
            If an eigenvector fails to converge, then that column of Z   
            contains the latest approximation to the eigenvector, and the   
            index of the eigenvector is returned in IFAIL.   
            If JOBZ = 'N', then Z is not referenced.   
            Note: the user must ensure that at least max(1,M) columns are   
            supplied in the array Z; if RANGE = 'V', the exact value of M   
            is not known in advance and an upper bound must be used.   

    LDZ     (input) INTEGER   
            The leading dimension of the array Z.  LDZ >= 1, and if   
            JOBZ = 'V', LDZ >= max(1,N).   

    WORK    (workspace/output) COMPLEX*16 array, dimension (LWORK)   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The length of the array WORK.  LWORK >= max(1,2*N-1).   
            For optimal efficiency, LWORK >= (NB+1)*N,   
            where NB is the max of the blocksize for ZHETRD and for   
            ZUNMTR as returned by ILAENV.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    RWORK   (workspace) DOUBLE PRECISION array, dimension (7*N)   

    IWORK   (workspace) INTEGER array, dimension (5*N)   

    IFAIL   (output) INTEGER array, dimension (N)   
            If JOBZ = 'V', then if INFO = 0, the first M elements of   
            IFAIL are zero.  If INFO > 0, then IFAIL contains the   
            indices of the eigenvectors that failed to converge.   
            If JOBZ = 'N', then IFAIL is not referenced.   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, then i eigenvectors failed to converge.   
                  Their indices are stored in array IFAIL.     
   
*/
  
   
  char uplo_[2] = {uplo, 0};
  char jobz_[2] = {jobz, 0};
  char range_[2] = {range, 0};

  static magma_int_t izero = 0;
  static magma_int_t ione = 1;
  
  static char order[1];
  static magma_int_t indd, inde;
  static magma_int_t imax;
  static magma_int_t lopt, itmp1, indee;
  static magma_int_t lower, wantz;
  static magma_int_t i, j, jj, i__1;
  static magma_int_t alleig, valeig, indeig;
  static magma_int_t iscale, indibl;
  static magma_int_t indiwk, indisp, indtau;
  static magma_int_t indrwk, indwrk;
  static magma_int_t llwork, nsplit;
  static magma_int_t lquery;
  static magma_int_t iinfo;
  static double safmin;
  static double bignum;
  static double smlnum;
  static double eps, tmp1;
  static double anrm;
  static double sigma, d__1;
  static double rmin, rmax;

  
  /* Function Body */
  lower = lapackf77_lsame(uplo_, MagmaLowerStr);
  wantz = lapackf77_lsame(jobz_, MagmaVectorsStr);
  alleig = lapackf77_lsame(range_, "A");
  valeig = lapackf77_lsame(range_, "V");
  indeig = lapackf77_lsame(range_, "I");
  lquery = lwork == -1;
  
  *info = 0;
  if (! (wantz || lapackf77_lsame(jobz_, MagmaNoVectorsStr))) {
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
  
  lopt = n * (nb + 1);
  
  MAGMA_Z_SET2REAL(work[0],(double)lopt);
  
  
  if (lwork < lopt && ! lquery) {
    *info = -17;
  }
  
  if (*info != 0) {
      magma_xerbla( __func__, -(*info));
      return MAGMA_ERR_ILLEGAL_VALUE;
  } else if (lquery) {
      return MAGMA_SUCCESS;
  }
  
  /* Quick return if possible */
  *m = 0;
  if (n == 0) {
    return MAGMA_SUCCESS;
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
    return MAGMA_SUCCESS;
  }

  --w;
  --work;
  --rwork;
  --iwork;
  --ifail; 
  
  /*     Get machine constants. */
  safmin = lapackf77_dlamch("Safe minimum");
  eps = lapackf77_dlamch("Precision");
  smlnum = safmin / eps;
  bignum = 1. / smlnum;
  rmin = magma_dsqrt(smlnum);
  rmax = magma_dsqrt(bignum);
  
  /*     Scale matrix to allowable range, if necessary. */
  

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
  
  /*     Call ZHETRD to reduce Hermitian matrix to tridiagonal form. */
  
  indd = 1;
  inde = indd + n;
  indrwk = inde + n;
  indtau = 1;
  indwrk = indtau + n;
  llwork = lwork - indwrk + 1;

  magma_zhetrd(uplo, n, a, lda, &rwork[indd], &rwork[inde], &work[indtau], &work[indwrk], llwork, &iinfo);

  lopt = n + (magma_int_t)MAGMA_Z_REAL(work[indwrk]);

  
  /*     If all eigenvalues are desired and ABSTOL is less than or equal to   
   zero, then call DSTERF or ZUNGTR and ZSTEQR.  If this fails for   
   some eigenvalue, then try DSTEBZ. */
  
  if ((alleig || indeig && il == 1 && iu == n) && abstol <= 0.) {
    blasf77_dcopy(&n, &rwork[indd], &ione, &w[1], &ione);
    indee = indrwk + 2*n;
    if (! wantz) {
      i__1 = n - 1;
      blasf77_dcopy(&i__1, &rwork[inde], &ione, &rwork[indee], &ione);
      lapackf77_dsterf(&n, &w[1], &rwork[indee], info);
    } else {
      lapackf77_zlacpy("A", &n, &n, a, &lda, z, &ldz);
      lapackf77_zungtr(uplo_, &n, z, &ldz, &work[indtau], &work[indwrk], &llwork, &iinfo);
      i__1 = n - 1;
      blasf77_dcopy(&i__1, &rwork[inde], &ione, &rwork[indee], &ione);
      lapackf77_zsteqr(jobz_, &n, &w[1], &rwork[indee], z, &ldz, &rwork[indrwk], info);
      if (*info == 0) {
        for (i = 1; i <= n; ++i) {
          ifail[i] = 0;
        }
      }
    }
    if (*info == 0) {
      *m = n;
    }
  }

  /*     Otherwise, call DSTEBZ and, if eigenvectors are desired, ZSTEIN. */
  if (*m == 0) {
    *info = 0;
    if (wantz) {
      *(unsigned char *)order = 'B';
    } else {
      *(unsigned char *)order = 'E';
    }
    indibl = 1;
    indisp = indibl + n;
    indiwk = indisp + n;
    lapackf77_dstebz(range_, order, &n, &vl, &vu, &il, &iu, &abstol, &rwork[indd], &rwork[inde], m,
                     &nsplit, &w[1], &iwork[indibl], &iwork[indisp], &rwork[indrwk], &iwork[indiwk], info);
    
    if (wantz) {
      lapackf77_zstein(&n, &rwork[indd], &rwork[inde], m, &w[1], &iwork[indibl], &iwork[indisp],
                       z, &ldz, &rwork[indrwk], &iwork[indiwk], &ifail[1], info);
      
      /*        Apply unitary matrix used in reduction to tridiagonal   
       form to eigenvectors returned by ZSTEIN. */
      magma_zunmtr(MagmaLeft, uplo, MagmaNoTrans, n, *m, a, lda, &work[indtau],
                   z, ldz, &work[indwrk], llwork, &iinfo);
    }
  }
  /*     If matrix was scaled, then rescale eigenvalues appropriately. */
  if (iscale == 1) {
    if (*info == 0) {
      imax = *m;
    } else {
      imax = *info - 1;
    }
    d__1 = 1. / sigma;
    dscal_(&imax, &d__1, &w[1], &ione);
  }
  
  /*     If eigenvalues are not in order, then sort them, along with   
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
        if (*info != 0) {
          itmp1 = ifail[i];
          ifail[i] = ifail[j];
          ifail[j] = itmp1;
        }
      }
    }
  }
  
  /*     Set WORK(1) to optimal complex workspace size. */
  
  work[1] = MAGMA_Z_MAKE((double) lopt, 0.);
  
  return MAGMA_SUCCESS;
  
} /* magma_zheevx_ */

