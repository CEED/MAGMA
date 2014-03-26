/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal d -> s

*/
#include "common_magma.h"

#define PRECISION_d

/**
    Purpose
    -------
    DGESDD computes the singular value decomposition (SVD) of a real
    M-by-N matrix A, optionally computing the left and right singular
    vectors, by using divide-and-conquer method. The SVD is written

        A = U * SIGMA * transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
    V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note that the routine returns VT = V**T, not V.

    The divide and conquer algorithm makes very mild assumptions about
    floating point arithmetic. It will work on machines with a guard
    digit in add/subtract, or on those binary machines without guard
    digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
    Cray-2. It could conceivably fail on hexadecimal or decimal machines
    without guard digits, but we know of none.

    Arguments
    ---------
    @param[in]
    jobz    CHARACTER*1
            Specifies options for computing all or part of the matrix U:
      -     = 'A':  all M columns of U and all N rows of V**T are
                    returned in the arrays U and VT;
      -     = 'S':  the first min(M,N) columns of U and the first
                    min(M,N) rows of V**T are returned in the arrays U
                    and VT;
      -     = 'O':  If M >= N, the first N columns of U are overwritten
                    on the array A and all rows of V**T are returned in
                    the array VT;
                    otherwise, all columns of U are returned in the
                    array U and the first M rows of V**T are overwritten
                    on the array A;
      -     = 'N':  no columns of U or rows of V**T are computed.

    @param[in]
    m       INTEGER
            The number of rows of the input matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the input matrix A.  N >= 0.

    @param[in,out]
    A       DOUBLE PRECISION array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit,
            if JOBZ = 'O',  A is overwritten with the first N columns
                            of U (the left singular vectors, stored
                            columnwise) if M >= N;
                            A is overwritten with the first M rows
                            of V**T (the right singular vectors, stored
                            rowwise) otherwise.
            if JOBZ != 'O', the contents of A are destroyed.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    S       DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i + 1).

    @param[out]
    U       DOUBLE PRECISION array, dimension (LDU,UCOL)
            UCOL = M if JOBZ = 'A' or JOBZ = 'O' and M < N;
            UCOL = min(M,N) if JOBZ = 'S'.
      -     If JOBZ = 'A' or JOBZ = 'O' and M < N, U contains the M-by-M
            orthogonal matrix U;
      -     if JOBZ = 'S', U contains the first min(M,N) columns of U
            (the left singular vectors, stored columnwise);
      -     if JOBZ = 'O' and M >= N, or JOBZ = 'N', U is not referenced.

    @param[in]
    ldu     INTEGER
            The leading dimension of the array U.  LDU >= 1; if
            JOBZ = 'S' or 'A' or JOBZ = 'O' and M < N, LDU >= M.

    @param[out]
    VT      DOUBLE PRECISION array, dimension (LDVT,N)
      -     If JOBZ = 'A' or JOBZ = 'O' and M >= N, VT contains the
            N-by-N orthogonal matrix V**T;
      -     if JOBZ = 'S', VT contains the first min(M,N) rows of
            V**T (the right singular vectors, stored rowwise);
      -     if JOBZ = 'O' and M < N, or JOBZ = 'N', VT is not referenced.

    @param[in]
    ldvt    INTEGER
            The leading dimension of the array VT.  LDVT >= 1; if
            JOBZ = 'A' or JOBZ = 'O' and M >= N, LDVT >= N;
            if JOBZ = 'S', LDVT >= min(M,N).

    @param[out]
    work    (workspace) DOUBLE PRECISION array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[1] returns the optimal LWORK.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK. LWORK >= 1.
      -     If JOBZ = 'N',
                LWORK >= 3*min(M,N) + max(max(M,N), 7*min(M,N)).
      -     If JOBZ = 'O',
                LWORK >= 3*min(M,N) + max(max(M,N), 5*min(M,N)*min(M,N) + 4*min(M,N)).
      -     If JOBZ = 'S' or 'A',
                LWORK >= 3*min(M,N) + max(max(M,N), 4*min(M,N)*min(M,N) + 4*min(M,N)).
            For good performance, LWORK should generally be larger.
    \n
            If LWORK = -1 but other input arguments are legal, WORK[1]
            returns the optimal LWORK.

    @param
    iwork   (workspace) INTEGER array, dimension (8*min(M,N))

    @param[out]
    info    INTEGER
      -     = 0:  successful exit.
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
      -     > 0:  DBDSDC did not converge, updating process failed.

    Further Details
    ---------------
    Based on contributions by
    Ming Gu and Huan Ren, Computer Science Division, University of
    California at Berkeley, USA

    @ingroup magma_d
    ********************************************************************/
magma_int_t magma_dgesdd(
    magma_vec_t jobz, magma_int_t m, magma_int_t n,
    double *A, magma_int_t lda,
    double *s,
    double *U, magma_int_t ldu,
    double *VT, magma_int_t ldvt,
    double *work, magma_int_t lwork,
    magma_int_t *iwork, magma_int_t *info)
{
#define  A(i_,j_) (A  + (i_) + (j_)*lda)
#define  U(i_,j_) (U  + (i_) + (j_)*ldu)
#define VT(i_,j_) (VT + (i_) + (j_)*ldvt)

    /* Constants */
    const double c_zero = MAGMA_D_ZERO;
    const double c_one  = MAGMA_D_ONE;
    const magma_int_t izero = 0;
    const magma_int_t ione  = 1;
    
    /* Local variables */
    magma_int_t i__1, i__2;
    magma_int_t i, ie, il, ir, iu, blk;
    double dum[1], eps;
    magma_int_t ivt, iscl;
    double anrm;
    magma_int_t idum[1], ierr, itau;


    magma_int_t chunk, minmn, wrkbl, itaup, itauq, mnthr;
    magma_int_t wantqa;  /* logical */
    magma_int_t nwork;
    magma_int_t wantqn, wantqo, wantqs;  /* logical */


    magma_int_t bdspac;


    double bignum;

    magma_int_t ldwrkl, ldwrkr, minwrk, ldwrku, maxwrk, ldwkvt;
    double smlnum;
    magma_int_t wantqas, lquery;  /* logical */

    /* Parameter adjustments */
    A  -= 1 + lda;
    U  -= 1 + ldu;
    VT -= 1 + ldvt;
    --s;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;
    minmn = min(m,n);
    wantqa  = (jobz == MagmaAllVec);
    wantqs  = (jobz == MagmaSomeVec);
    wantqas = (wantqa || wantqs);
    wantqo  = (jobz == MagmaOverwriteVec);
    wantqn  = (jobz == MagmaNoVec);
    lquery  = (lwork == -1);

    /* Test the input arguments */
    if (! (wantqa || wantqs || wantqo || wantqn)) {
        *info = -1;
    }
    else if (m < 0) {
        *info = -2;
    }
    else if (n < 0) {
        *info = -3;
    }
    else if (lda < max(1,m)) {
        *info = -5;
    }
    else if (ldu < 1 || (wantqas && ldu < m) || (wantqo && m < n && ldu < m)) {
        *info = -8;
    }
    else if (ldvt < 1 || (wantqa && ldvt < n) || (wantqs && ldvt < minmn)
                      || (wantqo && m >= n && ldvt < n)) {
        *info = -10;
    }

    /* Compute workspace */
    /* (Note: Comments in the code beginning "Workspace:" describe the */
    /* minimal amount of workspace needed at that point in the code, */
    /* as well as the preferred amount for good performance. */
    /* NB refers to the optimal block size for the immediately */
    /* following subroutine, as returned by ILAENV.) */
    if (*info == 0) {
        minwrk = 1;
        maxwrk = 1;
        if (m >= n && minmn > 0) {
            /* Compute space needed for DBDSDC */
            mnthr = (magma_int_t) (minmn*11. / 6.);
            if (wantqn) {
                bdspac = 7*n;
            }
            else {
                bdspac = 3*n*n + 4*n;
            }
            if (m >= mnthr) {
                if (wantqn) {
                    /* Path 1 (M much larger than N, JOBZ='N') */
                    wrkbl =              n +   n * magma_ilaenv( 1, "DGEQRF", " ",   m, n, -1, -1 );
                    wrkbl = max(wrkbl, 3*n + 2*n * magma_ilaenv( 1, "DGEBRD", " ",   n, n, -1, -1 ));
                    maxwrk = max(wrkbl, bdspac + n);
                    minwrk = bdspac + n;
                }
                else if (wantqo) {
                    /* Path 2 (M much larger than N, JOBZ='O') */
                    wrkbl =              n +   n * magma_ilaenv( 1, "DGEQRF", " ",   m, n, -1, -1 );
                    wrkbl = max(wrkbl,   n +   n * magma_ilaenv( 1, "DORGQR", " ",   m, n,  n, -1 ));
                    wrkbl = max(wrkbl, 3*n + 2*n * magma_ilaenv( 1, "DGEBRD", " ",   n, n, -1, -1 ));
                    wrkbl = max(wrkbl, 3*n +   n * magma_ilaenv( 1, "DORMBR", "QLN", n, n,  n, -1 ));
                    wrkbl = max(wrkbl, 3*n +   n * magma_ilaenv( 1, "DORMBR", "PRT", n, n,  n, -1 ));
                    wrkbl = max(wrkbl, bdspac + 3*n);
                    maxwrk = wrkbl + 2*n*n;
                    minwrk = bdspac + 2*n*n + 3*n;
                }
                else if (wantqs) {
                    /* Path 3 (M much larger than N, JOBZ='S') */
                    wrkbl =              n +   n * magma_ilaenv( 1, "DGEQRF", " ",   m, n, -1, -1 );
                    wrkbl = max(wrkbl,   n +   n * magma_ilaenv( 1, "DORGQR", " ",   m, n,  n, -1 ));
                    wrkbl = max(wrkbl, 3*n + 2*n * magma_ilaenv( 1, "DGEBRD", " ",   n, n, -1, -1 ));
                    wrkbl = max(wrkbl, 3*n +   n * magma_ilaenv( 1, "DORMBR", "QLN", n, n,  n, -1 ));
                    wrkbl = max(wrkbl, 3*n +   n * magma_ilaenv( 1, "DORMBR", "PRT", n, n,  n, -1 ));
                    wrkbl = max(wrkbl, bdspac + 3*n);
                    maxwrk = wrkbl + n*n;
                    minwrk = bdspac + n*n + 3*n;
                }
                else if (wantqa) {
                    /* Path 4 (M much larger than N, JOBZ='A') */
                    wrkbl =              n +   n * magma_ilaenv( 1, "DGEQRF", " ",   m, n, -1, -1 );
                    wrkbl = max(wrkbl,   n +   m * magma_ilaenv( 1, "DORGQR", " ",   m, m,  n, -1 ));
                    wrkbl = max(wrkbl, 3*n + 2*n * magma_ilaenv( 1, "DGEBRD", " ",   n, n, -1, -1 ));
                    wrkbl = max(wrkbl, 3*n +   n * magma_ilaenv( 1, "DORMBR", "QLN", n, n,  n, -1 ));
                    wrkbl = max(wrkbl, 3*n +   n * magma_ilaenv( 1, "DORMBR", "PRT", n, n,  n, -1 ));
                    wrkbl = max(wrkbl, bdspac + 3*n);
                    maxwrk = wrkbl + n*n;
                    minwrk = bdspac + n*n + 2*n + m;  /* was 3*n in clapack 3.2.1 */
                }
            }
            else {
                /* Path 5 (M at least N, but not much larger) */
                wrkbl = 3*n + (m + n) * magma_ilaenv( 1, "DGEBRD", " ",   m, n, -1, -1);
                if (wantqn) {
                    maxwrk = max(wrkbl, bdspac + 3*n);
                    minwrk = 3*n + max(m,bdspac);
                }
                else if (wantqo) {
                    wrkbl = max(wrkbl, 3*n + n * magma_ilaenv( 1, "DORMBR", "QLN", m, n, n, -1 ));
                    wrkbl = max(wrkbl, 3*n + n * magma_ilaenv( 1, "DORMBR", "PRT", n, n, n, -1 ));
                    wrkbl = max(wrkbl, bdspac + 3*n);
                    maxwrk = wrkbl + m*n;
                    minwrk = 3*n + max(m, n*n + bdspac);
                }
                else if (wantqs) {
                    wrkbl = max(wrkbl, 3*n + n * magma_ilaenv( 1, "DORMBR", "QLN", m, n, n, -1 ));
                    wrkbl = max(wrkbl, 3*n + n * magma_ilaenv( 1, "DORMBR", "PRT", n, n, n, -1 ));
                    maxwrk = max(wrkbl, bdspac + 3*n);
                    minwrk = 3*n + max(m,bdspac);
                }
                else if (wantqa) {
                    wrkbl = max(wrkbl, 3*n + m * magma_ilaenv( 1, "DORMBR", "QLN", m, m, n, -1 ));
                    wrkbl = max(wrkbl, 3*n + n * magma_ilaenv( 1, "DORMBR", "PRT", n, n, n, -1 ));
                    maxwrk = max(maxwrk, bdspac + 3*n);
                    minwrk = 3*n + max(m,bdspac);
                }
            }
        }
        else if (minmn > 0) {
            /* Compute space needed for DBDSDC */
            mnthr = (magma_int_t) (minmn*11. / 6.);
            if (wantqn) {
                bdspac = 7*m;
            }
            else {
                bdspac = 3*m*m + 4*m;
            }
            if (n >= mnthr) {
                if (wantqn) {
                    /* Path 1t (N much larger than M, JOBZ='N') */
                    wrkbl =              m +   m * magma_ilaenv( 1, "DGELQF", " ",   m, n, -1, -1 );
                    wrkbl = max(wrkbl, 3*m + 2*m * magma_ilaenv( 1, "DGEBRD", " ",   m, m, -1, -1 ));
                    maxwrk = max(wrkbl, bdspac + m);
                    minwrk = bdspac + m;
                }
                else if (wantqo) {
                    /* Path 2t (N much larger than M, JOBZ='O') */
                    wrkbl =              m +   m * magma_ilaenv( 1, "DGELQF", " ",   m, n, -1, -1 );
                    wrkbl = max(wrkbl,   m +   m * magma_ilaenv( 1, "DORGLQ", " ",   m, n,  m, -1 ));
                    wrkbl = max(wrkbl, 3*m + 2*m * magma_ilaenv( 1, "DGEBRD", " ",   m, m, -1, -1 ));
                    wrkbl = max(wrkbl, 3*m +   m * magma_ilaenv( 1, "DORMBR", "QLN", m, m,  m, -1 ));
                    wrkbl = max(wrkbl, 3*m +   m * magma_ilaenv( 1, "DORMBR", "PRT", m, m,  m, -1 ));
                    wrkbl = max(wrkbl, bdspac + 3*m);
                    maxwrk = wrkbl + 2*m*m;
                    minwrk = bdspac + 2*m*m + 3*m;
                }
                else if (wantqs) {
                    /* Path 3t (N much larger than M, JOBZ='S') */
                    wrkbl =              m +   m * magma_ilaenv( 1, "DGELQF", " ",   m, n, -1, -1 );
                    wrkbl = max(wrkbl,   m +   m * magma_ilaenv( 1, "DORGLQ", " ",   m, n,  m, -1 ));
                    wrkbl = max(wrkbl, 3*m + 2*m * magma_ilaenv( 1, "DGEBRD", " ",   m, m, -1, -1 ));
                    wrkbl = max(wrkbl, 3*m +   m * magma_ilaenv( 1, "DORMBR", "QLN", m, m,  m, -1 ));
                    wrkbl = max(wrkbl, 3*m +   m * magma_ilaenv( 1, "DORMBR", "PRT", m, m,  m, -1 ));
                    wrkbl = max(wrkbl, bdspac + 3*m);
                    maxwrk = wrkbl + m*m;
                    minwrk = bdspac + m*m + 3*m;
                }
                else if (wantqa) {
                    /* Path 4t (N much larger than M, JOBZ='A') */
                    wrkbl =              m +   m * magma_ilaenv( 1, "DGELQF", " ",   m, n, -1, -1 );
                    wrkbl = max(wrkbl,   m +   n * magma_ilaenv( 1, "DORGLQ", " ",   n, n,  m, -1 ));
                    wrkbl = max(wrkbl, 3*m + 2*m * magma_ilaenv( 1, "DGEBRD", " ",   m, m, -1, -1 ));
                    wrkbl = max(wrkbl, 3*m +   m * magma_ilaenv( 1, "DORMBR", "QLN", m, m,  m, -1 ));
                    wrkbl = max(wrkbl, 3*m +   m * magma_ilaenv( 1, "DORMBR", "PRT", m, m,  m, -1 ));
                    wrkbl = max(wrkbl, bdspac + 3*m);
                    maxwrk = wrkbl + m*m;
                    minwrk = bdspac + m*m + 3*m;
                }
            }
            else {
                /* Path 5t (N greater than M, but not much larger) */
                wrkbl = 3*m + (m + n) * magma_ilaenv( 1, "DGEBRD", " ",   m, n, -1, -1);
                if (wantqn) {
                    maxwrk = max(wrkbl, bdspac + 3*m);
                    minwrk = 3*m + max(n,bdspac);
                }
                else if (wantqo) {
                    wrkbl = max(wrkbl, 3*m + m * magma_ilaenv( 1, "DORMBR", "QLN", m, m, n, -1 ));
                    wrkbl = max(wrkbl, 3*m + m * magma_ilaenv( 1, "DORMBR", "PRT", m, n, m, -1 ));
                    wrkbl = max(wrkbl, bdspac + 3*m);
                    maxwrk = wrkbl + m*n;
                    minwrk = 3*m + max(n, m*m + bdspac);
                }
                else if (wantqs) {
                    wrkbl = max(wrkbl, 3*m + m * magma_ilaenv( 1, "DORMBR", "QLN", m, m, n, -1 ));
                    wrkbl = max(wrkbl, 3*m + m * magma_ilaenv( 1, "DORMBR", "PRT", m, n, m, -1 ));
                    maxwrk = max(wrkbl, bdspac + 3*m);
                    minwrk = 3*m + max(n,bdspac);
                }
                else if (wantqa) {
                    wrkbl = max(wrkbl, 3*m + m * magma_ilaenv( 1, "DORMBR", "QLN", m, m, n, -1 ));
                    wrkbl = max(wrkbl, 3*m + m * magma_ilaenv( 1, "DORMBR", "PRT", n, n, m, -1 ));
                    maxwrk = max(wrkbl, bdspac + 3*m);
                    minwrk = 3*m + max(n,bdspac);
                }
            }
        }
        maxwrk = max(maxwrk,minwrk);
        work[1] = (double) maxwrk;

        if (lwork < minwrk && ! lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery) {
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        return *info;
    }

    /* Get machine constants */
    eps = lapackf77_dlamch("P");
    smlnum = sqrt(lapackf77_dlamch("S")) / eps;
    bignum = 1. / smlnum;

    /* Scale A if max element outside range [SMLNUM,BIGNUM] */
    anrm = lapackf77_dlange("M", &m, &n, A(1,1), &lda, dum);
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
        iscl = 1;
        lapackf77_dlascl("G", &izero, &izero, &anrm, &smlnum, &m, &n, A(1,1), &lda, &ierr);
    }
    else if (anrm > bignum) {
        iscl = 1;
        lapackf77_dlascl("G", &izero, &izero, &anrm, &bignum, &m, &n, A(1,1), &lda, &ierr);
    }

    if (m >= n) {
        /* A has at least as many rows as columns. If A has sufficiently */
        /* more rows than columns, first reduce using the QR */
        /* decomposition (if sufficient workspace available) */
        if (m >= mnthr) {
            if (wantqn) {
                /* Path 1 (M much larger than N, JOBZ='N') */
                /* No singular vectors to be computed */
                itau = 1;
                nwork = itau + n;

                /* Compute A=Q*R */
                /* (Workspace: need 2*N, prefer N + N*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);

                /* Zero out below R */
                i__1 = n - 1;
                lapackf77_dlaset("L", &i__1, &i__1, &c_zero, &c_zero, &A[lda + 2], &lda);
                ie = 1;
                itauq = ie + n;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in A */
                /* (Workspace: need 4*N, prefer 3*N + 2*N*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dgebrd(&n, &n, A(1,1), &lda, &s[1], &work[ie], &work[itauq], &work[itaup], &work[nwork], &i__1, &ierr);
                nwork = ie + n;

                /* Perform bidiagonal SVD, computing singular values only */
                /* (Workspace: need N + BDSPAC) */
                lapackf77_dbdsdc("U", "N", &n, &s[1], &work[ie], dum, &ione, dum, &ione, dum, idum, &work[nwork], &iwork[1], info);
            }
            else if (wantqo) {
                /* Path 2 (M much larger than N, JOBZ = 'O') */
                /* N left singular vectors to be overwritten on A and */
                /* N right singular vectors to be computed in VT */
                ir = 1;

                /* WORK[IR] is LDWRKR by N */
                if (lwork >= lda*n + n*n + 3*n + bdspac) {
                    ldwrkr = lda;
                }
                else {
                    ldwrkr = (lwork - n*n - 3*n - bdspac) / n;
                }
                itau = ir + ldwrkr*n;
                nwork = itau + n;

                /* Compute A=Q*R */
                /* (Workspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);

                /* Copy R to WORK[IR], zeroing out below it */
                lapackf77_dlacpy("U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr);
                i__1 = n - 1;
                lapackf77_dlaset("L", &i__1, &i__1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr);

                /* Generate Q in A */
                /* (Workspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dorgqr(&m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);
                ie = itau;
                itauq = ie + n;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in VT, copying result to WORK[IR] */
                /* (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dgebrd(&n, &n, &work[ir], &ldwrkr, &s[1], &work[ie], &work[itauq], &work[itaup], &work[nwork], &i__1, &ierr);

                /* WORK[IU] is N by N */
                iu = nwork;
                nwork = iu + n*n;

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in WORK[IU] and computing right */
                /* singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + N*N + BDSPAC) */
                lapackf77_dbdsdc("U", "I", &n, &s[1], &work[ie], &work[iu], &n, VT(1,1), &ldvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Overwrite WORK[IU] by left singular vectors of R */
                /* and VT by right singular vectors of R */
                /* (Workspace: need 2*N*N + 3*N, prefer 2*N*N + 2*N + N*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], &work[iu], &n, &work[nwork], &i__1, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__1, &ierr);

                /* Multiply Q in A by left singular vectors of R in */
                /* WORK[IU], storing result in WORK[IR] and copying to A */
                /* (Workspace: need 2*N*N, prefer N*N + M*N) */
                for (i = 1; (ldwrkr < 0 ? i >= m : i <= m); i += ldwrkr) {
                    chunk = min(m - i + 1, ldwrkr);
                    blasf77_dgemm("N", "N", &chunk, &n, &n, &c_one, &A[i + lda], &lda, &work[iu], &n, &c_zero, &work[ir], &ldwrkr);
                    lapackf77_dlacpy("F", &chunk, &n, &work[ir], &ldwrkr, &A[i + lda], &lda);
                }
            }
            else if (wantqs) {
                /* Path 3 (M much larger than N, JOBZ='S') */
                /* N left singular vectors to be computed in U and */
                /* N right singular vectors to be computed in VT */
                ir = 1;

                /* WORK[IR] is N by N */
                ldwrkr = n;
                itau = ir + ldwrkr*n;
                nwork = itau + n;

                /* Compute A=Q*R */
                /* (Workspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);

                /* Copy R to WORK[IR], zeroing out below it */
                lapackf77_dlacpy("U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr);
                i__1 = n - 1;
                lapackf77_dlaset("L", &i__1, &i__1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr);

                /* Generate Q in A */
                /* (Workspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dorgqr(&m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);
                ie = itau;
                itauq = ie + n;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in WORK[IR] */
                /* (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dgebrd(&n, &n, &work[ir], &ldwrkr, &s[1], &work[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in U and computing right singular */
                /* vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + BDSPAC) */
                lapackf77_dbdsdc("U", "I", &n, &s[1], &work[ie], U(1,1), &ldu, VT(1,1), &ldvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Overwrite U by left singular vectors of R and VT */
                /* by right singular vectors of R */
                /* (Workspace: need N*N + 3*N, prefer N*N + 2*N + N*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], U(1,1), &ldu, &work[nwork], &i__2, &ierr);

                i__2 = lwork - nwork + 1;
                lapackf77_dormbr("P", "R", "T", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__2, &ierr);

                /* Multiply Q in A by left singular vectors of R in */
                /* WORK[IR], storing result in U */
                /* (Workspace: need N*N) */
                lapackf77_dlacpy("F", &n, &n, U(1,1), &ldu, &work[ir], &ldwrkr);
                blasf77_dgemm("N", "N", &m, &n, &n, &c_one, A(1,1), &lda, &work[ir], &ldwrkr, &c_zero, U(1,1), &ldu);
            }
            else if (wantqa) {
                /* Path 4 (M much larger than N, JOBZ='A') */
                /* M left singular vectors to be computed in U and */
                /* N right singular vectors to be computed in VT */
                iu = 1;

                /* WORK[IU] is N by N */
                ldwrku = n;
                itau = iu + ldwrku*n;
                nwork = itau + n;

                /* Compute A=Q*R, copying result to U */
                /* (Workspace: need N*N + N + M, prefer N*N + N + M*NB);  was N*N + 2*N, prefer N*N + N + N*NB in clapack 3.2.1 */
                i__2 = lwork - nwork + 1;
                lapackf77_dgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);
                lapackf77_dlacpy("L", &m, &n, A(1,1), &lda, U(1,1), &ldu);

                /* Generate Q in U */
                /* (Workspace: need N*N + N + M, prefer N*N + N + M*NB);  was N*N + 2*N, prefer N*N + N + N*NB in clapack 3.2.1 */
                i__2 = lwork - nwork + 1;
                lapackf77_dorgqr(&m, &m, &n, U(1,1), &ldu, &work[itau], &work[nwork], &i__2, &ierr);

                /* Produce R in A, zeroing out other entries */
                i__1 = n - 1;
                lapackf77_dlaset("L", &i__1, &i__1, &c_zero, &c_zero, &A[lda + 2], &lda);
                ie = itau;
                itauq = ie + n;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in A */
                /* (Workspace: need N*N + 4*N, prefer N*N + 3*N + 2*N*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dgebrd(&n, &n, A(1,1), &lda, &s[1], &work[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in WORK[IU] and computing right */
                /* singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + N*N + BDSPAC) */
                lapackf77_dbdsdc("U", "I", &n, &s[1], &work[ie], &work[iu], &n, VT(1,1), &ldvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Overwrite WORK[IU] by left singular vectors of R and VT */
                /* by right singular vectors of R */
                /* (Workspace: need N*N + 3*N, prefer N*N + 2*N + N*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &n, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &n, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__2, &ierr);

                /* Multiply Q in U by left singular vectors of R in */
                /* WORK[IU], storing result in A */
                /* (Workspace: need N*N) */
                blasf77_dgemm("N", "N", &m, &n, &n, &c_one, U(1,1), &ldu, &work[iu], &ldwrku, &c_zero, A(1,1), &lda);

                /* Copy left singular vectors of A from A to U */
                lapackf77_dlacpy("F", &m, &n, A(1,1), &lda, U(1,1), &ldu);
            }
        }
        else {
            /* M < MNTHR */
            /* Path 5 (M at least N, but not much larger) */
            /* Reduce to bidiagonal form without QR decomposition */
            ie = 1;
            itauq = ie + n;
            itaup = itauq + n;
            nwork = itaup + n;

            /* Bidiagonalize A */
            /* (Workspace: need 3*N + M, prefer 3*N + (M + N)*NB) */
            i__2 = lwork - nwork + 1;
            lapackf77_dgebrd(&m, &n, A(1,1), &lda, &s[1], &work[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);
            if (wantqn) {
                /* Perform bidiagonal SVD, only computing singular values */
                /* (Workspace: need N + BDSPAC) */
                lapackf77_dbdsdc("U", "N", &n, &s[1], &work[ie], dum, &ione, dum, &ione, dum, idum, &work[nwork], &iwork[1], info);
            }
            else if (wantqo) {
                iu = nwork;
                if (lwork >= m*n + 3*n + bdspac) {
                    /* WORK[ IU ] is M by N */
                    ldwrku = m;
                    nwork = iu + ldwrku*n;
                    lapackf77_dlaset("F", &m, &n, &c_zero, &c_zero, &work[iu], &ldwrku);
                }
                else {
                    /* WORK[ IU ] is N by N */
                    ldwrku = n;
                    nwork = iu + ldwrku*n;

                    /* WORK[IR] is LDWRKR by N */
                    ir = nwork;
                    ldwrkr = (lwork - n*n - 3*n) / n;
                }
                nwork = iu + ldwrku*n;

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in WORK[IU] and computing right */
                /* singular vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + N*N + BDSPAC) */
                lapackf77_dbdsdc("U", "I", &n, &s[1], &work[ie], &work[iu], &ldwrku, VT(1,1), &ldvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Overwrite VT by right singular vectors of A */
                /* (Workspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dormbr("P", "R", "T", &n, &n, &n, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__2, &ierr);

                if (lwork >= m*n + 3*n + bdspac) {
                    /* Overwrite WORK[IU] by left singular vectors of A */
                    /* (Workspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                    i__2 = lwork - nwork + 1;
                    lapackf77_dormbr("Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &ierr);

                    /* Copy left singular vectors of A from WORK[IU] to A */
                    lapackf77_dlacpy("F", &m, &n, &work[iu], &ldwrku, A(1,1), &lda);
                }
                else {
                    /* Generate Q in A */
                    /* (Workspace: need N*N + 2*N, prefer N*N + N + N*NB) */
                    i__2 = lwork - nwork + 1;
                    lapackf77_dorgbr("Q", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[nwork], &i__2, &ierr);

                    /* Multiply Q in A by left singular vectors of */
                    /* bidiagonal matrix in WORK[IU], storing result in */
                    /* WORK[IR] and copying to A */
                    /* (Workspace: need 2*N*N, prefer N*N + M*N) */
                    for (i = 1; (ldwrkr < 0 ? i >= m : i <= m); i += ldwrkr) {
                        chunk = min(m - i + 1, ldwrkr);
                        blasf77_dgemm("N", "N", &chunk, &n, &n, &c_one, &A[i + lda], &lda, &work[iu], &ldwrku, &c_zero, &work[ir], &ldwrkr);
                        lapackf77_dlacpy("F", &chunk, &n, &work[ir], &ldwrkr, &A[i + lda], &lda);
                    }
                }
            }
            else if (wantqs) {
                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in U and computing right singular */
                /* vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + BDSPAC) */
                lapackf77_dlaset("F", &m, &n, &c_zero, &c_zero, U(1,1), &ldu);
                lapackf77_dbdsdc("U", "I", &n, &s[1], &work[ie], U(1,1), &ldu, VT(1,1), &ldvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Overwrite U by left singular vectors of A and VT */
                /* by right singular vectors of A */
                /* (Workspace: need 3*N, prefer 2*N + N*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__1, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &n, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__1, &ierr);
            }
            else if (wantqa) {
                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in U and computing right singular */
                /* vectors of bidiagonal matrix in VT */
                /* (Workspace: need N + BDSPAC) */
                lapackf77_dlaset("F", &m, &m, &c_zero, &c_zero, U(1,1), &ldu);
                lapackf77_dbdsdc("U", "I", &n, &s[1], &work[ie], U(1,1), &ldu, VT(1,1), &ldvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Set the right corner of U to identity matrix */
                if (m > n) {
                    i__1 = m - n;
                    lapackf77_dlaset("F", &i__1, &i__1, &c_zero, &c_one, &U[n + 1 + (n + 1)*ldu], &ldu);
                }

                /* Overwrite U by left singular vectors of A and VT */
                /* by right singular vectors of A */
                /* (Workspace: need N*N + 2*N + M, prefer N*N + 2*N + M*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__1, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &m, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__1, &ierr);
            }
        }
    }
    else {
        /* A has more columns than rows. If A has sufficiently more */
        /* columns than rows, first reduce using the LQ decomposition (if */
        /* sufficient workspace available) */
        if (n >= mnthr) {
            if (wantqn) {
                /* Path 1t (N much larger than M, JOBZ='N') */
                /* No singular vectors to be computed */
                itau = 1;
                nwork = itau + m;

                /* Compute A=L*Q */
                /* (Workspace: need 2*M, prefer M + M*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);

                /* Zero out above L */
                i__1 = m - 1;
                lapackf77_dlaset("U", &i__1, &i__1, &c_zero, &c_zero, &A[(2*lda) + 1], &lda);
                ie = 1;
                itauq = ie + m;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in A */
                /* (Workspace: need 4*M, prefer 3*M + 2*M*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dgebrd(&m, &m, A(1,1), &lda, &s[1], &work[ie], &work[itauq], &work[itaup], &work[nwork], &i__1, &ierr);
                nwork = ie + m;

                /* Perform bidiagonal SVD, computing singular values only */
                /* (Workspace: need M + BDSPAC) */
                lapackf77_dbdsdc("U", "N", &m, &s[1], &work[ie], dum, &ione, dum, &ione, dum, idum, &work[nwork], &iwork[1], info);
            }
            else if (wantqo) {
                /* Path 2t (N much larger than M, JOBZ='O') */
                /* M right singular vectors to be overwritten on A and */
                /* M left singular vectors to be computed in U */
                ivt = 1;

                /* IVT is M by M */
                il = ivt + m*m;
                if (lwork >= m*n + m*m + 3*m + bdspac) {
                    /* WORK[IL] is M by N */
                    ldwrkl = m;
                    chunk = n;
                }
                else {
                    ldwrkl = m;
                    chunk = (lwork - m*m) / m;
                }
                itau = il + ldwrkl*m;
                nwork = itau + m;

                /* Compute A=L*Q */
                /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);

                /* Copy L to WORK[IL], zeroing about above it */
                lapackf77_dlacpy("L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl);
                i__1 = m - 1;
                lapackf77_dlaset("U", &i__1, &i__1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl);

                /* Generate Q in A */
                /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dorglq(&m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);
                ie = itau;
                itauq = ie + m;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in WORK[IL] */
                /* (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dgebrd(&m, &m, &work[il], &ldwrkl, &s[1], &work[ie], &work[itauq], &work[itaup], &work[nwork], &i__1, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in U, and computing right singular */
                /* vectors of bidiagonal matrix in WORK[IVT] */
                /* (Workspace: need M + M*M + BDSPAC) */
                lapackf77_dbdsdc("U", "I", &m, &s[1], &work[ie], U(1,1), &ldu, &work[ivt], &m, dum, idum, &work[nwork], &iwork[1], info);

                /* Overwrite U by left singular vectors of L and WORK[IVT] */
                /* by right singular vectors of L */
                /* (Workspace: need 2*M*M + 3*M, prefer 2*M*M + 2*M + M*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U(1,1), &ldu, &work[nwork], &i__1, &ierr);
                lapackf77_dormbr("P", "R", "T", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], &work[ivt], &m, &work[nwork], &i__1, &ierr);

                /* Multiply right singular vectors of L in WORK[IVT] by Q */
                /* in A, storing result in WORK[IL] and copying to A */
                /* (Workspace: need 2*M*M, prefer M*M + M*N) */
                for (i = 1; (chunk < 0 ? i >= n : i <= n); i += chunk) {
                    blk = min(n - i + 1, chunk);
                    blasf77_dgemm("N", "N", &m, &blk, &m, &c_one, &work[ivt], &m, &A[i*lda + 1], &lda, &c_zero, &work[il], &ldwrkl);
                    lapackf77_dlacpy("F", &m, &blk, &work[il], &ldwrkl, &A[i*lda + 1], &lda);
                }
            }
            else if (wantqs) {
                /* Path 3t (N much larger than M, JOBZ='S') */
                /* M right singular vectors to be computed in VT and */
                /* M left singular vectors to be computed in U */
                il = 1;

                /* WORK[IL] is M by M */
                ldwrkl = m;
                itau = il + ldwrkl*m;
                nwork = itau + m;

                /* Compute A=L*Q */
                /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);

                /* Copy L to WORK[IL], zeroing out above it */
                lapackf77_dlacpy("L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl);
                i__1 = m - 1;
                lapackf77_dlaset("U", &i__1, &i__1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl);

                /* Generate Q in A */
                /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dorglq(&m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);
                ie = itau;
                itauq = ie + m;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in WORK[IU], copying result to U */
                /* (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dgebrd(&m, &m, &work[il], &ldwrkl, &s[1], &work[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in U and computing right singular */
                /* vectors of bidiagonal matrix in VT */
                /* (Workspace: need M + BDSPAC) */
                lapackf77_dbdsdc("U", "I", &m, &s[1], &work[ie], U(1,1), &ldu, VT(1,1), &ldvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Overwrite U by left singular vectors of L and VT */
                /* by right singular vectors of L */
                /* (Workspace: need M*M + 3*M, prefer M*M + 2*M + M*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U(1,1), &ldu, &work[nwork], &i__2, &ierr);
                lapackf77_dormbr("P", "R", "T", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__2, &ierr);

                /* Multiply right singular vectors of L in WORK[IL] by */
                /* Q in A, storing result in VT */
                /* (Workspace: need M*M) */
                lapackf77_dlacpy("F", &m, &m, VT(1,1), &ldvt, &work[il], &ldwrkl);
                blasf77_dgemm("N", "N", &m, &n, &m, &c_one, &work[il], &ldwrkl, A(1,1), &lda, &c_zero, VT(1,1), &ldvt);
            }
            else if (wantqa) {
                /* Path 4t (N much larger than M, JOBZ='A') */
                /* N right singular vectors to be computed in VT and */
                /* M left singular vectors to be computed in U */
                ivt = 1;

                /* WORK[IVT] is M by M */
                ldwkvt = m;
                itau = ivt + ldwkvt*m;
                nwork = itau + m;

                /* Compute A=L*Q, copying result to VT */
                /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);
                lapackf77_dlacpy("U", &m, &n, A(1,1), &lda, VT(1,1), &ldvt);

                /* Generate Q in VT */
                /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dorglq(&n, &n, &m, VT(1,1), &ldvt, &work[itau], &work[nwork], &i__2, &ierr);

                /* Produce L in A, zeroing out other entries */
                i__1 = m - 1;
                lapackf77_dlaset("U", &i__1, &i__1, &c_zero, &c_zero, &A[(2*lda) + 1], &lda);
                ie = itau;
                itauq = ie + m;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in A */
                /* (Workspace: need M*M + 4*M, prefer M*M + 3*M + 2*M*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dgebrd(&m, &m, A(1,1), &lda, &s[1], &work[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in U and computing right singular */
                /* vectors of bidiagonal matrix in WORK[IVT] */
                /* (Workspace: need M + M*M + BDSPAC) */
                lapackf77_dbdsdc("U", "I", &m, &s[1], &work[ie], U(1,1), &ldu, &work[ivt], &ldwkvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Overwrite U by left singular vectors of L and WORK[IVT] */
                /* by right singular vectors of L */
                /* (Workspace: need M*M + 3*M, prefer M*M + 2*M + M*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &m, &m, &m, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__2, &ierr);
                lapackf77_dormbr("P", "R", "T", &m, &m, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, &ierr);

                /* Multiply right singular vectors of L in WORK[IVT] by */
                /* Q in VT, storing result in A */
                /* (Workspace: need M*M) */
                blasf77_dgemm("N", "N", &m, &n, &m, &c_one, &work[ivt], &ldwkvt, VT(1,1), &ldvt, &c_zero, A(1,1), &lda);

                /* Copy right singular vectors of A from A to VT */
                lapackf77_dlacpy("F", &m, &n, A(1,1), &lda, VT(1,1), &ldvt);
            }
        }
        else {
            /* N < MNTHR */
            /* Path 5t (N greater than M, but not much larger) */
            /* Reduce to bidiagonal form without LQ decomposition */
            ie = 1;
            itauq = ie + m;
            itaup = itauq + m;
            nwork = itaup + m;

            /* Bidiagonalize A */
            /* (Workspace: need 3*M + N, prefer 3*M + (M + N)*NB) */
            i__2 = lwork - nwork + 1;
            lapackf77_dgebrd(&m, &n, A(1,1), &lda, &s[1], &work[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);
            if (wantqn) {
                /* Perform bidiagonal SVD, only computing singular values */
                /* (Workspace: need M + BDSPAC) */
                lapackf77_dbdsdc("L", "N", &m, &s[1], &work[ie], dum, &ione, dum, &ione, dum, idum, &work[nwork], &iwork[1], info);
            }
            else if (wantqo) {
                ldwkvt = m;
                ivt = nwork;
                if (lwork >= m*n + 3*m + bdspac) {
                    /* WORK[ IVT ] is M by N */
                    lapackf77_dlaset("F", &m, &n, &c_zero, &c_zero, &work[ivt], &ldwkvt);
                    nwork = ivt + ldwkvt*n;
                }
                else {
                    /* WORK[ IVT ] is M by M */
                    nwork = ivt + ldwkvt*m;
                    il = nwork;

                    /* WORK[IL] is M by CHUNK */
                    chunk = (lwork - m*m - 3*m) / m;
                }

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in U and computing right singular */
                /* vectors of bidiagonal matrix in WORK[IVT] */
                /* (Workspace: need M*M + BDSPAC) */
                lapackf77_dbdsdc("L", "I", &m, &s[1], &work[ie], U(1,1), &ldu, &work[ivt], &ldwkvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Overwrite U by left singular vectors of A */
                /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                i__2 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__2, &ierr);

                if (lwork >= m*n + 3*m + bdspac) {
                    /* Overwrite WORK[IVT] by left singular vectors of A */
                    /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                    i__2 = lwork - nwork + 1;
                    lapackf77_dormbr("P", "R", "T", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, &ierr);

                    /* Copy right singular vectors of A from WORK[IVT] to A */
                    lapackf77_dlacpy("F", &m, &n, &work[ivt], &ldwkvt, A(1,1), &lda);
                }
                else {
                    /* Generate P**T in A */
                    /* (Workspace: need M*M + 2*M, prefer M*M + M + M*NB) */
                    i__2 = lwork - nwork + 1;
                    lapackf77_dorgbr("P", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[nwork], &i__2, &ierr);

                    /* Multiply Q in A by right singular vectors of */
                    /* bidiagonal matrix in WORK[IVT], storing result in */
                    /* WORK[IL] and copying to A */
                    /* (Workspace: need 2*M*M, prefer M*M + M*N) */
                    for (i = 1; (chunk < 0 ? i >= n : i <= n); i += chunk) {
                        blk = min(n - i + 1, chunk);
                        blasf77_dgemm("N", "N", &m, &blk, &m, &c_one, &work[ivt], &ldwkvt, &A[i*lda + 1], &lda, &c_zero, &work[il], &m);
                        lapackf77_dlacpy("F", &m, &blk, &work[il], &m, &A[i*lda + 1], &lda);
                    }
                }
            }
            else if (wantqs) {
                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in U and computing right singular */
                /* vectors of bidiagonal matrix in VT */
                /* (Workspace: need M + BDSPAC) */
                lapackf77_dlaset("F", &m, &n, &c_zero, &c_zero, VT(1,1), &ldvt);
                lapackf77_dbdsdc("L", "I", &m, &s[1], &work[ie], U(1,1), &ldu, VT(1,1), &ldvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Overwrite U by left singular vectors of A and VT */
                /* by right singular vectors of A */
                /* (Workspace: need 3*M, prefer 2*M + M*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__1, &ierr);
                lapackf77_dormbr("P", "R", "T", &m, &n, &m, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__1, &ierr);
            }
            else if (wantqa) {
                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in U and computing right singular */
                /* vectors of bidiagonal matrix in VT */
                /* (Workspace: need M + BDSPAC) */
                lapackf77_dlaset("F", &n, &n, &c_zero, &c_zero, VT(1,1), &ldvt);
                lapackf77_dbdsdc("L", "I", &m, &s[1], &work[ie], U(1,1), &ldu, VT(1,1), &ldvt, dum, idum, &work[nwork], &iwork[1], info);

                /* Set the right corner of VT to identity matrix */
                if (n > m) {
                    i__1 = n - m;
                    lapackf77_dlaset("F", &i__1, &i__1, &c_zero, &c_one, &VT[m + 1 + (m + 1)*ldvt], &ldvt);
                }

                /* Overwrite U by left singular vectors of A and VT */
                /* by right singular vectors of A */
                /* (Workspace: need 2*M + N, prefer 2*M + N*NB) */
                i__1 = lwork - nwork + 1;
                lapackf77_dormbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__1, &ierr);
                lapackf77_dormbr("P", "R", "T", &n, &n, &m, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__1, &ierr);
            }
        }
    }

    /* Undo scaling if necessary */
    if (iscl == 1) {
        if (anrm > bignum) {
            lapackf77_dlascl("G", &izero, &izero, &bignum, &anrm, &minmn, &ione, &s[1], &minmn, &ierr);
        }
        if (anrm < smlnum) {
            lapackf77_dlascl("G", &izero, &izero, &smlnum, &anrm, &minmn, &ione, &s[1], &minmn, &ierr);
        }
    }

    /* Return optimal workspace in WORK[1] */
    work[1] = (double) maxwrk;

    return *info;
} /* magma_dgesdd */
