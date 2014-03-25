/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal z -> c

*/
#include "common_magma.h"

#define PRECISION_z

/**
    Purpose
    -------
    ZGESDD computes the singular value decomposition (SVD) of a complex
    M-by-N matrix A, optionally computing the left and/or right singular
    vectors, by using divide-and-conquer method. The SVD is written

         A = U * SIGMA * conjugate-transpose(V)

    where SIGMA is an M-by-N matrix which is zero except for its
    min(m,n) diagonal elements, U is an M-by-M unitary matrix, and
    V is an N-by-N unitary matrix.  The diagonal elements of SIGMA
    are the singular values of A; they are real and non-negative, and
    are returned in descending order.  The first min(m,n) columns of
    U and V are the left and right singular vectors of A.

    Note that the routine returns VT = V**H, not V.

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
      -     = 'A':  all M columns of U and all N rows of V**H are
                    returned in the arrays U and VT;
      -     = 'S':  the first min(M,N) columns of U and the first
                    min(M,N) rows of V**H are returned in the arrays U
                    and VT;
      -     = 'O':  If M >= N, the first N columns of U are overwritten
                    in the array A and all rows of V**H are returned in
                    the array VT;
                    otherwise, all columns of U are returned in the
                    array U and the first M rows of V**H are overwritten
                    in the array A;
      -     = 'N':  no columns of U or rows of V**H are computed.

    @param[in]
    m       MAGMA_INT_T
            The number of rows of the input matrix A.  M >= 0.

    @param[in]
    n       MAGMA_INT_T
            The number of columns of the input matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX*16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit,
            if JOBZ = 'O',  A is overwritten with the first N columns
                            of U (the left singular vectors, stored
                            columnwise) if M >= N;
                            A is overwritten with the first M rows
                            of V**H (the right singular vectors, stored
                            rowwise) otherwise.
            if JOBZ != 'O', the contents of A are destroyed.

    @param[in]
    lda     MAGMA_INT_T
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    S       DOUBLE PRECISION array, dimension (min(M,N))
            The singular values of A, sorted so that S(i) >= S(i+1).

    @param[out]
    U       COMPLEX*16 array, dimension (LDU,UCOL)
            UCOL = M if JOBZ = 'A' or JOBZ = 'O' and M < N;
            UCOL = min(M,N) if JOBZ = 'S'.
            If JOBZ = 'A' or JOBZ = 'O' and M < N, U contains the M-by-M
            unitary matrix U;
            if JOBZ = 'S', U contains the first min(M,N) columns of U
            (the left singular vectors, stored columnwise);
            if JOBZ = 'O' and M >= N, or JOBZ = 'N', U is not referenced.

    @param[in]
    ldu     MAGMA_INT_T
            The leading dimension of the array U.  LDU >= 1; if
            JOBZ = 'S' or 'A' or JOBZ = 'O' and M < N, LDU >= M.

    @param[out]
    VT      COMPLEX*16 array, dimension (LDVT,N)
            If JOBZ = 'A' or JOBZ = 'O' and M >= N, VT contains the
            N-by-N unitary matrix V**H;
            if JOBZ = 'S', VT contains the first min(M,N) rows of
            V**H (the right singular vectors, stored rowwise);
            if JOBZ = 'O' and M < N, or JOBZ = 'N', VT is not referenced.

    @param[in]
    ldvt    MAGMA_INT_T
            The leading dimension of the array VT.  LDVT >= 1; if
            JOBZ = 'A' or JOBZ = 'O' and M >= N, LDVT >= N;
            if JOBZ = 'S', LDVT >= min(M,N).

    @param[out]
    work    (workspace) COMPLEX*16 array, dimension (MAX(1,&lwork))
            On exit, if INFO = 0, WORK[1] returns the optimal &lwork.

    &lwork   (input) MAGMA_INT_T
            The dimension of the array WORK. &lwork >= 1.
            if JOBZ = 'N', &lwork >= 2*min(M,N)+max(M,N).
            if JOBZ = 'O',
                  &lwork >= 2*min(M,N)*min(M,N)+2*min(M,N)+max(M,N).
            if JOBZ = 'S' or 'A',
                  &lwork >= min(M,N)*min(M,N)+2*min(M,N)+max(M,N).
            For good performance, &lwork should generally be larger.
    \n
            If &lwork = -1, a workspace query is assumed.  The optimal
            size for the WORK array is calculated and stored in WORK[1],
            and no other work except argument checking is performed.

    @param
    rwork   (workspace) DOUBLE PRECISION array, dimension (MAX(1,LRWORK))
            If JOBZ = 'N', LRWORK >= 5*min(M,N).
            Otherwise, LRWORK >= 5*min(M,N)*min(M,N) + 7*min(M,N)

    @param
    iwork   (workspace) MAGMA_INT_T array, dimension (8*min(M,N))

    @param[out]
    info    MAGMA_INT_T
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  The updating process of DBDSDC did not converge.

    Further Details
    ---------------
    Based on contributions by
    Ming Gu and Huan Ren, Computer Science Division, University of
    California at Berkeley, USA

    @ingroup magma_z
    ********************************************************************/
magma_int_t magma_zgesdd(
    const char *jobz, magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    double *s,
    magmaDoubleComplex *U, magma_int_t ldu,
    magmaDoubleComplex *VT, magma_int_t ldvt,
    magmaDoubleComplex *work, magma_int_t lwork,
    double *rwork, magma_int_t *iwork, magma_int_t *info)
{
#define  A(i_,j_) (A  + (i_) + (j_)*lda)
#define  U(i_,j_) (U  + (i_) + (j_)*ldu)
#define VT(i_,j_) (VT + (i_) + (j_)*ldvt)

    /* Constants */
    const magmaDoubleComplex c_zero = {0., 0.};
    const magmaDoubleComplex c_one  = {1., 0.};
    const magma_int_t izero = 0;
    const magma_int_t ione  = 1;

    /* Local variables */
    magma_int_t i__1, i__2;
    magma_int_t i, ie, il, ir, iu, blk;
    double anrm, dum[1], eps, bignum, smlnum;
    magma_int_t iru, ivt, iscl;
    magma_int_t idum[1], ierr, itau, irvt;
    magma_int_t chunk, minmn;
    magma_int_t wrkbl, itaup, itauq;
    magma_int_t nwork;
    magma_int_t mnthr1, mnthr2;
    magma_int_t ldwrkl;
    magma_int_t ldwrkr, minwrk, ldwrku, maxwrk;
    magma_int_t ldwkvt;
    magma_int_t nrwork;
    magma_int_t wantqa, wantqn, wantqo, wantqs, wantqas;    /* logical */

    /* Parameter adjustments */
    A  -= 1 + lda;
    U  -= 1 + ldu;
    VT -= 1 + ldvt;
    --s;
    --work;
    --rwork;
    --iwork;

    /* Function Body */
    *info = 0;
    minmn = min(m,n);
    mnthr1 = (magma_int_t) (minmn * 17. / 9.);
    mnthr2 = (magma_int_t) (minmn * 5. / 3.);
    wantqa = lapackf77_lsame(jobz, "A");
    wantqs = lapackf77_lsame(jobz, "S");
    wantqas = (wantqa || wantqs);
    wantqo = lapackf77_lsame(jobz, "O");
    wantqn = lapackf77_lsame(jobz, "N");
    minwrk = 1;
    maxwrk = 1;

    /* Test the input arguments */
    if (! (wantqa || wantqs || wantqo || wantqn)) {
        *info = -1;
    } else if (m < 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < max(1,m)) {
        *info = -5;
    } else if (ldu < 1 || (wantqas && ldu < m) || (wantqo && m < n && ldu < m)) {
        *info = -8;
    } else if (ldvt < 1 || (wantqa && ldvt < n) || (wantqs && ldvt < minmn)
                        || (wantqo && m >= n && ldvt < n)) {
        *info = -10;
    }

    /* Compute workspace */
    /*  (Note: Comments in the code beginning "Workspace:" describe the */
    /*   minimal amount of workspace needed at that point in the code, */
    /*   as well as the preferred amount for good performance. */
    /*   CWorkspace refers to complex workspace, and RWorkspace to */
    /*   real workspace. NB refers to the optimal block size for the */
    /*   immediately following subroutine, as returned by ILAENV.) */
    if (*info == 0 && m > 0 && n > 0) {
        if (m >= n) {
            /* There is no complex work space needed for bidiagonal SVD */
            /* The real work space needed for bidiagonal SVD is BDSPAC */
            /* for computing singular values and singular vectors; BDSPAN */
            /* for computing singular values only. */
            /* BDSPAC = 5*N*N + 7*N */
            /* BDSPAN = MAX(7*N+4, 3*N+2+SMLSIZ*(SMLSIZ+8)) */
            if (m >= mnthr1) {
                if (wantqn) {
                    /* Path 1 (M much larger than N, JOBZ='N') */
                    maxwrk =                n +   n * magma_ilaenv( 1, "ZGEQRF", " ",   m, n, -1, -1 );
                    maxwrk = max( maxwrk, 2*n + 2*n * magma_ilaenv( 1, "ZGEBRD", " ",   n, n, -1, -1 ));
                    minwrk = 3*n;
                }
                else if (wantqo) {
                    /* Path 2 (M much larger than N, JOBZ='O') */
                    wrkbl =               n +   n * magma_ilaenv( 1, "ZGEQRF", " ",   m, n, -1, -1 );
                    wrkbl = max( wrkbl,   n +   n * magma_ilaenv( 1, "ZUNGQR", " ",   m, n,  n, -1 ));
                    wrkbl = max( wrkbl, 2*n + 2*n * magma_ilaenv( 1, "ZGEBRD", " ",   n, n, -1, -1 ));
                    wrkbl = max( wrkbl, 2*n +   n * magma_ilaenv( 1, "ZUNMBR", "QLN", n, n,  n, -1 ));
                    wrkbl = max( wrkbl, 2*n +   n * magma_ilaenv( 1, "ZUNMBR", "PRC", n, n,  n, -1 ));
                    maxwrk = m*n + n*n + wrkbl;
                    minwrk = 2*n*n + 3*n;
                }
                else if (wantqs) {
                    /* Path 3 (M much larger than N, JOBZ='S') */
                    wrkbl =               n +   n * magma_ilaenv( 1, "ZGEQRF", " ",   m, n, -1, -1 );
                    wrkbl = max( wrkbl,   n +   n * magma_ilaenv( 1, "ZUNGQR", " ",   m, n,  n, -1 ));
                    wrkbl = max( wrkbl, 2*n + 2*n * magma_ilaenv( 1, "ZGEBRD", " ",   n, n, -1, -1 ));
                    wrkbl = max( wrkbl, 2*n +   n * magma_ilaenv( 1, "ZUNMBR", "QLN", n, n,  n, -1 ));
                    wrkbl = max( wrkbl, 2*n +   n * magma_ilaenv( 1, "ZUNMBR", "PRC", n, n,  n, -1 ));
                    maxwrk = n*n + wrkbl;
                    minwrk = n*n + 3*n;
                }
                else if (wantqa) {
                    /* Path 4 (M much larger than N, JOBZ='A') */
                    wrkbl =               n +   n * magma_ilaenv( 1, "ZGEQRF", " ",   m, n, -1, -1 );
                    wrkbl = max( wrkbl,   n +   m * magma_ilaenv( 1, "ZUNGQR", " ",   m, m,  n, -1 ));
                    wrkbl = max( wrkbl, 2*n + 2*n * magma_ilaenv( 1, "ZGEBRD", " ",   n, n, -1, -1 ));
                    wrkbl = max( wrkbl, 2*n +   n * magma_ilaenv( 1, "ZUNMBR", "QLN", n, n,  n, -1 ));
                    wrkbl = max( wrkbl, 2*n +   n * magma_ilaenv( 1, "ZUNMBR", "PRC", n, n,  n, -1 ));
                    maxwrk = n*n + wrkbl;
                    minwrk = n*n + 2*n + m;
                }
            }
            else if (m >= mnthr2) {
                /* Path 5 (M much larger than N, but not as much as MNTHR1) */
                maxwrk = 2*n + (m + n) * magma_ilaenv( 1, "ZGEBRD", " ",   m, n, -1, -1 );
                minwrk = 2*n + m;
                if (wantqo) {
                    maxwrk = max( maxwrk, 2*n + n * magma_ilaenv( 1, "ZUNGBR", "P", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + n * magma_ilaenv( 1, "ZUNGBR", "Q", m, n, n, -1 ));
                    maxwrk += m*n;
                    minwrk += n*n;
                }
                else if (wantqs) {
                    maxwrk = max( maxwrk, 2*n + n * magma_ilaenv( 1, "ZUNGBR", "P", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + n * magma_ilaenv( 1, "ZUNGBR", "Q", m, n, n, -1 ));
                }
                else if (wantqa) {
                    maxwrk = max( maxwrk, 2*n + n * magma_ilaenv( 1, "ZUNGBR", "P", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + m * magma_ilaenv( 1, "ZUNGBR", "Q", m, m, n, -1 ));
                }
            }
            else {
                /* Path 6 (M at least N, but not much larger) */
                maxwrk = 2*n + (m + n) * magma_ilaenv( 1, "ZGEBRD", " ",   m, n, -1, -1 );
                minwrk = 2*n + m;
                if (wantqo) {
                    maxwrk = max( maxwrk, 2*n + n * magma_ilaenv( 1, "ZUNMBR", "PRC", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + n * magma_ilaenv( 1, "ZUNMBR", "QLN", m, n, n, -1 ));
                    maxwrk += m*n;
                    minwrk += n*n;
                }
                else if (wantqs) {
                    maxwrk = max( maxwrk, 2*n + n * magma_ilaenv( 1, "ZUNMBR", "PRC", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + n * magma_ilaenv( 1, "ZUNMBR", "QLN", m, n, n, -1 ));
                }
                else if (wantqa) {
                    maxwrk = max( maxwrk, 2*n + n * magma_ilaenv( 1, "ZUNGBR", "PRC", n, n, n, -1 ));
                    maxwrk = max( maxwrk, 2*n + m * magma_ilaenv( 1, "ZUNGBR", "QLN", m, m, n, -1 ));
                }
            }
        }
        else {
            /* There is no complex work space needed for bidiagonal SVD */
            /* The real work space needed for bidiagonal SVD is BDSPAC */
            /* for computing singular values and singular vectors; BDSPAN */
            /* for computing singular values only. */
            /* BDSPAC = 5*M*M + 7*M */
            /* BDSPAN = MAX(7*M+4, 3*M+2+SMLSIZ*(SMLSIZ+8)) */
            if (n >= mnthr1) {
                if (wantqn) {
                    /* Path 1t (N much larger than M, JOBZ='N') */
                    maxwrk =                m +   m * magma_ilaenv( 1, "ZGELQF", " ",   m, n, -1, -1 );
                    maxwrk = max( maxwrk, 2*m + 2*m * magma_ilaenv( 1, "ZGEBRD", " ",   m, m, -1, -1 ));
                    minwrk = 3*m;
                }
                else if (wantqo) {
                    /* Path 2t (N much larger than M, JOBZ='O') */
                    wrkbl =               m +   m * magma_ilaenv( 1, "ZGELQF", " ",   m, n, -1, -1 );
                    wrkbl = max( wrkbl,   m +   m * magma_ilaenv( 1, "ZUNGLQ", " ",   m, n,  m, -1 ));
                    wrkbl = max( wrkbl, 2*m + 2*m * magma_ilaenv( 1, "ZGEBRD", " ",   m, m, -1, -1 ));
                    wrkbl = max( wrkbl, 2*m +   m * magma_ilaenv( 1, "ZUNMBR", "PRC", m, m,  m, -1 ));
                    wrkbl = max( wrkbl, 2*m +   m * magma_ilaenv( 1, "ZUNMBR", "QLN", m, m,  m, -1 ));
                    maxwrk = m*n + m*m + wrkbl;
                    minwrk = 2*m*m + 3*m;
                }
                else if (wantqs) {
                    /* Path 3t (N much larger than M, JOBZ='S') */
                    wrkbl =               m +   m * magma_ilaenv( 1, "ZGELQF", " ",   m, n, -1, -1 );
                    wrkbl = max( wrkbl,   m +   m * magma_ilaenv( 1, "ZUNGLQ", " ",   m, n,  m, -1 ));
                    wrkbl = max( wrkbl, 2*m + 2*m * magma_ilaenv( 1, "ZGEBRD", " ",   m, m, -1, -1 ));
                    wrkbl = max( wrkbl, 2*m +   m * magma_ilaenv( 1, "ZUNMBR", "PRC", m, m,  m, -1 ));
                    wrkbl = max( wrkbl, 2*m +   m * magma_ilaenv( 1, "ZUNMBR", "QLN", m, m,  m, -1 ));
                    maxwrk = m*m + wrkbl;
                    minwrk = m*m + 3*m;
                }
                else if (wantqa) {
                    /* Path 4t (N much larger than M, JOBZ='A') */
                    wrkbl =               m +   m * magma_ilaenv( 1, "ZGELQF", " ",   m, n, -1, -1 );
                    wrkbl = max( wrkbl,   m +   n * magma_ilaenv( 1, "ZUNGLQ", " ",   n, n,  m, -1 ));
                    wrkbl = max( wrkbl, 2*m + 2*m * magma_ilaenv( 1, "ZGEBRD", " ",   m, m, -1, -1 ));
                    wrkbl = max( wrkbl, 2*m +   m * magma_ilaenv( 1, "ZUNMBR", "PRC", m, m,  m, -1 ));
                    wrkbl = max( wrkbl, 2*m +   m * magma_ilaenv( 1, "ZUNMBR", "QLN", m, m,  m, -1 ));
                    maxwrk = m*m + wrkbl;
                    minwrk = m*m + 2*m + n;
                }
            }
            else if (n >= mnthr2) {
                /* Path 5t (N much larger than M, but not as much as MNTHR1) */
                maxwrk = 2*m + (m + n) * magma_ilaenv( 1, "ZGEBRD", " ",   m, n, -1, -1 );
                minwrk = 2*m + n;
                if (wantqo) {
                    maxwrk = max( maxwrk, 2*m + m * magma_ilaenv( 1, "ZUNGBR", "P", m, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m * magma_ilaenv( 1, "ZUNGBR", "Q", m, m, n, -1 ));
                    maxwrk += m*n;
                    minwrk += m*m;
                }
                else if (wantqs) {
                    maxwrk = max( maxwrk, 2*m + m * magma_ilaenv( 1, "ZUNGBR", "P", m, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m * magma_ilaenv( 1, "ZUNGBR", "Q", m, m, n, -1 ));
                }
                else if (wantqa) {
                    maxwrk = max( maxwrk, 2*m + n * magma_ilaenv( 1, "ZUNGBR", "P", n, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m * magma_ilaenv( 1, "ZUNGBR", "Q", m, m, n, -1 ));
                }
            }
            else {
                /* Path 6t (N greater than M, but not much larger) */
                maxwrk = 2*m + (m + n) * magma_ilaenv( 1, "ZGEBRD", " ",   m, n, -1, -1 );
                minwrk = 2*m + n;
                if (wantqo) {
                    maxwrk = max( maxwrk, 2*m + m * magma_ilaenv( 1, "ZUNMBR", "PRC", m, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m * magma_ilaenv( 1, "ZUNMBR", "QLN", m, m, n, -1 ));
                    maxwrk += m*n;
                    minwrk += m*m;
                }
                else if (wantqs) {
                    maxwrk = max( maxwrk, 2*m + m * magma_ilaenv( 1, "ZUNGBR", "PRC", m, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m * magma_ilaenv( 1, "ZUNGBR", "QLN", m, m, n, -1 ));
                }
                else if (wantqa) {
                    maxwrk = max( maxwrk, 2*m + n * magma_ilaenv( 1, "ZUNGBR", "PRC", n, n, m, -1 ));
                    maxwrk = max( maxwrk, 2*m + m * magma_ilaenv( 1, "ZUNGBR", "QLN", m, m, n, -1 ));
                }
            }
        }
        maxwrk = max(maxwrk, minwrk);
    }
    if (*info == 0) {
        work[1] = MAGMA_Z_MAKE( maxwrk, 0 );
        if (lwork < minwrk && lwork != -1) {
            *info = -13;
        }
    }

    /* Quick returns */
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    if (lwork == -1) {
        return *info;
    }
    if (m == 0 || n == 0) {
        return *info;
    }

    /* Get machine constants */
    eps = lapackf77_dlamch("P");
    smlnum = sqrt(lapackf77_dlamch("S")) / eps;
    bignum = 1. / smlnum;

    /* Scale A if max element outside range [SMLNUM, BIGNUM] */
    anrm = lapackf77_zlange("M", &m, &n, A(1,1), &lda, dum);
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
        iscl = 1;
        lapackf77_zlascl("G", &izero, &izero, &anrm, &smlnum, &m, &n, A(1,1), &lda, &ierr);
    }
    else if (anrm > bignum) {
        iscl = 1;
        lapackf77_zlascl("G", &izero, &izero, &anrm, &bignum, &m, &n, A(1,1), &lda, &ierr);
    }

    if (m >= n) {
        /* A has at least as many rows as columns. If A has sufficiently */
        /* more rows than columns, first reduce using the QR */
        /* decomposition (if sufficient workspace available) */
        if (m >= mnthr1) {
            if (wantqn) {
                /* Path 1 (M much larger than N, JOBZ='N') */
                /* No singular vectors to be computed */
                itau = 1;
                nwork = itau + n;

                /* Compute A=Q*R */
                /* (CWorkspace: need 2*N, prefer N+N*NB) */
                /* (RWorkspace: need 0) */
                i__1 = lwork - nwork + 1;
                lapackf77_zgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);

                /* Zero out below R */
                i__1 = n - 1;
                lapackf77_zlaset("L", &i__1, &i__1, &c_zero, &c_zero, &A[lda + 2], &lda);
                ie = 1;
                itauq = 1;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in A */
                /* (CWorkspace: need 3*N, prefer 2*N+2*N*NB) */
                /* (RWorkspace: need N) */
                i__1 = lwork - nwork + 1;
                lapackf77_zgebrd(&n, &n, A(1,1), &lda, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__1, &ierr);
                nrwork = ie + n;

                /* Perform bidiagonal SVD, compute singular values only */
                /* (CWorkspace: 0) */
                /* (RWorkspace: need BDSPAN) */
                lapackf77_dbdsdc("U", "N", &n, &s[1], &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], &iwork[1], info);
            }
            else if (wantqo) {
                /* Path 2 (M much larger than N, JOBZ='O') */
                /* N left singular vectors to be overwritten on A and */
                /* N right singular vectors to be computed in VT */
                iu = 1;

                /* WORK[IU] is N by N */
                ldwrku = n;
                ir = iu + ldwrku*n;
                if (lwork >= m*n + n*n + 3*n) {
                    /* WORK[IR] is M by N */
                    ldwrkr = m;
                }
                else {
                    ldwrkr = (lwork - n*n - 3*n) / n;
                }
                itau = ir + ldwrkr*n;
                nwork = itau + n;

                /* Compute A=Q*R */
                /* (CWorkspace: need N*N+2*N, prefer M*N+N+N*NB) */
                /* (RWorkspace: 0) */
                i__1 = lwork - nwork + 1;
                lapackf77_zgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);

                /* Copy R to WORK[ IR ], zeroing out below it */
                lapackf77_zlacpy("U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr);
                i__1 = n - 1;
                lapackf77_zlaset("L", &i__1, &i__1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr);

                /* Generate Q in A */
                /* (CWorkspace: need 2*N, prefer N+N*NB) */
                /* (RWorkspace: 0) */
                i__1 = lwork - nwork + 1;
                lapackf77_zungqr(&m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);
                ie = 1;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in WORK[IR] */
                /* (CWorkspace: need N*N+3*N, prefer M*N+2*N+2*N*NB) */
                /* (RWorkspace: need N) */
                i__1 = lwork - nwork + 1;
                lapackf77_zgebrd(&n, &n, &work[ir], &ldwrkr, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__1, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of R in WORK[IRU] and computing right singular vectors */
                /* of R in WORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru = ie + n;
                irvt = iru + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc("U", "I", &n, &s[1], &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRU] to complex matrix WORK[IU] */
                /* Overwrite WORK[IU] by the left singular vectors of R */
                /* (CWorkspace: need 2*N*N+3*N, prefer M*N+N*N+2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &n, &n, &rwork[iru], &n, &work[iu], &ldwrku);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], &work[iu], &ldwrku, &work[nwork], &i__1, &ierr);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by the right singular vectors of R */
                /* (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &n, &n, &rwork[irvt], &n, VT(1,1), &ldvt);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__1, &ierr);

                /* Multiply Q in A by left singular vectors of R in */
                /* WORK[IU], storing result in WORK[IR] and copying to A */
                /* (CWorkspace: need 2*N*N, prefer N*N+M*N) */
                /* (RWorkspace: 0) */
                for (i = 1; (ldwrkr < 0 ? i >= m : i <= m); i += ldwrkr) {
                    chunk = min(m - i + 1, ldwrkr);
                    blasf77_zgemm("N", "N", &chunk, &n, &n, &c_one, &A[i + lda], &lda, &work[iu], &ldwrku, &c_zero, &work[ir], &ldwrkr);
                    lapackf77_zlacpy("F", &chunk, &n, &work[ir], &ldwrkr, &A[i + lda], &lda);
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
                /* (CWorkspace: need N*N+2*N, prefer N*N+N+N*NB) */
                /* (RWorkspace: 0) */
                i__2 = lwork - nwork + 1;
                lapackf77_zgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);

                /* Copy R to WORK[IR], zeroing out below it */
                lapackf77_zlacpy("U", &n, &n, A(1,1), &lda, &work[ir], &ldwrkr);
                i__1 = n - 1;
                lapackf77_zlaset("L", &i__1, &i__1, &c_zero, &c_zero, &work[ir + 1], &ldwrkr);

                /* Generate Q in A */
                /* (CWorkspace: need 2*N, prefer N+N*NB) */
                /* (RWorkspace: 0) */
                i__2 = lwork - nwork + 1;
                lapackf77_zungqr(&m, &n, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);
                ie = 1;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in WORK[IR] */
                /* (CWorkspace: need N*N+3*N, prefer N*N+2*N+2*N*NB) */
                /* (RWorkspace: need N) */
                i__2 = lwork - nwork + 1;
                lapackf77_zgebrd(&n, &n, &work[ir], &ldwrkr, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru = ie + n;
                irvt = iru + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc("U", "I", &n, &s[1], &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of R */
                /* (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &n, &n, &rwork[iru], &n, U(1,1), &ldu);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &n, &n, &n, &work[ir], &ldwrkr, &work[itauq], U(1,1), &ldu, &work[nwork], &i__2, &ierr);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of R */
                /* (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &n, &n, &rwork[irvt], &n, VT(1,1), &ldvt);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &n, &n, &n, &work[ir], &ldwrkr, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__2, &ierr);

                /* Multiply Q in A by left singular vectors of R in */
                /* WORK[IR], storing result in U */
                /* (CWorkspace: need N*N) */
                /* (RWorkspace: 0) */
                lapackf77_zlacpy("F", &n, &n, U(1,1), &ldu, &work[ir], &ldwrkr);
                blasf77_zgemm("N", "N", &m, &n, &n, &c_one, A(1,1), &lda, &work[ir], &ldwrkr, &c_zero, U(1,1), &ldu);
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
                /* (CWorkspace: need 2*N, prefer N+N*NB) */
                /* (RWorkspace: 0) */
                i__2 = lwork - nwork + 1;
                lapackf77_zgeqrf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);
                lapackf77_zlacpy("L", &m, &n, A(1,1), &lda, U(1,1), &ldu);

                /* Generate Q in U */
                /* (CWorkspace: need N+M, prefer N+M*NB) */
                /* (RWorkspace: 0) */
                i__2 = lwork - nwork + 1;
                lapackf77_zungqr(&m, &m, &n, U(1,1), &ldu, &work[itau], &work[nwork], &i__2, &ierr);

                /* Produce R in A, zeroing out below it */
                i__1 = n - 1;
                lapackf77_zlaset("L", &i__1, &i__1, &c_zero, &c_zero, &A[lda + 2], &lda);
                ie = 1;
                itauq = itau;
                itaup = itauq + n;
                nwork = itaup + n;

                /* Bidiagonalize R in A */
                /* (CWorkspace: need 3*N, prefer 2*N+2*N*NB) */
                /* (RWorkspace: need N) */
                i__2 = lwork - nwork + 1;
                lapackf77_zgebrd(&n, &n, A(1,1), &lda, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);
                iru = ie + n;
                irvt = iru + n*n;
                nrwork = irvt + n*n;

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                lapackf77_dbdsdc("U", "I", &n, &s[1], &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRU] to complex matrix WORK[IU] */
                /* Overwrite WORK[IU] by left singular vectors of R */
                /* (CWorkspace: need N*N+3*N, prefer N*N+2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &n, &n, &rwork[iru], &n, &work[iu], &ldwrku);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &n, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &ierr);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of R */
                /* (CWorkspace: need 3*N, prefer 2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &n, &n, &rwork[irvt], &n, VT(1,1), &ldvt);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__2, &ierr);

                /* Multiply Q in U by left singular vectors of R in */
                /* WORK[IU], storing result in A */
                /* (CWorkspace: need N*N) */
                /* (RWorkspace: 0) */
                blasf77_zgemm("N", "N", &m, &n, &n, &c_one, U(1,1), &ldu, &work[iu], &ldwrku, &c_zero, A(1,1), &lda);

                /* Copy left singular vectors of A from A to U */
                lapackf77_zlacpy("F", &m, &n, A(1,1), &lda, U(1,1), &ldu);
            }
        }
        else if (m >= mnthr2) {
            /* MNTHR2 <= M < MNTHR1 */
            /* Path 5 (M much larger than N, but not as much as MNTHR1) */
            /* Reduce to bidiagonal form without QR decomposition, use */
            /* ZUNGBR and matrix multiplication to compute singular vectors */
            ie = 1;
            nrwork = ie + n;
            itauq = 1;
            itaup = itauq + n;
            nwork = itaup + n;

            /* Bidiagonalize A */
            /* (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB) */
            /* (RWorkspace: need N) */
            i__2 = lwork - nwork + 1;
            lapackf77_zgebrd(&m, &n, A(1,1), &lda, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);
            if (wantqn) {
                /* Compute singular values only */
                /* (Cworkspace: 0) */
                /* (Rworkspace: need BDSPAN) */
                lapackf77_dbdsdc("U", "N", &n, &s[1], &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], &iwork[1], info);
            }
            else if (wantqo) {
                iu = nwork;
                iru = nrwork;
                irvt = iru + n*n;
                nrwork = irvt + n*n;

                /* Copy A to VT, generate P**H */
                /* (Cworkspace: need 2*N, prefer N+N*NB) */
                /* (Rworkspace: 0) */
                lapackf77_zlacpy("U", &n, &n, A(1,1), &lda, VT(1,1), &ldvt);
                i__2 = lwork - nwork + 1;
                lapackf77_zungbr("P", &n, &n, &n, VT(1,1), &ldvt, &work[itaup], &work[nwork], &i__2, &ierr);

                /* Generate Q in A */
                /* (CWorkspace: need 2*N, prefer N+N*NB) */
                /* (RWorkspace: 0) */
                i__2 = lwork - nwork + 1;
                lapackf77_zungbr("Q", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[nwork], &i__2, &ierr);

                if (lwork >= m*n + 3*n) {
                    /* WORK[ IU ] is M by N */
                    ldwrku = m;
                }
                else {
                    /* WORK[IU] is LDWRKU by N */
                    ldwrku = (lwork - 3*n) / n;
                }
                nwork = iu + ldwrku*n;

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                lapackf77_dbdsdc("U", "I", &n, &s[1], &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Multiply real matrix RWORK[IRVT] by P**H in VT, */
                /* storing the result in WORK[IU], copying to VT */
                /* (Cworkspace: need 0) */
                /* (Rworkspace: need 3*N*N) */
                lapackf77_zlarcm(&n, &n, &rwork[irvt], &n, VT(1,1), &ldvt, &work[iu], &ldwrku, &rwork[nrwork]);
                lapackf77_zlacpy("F", &n, &n, &work[iu], &ldwrku, VT(1,1), &ldvt);

                /* Multiply Q in A by real matrix RWORK[IRU], storing the */
                /* result in WORK[IU], copying to A */
                /* (CWorkspace: need N*N, prefer M*N) */
                /* (Rworkspace: need 3*N*N, prefer N*N+2*M*N) */
                nrwork = irvt;
                for (i = 1; (ldwrku < 0 ? i >= m : i <= m); i += ldwrku) {
                    chunk = min(m - i + 1, ldwrku);
                    lapackf77_zlacrm(&chunk, &n, &A[i + lda], &lda, &rwork[iru], &n, &work[iu], &ldwrku, &rwork[nrwork]);
                    lapackf77_zlacpy("F", &chunk, &n, &work[iu], &ldwrku, &A[i + lda], &lda);
                }
            }
            else if (wantqs) {
                /* Copy A to VT, generate P**H */
                /* (Cworkspace: need 2*N, prefer N+N*NB) */
                /* (Rworkspace: 0) */
                lapackf77_zlacpy("U", &n, &n, A(1,1), &lda, VT(1,1), &ldvt);
                i__1 = lwork - nwork + 1;
                lapackf77_zungbr("P", &n, &n, &n, VT(1,1), &ldvt, &work[itaup], &work[nwork], &i__1, &ierr);

                /* Copy A to U, generate Q */
                /* (Cworkspace: need 2*N, prefer N+N*NB) */
                /* (Rworkspace: 0) */
                lapackf77_zlacpy("L", &m, &n, A(1,1), &lda, U(1,1), &ldu);
                i__1 = lwork - nwork + 1;
                lapackf77_zungbr("Q", &m, &n, &n, U(1,1), &ldu, &work[itauq], &work[nwork], &i__1, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru = nrwork;
                irvt = iru + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc("U", "I", &n, &s[1], &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Multiply real matrix RWORK[IRVT] by P**H in VT, */
                /* storing the result in A, copying to VT */
                /* (Cworkspace: need 0) */
                /* (Rworkspace: need 3*N*N) */
                lapackf77_zlarcm(&n, &n, &rwork[irvt], &n, VT(1,1), &ldvt, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_zlacpy("F", &n, &n, A(1,1), &lda, VT(1,1), &ldvt);

                /* Multiply Q in U by real matrix RWORK[IRU], storing the */
                /* result in A, copying to U */
                /* (CWorkspace: need 0) */
                /* (Rworkspace: need N*N+2*M*N) */
                nrwork = irvt;
                lapackf77_zlacrm(&m, &n, U(1,1), &ldu, &rwork[iru], &n, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_zlacpy("F", &m, &n, A(1,1), &lda, U(1,1), &ldu);
            }
            else {
                /* Copy A to VT, generate P**H */
                /* (Cworkspace: need 2*N, prefer N+N*NB) */
                /* (Rworkspace: 0) */
                lapackf77_zlacpy("U", &n, &n, A(1,1), &lda, VT(1,1), &ldvt);
                i__1 = lwork - nwork + 1;
                lapackf77_zungbr("P", &n, &n, &n, VT(1,1), &ldvt, &work[itaup], &work[nwork], &i__1, &ierr);

                /* Copy A to U, generate Q */
                /* (Cworkspace: need 2*N, prefer N+N*NB) */
                /* (Rworkspace: 0) */
                lapackf77_zlacpy("L", &m, &n, A(1,1), &lda, U(1,1), &ldu);
                i__1 = lwork - nwork + 1;
                lapackf77_zungbr("Q", &m, &m, &n, U(1,1), &ldu, &work[itauq], &work[nwork], &i__1, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru = nrwork;
                irvt = iru + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc("U", "I", &n, &s[1], &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Multiply real matrix RWORK[IRVT] by P**H in VT, */
                /* storing the result in A, copying to VT */
                /* (Cworkspace: need 0) */
                /* (Rworkspace: need 3*N*N) */
                lapackf77_zlarcm(&n, &n, &rwork[irvt], &n, VT(1,1), &ldvt, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_zlacpy("F", &n, &n, A(1,1), &lda, VT(1,1), &ldvt);

                /* Multiply Q in U by real matrix RWORK[IRU], storing the */
                /* result in A, copying to U */
                /* (CWorkspace: 0) */
                /* (Rworkspace: need 3*N*N) */
                nrwork = irvt;
                lapackf77_zlacrm(&m, &n, U(1,1), &ldu, &rwork[iru], &n, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_zlacpy("F", &m, &n, A(1,1), &lda, U(1,1), &ldu);
            }
        }
        else {
            /* M < MNTHR2 */
            /* Path 6 (M at least N, but not much larger) */
            /* Reduce to bidiagonal form without QR decomposition */
            /* Use ZUNMBR to compute singular vectors */
            ie = 1;
            nrwork = ie + n;
            itauq = 1;
            itaup = itauq + n;
            nwork = itaup + n;

            /* Bidiagonalize A */
            /* (CWorkspace: need 2*N+M, prefer 2*N+(M+N)*NB) */
            /* (RWorkspace: need N) */
            i__1 = lwork - nwork + 1;
            lapackf77_zgebrd(&m, &n, A(1,1), &lda, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__1, &ierr);
            if (wantqn) {
                /* Compute singular values only */
                /* (Cworkspace: 0) */
                /* (Rworkspace: need BDSPAN) */
                lapackf77_dbdsdc("U", "N", &n, &s[1], &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], &iwork[1], info);
            }
            else if (wantqo) {
                iu = nwork;
                iru = nrwork;
                irvt = iru + n*n;
                nrwork = irvt + n*n;
                if (lwork >= m*n + 3*n) {
                    /* WORK[ IU ] is M by N */
                    ldwrku = m;
                }
                else {
                    /* WORK[ IU ] is LDWRKU by N */
                    ldwrku = (lwork - 3*n) / n;
                }
                nwork = iu + ldwrku*n;

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                lapackf77_dbdsdc("U", "I", &n, &s[1], &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of A */
                /* (Cworkspace: need 2*N, prefer N+N*NB) */
                /* (Rworkspace: need 0) */
                lapackf77_zlacp2("F", &n, &n, &rwork[irvt], &n, VT(1,1), &ldvt);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__1, &ierr);

                if (lwork >= m*n + 3*n) {
                /* Copy real matrix RWORK[IRU] to complex matrix WORK[IU] */
                /* Overwrite WORK[IU] by left singular vectors of A, copying */
                /* to A */
                /* (Cworkspace: need M*N+2*N, prefer M*N+N+N*NB) */
                /* (Rworkspace: need 0) */
                    lapackf77_zlaset("F", &m, &n, &c_zero, &c_zero, &work[iu], &ldwrku);
                    lapackf77_zlacp2("F", &n, &n, &rwork[iru], &n, &work[iu], &ldwrku);
                    i__1 = lwork - nwork + 1;
                    lapackf77_zunmbr("Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[iu], &ldwrku, &work[nwork], &i__1, &ierr);
                    lapackf77_zlacpy("F", &m, &n, &work[iu], &ldwrku, A(1,1), &lda);
                }
                else {
                    /* Generate Q in A */
                    /* (Cworkspace: need 2*N, prefer N+N*NB) */
                    /* (Rworkspace: need 0) */
                    i__1 = lwork - nwork + 1;
                    lapackf77_zungbr("Q", &m, &n, &n, A(1,1), &lda, &work[itauq], &work[nwork], &i__1, &ierr);

                    /* Multiply Q in A by real matrix RWORK[IRU], storing the */
                    /* result in WORK[IU], copying to A */
                    /* (CWorkspace: need N*N, prefer M*N) */
                    /* (Rworkspace: need 3*N*N, prefer N*N+2*M*N) */
                    nrwork = irvt;
                    for (i = 1; (ldwrku < 0 ? i >= m : i <= m); i += ldwrku) {
                        chunk = min(m - i + 1, ldwrku);
                        lapackf77_zlacrm(&chunk, &n, &A[i + lda], &lda, &rwork[iru], &n, &work[iu], &ldwrku, &rwork[nrwork]);
                        lapackf77_zlacpy("F", &chunk, &n, &work[iu], &ldwrku, &A[i + lda], &lda);
                    }
                }
            }
            else if (wantqs) {
                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru = nrwork;
                irvt = iru + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc("U", "I", &n, &s[1], &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of A */
                /* (CWorkspace: need 3*N, prefer 2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlaset("F", &m, &n, &c_zero, &c_zero, U(1,1), &ldu)
                        ;
                lapackf77_zlacp2("F", &n, &n, &rwork[iru], &n, U(1,1), &ldu);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &m, &n, &n, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__2, &ierr);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of A */
                /* (CWorkspace: need 3*N, prefer 2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &n, &n, &rwork[irvt], &n, VT(1,1), &ldvt);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__2, &ierr);
            }
            else {
                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru = nrwork;
                irvt = iru + n*n;
                nrwork = irvt + n*n;
                lapackf77_dbdsdc("U", "I", &n, &s[1], &rwork[ie], &rwork[iru], &n, &rwork[irvt], &n, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Set the right corner of U to identity matrix */
                lapackf77_zlaset("F", &m, &m, &c_zero, &c_zero, U(1,1), &ldu)
                        ;
                if (m > n) {
                    i__1 = m - n;
                    lapackf77_zlaset("F", &i__1, &i__1, &c_zero, &c_one, &U[n + 1 + (n + 1)*ldu], &ldu);
                }

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of A */
                /* (CWorkspace: need 2*N+M, prefer 2*N+M*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &n, &n, &rwork[iru], &n, U(1,1), &ldu);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__2, &ierr);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of A */
                /* (CWorkspace: need 3*N, prefer 2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &n, &n, &rwork[irvt], &n, VT(1,1), &ldvt);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &n, &n, &n, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__2, &ierr);
            }
        }
    }
    else {
        /* A has more columns than rows. If A has sufficiently more */
        /* columns than rows, first reduce using the LQ decomposition (if */
        /* sufficient workspace available) */
        if (n >= mnthr1) {
            if (wantqn) {
                /* Path 1t (N much larger than M, JOBZ='N') */
                /* No singular vectors to be computed */
                itau = 1;
                nwork = itau + m;

                /* Compute A=L*Q */
                /* (CWorkspace: need 2*M, prefer M+M*NB) */
                /* (RWorkspace: 0) */
                i__2 = lwork - nwork + 1;
                lapackf77_zgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);

                /* Zero out above L */
                i__1 = m - 1;
                lapackf77_zlaset("U", &i__1, &i__1, &c_zero, &c_zero, &A[(2*lda) + 1], &lda);
                ie = 1;
                itauq = 1;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in A */
                /* (CWorkspace: need 3*M, prefer 2*M+2*M*NB) */
                /* (RWorkspace: need M) */
                i__2 = lwork - nwork + 1;
                lapackf77_zgebrd(&m, &m, A(1,1), &lda, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);
                nrwork = ie + m;

                /* Perform bidiagonal SVD, compute singular values only */
                /* (CWorkspace: 0) */
                /* (RWorkspace: need BDSPAN) */
                lapackf77_dbdsdc("U", "N", &m, &s[1], &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], &iwork[1], info);
            }
            else if (wantqo) {
                /* Path 2t (N much larger than M, JOBZ='O') */
                /* M right singular vectors to be overwritten on A and */
                /* M left singular vectors to be computed in U */
                ivt = 1;
                ldwkvt = m;

                /* WORK[IVT] is M by M */
                il = ivt + ldwkvt*m;
                if (lwork >= m*n + m*m + 3*m) {
                    /* WORK[IL] M by N */
                    ldwrkl = m;
                    chunk = n;
                }
                else {
                    /* WORK[IL] is M by CHUNK */
                    ldwrkl = m;
                    chunk = (lwork - m*m - 3*m) / m;
                }
                itau = il + ldwrkl*chunk;
                nwork = itau + m;

                /* Compute A=L*Q */
                /* (CWorkspace: need 2*M, prefer M+M*NB) */
                /* (RWorkspace: 0) */
                i__2 = lwork - nwork + 1;
                lapackf77_zgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);

                /* Copy L to WORK[IL], zeroing about above it */
                lapackf77_zlacpy("L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl);
                i__1 = m - 1;
                lapackf77_zlaset("U", &i__1, &i__1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl);

                /* Generate Q in A */
                /* (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB) */
                /* (RWorkspace: 0) */
                i__2 = lwork - nwork + 1;
                lapackf77_zunglq(&m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &i__2, &ierr);
                ie = 1;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in WORK[IL] */
                /* (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB) */
                /* (RWorkspace: need M) */
                i__2 = lwork - nwork + 1;
                lapackf77_zgebrd(&m, &m, &work[il], &ldwrkl, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru = ie + m;
                irvt = iru + m*m;
                nrwork = irvt + m*m;
                lapackf77_dbdsdc("U", "I", &m, &s[1], &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRU] to complex matrix WORK[IU] */
                /* Overwrite WORK[IU] by the left singular vectors of L */
                /* (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &m, &m, &rwork[iru], &m, U(1,1), &ldu);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U(1,1), &ldu, &work[nwork], &i__2, &ierr);

                /* Copy real matrix RWORK[IRVT] to complex matrix WORK[IVT] */
                /* Overwrite WORK[IVT] by the right singular vectors of L */
                /* (CWorkspace: need N*N+3*N, prefer M*N+2*N+N*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &m, &m, &rwork[irvt], &m, &work[ivt], &ldwkvt);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, &ierr);

                /* Multiply right singular vectors of L in WORK[IL] by Q */
                /* in A, storing result in WORK[IL] and copying to A */
                /* (CWorkspace: need 2*M*M, prefer M*M+M*N)) */
                /* (RWorkspace: 0) */
                for (i = 1; (chunk < 0 ? i >= n : i <= n); i += chunk) {
                    blk = min(n - i + 1, chunk);
                    blasf77_zgemm("N", "N", &m, &blk, &m, &c_one, &work[ivt], &m, &A[i*lda + 1], &lda, &c_zero, &work[il], &ldwrkl);
                    lapackf77_zlacpy("F", &m, &blk, &work[il], &ldwrkl, &A[i*lda + 1], &lda);
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
                /* (CWorkspace: need 2*M, prefer M+M*NB) */
                /* (RWorkspace: 0) */
                i__1 = lwork - nwork + 1;
                lapackf77_zgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);

                /* Copy L to WORK[IL], zeroing out above it */
                lapackf77_zlacpy("L", &m, &m, A(1,1), &lda, &work[il], &ldwrkl);
                i__1 = m - 1;
                lapackf77_zlaset("U", &i__1, &i__1, &c_zero, &c_zero, &work[il + ldwrkl], &ldwrkl);

                /* Generate Q in A */
                /* (CWorkspace: need M*M+2*M, prefer M*M+M+M*NB) */
                /* (RWorkspace: 0) */
                i__1 = lwork - nwork + 1;
                lapackf77_zunglq(&m, &n, &m, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);
                ie = 1;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in WORK[IL] */
                /* (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB) */
                /* (RWorkspace: need M) */
                i__1 = lwork - nwork + 1;
                lapackf77_zgebrd(&m, &m, &work[il], &ldwrkl, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__1, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru = ie + m;
                irvt = iru + m*m;
                nrwork = irvt + m*m;
                lapackf77_dbdsdc("U", "I", &m, &s[1], &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of L */
                /* (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &m, &m, &rwork[iru], &m, U(1,1), &ldu);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &m, &m, &m, &work[il], &ldwrkl, &work[itauq], U(1,1), &ldu, &work[nwork], &i__1, &ierr);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by left singular vectors of L */
                /* (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &m, &m, &rwork[irvt], &m, VT(1,1), &ldvt);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &m, &m, &m, &work[il], &ldwrkl, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__1, &ierr);

                /* Copy VT to WORK[IL], multiply right singular vectors of L */
                /* in WORK[IL] by Q in A, storing result in VT */
                /* (CWorkspace: need M*M) */
                /* (RWorkspace: 0) */
                lapackf77_zlacpy("F", &m, &m, VT(1,1), &ldvt, &work[il], &ldwrkl);
                blasf77_zgemm("N", "N", &m, &n, &m, &c_one, &work[il], &ldwrkl, A(1,1), &lda, &c_zero, VT(1,1), &ldvt);
            }
            else if (wantqa) {
                /* Path 9t (N much larger than M, JOBZ='A') */
                /* N right singular vectors to be computed in VT and */
                /* M left singular vectors to be computed in U */
                ivt = 1;

                /* WORK[IVT] is M by M */
                ldwkvt = m;
                itau = ivt + ldwkvt*m;
                nwork = itau + m;

                /* Compute A=L*Q, copying result to VT */
                /* (CWorkspace: need 2*M, prefer M+M*NB) */
                /* (RWorkspace: 0) */
                i__1 = lwork - nwork + 1;
                lapackf77_zgelqf(&m, &n, A(1,1), &lda, &work[itau], &work[nwork], &i__1, &ierr);
                lapackf77_zlacpy("U", &m, &n, A(1,1), &lda, VT(1,1), &ldvt);

                /* Generate Q in VT */
                /* (CWorkspace: need M+N, prefer M+N*NB) */
                /* (RWorkspace: 0) */
                i__1 = lwork - nwork + 1;
                lapackf77_zunglq(&n, &n, &m, VT(1,1), &ldvt, &work[itau], &work[nwork], &i__1, &ierr);

                /* Produce L in A, zeroing out above it */
                i__1 = m - 1;
                lapackf77_zlaset("U", &i__1, &i__1, &c_zero, &c_zero, &A[(lda*2) + 1], &lda);
                ie = 1;
                itauq = itau;
                itaup = itauq + m;
                nwork = itaup + m;

                /* Bidiagonalize L in A */
                /* (CWorkspace: need M*M+3*M, prefer M*M+2*M+2*M*NB) */
                /* (RWorkspace: need M) */
                i__1 = lwork - nwork + 1;
                lapackf77_zgebrd(&m, &m, A(1,1), &lda, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__1, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                iru = ie + m;
                irvt = iru + m*m;
                nrwork = irvt + m*m;
                lapackf77_dbdsdc("U", "I", &m, &s[1], &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of L */
                /* (CWorkspace: need 3*M, prefer 2*M+M*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &m, &m, &rwork[iru], &m, U(1,1), &ldu);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &m, &m, &m, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__1, &ierr);

                /* Copy real matrix RWORK[IRVT] to complex matrix WORK[IVT] */
                /* Overwrite WORK[IVT] by right singular vectors of L */
                /* (CWorkspace: need M*M+3*M, prefer M*M+2*M+M*NB) */
                /* (RWorkspace: 0) */
                lapackf77_zlacp2("F", &m, &m, &rwork[irvt], &m, &work[ivt], &ldwkvt);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &m, &m, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwkvt, &work[nwork], &i__1, &ierr);

                /* Multiply right singular vectors of L in WORK[IVT] by */
                /* Q in VT, storing result in A */
                /* (CWorkspace: need M*M) */
                /* (RWorkspace: 0) */
                blasf77_zgemm("N", "N", &m, &n, &m, &c_one, &work[ivt], &ldwkvt, VT(1,1), &ldvt, &c_zero, A(1,1), &lda);

                /* Copy right singular vectors of A from A to VT */
                lapackf77_zlacpy("F", &m, &n, A(1,1), &lda, VT(1,1), &ldvt);
            }
        }
        else if (n >= mnthr2) {
            /* MNTHR2 <= N < MNTHR1 */
            /* Path 5t (N much larger than M, but not as much as MNTHR1) */
            /* Reduce to bidiagonal form without QR decomposition, use */
            /* ZUNGBR and matrix multiplication to compute singular vectors */
            ie = 1;
            nrwork = ie + m;
            itauq = 1;
            itaup = itauq + m;
            nwork = itaup + m;

            /* Bidiagonalize A */
            /* (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB) */
            /* (RWorkspace: M) */
            i__1 = lwork - nwork + 1;
            lapackf77_zgebrd(&m, &n, A(1,1), &lda, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__1, &ierr);

            if (wantqn) {
                /* Compute singular values only */
                /* (Cworkspace: 0) */
                /* (Rworkspace: need BDSPAN) */
                lapackf77_dbdsdc("L", "N", &m, &s[1], &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], &iwork[1], info);
            }
            else if (wantqo) {
                irvt = nrwork;
                iru = irvt + m*m;
                nrwork = iru + m*m;
                ivt = nwork;

                /* Copy A to U, generate Q */
                /* (Cworkspace: need 2*M, prefer M+M*NB) */
                /* (Rworkspace: 0) */
                lapackf77_zlacpy("L", &m, &m, A(1,1), &lda, U(1,1), &ldu);
                i__1 = lwork - nwork + 1;
                lapackf77_zungbr("Q", &m, &m, &n, U(1,1), &ldu, &work[itauq], &work[nwork], &i__1, &ierr);

                /* Generate P**H in A */
                /* (Cworkspace: need 2*M, prefer M+M*NB) */
                /* (Rworkspace: 0) */
                i__1 = lwork - nwork + 1;
                lapackf77_zungbr("P", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[nwork], &i__1, &ierr);

                ldwkvt = m;
                if (lwork >= m*n + 3*m) {
                    /* WORK[ IVT ] is M by N */
                    nwork = ivt + ldwkvt*n;
                    chunk = n;
                }
                else {
                    /* WORK[ IVT ] is M by CHUNK */
                    chunk = (lwork - 3*m) / m;
                    nwork = ivt + ldwkvt*chunk;
                }

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                lapackf77_dbdsdc("L", "I", &m, &s[1], &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Multiply Q in U by real matrix RWORK[IRVT] */
                /* storing the result in WORK[IVT], copying to U */
                /* (Cworkspace: need 0) */
                /* (Rworkspace: need 2*M*M) */
                lapackf77_zlacrm(&m, &m, U(1,1), &ldu, &rwork[iru], &m, &work[ivt], &ldwkvt, &rwork[nrwork]);
                lapackf77_zlacpy("F", &m, &m, &work[ivt], &ldwkvt, U(1,1), &ldu);

                /* Multiply RWORK[IRVT] by P**H in A, storing the */
                /* result in WORK[IVT], copying to A */
                /* (CWorkspace: need M*M, prefer M*N) */
                /* (Rworkspace: need 2*M*M, prefer 2*M*N) */
                nrwork = iru;
                for (i = 1; (chunk < 0 ? i >= n : i <= n); i += chunk) {
                    blk = min(n - i + 1, chunk);
                    lapackf77_zlarcm(&m, &blk, &rwork[irvt], &m, &A[i*lda + 1], &lda, &work[ivt], &ldwkvt, &rwork[nrwork]);
                    lapackf77_zlacpy("F", &m, &blk, &work[ivt], &ldwkvt, &A[i*lda + 1], &lda);
                }
            }
            else if (wantqs) {
                /* Copy A to U, generate Q */
                /* (Cworkspace: need 2*M, prefer M+M*NB) */
                /* (Rworkspace: 0) */
                lapackf77_zlacpy("L", &m, &m, A(1,1), &lda, U(1,1), &ldu);
                i__2 = lwork - nwork + 1;
                lapackf77_zungbr("Q", &m, &m, &n, U(1,1), &ldu, &work[itauq], &work[nwork], &i__2, &ierr);

                /* Copy A to VT, generate P**H */
                /* (Cworkspace: need 2*M, prefer M+M*NB) */
                /* (Rworkspace: 0) */
                lapackf77_zlacpy("U", &m, &n, A(1,1), &lda, VT(1,1), &ldvt);
                i__2 = lwork - nwork + 1;
                lapackf77_zungbr("P", &m, &n, &m, VT(1,1), &ldvt, &work[itaup], &work[nwork], &i__2, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                irvt = nrwork;
                iru = irvt + m*m;
                nrwork = iru + m*m;
                lapackf77_dbdsdc("L", "I", &m, &s[1], &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Multiply Q in U by real matrix RWORK[IRU], storing the */
                /* result in A, copying to U */
                /* (CWorkspace: need 0) */
                /* (Rworkspace: need 3*M*M) */
                lapackf77_zlacrm(&m, &m, U(1,1), &ldu, &rwork[iru], &m, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_zlacpy("F", &m, &m, A(1,1), &lda, U(1,1), &ldu);

                /* Multiply real matrix RWORK[IRVT] by P**H in VT, */
                /* storing the result in A, copying to VT */
                /* (Cworkspace: need 0) */
                /* (Rworkspace: need M*M+2*M*N) */
                nrwork = iru;
                lapackf77_zlarcm(&m, &n, &rwork[irvt], &m, VT(1,1), &ldvt, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_zlacpy("F", &m, &n, A(1,1), &lda, VT(1,1), &ldvt);
            }
            else {
                /* Copy A to U, generate Q */
                /* (Cworkspace: need 2*M, prefer M+M*NB) */
                /* (Rworkspace: 0) */
                lapackf77_zlacpy("L", &m, &m, A(1,1), &lda, U(1,1), &ldu);
                i__2 = lwork - nwork + 1;
                lapackf77_zungbr("Q", &m, &m, &n, U(1,1), &ldu, &work[itauq], &work[nwork], &i__2, &ierr);

                /* Copy A to VT, generate P**H */
                /* (Cworkspace: need 2*M, prefer M+M*NB) */
                /* (Rworkspace: 0) */
                lapackf77_zlacpy("U", &m, &n, A(1,1), &lda, VT(1,1), &ldvt);
                i__2 = lwork - nwork + 1;
                lapackf77_zungbr("P", &n, &n, &m, VT(1,1), &ldvt, &work[itaup], &work[nwork], &i__2, &ierr);

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                irvt = nrwork;
                iru = irvt + m*m;
                nrwork = iru + m*m;
                lapackf77_dbdsdc("L", "I", &m, &s[1], &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Multiply Q in U by real matrix RWORK[IRU], storing the */
                /* result in A, copying to U */
                /* (CWorkspace: need 0) */
                /* (Rworkspace: need 3*M*M) */
                lapackf77_zlacrm(&m, &m, U(1,1), &ldu, &rwork[iru], &m, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_zlacpy("F", &m, &m, A(1,1), &lda, U(1,1), &ldu);

                /* Multiply real matrix RWORK[IRVT] by P**H in VT, */
                /* storing the result in A, copying to VT */
                /* (Cworkspace: need 0) */
                /* (Rworkspace: need M*M+2*M*N) */
                lapackf77_zlarcm(&m, &n, &rwork[irvt], &m, VT(1,1), &ldvt, A(1,1), &lda, &rwork[nrwork]);
                lapackf77_zlacpy("F", &m, &n, A(1,1), &lda, VT(1,1), &ldvt);
            }
        }
        else {
            /* N < MNTHR2 */
            /* Path 6t (N greater than M, but not much larger) */
            /* Reduce to bidiagonal form without LQ decomposition */
            /* Use ZUNMBR to compute singular vectors */
            ie = 1;
            nrwork = ie + m;
            itauq = 1;
            itaup = itauq + m;
            nwork = itaup + m;

            /* Bidiagonalize A */
            /* (CWorkspace: need 2*M+N, prefer 2*M+(M+N)*NB) */
            /* (RWorkspace: M) */
            i__2 = lwork - nwork + 1;
            lapackf77_zgebrd(&m, &n, A(1,1), &lda, &s[1], &rwork[ie], &work[itauq], &work[itaup], &work[nwork], &i__2, &ierr);
            if (wantqn) {
                /* Compute singular values only */
                /* (Cworkspace: 0) */
                /* (Rworkspace: need BDSPAN) */
                lapackf77_dbdsdc("L", "N", &m, &s[1], &rwork[ie], dum, &ione, dum, &ione, dum, idum, &rwork[nrwork], &iwork[1], info);
            }
            else if (wantqo) {
                ldwkvt = m;
                ivt = nwork;
                if (lwork >= m*n + 3*m) {
                    /* WORK[ IVT ] is M by N */
                    lapackf77_zlaset("F", &m, &n, &c_zero, &c_zero, &work[ivt], &ldwkvt);
                    nwork = ivt + ldwkvt*n;
                }
                else {
                    /* WORK[ IVT ] is M by CHUNK */
                    chunk = (lwork - 3*m) / m;
                    nwork = ivt + ldwkvt*chunk;
                }

                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                irvt = nrwork;
                iru = irvt + m*m;
                nrwork = iru + m*m;
                lapackf77_dbdsdc("L", "I", &m, &s[1], &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of A */
                /* (Cworkspace: need 2*M, prefer M+M*NB) */
                /* (Rworkspace: need 0) */
                lapackf77_zlacp2("F", &m, &m, &rwork[iru], &m, U(1,1), &ldu);
                i__2 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__2, &ierr);

                if (lwork >= m*n + 3*m) {
                /* Copy real matrix RWORK[IRVT] to complex matrix WORK[IVT] */
                /* Overwrite WORK[IVT] by right singular vectors of A, */
                /* copying to A */
                /* (Cworkspace: need M*N+2*M, prefer M*N+M+M*NB) */
                /* (Rworkspace: need 0) */
                    lapackf77_zlacp2("F", &m, &m, &rwork[irvt], &m, &work[ivt], &ldwkvt);
                    i__2 = lwork - nwork + 1;
                    lapackf77_zunmbr("P", "R", "C", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, &ierr);
                    lapackf77_zlacpy("F", &m, &n, &work[ivt], &ldwkvt, A(1,1), &lda);
                }
                else {
                    /* Generate P**H in A */
                    /* (Cworkspace: need 2*M, prefer M+M*NB) */
                    /* (Rworkspace: need 0) */
                    i__2 = lwork - nwork + 1;
                    lapackf77_zungbr("P", &m, &n, &m, A(1,1), &lda, &work[itaup], &work[nwork], &i__2, &ierr);

                    /* Multiply Q in A by real matrix RWORK[IRU], storing the */
                    /* result in WORK[IU], copying to A */
                    /* (CWorkspace: need M*M, prefer M*N) */
                    /* (Rworkspace: need 3*M*M, prefer M*M+2*M*N) */
                    nrwork = iru;
                    for (i = 1; (chunk < 0 ? i >= n : i <= n); i += chunk) {
                        blk = min(n - i + 1, chunk);
                        lapackf77_zlarcm(&m, &blk, &rwork[irvt], &m, &A[i*lda + 1], &lda, &work[ivt], &ldwkvt, &rwork[nrwork]);
                        lapackf77_zlacpy("F", &m, &blk, &work[ivt], &ldwkvt, &A[i*lda + 1], &lda);
                    }
                }
            }
            else if (wantqs) {
                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                irvt = nrwork;
                iru = irvt + m*m;
                nrwork = iru + m*m;
                lapackf77_dbdsdc("L", "I", &m, &s[1], &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of A */
                /* (CWorkspace: need 3*M, prefer 2*M+M*NB) */
                /* (RWorkspace: M*M) */
                lapackf77_zlacp2("F", &m, &m, &rwork[iru], &m, U(1,1), &ldu);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__1, &ierr);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of A */
                /* (CWorkspace: need 3*M, prefer 2*M+M*NB) */
                /* (RWorkspace: M*M) */
                lapackf77_zlaset("F", &m, &n, &c_zero, &c_zero, VT(1,1), &ldvt);
                lapackf77_zlacp2("F", &m, &m, &rwork[irvt], &m, VT(1,1), &ldvt);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &m, &n, &m, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__1, &ierr);
            }
            else {
                /* Perform bidiagonal SVD, computing left singular vectors */
                /* of bidiagonal matrix in RWORK[IRU] and computing right */
                /* singular vectors of bidiagonal matrix in RWORK[IRVT] */
                /* (CWorkspace: need 0) */
                /* (RWorkspace: need BDSPAC) */
                irvt = nrwork;
                iru = irvt + m*m;
                nrwork = iru + m*m;

                lapackf77_dbdsdc("L", "I", &m, &s[1], &rwork[ie], &rwork[iru], &m, &rwork[irvt], &m, dum, idum, &rwork[nrwork], &iwork[1], info);

                /* Copy real matrix RWORK[IRU] to complex matrix U */
                /* Overwrite U by left singular vectors of A */
                /* (CWorkspace: need 3*M, prefer 2*M+M*NB) */
                /* (RWorkspace: M*M) */
                lapackf77_zlacp2("F", &m, &m, &rwork[iru], &m, U(1,1), &ldu);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("Q", "L", "N", &m, &m, &n, A(1,1), &lda, &work[itauq], U(1,1), &ldu, &work[nwork], &i__1, &ierr);

                /* Set all of VT to identity matrix */
                lapackf77_zlaset("F", &n, &n, &c_zero, &c_one, VT(1,1), &ldvt);

                /* Copy real matrix RWORK[IRVT] to complex matrix VT */
                /* Overwrite VT by right singular vectors of A */
                /* (CWorkspace: need 2*M+N, prefer 2*M+N*NB) */
                /* (RWorkspace: M*M) */
                lapackf77_zlacp2("F", &m, &m, &rwork[irvt], &m, VT(1,1), &ldvt);
                i__1 = lwork - nwork + 1;
                lapackf77_zunmbr("P", "R", "C", &n, &n, &m, A(1,1), &lda, &work[itaup], VT(1,1), &ldvt, &work[nwork], &i__1, &ierr);
            }
        }
    }

    /* Undo scaling if necessary */
    if (iscl == 1) {
        if (anrm > bignum) {
            lapackf77_dlascl("G", &izero, &izero, &bignum, &anrm, &minmn, &ione, &s[1], &minmn, &ierr);
        }
        if (*info != 0 && anrm > bignum) {
            i__1 = minmn - 1;
            lapackf77_dlascl("G", &izero, &izero, &bignum, &anrm, &i__1, &ione, &rwork[ie], &minmn, &ierr);
        }
        if (anrm < smlnum) {
            lapackf77_dlascl("G", &izero, &izero, &smlnum, &anrm, &minmn, &ione, &s[1], &minmn, &ierr);
        }
        if (*info != 0 && anrm < smlnum) {
            i__1 = minmn - 1;
            lapackf77_dlascl("G", &izero, &izero, &smlnum, &anrm, &i__1, &ione, &rwork[ie], &minmn, &ierr);
        }
    }

    /* Return optimal workspace in WORK[1] */
    work[1] = MAGMA_Z_MAKE( maxwrk, 0 );

    return *info;
} /* magma_zgesdd */
