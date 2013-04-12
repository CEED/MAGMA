/*  -- MAGMA (version 1.1) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    November 2011

    @author Raffaele Solca

    @precisions normal d -> s
*/
#include "common_magma.h"

#define Q(ix, iy) (q + (ix) + ldq * (iy))

#define magma_dlaed2 FORTRAN_NAME( magma_dlaed2, MAGMA_DLAED2 )

extern "C" {

    magma_int_t magma_dlaex3_m(magma_int_t nrgpu,
                               magma_int_t k, magma_int_t n, magma_int_t n1, double* d,
                               double* q, magma_int_t ldq, double rho,
                               double* dlamda, double* q2, magma_int_t* indx,
                               magma_int_t* ctot, double* w, double* s, magma_int_t* indxq,
                               double** dwork, magma_queue_t stream[MagmaMaxGPUs][2],
                               char range, double vl, double vu, magma_int_t il, magma_int_t iu,
                               magma_int_t* info );

    void magma_dlaed2( magma_int_t* k, magma_int_t* n, magma_int_t* cutpnt,
                       double* d, double* q, magma_int_t* ldq, magma_int_t* indxq,
                       double* rho, double* z,
                       double*  dlmda, double* w, double* q2,
                       magma_int_t* indx, magma_int_t* indxc, magma_int_t* indxp,
                       magma_int_t* coltyp, magma_int_t* info);

}


extern "C" magma_int_t
magma_dlaex1_m(magma_int_t nrgpu, magma_int_t n, double* d, double* q, magma_int_t ldq,
               magma_int_t* indxq, double rho, magma_int_t cutpnt,
               double* work, magma_int_t* iwork, double** dwork,
               magma_queue_t stream[MagmaMaxGPUs][2],
               char range, double vl, double vu,
               magma_int_t il, magma_int_t iu, magma_int_t* info)
{
/*
    -- MAGMA (version 1.1) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    November 2011

       .. Scalar Arguments ..
      CHARACTER          RANGE
      INTEGER            IL, IU, CUTPNT, INFO, LDQ, N
      DOUBLE PRECISION   RHO, VL, VU
       ..
       .. Array Arguments ..
      INTEGER            INDXQ( * ), iwork[* )
      DOUBLE PRECISION   D( * ), Q( LDQ, * ), WORK( * ), DWORK( * )
       ..

    Purpose
    =======

    DLAEX1 computes the updated eigensystem of a diagonal
    matrix after modification by a rank-one symmetric matrix.

      T = Q(in) ( D(in) + RHO * Z*Z' ) Q'(in) = Q(out) * D(out) * Q'(out)

       where Z = Q'u, u is a vector of length N with ones in the
       CUTPNT and CUTPNT + 1 th elements and zeros elsewhere.

       The eigenvectors of the original matrix are stored in Q, and the
       eigenvalues are in D.  The algorithm consists of three stages:

          The first stage consists of deflating the size of the problem
          when there are multiple eigenvalues or if there is a zero in
          the Z vector.  For each such occurence the dimension of the
          secular equation problem is reduced by one.  This stage is
          performed by the routine DLAED2.

          The second stage consists of calculating the updated
          eigenvalues. This is done by finding the roots of the secular
          equation via the routine DLAED4 (as called by DLAED3).
          This routine also calculates the eigenvectors of the current
          problem.

          The final stage consists of computing the updated eigenvectors
          directly using the updated eigenvalues.  The eigenvectors for
          the current problem are multiplied with the eigenvectors from
          the overall problem.

    Arguments
    =========

    N      (input) INTEGER
           The dimension of the symmetric tridiagonal matrix.  N >= 0.

    D      (input/output) DOUBLE PRECISION array, dimension (N)
           On entry, the eigenvalues of the rank-1-perturbed matrix.
           On exit, the eigenvalues of the repaired matrix.

    Q      (input/output) DOUBLE PRECISION array, dimension (LDQ,N)
           On entry, the eigenvectors of the rank-1-perturbed matrix.
           On exit, the eigenvectors of the repaired tridiagonal matrix.

    LDQ    (input) INTEGER
           The leading dimension of the array Q.  LDQ >= max(1,N).

    INDXQ  (input/output) INTEGER array, dimension (N)
           On entry, the permutation which separately sorts the two
           subproblems in D into ascending order.
           On exit, the permutation which will reintegrate the
           subproblems back into sorted order,
           i.e. D( INDXQ( I = 1, N ) ) will be in ascending order.

    RHO    (input) DOUBLE PRECISION
           The subdiagonal entry used to create the rank-1 modification.

    CUTPNT (input) INTEGER
           The location of the last eigenvalue in the leading sub-matrix.
           min(1,N) <= CUTPNT <= N/2.

    WORK   (workspace) DOUBLE PRECISION array, dimension (4*N + N**2)

    IWORK  (workspace) INTEGER array, dimension (4*N)

    DWORK  (devices workspaces) DOUBLE PRECISION array of arrays,
            dimension NRGPU.
            if NRGPU = 1 the dimension of the first workspace
            should be (3*N*N/2+3*N)
            otherwise the NRGPU workspaces should have the size
            ceil((N-N1) * (N-N1) / floor(nrgpu/2)) +
            NB * ((N-N1) + (N-N1) / floor(nrgpu/2))

    STREAM  (device stream) magma_queue_t array,
            dimension (MagmaMaxGPUs,2)

    RANGE   (input) CHARACTER*1
            = 'A': all eigenvalues will be found.
            = 'V': all eigenvalues in the half-open interval (VL,VU]
                   will be found.
            = 'I': the IL-th through IU-th eigenvalues will be found.

    VL      (input) DOUBLE PRECISION
    VU      (input) DOUBLE PRECISION
            if RANGE='V', the lower and upper bounds of the interval to
            be searched for eigenvalues. VL < VU.
            Not referenced if RANGE = 'A' or 'I'.

    IL      (input) INTEGER
    IU      (input) INTEGER
            if RANGE='I', the indices (in ascending order) of the
            smallest and largest eigenvalues to be returned.
            1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
            Not referenced if RANGE = 'A' or 'V'.

    INFO   (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value.
            > 0:  if INFO = 1, an eigenvalue did not converge

    Further Details
    ===============

    Based on contributions by
       Jeff Rutter, Computer Science Division, University of California
       at Berkeley, USA
    Modified by Francoise Tisseur, University of Tennessee.

    =====================================================================
*/
    magma_int_t coltyp, i, idlmda;
    magma_int_t indx, indxc, indxp;
    magma_int_t iq2, is, iw, iz, k, tmp;
    magma_int_t ione = 1;
    //  Test the input parameters.

    *info = 0;

    if( n < 0 )
        *info = -1;
    else if( ldq < max(1, n) )
        *info = -4;
    else if( min( 1, n/2 ) > cutpnt || n/2 < cutpnt )
        *info = -7;
    if( *info != 0 ){
        magma_xerbla( __func__, -*info );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }

    //  Quick return if possible

    if( n == 0 )
        return MAGMA_SUCCESS;

    //  The following values are integer pointers which indicate
    //  the portion of the workspace
    //  used by a particular array in DLAED2 and DLAED3.

    iz = 0;
    idlmda = iz + n;
    iw = idlmda + n;
    iq2 = iw + n;

    indx = 0;
    indxc = indx + n;
    coltyp = indxc + n;
    indxp = coltyp + n;

    //  Form the z-vector which consists of the last row of Q_1 and the
    //  first row of Q_2.

    blasf77_dcopy( &cutpnt, Q(cutpnt-1, 0), &ldq, &work[iz], &ione);
    tmp = n-cutpnt;
    blasf77_dcopy( &tmp, Q(cutpnt, cutpnt), &ldq, &work[iz+cutpnt], &ione);

    //  Deflate eigenvalues.

    magma_dlaed2(&k, &n, &cutpnt, d, q, &ldq, indxq, &rho, &work[iz],
                 &work[idlmda], &work[iw], &work[iq2],
                 &iwork[indx], &iwork[indxc], &iwork[indxp],
                 &iwork[coltyp], info);

    if( *info != 0 )
        return MAGMA_SUCCESS;

    //  Solve Secular Equation.

    if( k != 0 ){
        is = (iwork[coltyp]+iwork[coltyp+1])*cutpnt + (iwork[coltyp+1]+iwork[coltyp+2])*(n-cutpnt) + iq2;
        magma_dlaex3_m(nrgpu, k, n, cutpnt, d, q, ldq, rho,
                       &work[idlmda], &work[iq2], &iwork[indxc],
                       &iwork[coltyp], &work[iw], &work[is],
                       indxq, dwork, stream, range, vl, vu, il, iu, info );
        if( *info != 0 )
            return MAGMA_SUCCESS;
    }
    else {
        for (i = 0; i<n; ++i)
            indxq[i] = i+1;
    }

    return MAGMA_SUCCESS;

} /* magma_dlaex1_m */

