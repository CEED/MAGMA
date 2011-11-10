      REAL             FUNCTION CQRT12( M, N, A, LDA, S, WORK, LWORK,
     $                 RWORK )
*
*  -- LAPACK test routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            LDA, LWORK, M, N
*     ..
*     .. Array Arguments ..
      REAL               RWORK( * ), S( * )
      COMPLEX            A( LDA, * ), WORK( LWORK )
*     ..
*
*  Purpose
*  =======
*
*  CQRT12 computes the singular values `svlues' of the upper trapezoid
*  of A(1:M,1:N) and returns the ratio
*
*       || s - svlues||/(||svlues||*eps*max(M,N))
*
*  Arguments
*  =========
*
*  M       (input) INTEGER
*          The number of rows of the matrix A.
*
*  N       (input) INTEGER
*          The number of columns of the matrix A.
*
*  A       (input) COMPLEX array, dimension (LDA,N)
*          The M-by-N matrix A. Only the upper trapezoid is referenced.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.
*
*  S       (input) REAL array, dimension (min(M,N))
*          The singular values of the matrix A.
*
*  WORK    (workspace) COMPLEX array, dimension (LWORK)
*
*  LWORK   (input) INTEGER
*          The length of the array WORK. LWORK >= M*N + 2*min(M,N) +
*          max(M,N).
*
*  RWORK   (workspace) REAL array, dimension (4*min(M,N))
*
*  =====================================================================
*
*     .. Parameters ..
      REAL               ZERO, ONE
      PARAMETER          ( ZERO = 0.0E0, ONE = 1.0E0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, INFO, ISCL, J, MN
      REAL               ANRM, BIGNUM, NRMSVL, SMLNUM
*     ..
*     .. Local Arrays ..
      REAL               DUMMY( 1 )
*     ..
*     .. External Functions ..
      REAL               CLANGE, SASUM, SLAMCH, SNRM2
      EXTERNAL           CLANGE, SASUM, SLAMCH, SNRM2
*     ..
*     .. External Subroutines ..
      EXTERNAL           CGEBD2, CLASCL, CLASET, SAXPY, SBDSQR, SLABAD,
     $                   SLASCL, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          CMPLX, MAX, MIN, REAL
*     ..
*     .. Executable Statements ..
*
      CQRT12 = ZERO
*
*     Test that enough workspace is supplied
*
      IF( LWORK.LT.M*N+2*MIN( M, N )+MAX( M, N ) ) THEN
         CALL XERBLA( 'CQRT12', 7 )
         RETURN
      END IF
*
*     Quick return if possible
*
      MN = MIN( M, N )
      IF( MN.LE.ZERO )
     $   RETURN
*
      NRMSVL = SNRM2( MN, S, 1 )
*
*     Copy upper triangle of A into work
*
      CALL CLASET( 'Full', M, N, CMPLX( ZERO ), CMPLX( ZERO ), WORK, M )
      DO 20 J = 1, N
         DO 10 I = 1, MIN( J, M )
            WORK( ( J-1 )*M+I ) = A( I, J )
   10    CONTINUE
   20 CONTINUE
*
*     Get machine parameters
*
      SMLNUM = SLAMCH( 'S' ) / SLAMCH( 'P' )
      BIGNUM = ONE / SMLNUM
      CALL SLABAD( SMLNUM, BIGNUM )
*
*     Scale work if max entry outside range [SMLNUM,BIGNUM]
*
      ANRM = CLANGE( 'M', M, N, WORK, M, DUMMY )
      ISCL = 0
      IF( ANRM.GT.ZERO .AND. ANRM.LT.SMLNUM ) THEN
*
*        Scale matrix norm up to SMLNUM
*
         CALL CLASCL( 'G', 0, 0, ANRM, SMLNUM, M, N, WORK, M, INFO )
         ISCL = 1
      ELSE IF( ANRM.GT.BIGNUM ) THEN
*
*        Scale matrix norm down to BIGNUM
*
         CALL CLASCL( 'G', 0, 0, ANRM, BIGNUM, M, N, WORK, M, INFO )
         ISCL = 1
      END IF
*
      IF( ANRM.NE.ZERO ) THEN
*
*        Compute SVD of work
*
         CALL CGEBD2( M, N, WORK, M, RWORK( 1 ), RWORK( MN+1 ),
     $                WORK( M*N+1 ), WORK( M*N+MN+1 ),
     $                WORK( M*N+2*MN+1 ), INFO )
         CALL SBDSQR( 'Upper', MN, 0, 0, 0, RWORK( 1 ), RWORK( MN+1 ),
     $                DUMMY, MN, DUMMY, 1, DUMMY, MN, RWORK( 2*MN+1 ),
     $                INFO )
*
         IF( ISCL.EQ.1 ) THEN
            IF( ANRM.GT.BIGNUM ) THEN
               CALL SLASCL( 'G', 0, 0, BIGNUM, ANRM, MN, 1, RWORK( 1 ),
     $                      MN, INFO )
            END IF
            IF( ANRM.LT.SMLNUM ) THEN
               CALL SLASCL( 'G', 0, 0, SMLNUM, ANRM, MN, 1, RWORK( 1 ),
     $                      MN, INFO )
            END IF
         END IF
*
      ELSE
*
         DO 30 I = 1, MN
            RWORK( I ) = ZERO
   30    CONTINUE
      END IF
*
*     Compare s and singular values of work
*
      CALL SAXPY( MN, -ONE, S, 1, RWORK( 1 ), 1 )
      CQRT12 = SASUM( MN, RWORK( 1 ), 1 ) /
     $         ( SLAMCH( 'Epsilon' )*REAL( MAX( M, N ) ) )
      IF( NRMSVL.NE.ZERO )
     $   CQRT12 = CQRT12 / NRMSVL
*
      RETURN
*
*     End of CQRT12
*
      END
