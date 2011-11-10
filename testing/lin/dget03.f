      SUBROUTINE DGET03( N, A, LDA, AINV, LDAINV, WORK, LDWORK, RWORK,
     $                   RCOND, RESID )
*
*  -- LAPACK test routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            LDA, LDAINV, LDWORK, N
      DOUBLE PRECISION   RCOND, RESID
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION   A( LDA, * ), AINV( LDAINV, * ), RWORK( * ),
     $                   WORK( LDWORK, * )
*     ..
*
*  Purpose
*  =======
*
*  DGET03 computes the residual for a general matrix times its inverse:
*     norm( I - AINV*A ) / ( N * norm(A) * norm(AINV) * EPS ),
*  where EPS is the machine epsilon.
*
*  Arguments
*  ==========
*
*  N       (input) INTEGER
*          The number of rows and columns of the matrix A.  N >= 0.
*
*  A       (input) DOUBLE PRECISION array, dimension (LDA,N)
*          The original N x N matrix A.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,N).
*
*  AINV    (input) DOUBLE PRECISION array, dimension (LDAINV,N)
*          The inverse of the matrix A.
*
*  LDAINV  (input) INTEGER
*          The leading dimension of the array AINV.  LDAINV >= max(1,N).
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (LDWORK,N)
*
*  LDWORK  (input) INTEGER
*          The leading dimension of the array WORK.  LDWORK >= max(1,N).
*
*  RWORK   (workspace) DOUBLE PRECISION array, dimension (N)
*
*  RCOND   (output) DOUBLE PRECISION
*          The reciprocal of the condition number of A, computed as
*          ( 1/norm(A) ) / norm(AINV).
*
*  RESID   (output) DOUBLE PRECISION
*          norm(I - AINV*A) / ( N * norm(A) * norm(AINV) * EPS )
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I
      DOUBLE PRECISION   AINVNM, ANORM, EPS
*     ..
*     .. External Functions ..
      DOUBLE PRECISION   DLAMCH, DLANGE
      EXTERNAL           DLAMCH, DLANGE
*     ..
*     .. External Subroutines ..
      EXTERNAL           DGEMM
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          DBLE
*     ..
*     .. Executable Statements ..
*
*     Quick exit if N = 0.
*
      IF( N.LE.0 ) THEN
         RCOND = ONE
         RESID = ZERO
         RETURN
      END IF
*
*     Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0.
*
      EPS = DLAMCH( 'Epsilon' )
      ANORM = DLANGE( '1', N, N, A, LDA, RWORK )
      AINVNM = DLANGE( '1', N, N, AINV, LDAINV, RWORK )
      IF( ANORM.LE.ZERO .OR. AINVNM.LE.ZERO ) THEN
         RCOND = ZERO
         RESID = ONE / EPS
         RETURN
      END IF
      RCOND = ( ONE / ANORM ) / AINVNM
*
*     Compute I - A * AINV
*
      CALL DGEMM( 'No transpose', 'No transpose', N, N, N, -ONE, AINV,
     $            LDAINV, A, LDA, ZERO, WORK, LDWORK )
      DO 10 I = 1, N
         WORK( I, I ) = ONE + WORK( I, I )
   10 CONTINUE
*
*     Compute norm(I - AINV*A) / (N * norm(A) * norm(AINV) * EPS)
*
      RESID = DLANGE( '1', N, N, WORK, LDWORK, RWORK )
*
      RESID = ( ( RESID*RCOND ) / EPS ) / DBLE( N )
*
      RETURN
*
*     End of DGET03
*
      END
