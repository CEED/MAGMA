      SUBROUTINE CPBT02( UPLO, N, KD, NRHS, A, LDA, X, LDX, B, LDB,
     $                   RWORK, RESID )
*
*  -- LAPACK test routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          UPLO
      INTEGER            KD, LDA, LDB, LDX, N, NRHS
      REAL               RESID
*     ..
*     .. Array Arguments ..
      REAL               RWORK( * )
      COMPLEX            A( LDA, * ), B( LDB, * ), X( LDX, * )
*     ..
*
*  Purpose
*  =======
*
*  CPBT02 computes the residual for a solution of a Hermitian banded
*  system of equations  A*x = b:
*     RESID = norm( B - A*X ) / ( norm(A) * norm(X) * EPS)
*  where EPS is the machine precision.
*
*  Arguments
*  =========
*
*  UPLO    (input) CHARACTER*1
*          Specifies whether the upper or lower triangular part of the
*          Hermitian matrix A is stored:
*          = 'U':  Upper triangular
*          = 'L':  Lower triangular
*
*  N       (input) INTEGER
*          The number of rows and columns of the matrix A.  N >= 0.
*
*  KD      (input) INTEGER
*          The number of super-diagonals of the matrix A if UPLO = 'U',
*          or the number of sub-diagonals if UPLO = 'L'.  KD >= 0.
*
*  A       (input) COMPLEX array, dimension (LDA,N)
*          The original Hermitian band matrix A.  If UPLO = 'U', the
*          upper triangular part of A is stored as a band matrix; if
*          UPLO = 'L', the lower triangular part of A is stored.  The
*          columns of the appropriate triangle are stored in the columns
*          of A and the diagonals of the triangle are stored in the rows
*          of A.  See CPBTRF for further details.
*
*  LDA     (input) INTEGER.
*          The leading dimension of the array A.  LDA >= max(1,KD+1).
*
*  X       (input) COMPLEX array, dimension (LDX,NRHS)
*          The computed solution vectors for the system of linear
*          equations.
*
*  LDX     (input) INTEGER
*          The leading dimension of the array X.   LDX >= max(1,N).
*
*  B       (input/output) COMPLEX array, dimension (LDB,NRHS)
*          On entry, the right hand side vectors for the system of
*          linear equations.
*          On exit, B is overwritten with the difference B - A*X.
*
*  LDB     (input) INTEGER
*          The leading dimension of the array B.  LDB >= max(1,N).
*
*  RWORK   (workspace) REAL array, dimension (N)
*
*  RESID   (output) REAL
*          The maximum over the number of right hand sides of
*          norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
*
*  =====================================================================
*
*     .. Parameters ..
      REAL               ZERO, ONE
      PARAMETER          ( ZERO = 0.0E+0, ONE = 1.0E+0 )
      COMPLEX            CONE
      PARAMETER          ( CONE = ( 1.0E+0, 0.0E+0 ) )
*     ..
*     .. Local Scalars ..
      INTEGER            J
      REAL               ANORM, BNORM, EPS, XNORM
*     ..
*     .. External Functions ..
      REAL               CLANHB, SCASUM, SLAMCH
      EXTERNAL           CLANHB, SCASUM, SLAMCH
*     ..
*     .. External Subroutines ..
      EXTERNAL           CHBMV
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX
*     ..
*     .. Executable Statements ..
*
*     Quick exit if N = 0 or NRHS = 0.
*
      IF( N.LE.0 .OR. NRHS.LE.0 ) THEN
         RESID = ZERO
         RETURN
      END IF
*
*     Exit with RESID = 1/EPS if ANORM = 0.
*
      EPS = SLAMCH( 'Epsilon' )
      ANORM = CLANHB( '1', UPLO, N, KD, A, LDA, RWORK )
      IF( ANORM.LE.ZERO ) THEN
         RESID = ONE / EPS
         RETURN
      END IF
*
*     Compute  B - A*X
*
      DO 10 J = 1, NRHS
         CALL CHBMV( UPLO, N, KD, -CONE, A, LDA, X( 1, J ), 1, CONE,
     $               B( 1, J ), 1 )
   10 CONTINUE
*
*     Compute the maximum over the number of right hand sides of
*          norm( B - A*X ) / ( norm(A) * norm(X) * EPS )
*
      RESID = ZERO
      DO 20 J = 1, NRHS
         BNORM = SCASUM( N, B( 1, J ), 1 )
         XNORM = SCASUM( N, X( 1, J ), 1 )
         IF( XNORM.LE.ZERO ) THEN
            RESID = ONE / EPS
         ELSE
            RESID = MAX( RESID, ( ( BNORM/ANORM )/XNORM )/EPS )
         END IF
   20 CONTINUE
*
      RETURN
*
*     End of CPBT02
*
      END
