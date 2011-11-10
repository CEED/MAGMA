      SUBROUTINE CPTT01( N, D, E, DF, EF, WORK, RESID )
*
*  -- LAPACK test routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            N
      REAL               RESID
*     ..
*     .. Array Arguments ..
      REAL               D( * ), DF( * )
      COMPLEX            E( * ), EF( * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  CPTT01 reconstructs a tridiagonal matrix A from its L*D*L'
*  factorization and computes the residual
*     norm(L*D*L' - A) / ( n * norm(A) * EPS ),
*  where EPS is the machine epsilon.
*
*  Arguments
*  =========
*
*  N       (input) INTEGTER
*          The order of the matrix A.
*
*  D       (input) REAL array, dimension (N)
*          The n diagonal elements of the tridiagonal matrix A.
*
*  E       (input) COMPLEX array, dimension (N-1)
*          The (n-1) subdiagonal elements of the tridiagonal matrix A.
*
*  DF      (input) REAL array, dimension (N)
*          The n diagonal elements of the factor L from the L*D*L'
*          factorization of A.
*
*  EF      (input) COMPLEX array, dimension (N-1)
*          The (n-1) subdiagonal elements of the factor L from the
*          L*D*L' factorization of A.
*
*  WORK    (workspace) COMPLEX array, dimension (2*N)
*
*  RESID   (output) REAL
*          norm(L*D*L' - A) / (n * norm(A) * EPS)
*
*  =====================================================================
*
*     .. Parameters ..
      REAL               ONE, ZERO
      PARAMETER          ( ONE = 1.0E+0, ZERO = 0.0E+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I
      REAL               ANORM, EPS
      COMPLEX            DE
*     ..
*     .. External Functions ..
      REAL               SLAMCH
      EXTERNAL           SLAMCH
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, CONJG, MAX, REAL
*     ..
*     .. Executable Statements ..
*
*     Quick return if possible
*
      IF( N.LE.0 ) THEN
         RESID = ZERO
         RETURN
      END IF
*
      EPS = SLAMCH( 'Epsilon' )
*
*     Construct the difference L*D*L' - A.
*
      WORK( 1 ) = DF( 1 ) - D( 1 )
      DO 10 I = 1, N - 1
         DE = DF( I )*EF( I )
         WORK( N+I ) = DE - E( I )
         WORK( 1+I ) = DE*CONJG( EF( I ) ) + DF( I+1 ) - D( I+1 )
   10 CONTINUE
*
*     Compute the 1-norms of the tridiagonal matrices A and WORK.
*
      IF( N.EQ.1 ) THEN
         ANORM = D( 1 )
         RESID = ABS( WORK( 1 ) )
      ELSE
         ANORM = MAX( D( 1 )+ABS( E( 1 ) ), D( N )+ABS( E( N-1 ) ) )
         RESID = MAX( ABS( WORK( 1 ) )+ABS( WORK( N+1 ) ),
     $           ABS( WORK( N ) )+ABS( WORK( 2*N-1 ) ) )
         DO 20 I = 2, N - 1
            ANORM = MAX( ANORM, D( I )+ABS( E( I ) )+ABS( E( I-1 ) ) )
            RESID = MAX( RESID, ABS( WORK( I ) )+ABS( WORK( N+I-1 ) )+
     $              ABS( WORK( N+I ) ) )
   20    CONTINUE
      END IF
*
*     Compute norm(L*D*L' - A) / (n * norm(A) * EPS)
*
      IF( ANORM.LE.ZERO ) THEN
         IF( RESID.NE.ZERO )
     $      RESID = ONE / EPS
      ELSE
         RESID = ( ( RESID / REAL( N ) ) / ANORM ) / EPS
      END IF
*
      RETURN
*
*     End of CPTT01
*
      END
