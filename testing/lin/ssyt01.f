      SUBROUTINE SSYT01( UPLO, N, A, LDA, AFAC, LDAFAC, IPIV, C, LDC,
     $                   RWORK, RESID )
*
*  -- LAPACK test routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          UPLO
      INTEGER            LDA, LDAFAC, LDC, N
      REAL               RESID
*     ..
*     .. Array Arguments ..
      INTEGER            IPIV( * )
      REAL               A( LDA, * ), AFAC( LDAFAC, * ), C( LDC, * ),
     $                   RWORK( * )
*     ..
*
*  Purpose
*  =======
*
*  SSYT01 reconstructs a symmetric indefinite matrix A from its
*  block L*D*L' or U*D*U' factorization and computes the residual
*     norm( C - A ) / ( N * norm(A) * EPS ),
*  where C is the reconstructed matrix and EPS is the machine epsilon.
*
*  Arguments
*  ==========
*
*  UPLO    (input) CHARACTER*1
*          Specifies whether the upper or lower triangular part of the
*          symmetric matrix A is stored:
*          = 'U':  Upper triangular
*          = 'L':  Lower triangular
*
*  N       (input) INTEGER
*          The number of rows and columns of the matrix A.  N >= 0.
*
*  A       (input) REAL array, dimension (LDA,N)
*          The original symmetric matrix A.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= max(1,N)
*
*  AFAC    (input) REAL array, dimension (LDAFAC,N)
*          The factored form of the matrix A.  AFAC contains the block
*          diagonal matrix D and the multipliers used to obtain the
*          factor L or U from the block L*D*L' or U*D*U' factorization
*          as computed by SSYTRF.
*
*  LDAFAC  (input) INTEGER
*          The leading dimension of the array AFAC.  LDAFAC >= max(1,N).
*
*  IPIV    (input) INTEGER array, dimension (N)
*          The pivot indices from SSYTRF.
*
*  C       (workspace) REAL array, dimension (LDC,N)
*
*  LDC     (integer) INTEGER
*          The leading dimension of the array C.  LDC >= max(1,N).
*
*  RWORK   (workspace) REAL array, dimension (N)
*
*  RESID   (output) REAL
*          If UPLO = 'L', norm(L*D*L' - A) / ( N * norm(A) * EPS )
*          If UPLO = 'U', norm(U*D*U' - A) / ( N * norm(A) * EPS )
*
*  =====================================================================
*
*     .. Parameters ..
      REAL               ZERO, ONE
      PARAMETER          ( ZERO = 0.0E+0, ONE = 1.0E+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, INFO, J
      REAL               ANORM, EPS
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      REAL               SLAMCH, SLANSY
      EXTERNAL           LSAME, SLAMCH, SLANSY
*     ..
*     .. External Subroutines ..
      EXTERNAL           SLAVSY, SLASET
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          REAL
*     ..
*     .. Executable Statements ..
*
*     Quick exit if N = 0.
*
      IF( N.LE.0 ) THEN
         RESID = ZERO
         RETURN
      END IF
*
*     Determine EPS and the norm of A.
*
      EPS = SLAMCH( 'Epsilon' )
      ANORM = SLANSY( '1', UPLO, N, A, LDA, RWORK )
*
*     Initialize C to the identity matrix.
*
      CALL SLASET( 'Full', N, N, ZERO, ONE, C, LDC )
*
*     Call SLAVSY to form the product D * U' (or D * L' ).
*
      CALL SLAVSY( UPLO, 'Transpose', 'Non-unit', N, N, AFAC, LDAFAC,
     $             IPIV, C, LDC, INFO )
*
*     Call SLAVSY again to multiply by U (or L ).
*
      CALL SLAVSY( UPLO, 'No transpose', 'Unit', N, N, AFAC, LDAFAC,
     $             IPIV, C, LDC, INFO )
*
*     Compute the difference  C - A .
*
      IF( LSAME( UPLO, 'U' ) ) THEN
         DO 20 J = 1, N
            DO 10 I = 1, J
               C( I, J ) = C( I, J ) - A( I, J )
   10       CONTINUE
   20    CONTINUE
      ELSE
         DO 40 J = 1, N
            DO 30 I = J, N
               C( I, J ) = C( I, J ) - A( I, J )
   30       CONTINUE
   40    CONTINUE
      END IF
*
*     Compute norm( C - A ) / ( N * norm(A) * EPS )
*
      RESID = SLANSY( '1', UPLO, N, C, LDC, RWORK )
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
*     End of SSYT01
*
      END
