      SUBROUTINE STBT03( UPLO, TRANS, DIAG, N, KD, NRHS, AB, LDAB,
     $                   SCALE, CNORM, TSCAL, X, LDX, B, LDB, WORK,
     $                   RESID )
*
*  -- LAPACK test routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          DIAG, TRANS, UPLO
      INTEGER            KD, LDAB, LDB, LDX, N, NRHS
      REAL               RESID, SCALE, TSCAL
*     ..
*     .. Array Arguments ..
      REAL               AB( LDAB, * ), B( LDB, * ), CNORM( * ),
     $                   WORK( * ), X( LDX, * )
*     ..
*
*  Purpose
*  =======
*
*  STBT03 computes the residual for the solution to a scaled triangular
*  system of equations  A*x = s*b  or  A'*x = s*b  when A is a
*  triangular band matrix. Here A' is the transpose of A, s is a scalar,
*  and x and b are N by NRHS matrices.  The test ratio is the maximum
*  over the number of right hand sides of
*     norm(s*b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
*  where op(A) denotes A or A' and EPS is the machine epsilon.
*
*  Arguments
*  =========
*
*  UPLO    (input) CHARACTER*1
*          Specifies whether the matrix A is upper or lower triangular.
*          = 'U':  Upper triangular
*          = 'L':  Lower triangular
*
*  TRANS   (input) CHARACTER*1
*          Specifies the operation applied to A.
*          = 'N':  A *x = b  (No transpose)
*          = 'T':  A'*x = b  (Transpose)
*          = 'C':  A'*x = b  (Conjugate transpose = Transpose)
*
*  DIAG    (input) CHARACTER*1
*          Specifies whether or not the matrix A is unit triangular.
*          = 'N':  Non-unit triangular
*          = 'U':  Unit triangular
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  KD      (input) INTEGER
*          The number of superdiagonals or subdiagonals of the
*          triangular band matrix A.  KD >= 0.
*
*  NRHS    (input) INTEGER
*          The number of right hand sides, i.e., the number of columns
*          of the matrices X and B.  NRHS >= 0.
*
*  AB      (input) REAL array, dimension (LDAB,N)
*          The upper or lower triangular band matrix A, stored in the
*          first kd+1 rows of the array. The j-th column of A is stored
*          in the j-th column of the array AB as follows:
*          if UPLO = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
*          if UPLO = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
*
*  LDAB    (input) INTEGER
*          The leading dimension of the array AB.  LDAB >= KD+1.
*
*  SCALE   (input) REAL
*          The scaling factor s used in solving the triangular system.
*
*  CNORM   (input) REAL array, dimension (N)
*          The 1-norms of the columns of A, not counting the diagonal.
*
*  TSCAL   (input) REAL
*          The scaling factor used in computing the 1-norms in CNORM.
*          CNORM actually contains the column norms of TSCAL*A.
*
*  X       (input) REAL array, dimension (LDX,NRHS)
*          The computed solution vectors for the system of linear
*          equations.
*
*  LDX     (input) INTEGER
*          The leading dimension of the array X.  LDX >= max(1,N).
*
*  B       (input) REAL array, dimension (LDB,NRHS)
*          The right hand side vectors for the system of linear
*          equations.
*
*  LDB     (input) INTEGER
*          The leading dimension of the array B.  LDB >= max(1,N).
*
*  WORK    (workspace) REAL array, dimension (N)
*
*  RESID   (output) REAL
*          The maximum over the number of right hand sides of
*          norm(op(A)*x - s*b) / ( norm(op(A)) * norm(x) * EPS ).
*
*  =====================================================================
*
*     .. Parameters ..
      REAL               ONE, ZERO
      PARAMETER          ( ONE = 1.0E+0, ZERO = 0.0E+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            IX, J
      REAL               BIGNUM, EPS, ERR, SMLNUM, TNORM, XNORM, XSCAL
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      INTEGER            ISAMAX
      REAL               SLAMCH
      EXTERNAL           LSAME, ISAMAX, SLAMCH
*     ..
*     .. External Subroutines ..
      EXTERNAL           SAXPY, SCOPY, SLABAD, SSCAL, STBMV
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, REAL
*     ..
*     .. Executable Statements ..
*
*     Quick exit if N = 0
*
      IF( N.LE.0 .OR. NRHS.LE.0 ) THEN
         RESID = ZERO
         RETURN
      END IF
      EPS = SLAMCH( 'Epsilon' )
      SMLNUM = SLAMCH( 'Safe minimum' )
      BIGNUM = ONE / SMLNUM
      CALL SLABAD( SMLNUM, BIGNUM )
*
*     Compute the norm of the triangular matrix A using the column
*     norms already computed by SLATBS.
*
      TNORM = ZERO
      IF( LSAME( DIAG, 'N' ) ) THEN
         IF( LSAME( UPLO, 'U' ) ) THEN
            DO 10 J = 1, N
               TNORM = MAX( TNORM, TSCAL*ABS( AB( KD+1, J ) )+
     $                 CNORM( J ) )
   10       CONTINUE
         ELSE
            DO 20 J = 1, N
               TNORM = MAX( TNORM, TSCAL*ABS( AB( 1, J ) )+CNORM( J ) )
   20       CONTINUE
         END IF
      ELSE
         DO 30 J = 1, N
            TNORM = MAX( TNORM, TSCAL+CNORM( J ) )
   30    CONTINUE
      END IF
*
*     Compute the maximum over the number of right hand sides of
*        norm(op(A)*x - s*b) / ( norm(op(A)) * norm(x) * EPS ).
*
      RESID = ZERO
      DO 40 J = 1, NRHS
         CALL SCOPY( N, X( 1, J ), 1, WORK, 1 )
         IX = ISAMAX( N, WORK, 1 )
         XNORM = MAX( ONE, ABS( X( IX, J ) ) )
         XSCAL = ( ONE / XNORM ) / REAL( KD+1 )
         CALL SSCAL( N, XSCAL, WORK, 1 )
         CALL STBMV( UPLO, TRANS, DIAG, N, KD, AB, LDAB, WORK, 1 )
         CALL SAXPY( N, -SCALE*XSCAL, B( 1, J ), 1, WORK, 1 )
         IX = ISAMAX( N, WORK, 1 )
         ERR = TSCAL*ABS( WORK( IX ) )
         IX = ISAMAX( N, X( 1, J ), 1 )
         XNORM = ABS( X( IX, J ) )
         IF( ERR*SMLNUM.LE.XNORM ) THEN
            IF( XNORM.GT.ZERO )
     $         ERR = ERR / XNORM
         ELSE
            IF( ERR.GT.ZERO )
     $         ERR = ONE / EPS
         END IF
         IF( ERR*SMLNUM.LE.TNORM ) THEN
            IF( TNORM.GT.ZERO )
     $         ERR = ERR / TNORM
         ELSE
            IF( ERR.GT.ZERO )
     $         ERR = ONE / EPS
         END IF
         RESID = MAX( RESID, ERR )
   40 CONTINUE
*
      RETURN
*
*     End of STBT03
*
      END
