      SUBROUTINE SGTT05( TRANS, N, NRHS, DL, D, DU, B, LDB, X, LDX,
     $                   XACT, LDXACT, FERR, BERR, RESLTS )
*
*  -- LAPACK test routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER          TRANS
      INTEGER            LDB, LDX, LDXACT, N, NRHS
*     ..
*     .. Array Arguments ..
      REAL               B( LDB, * ), BERR( * ), D( * ), DL( * ),
     $                   DU( * ), FERR( * ), RESLTS( * ), X( LDX, * ),
     $                   XACT( LDXACT, * )
*     ..
*
*  Purpose
*  =======
*
*  SGTT05 tests the error bounds from iterative refinement for the
*  computed solution to a system of equations A*X = B, where A is a
*  general tridiagonal matrix of order n and op(A) = A or A**T,
*  depending on TRANS.
*
*  RESLTS(1) = test of the error bound
*            = norm(X - XACT) / ( norm(X) * FERR )
*
*  A large value is returned if this ratio is not less than one.
*
*  RESLTS(2) = residual from the iterative refinement routine
*            = the maximum of BERR / ( NZ*EPS + (*) ), where
*              (*) = NZ*UNFL / (min_i (abs(op(A))*abs(X) +abs(b))_i )
*              and NZ = max. number of nonzeros in any row of A, plus 1
*
*  Arguments
*  =========
*
*  TRANS   (input) CHARACTER*1
*          Specifies the form of the system of equations.
*          = 'N':  A * X = B     (No transpose)
*          = 'T':  A**T * X = B  (Transpose)
*          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
*
*  N       (input) INTEGER
*          The number of rows of the matrices X and XACT.  N >= 0.
*
*  NRHS    (input) INTEGER
*          The number of columns of the matrices X and XACT.  NRHS >= 0.
*
*  DL      (input) REAL array, dimension (N-1)
*          The (n-1) sub-diagonal elements of A.
*
*  D       (input) REAL array, dimension (N)
*          The diagonal elements of A.
*
*  DU      (input) REAL array, dimension (N-1)
*          The (n-1) super-diagonal elements of A.
*
*  B       (input) REAL array, dimension (LDB,NRHS)
*          The right hand side vectors for the system of linear
*          equations.
*
*  LDB     (input) INTEGER
*          The leading dimension of the array B.  LDB >= max(1,N).
*
*  X       (input) REAL array, dimension (LDX,NRHS)
*          The computed solution vectors.  Each vector is stored as a
*          column of the matrix X.
*
*  LDX     (input) INTEGER
*          The leading dimension of the array X.  LDX >= max(1,N).
*
*  XACT    (input) REAL array, dimension (LDX,NRHS)
*          The exact solution vectors.  Each vector is stored as a
*          column of the matrix XACT.
*
*  LDXACT  (input) INTEGER
*          The leading dimension of the array XACT.  LDXACT >= max(1,N).
*
*  FERR    (input) REAL array, dimension (NRHS)
*          The estimated forward error bounds for each solution vector
*          X.  If XTRUE is the true solution, FERR bounds the magnitude
*          of the largest entry in (X - XTRUE) divided by the magnitude
*          of the largest entry in X.
*
*  BERR    (input) REAL array, dimension (NRHS)
*          The componentwise relative backward error of each solution
*          vector (i.e., the smallest relative change in any entry of A
*          or B that makes X an exact solution).
*
*  RESLTS  (output) REAL array, dimension (2)
*          The maximum over the NRHS solution vectors of the ratios:
*          RESLTS(1) = norm(X - XACT) / ( norm(X) * FERR )
*          RESLTS(2) = BERR / ( NZ*EPS + (*) )
*
*  =====================================================================
*
*     .. Parameters ..
      REAL               ZERO, ONE
      PARAMETER          ( ZERO = 0.0E+0, ONE = 1.0E+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            NOTRAN
      INTEGER            I, IMAX, J, K, NZ
      REAL               AXBI, DIFF, EPS, ERRBND, OVFL, TMP, UNFL, XNORM
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      INTEGER            ISAMAX
      REAL               SLAMCH
      EXTERNAL           LSAME, ISAMAX, SLAMCH
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          ABS, MAX, MIN
*     ..
*     .. Executable Statements ..
*
*     Quick exit if N = 0 or NRHS = 0.
*
      IF( N.LE.0 .OR. NRHS.LE.0 ) THEN
         RESLTS( 1 ) = ZERO
         RESLTS( 2 ) = ZERO
         RETURN
      END IF
*
      EPS = SLAMCH( 'Epsilon' )
      UNFL = SLAMCH( 'Safe minimum' )
      OVFL = ONE / UNFL
      NOTRAN = LSAME( TRANS, 'N' )
      NZ = 4
*
*     Test 1:  Compute the maximum of
*        norm(X - XACT) / ( norm(X) * FERR )
*     over all the vectors X and XACT using the infinity-norm.
*
      ERRBND = ZERO
      DO 30 J = 1, NRHS
         IMAX = ISAMAX( N, X( 1, J ), 1 )
         XNORM = MAX( ABS( X( IMAX, J ) ), UNFL )
         DIFF = ZERO
         DO 10 I = 1, N
            DIFF = MAX( DIFF, ABS( X( I, J )-XACT( I, J ) ) )
   10    CONTINUE
*
         IF( XNORM.GT.ONE ) THEN
            GO TO 20
         ELSE IF( DIFF.LE.OVFL*XNORM ) THEN
            GO TO 20
         ELSE
            ERRBND = ONE / EPS
            GO TO 30
         END IF
*
   20    CONTINUE
         IF( DIFF / XNORM.LE.FERR( J ) ) THEN
            ERRBND = MAX( ERRBND, ( DIFF / XNORM ) / FERR( J ) )
         ELSE
            ERRBND = ONE / EPS
         END IF
   30 CONTINUE
      RESLTS( 1 ) = ERRBND
*
*     Test 2:  Compute the maximum of BERR / ( NZ*EPS + (*) ), where
*     (*) = NZ*UNFL / (min_i (abs(op(A))*abs(X) +abs(b))_i )
*
      DO 60 K = 1, NRHS
         IF( NOTRAN ) THEN
            IF( N.EQ.1 ) THEN
               AXBI = ABS( B( 1, K ) ) + ABS( D( 1 )*X( 1, K ) )
            ELSE
               AXBI = ABS( B( 1, K ) ) + ABS( D( 1 )*X( 1, K ) ) +
     $                ABS( DU( 1 )*X( 2, K ) )
               DO 40 I = 2, N - 1
                  TMP = ABS( B( I, K ) ) + ABS( DL( I-1 )*X( I-1, K ) )
     $                   + ABS( D( I )*X( I, K ) ) +
     $                  ABS( DU( I )*X( I+1, K ) )
                  AXBI = MIN( AXBI, TMP )
   40          CONTINUE
               TMP = ABS( B( N, K ) ) + ABS( DL( N-1 )*X( N-1, K ) ) +
     $               ABS( D( N )*X( N, K ) )
               AXBI = MIN( AXBI, TMP )
            END IF
         ELSE
            IF( N.EQ.1 ) THEN
               AXBI = ABS( B( 1, K ) ) + ABS( D( 1 )*X( 1, K ) )
            ELSE
               AXBI = ABS( B( 1, K ) ) + ABS( D( 1 )*X( 1, K ) ) +
     $                ABS( DL( 1 )*X( 2, K ) )
               DO 50 I = 2, N - 1
                  TMP = ABS( B( I, K ) ) + ABS( DU( I-1 )*X( I-1, K ) )
     $                   + ABS( D( I )*X( I, K ) ) +
     $                  ABS( DL( I )*X( I+1, K ) )
                  AXBI = MIN( AXBI, TMP )
   50          CONTINUE
               TMP = ABS( B( N, K ) ) + ABS( DU( N-1 )*X( N-1, K ) ) +
     $               ABS( D( N )*X( N, K ) )
               AXBI = MIN( AXBI, TMP )
            END IF
         END IF
         TMP = BERR( K ) / ( NZ*EPS+NZ*UNFL / MAX( AXBI, NZ*UNFL ) )
         IF( K.EQ.1 ) THEN
            RESLTS( 2 ) = TMP
         ELSE
            RESLTS( 2 ) = MAX( RESLTS( 2 ), TMP )
         END IF
   60 CONTINUE
*
      RETURN
*
*     End of SGTT05
*
      END
