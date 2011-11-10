      SUBROUTINE DLARGE( N, A, LDA, ISEED, WORK, INFO )
*
*  -- LAPACK auxiliary test routine (version 3.1)
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            INFO, LDA, N
*     ..
*     .. Array Arguments ..
      INTEGER            ISEED( 4 )
      DOUBLE PRECISION   A( LDA, * ), WORK( * )
*     ..
*
*  Purpose
*  =======
*
*  DLARGE pre- and post-multiplies a real general n by n matrix A
*  with a random orthogonal matrix: A = U*D*U'.
*
*  Arguments
*  =========
*
*  N       (input) INTEGER
*          The order of the matrix A.  N >= 0.
*
*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)
*          On entry, the original n by n matrix A.
*          On exit, A is overwritten by U*A*U' for some random
*          orthogonal matrix U.
*
*  LDA     (input) INTEGER
*          The leading dimension of the array A.  LDA >= N.
*
*  ISEED   (input/output) INTEGER array, dimension (4)
*          On entry, the seed of the random number generator; the array
*          elements must be between 0 and 4095, and ISEED(4) must be
*          odd.
*          On exit, the seed is updated.
*
*  WORK    (workspace) DOUBLE PRECISION array, dimension (2*N)
*
*  INFO    (output) INTEGER
*          = 0: successful exit
*          < 0: if INFO = -i, the i-th argument had an illegal value
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      INTEGER            I
      DOUBLE PRECISION   TAU, WA, WB, WN
*     ..
*     .. External Subroutines ..
      EXTERNAL           DGEMV, DGER, DLARNV, DSCAL, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, SIGN
*     ..
*     .. External Functions ..
      DOUBLE PRECISION   DNRM2
      EXTERNAL           DNRM2
*     ..
*     .. Executable Statements ..
*
*     Test the input arguments
*
      INFO = 0
      IF( N.LT.0 ) THEN
         INFO = -1
      ELSE IF( LDA.LT.MAX( 1, N ) ) THEN
         INFO = -3
      END IF
      IF( INFO.LT.0 ) THEN
         CALL XERBLA( 'DLARGE', -INFO )
         RETURN
      END IF
*
*     pre- and post-multiply A by random orthogonal matrix
*
      DO 10 I = N, 1, -1
*
*        generate random reflection
*
         CALL DLARNV( 3, ISEED, N-I+1, WORK )
         WN = DNRM2( N-I+1, WORK, 1 )
         WA = SIGN( WN, WORK( 1 ) )
         IF( WN.EQ.ZERO ) THEN
            TAU = ZERO
         ELSE
            WB = WORK( 1 ) + WA
            CALL DSCAL( N-I, ONE / WB, WORK( 2 ), 1 )
            WORK( 1 ) = ONE
            TAU = WB / WA
         END IF
*
*        multiply A(i:n,1:n) by random reflection from the left
*
         CALL DGEMV( 'Transpose', N-I+1, N, ONE, A( I, 1 ), LDA, WORK,
     $               1, ZERO, WORK( N+1 ), 1 )
         CALL DGER( N-I+1, N, -TAU, WORK, 1, WORK( N+1 ), 1, A( I, 1 ),
     $              LDA )
*
*        multiply A(1:n,i:n) by random reflection from the right
*
         CALL DGEMV( 'No transpose', N, N-I+1, ONE, A( 1, I ), LDA,
     $               WORK, 1, ZERO, WORK( N+1 ), 1 )
         CALL DGER( N, N-I+1, -TAU, WORK( N+1 ), 1, WORK, 1, A( 1, I ),
     $              LDA )
   10 CONTINUE
      RETURN
*
*     End of DLARGE
*
      END
