      SUBROUTINE SQLT02( M, N, K, A, AF, Q, L, LDA, TAU, WORK, LWORK,
     $                   RWORK, RESULT )
*
*  -- LAPACK test routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      INTEGER            K, LDA, LWORK, M, N
*     ..
*     .. Array Arguments ..
      REAL               A( LDA, * ), AF( LDA, * ), L( LDA, * ),
     $                   Q( LDA, * ), RESULT( * ), RWORK( * ), TAU( * ),
     $                   WORK( LWORK )
*     ..
*
*  Purpose
*  =======
*
*  SQLT02 tests SORGQL, which generates an m-by-n matrix Q with
*  orthonornmal columns that is defined as the product of k elementary
*  reflectors.
*
*  Given the QL factorization of an m-by-n matrix A, SQLT02 generates
*  the orthogonal matrix Q defined by the factorization of the last k
*  columns of A; it compares L(m-n+1:m,n-k+1:n) with
*  Q(1:m,m-n+1:m)'*A(1:m,n-k+1:n), and checks that the columns of Q are
*  orthonormal.
*
*  Arguments
*  =========
*
*  M       (input) INTEGER
*          The number of rows of the matrix Q to be generated.  M >= 0.
*
*  N       (input) INTEGER
*          The number of columns of the matrix Q to be generated.
*          M >= N >= 0.
*
*  K       (input) INTEGER
*          The number of elementary reflectors whose product defines the
*          matrix Q. N >= K >= 0.
*
*  A       (input) REAL array, dimension (LDA,N)
*          The m-by-n matrix A which was factorized by SQLT01.
*
*  AF      (input) REAL array, dimension (LDA,N)
*          Details of the QL factorization of A, as returned by SGEQLF.
*          See SGEQLF for further details.
*
*  Q       (workspace) REAL array, dimension (LDA,N)
*
*  L       (workspace) REAL array, dimension (LDA,N)
*
*  LDA     (input) INTEGER
*          The leading dimension of the arrays A, AF, Q and L. LDA >= M.
*
*  TAU     (input) REAL array, dimension (N)
*          The scalar factors of the elementary reflectors corresponding
*          to the QL factorization in AF.
*
*  WORK    (workspace) REAL array, dimension (LWORK)
*
*  LWORK   (input) INTEGER
*          The dimension of the array WORK.
*
*  RWORK   (workspace) REAL array, dimension (M)
*
*  RESULT  (output) REAL array, dimension (2)
*          The test ratios:
*          RESULT(1) = norm( L - Q'*A ) / ( M * norm(A) * EPS )
*          RESULT(2) = norm( I - Q'*Q ) / ( M * EPS )
*
*  =====================================================================
*
*     .. Parameters ..
      REAL               ZERO, ONE
      PARAMETER          ( ZERO = 0.0E+0, ONE = 1.0E+0 )
      REAL               ROGUE
      PARAMETER          ( ROGUE = -1.0E+10 )
*     ..
*     .. Local Scalars ..
      INTEGER            INFO
      REAL               ANORM, EPS, RESID
*     ..
*     .. External Functions ..
      REAL               SLAMCH, SLANGE, SLANSY
      EXTERNAL           SLAMCH, SLANGE, SLANSY
*     ..
*     .. External Subroutines ..
      EXTERNAL           SGEMM, SLACPY, SLASET, SORGQL, SSYRK
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, REAL
*     ..
*     .. Scalars in Common ..
      CHARACTER*32       SRNAMT
*     ..
*     .. Common blocks ..
      COMMON             / SRNAMC / SRNAMT
*     ..
*     .. Executable Statements ..
*
*     Quick return if possible
*
      IF( M.EQ.0 .OR. N.EQ.0 .OR. K.EQ.0 ) THEN
         RESULT( 1 ) = ZERO
         RESULT( 2 ) = ZERO
         RETURN
      END IF
*
      EPS = SLAMCH( 'Epsilon' )
*
*     Copy the last k columns of the factorization to the array Q
*
      CALL SLASET( 'Full', M, N, ROGUE, ROGUE, Q, LDA )
      IF( K.LT.M )
     $   CALL SLACPY( 'Full', M-K, K, AF( 1, N-K+1 ), LDA,
     $                Q( 1, N-K+1 ), LDA )
      IF( K.GT.1 )
     $   CALL SLACPY( 'Upper', K-1, K-1, AF( M-K+1, N-K+2 ), LDA,
     $                Q( M-K+1, N-K+2 ), LDA )
*
*     Generate the last n columns of the matrix Q
*
      SRNAMT = 'SORGQL'
      CALL SORGQL( M, N, K, Q, LDA, TAU( N-K+1 ), WORK, LWORK, INFO )
*
*     Copy L(m-n+1:m,n-k+1:n)
*
      CALL SLASET( 'Full', N, K, ZERO, ZERO, L( M-N+1, N-K+1 ), LDA )
      CALL SLACPY( 'Lower', K, K, AF( M-K+1, N-K+1 ), LDA,
     $             L( M-K+1, N-K+1 ), LDA )
*
*     Compute L(m-n+1:m,n-k+1:n) - Q(1:m,m-n+1:m)' * A(1:m,n-k+1:n)
*
      CALL SGEMM( 'Transpose', 'No transpose', N, K, M, -ONE, Q, LDA,
     $            A( 1, N-K+1 ), LDA, ONE, L( M-N+1, N-K+1 ), LDA )
*
*     Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) .
*
      ANORM = SLANGE( '1', M, K, A( 1, N-K+1 ), LDA, RWORK )
      RESID = SLANGE( '1', N, K, L( M-N+1, N-K+1 ), LDA, RWORK )
      IF( ANORM.GT.ZERO ) THEN
         RESULT( 1 ) = ( ( RESID / REAL( MAX( 1, M ) ) ) / ANORM ) / EPS
      ELSE
         RESULT( 1 ) = ZERO
      END IF
*
*     Compute I - Q'*Q
*
      CALL SLASET( 'Full', N, N, ZERO, ONE, L, LDA )
      CALL SSYRK( 'Upper', 'Transpose', N, M, -ONE, Q, LDA, ONE, L,
     $            LDA )
*
*     Compute norm( I - Q'*Q ) / ( M * EPS ) .
*
      RESID = SLANSY( '1', 'Upper', N, L, LDA, RWORK )
*
      RESULT( 2 ) = ( RESID / REAL( MAX( 1, M ) ) ) / EPS
*
      RETURN
*
*     End of SQLT02
*
      END
