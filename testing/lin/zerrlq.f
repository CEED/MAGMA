      SUBROUTINE ZERRLQ( PATH, NUNIT )
*
*  -- LAPACK test routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     November 2006
*
*     .. Scalar Arguments ..
      CHARACTER*3        PATH
      INTEGER            NUNIT
*     ..
*
*  Purpose
*  =======
*
*  ZERRLQ tests the error exits for the COMPLEX*16 routines
*  that use the LQ decomposition of a general matrix.
*
*  Arguments
*  =========
*
*  PATH    (input) CHARACTER*3
*          The LAPACK path name for the routines to be tested.
*
*  NUNIT   (input) INTEGER
*          The unit number for output.
*
*  =====================================================================
*
*     .. Parameters ..
      INTEGER            NMAX
      PARAMETER          ( NMAX = 2 )
*     ..
*     .. Local Scalars ..
      INTEGER            I, INFO, J
*     ..
*     .. Local Arrays ..
      COMPLEX*16         A( NMAX, NMAX ), AF( NMAX, NMAX ), B( NMAX ),
     $                   W( NMAX ), X( NMAX )
*     ..
*     .. External Subroutines ..
      EXTERNAL           ALAESM, CHKXER, ZGELQ2, ZGELQF, ZGELQS, ZUNGL2,
     $                   ZUNGLQ, ZUNML2, ZUNMLQ
*     ..
*     .. Scalars in Common ..
      LOGICAL            LERR, OK
      CHARACTER*32       SRNAMT
      INTEGER            INFOT, NOUT
*     ..
*     .. Common blocks ..
      COMMON             / INFOC / INFOT, NOUT, OK, LERR
      COMMON             / SRNAMC / SRNAMT
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          DBLE, DCMPLX
*     ..
*     .. Executable Statements ..
*
      NOUT = NUNIT
      WRITE( NOUT, FMT = * )
*
*     Set the variables to innocuous values.
*
      DO 20 J = 1, NMAX
         DO 10 I = 1, NMAX
            A( I, J ) = DCMPLX( 1.D0 / DBLE( I+J ),
     $                  -1.D0 / DBLE( I+J ) )
            AF( I, J ) = DCMPLX( 1.D0 / DBLE( I+J ),
     $                   -1.D0 / DBLE( I+J ) )
   10    CONTINUE
         B( J ) = 0.D0
         W( J ) = 0.D0
         X( J ) = 0.D0
   20 CONTINUE
      OK = .TRUE.
*
*     Error exits for LQ factorization
*
*     ZGELQF
*
      SRNAMT = 'ZGELQF'
      INFOT = 1
      CALL MAGMAF_ZGELQF( -1, 0, A, 1, B, W, 1, INFO )
      CALL CHKXER( 'ZGELQF', INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL MAGMAF_ZGELQF( 0, -1, A, 1, B, W, 1, INFO )
      CALL CHKXER( 'ZGELQF', INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL MAGMAF_ZGELQF( 2, 1, A, 1, B, W, 2, INFO )
      CALL CHKXER( 'ZGELQF', INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL MAGMAF_ZGELQF( 2, 1, A, 2, B, W, 1, INFO )
      CALL CHKXER( 'ZGELQF', INFOT, NOUT, LERR, OK )
*
*     ZGELQ2
*
      SRNAMT = 'ZGELQ2'
      INFOT = 1
      CALL ZGELQ2( -1, 0, A, 1, B, W, INFO )
      CALL CHKXER( 'ZGELQ2', INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL ZGELQ2( 0, -1, A, 1, B, W, INFO )
      CALL CHKXER( 'ZGELQ2', INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL ZGELQ2( 2, 1, A, 1, B, W, INFO )
      CALL CHKXER( 'ZGELQ2', INFOT, NOUT, LERR, OK )
*
*     ZGELQS
*
      SRNAMT = 'ZGELQS'
      INFOT = 1
      CALL ZGELQS( -1, 0, 0, A, 1, X, B, 1, W, 1, INFO )
      CALL CHKXER( 'ZGELQS', INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL ZGELQS( 0, -1, 0, A, 1, X, B, 1, W, 1, INFO )
      CALL CHKXER( 'ZGELQS', INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL ZGELQS( 2, 1, 0, A, 2, X, B, 1, W, 1, INFO )
      CALL CHKXER( 'ZGELQS', INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL ZGELQS( 0, 0, -1, A, 1, X, B, 1, W, 1, INFO )
      CALL CHKXER( 'ZGELQS', INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL ZGELQS( 2, 2, 0, A, 1, X, B, 2, W, 1, INFO )
      CALL CHKXER( 'ZGELQS', INFOT, NOUT, LERR, OK )
      INFOT = 8
      CALL ZGELQS( 1, 2, 0, A, 1, X, B, 1, W, 1, INFO )
      CALL CHKXER( 'ZGELQS', INFOT, NOUT, LERR, OK )
      INFOT = 10
      CALL ZGELQS( 1, 1, 2, A, 1, X, B, 1, W, 1, INFO )
      CALL CHKXER( 'ZGELQS', INFOT, NOUT, LERR, OK )
*
*     ZUNGLQ
*
      SRNAMT = 'ZUNGLQ'
      INFOT = 1
      CALL ZUNGLQ( -1, 0, 0, A, 1, X, W, 1, INFO )
      CALL CHKXER( 'ZUNGLQ', INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL ZUNGLQ( 0, -1, 0, A, 1, X, W, 1, INFO )
      CALL CHKXER( 'ZUNGLQ', INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL ZUNGLQ( 2, 1, 0, A, 2, X, W, 2, INFO )
      CALL CHKXER( 'ZUNGLQ', INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL ZUNGLQ( 0, 0, -1, A, 1, X, W, 1, INFO )
      CALL CHKXER( 'ZUNGLQ', INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL ZUNGLQ( 1, 1, 2, A, 1, X, W, 1, INFO )
      CALL CHKXER( 'ZUNGLQ', INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL ZUNGLQ( 2, 2, 0, A, 1, X, W, 2, INFO )
      CALL CHKXER( 'ZUNGLQ', INFOT, NOUT, LERR, OK )
      INFOT = 8
      CALL ZUNGLQ( 2, 2, 0, A, 2, X, W, 1, INFO )
      CALL CHKXER( 'ZUNGLQ', INFOT, NOUT, LERR, OK )
*
*     ZUNGL2
*
      SRNAMT = 'ZUNGL2'
      INFOT = 1
      CALL ZUNGL2( -1, 0, 0, A, 1, X, W, INFO )
      CALL CHKXER( 'ZUNGL2', INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL ZUNGL2( 0, -1, 0, A, 1, X, W, INFO )
      CALL CHKXER( 'ZUNGL2', INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL ZUNGL2( 2, 1, 0, A, 2, X, W, INFO )
      CALL CHKXER( 'ZUNGL2', INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL ZUNGL2( 0, 0, -1, A, 1, X, W, INFO )
      CALL CHKXER( 'ZUNGL2', INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL ZUNGL2( 1, 1, 2, A, 1, X, W, INFO )
      CALL CHKXER( 'ZUNGL2', INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL ZUNGL2( 2, 2, 0, A, 1, X, W, INFO )
      CALL CHKXER( 'ZUNGL2', INFOT, NOUT, LERR, OK )
*
*     ZUNMLQ
*
      SRNAMT = 'ZUNMLQ'
      INFOT = 1
      CALL ZUNMLQ( '/', 'N', 0, 0, 0, A, 1, X, AF, 1, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL ZUNMLQ( 'L', '/', 0, 0, 0, A, 1, X, AF, 1, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL ZUNMLQ( 'L', 'N', -1, 0, 0, A, 1, X, AF, 1, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL ZUNMLQ( 'L', 'N', 0, -1, 0, A, 1, X, AF, 1, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL ZUNMLQ( 'L', 'N', 0, 0, -1, A, 1, X, AF, 1, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL ZUNMLQ( 'L', 'N', 0, 1, 1, A, 1, X, AF, 1, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL ZUNMLQ( 'R', 'N', 1, 0, 1, A, 1, X, AF, 1, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL ZUNMLQ( 'L', 'N', 2, 0, 2, A, 1, X, AF, 2, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL ZUNMLQ( 'R', 'N', 0, 2, 2, A, 1, X, AF, 1, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 10
      CALL ZUNMLQ( 'L', 'N', 2, 1, 0, A, 2, X, AF, 1, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 12
      CALL ZUNMLQ( 'L', 'N', 1, 2, 0, A, 1, X, AF, 1, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
      INFOT = 12
      CALL ZUNMLQ( 'R', 'N', 2, 1, 0, A, 1, X, AF, 2, W, 1, INFO )
      CALL CHKXER( 'ZUNMLQ', INFOT, NOUT, LERR, OK )
*
*     ZUNML2
*
      SRNAMT = 'ZUNML2'
      INFOT = 1
      CALL ZUNML2( '/', 'N', 0, 0, 0, A, 1, X, AF, 1, W, INFO )
      CALL CHKXER( 'ZUNML2', INFOT, NOUT, LERR, OK )
      INFOT = 2
      CALL ZUNML2( 'L', '/', 0, 0, 0, A, 1, X, AF, 1, W, INFO )
      CALL CHKXER( 'ZUNML2', INFOT, NOUT, LERR, OK )
      INFOT = 3
      CALL ZUNML2( 'L', 'N', -1, 0, 0, A, 1, X, AF, 1, W, INFO )
      CALL CHKXER( 'ZUNML2', INFOT, NOUT, LERR, OK )
      INFOT = 4
      CALL ZUNML2( 'L', 'N', 0, -1, 0, A, 1, X, AF, 1, W, INFO )
      CALL CHKXER( 'ZUNML2', INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL ZUNML2( 'L', 'N', 0, 0, -1, A, 1, X, AF, 1, W, INFO )
      CALL CHKXER( 'ZUNML2', INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL ZUNML2( 'L', 'N', 0, 1, 1, A, 1, X, AF, 1, W, INFO )
      CALL CHKXER( 'ZUNML2', INFOT, NOUT, LERR, OK )
      INFOT = 5
      CALL ZUNML2( 'R', 'N', 1, 0, 1, A, 1, X, AF, 1, W, INFO )
      CALL CHKXER( 'ZUNML2', INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL ZUNML2( 'L', 'N', 2, 1, 2, A, 1, X, AF, 2, W, INFO )
      CALL CHKXER( 'ZUNML2', INFOT, NOUT, LERR, OK )
      INFOT = 7
      CALL ZUNML2( 'R', 'N', 1, 2, 2, A, 1, X, AF, 1, W, INFO )
      CALL CHKXER( 'ZUNML2', INFOT, NOUT, LERR, OK )
      INFOT = 10
      CALL ZUNML2( 'L', 'N', 2, 1, 0, A, 2, X, AF, 1, W, INFO )
      CALL CHKXER( 'ZUNML2', INFOT, NOUT, LERR, OK )
*
*     Print a summary line.
*
      CALL ALAESM( PATH, OK, NOUT )
*
      RETURN
*
*     End of ZERRLQ
*
      END
