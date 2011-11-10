      SUBROUTINE CERRHE( PATH, NUNIT )
*
*  -- LAPACK test routine (version 3.3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*  -- April 2011                                                      --
*
*     .. Scalar Arguments ..
      CHARACTER*3        PATH
      INTEGER            NUNIT
*     ..
*
*  Purpose
*  =======
*
*  CERRHE tests the error exits for the COMPLEX routines
*  for Hermitian indefinite matrices.
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
*
*     .. Parameters ..
      INTEGER            NMAX
      PARAMETER          ( NMAX = 4 )
*     ..
*     .. Local Scalars ..
      CHARACTER*2        C2
      INTEGER            I, INFO, J
      REAL               ANRM, RCOND
*     ..
*     .. Local Arrays ..
      INTEGER            IP( NMAX )
      REAL               R( NMAX ), R1( NMAX ), R2( NMAX )
      COMPLEX            A( NMAX, NMAX ), AF( NMAX, NMAX ), B( NMAX ),
     $                   W( 2*NMAX ), X( NMAX )
*     ..
*     .. External Functions ..
      LOGICAL            LSAMEN
      EXTERNAL           LSAMEN
*     ..
*     .. External Subroutines ..
      EXTERNAL           ALAESM, CHECON, CHERFS, CHETF2, CHETRF, CHETRI,
     $                   CHETRI2, CHETRS, CHKXER, CHPCON, CHPRFS,
     $                   CHPTRF, CHPTRI, CHPTRS
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
      INTRINSIC          CMPLX, REAL
*     ..
*     .. Executable Statements ..
*
      NOUT = NUNIT
      WRITE( NOUT, FMT = * )
      C2 = PATH( 2: 3 )
*
*     Set the variables to innocuous values.
*
      DO 20 J = 1, NMAX
         DO 10 I = 1, NMAX
            A( I, J ) = CMPLX( 1. / REAL( I+J ), -1. / REAL( I+J ) )
            AF( I, J ) = CMPLX( 1. / REAL( I+J ), -1. / REAL( I+J ) )
   10    CONTINUE
         B( J ) = 0.
         R1( J ) = 0.
         R2( J ) = 0.
         W( J ) = 0.
         X( J ) = 0.
         IP( J ) = J
   20 CONTINUE
      ANRM = 1.0
      OK = .TRUE.
*
*     Test error exits of the routines that use the diagonal pivoting
*     factorization of a Hermitian indefinite matrix.
*
      IF( LSAMEN( 2, C2, 'HE' ) ) THEN
*
*        CHETRF
*
         SRNAMT = 'CHETRF'
         INFOT = 1
         CALL CHETRF( '/', 0, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'CHETRF', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHETRF( 'U', -1, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'CHETRF', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL CHETRF( 'U', 2, A, 1, IP, W, 4, INFO )
         CALL CHKXER( 'CHETRF', INFOT, NOUT, LERR, OK )
*
*        CHETF2
*
         SRNAMT = 'CHETF2'
         INFOT = 1
         CALL CHETF2( '/', 0, A, 1, IP, INFO )
         CALL CHKXER( 'CHETF2', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHETF2( 'U', -1, A, 1, IP, INFO )
         CALL CHKXER( 'CHETF2', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL CHETF2( 'U', 2, A, 1, IP, INFO )
         CALL CHKXER( 'CHETF2', INFOT, NOUT, LERR, OK )
*
*        CHETRI
*
         SRNAMT = 'CHETRI'
         INFOT = 1
         CALL CHETRI( '/', 0, A, 1, IP, W, INFO )
         CALL CHKXER( 'CHETRI', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHETRI( 'U', -1, A, 1, IP, W, INFO )
         CALL CHKXER( 'CHETRI', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL CHETRI( 'U', 2, A, 1, IP, W, INFO )
         CALL CHKXER( 'CHETRI', INFOT, NOUT, LERR, OK )
*
*        CHETRI2
*
         SRNAMT = 'CHETRI2'
         INFOT = 1
         CALL CHETRI2( '/', 0, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'CHETRI2', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHETRI2( 'U', -1, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'CHETRI2', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL CHETRI2( 'U', 2, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'CHETRI2', INFOT, NOUT, LERR, OK )
*
*        CHETRS
*
         SRNAMT = 'CHETRS'
         INFOT = 1
         CALL CHETRS( '/', 0, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'CHETRS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHETRS( 'U', -1, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'CHETRS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL CHETRS( 'U', 0, -1, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'CHETRS', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL CHETRS( 'U', 2, 1, A, 1, IP, B, 2, INFO )
         CALL CHKXER( 'CHETRS', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL CHETRS( 'U', 2, 1, A, 2, IP, B, 1, INFO )
         CALL CHKXER( 'CHETRS', INFOT, NOUT, LERR, OK )
*
*        CHERFS
*
         SRNAMT = 'CHERFS'
         INFOT = 1
         CALL CHERFS( '/', 0, 0, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'CHERFS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHERFS( 'U', -1, 0, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2,
     $                W, R, INFO )
         CALL CHKXER( 'CHERFS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL CHERFS( 'U', 0, -1, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2,
     $                W, R, INFO )
         CALL CHKXER( 'CHERFS', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL CHERFS( 'U', 2, 1, A, 1, AF, 2, IP, B, 2, X, 2, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'CHERFS', INFOT, NOUT, LERR, OK )
         INFOT = 7
         CALL CHERFS( 'U', 2, 1, A, 2, AF, 1, IP, B, 2, X, 2, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'CHERFS', INFOT, NOUT, LERR, OK )
         INFOT = 10
         CALL CHERFS( 'U', 2, 1, A, 2, AF, 2, IP, B, 1, X, 2, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'CHERFS', INFOT, NOUT, LERR, OK )
         INFOT = 12
         CALL CHERFS( 'U', 2, 1, A, 2, AF, 2, IP, B, 2, X, 1, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'CHERFS', INFOT, NOUT, LERR, OK )
*
*        CHECON
*
         SRNAMT = 'CHECON'
         INFOT = 1
         CALL CHECON( '/', 0, A, 1, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CHECON', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHECON( 'U', -1, A, 1, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CHECON', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL CHECON( 'U', 2, A, 1, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CHECON', INFOT, NOUT, LERR, OK )
         INFOT = 6
         CALL CHECON( 'U', 1, A, 1, IP, -ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CHECON', INFOT, NOUT, LERR, OK )
*
*     Test error exits of the routines that use the diagonal pivoting
*     factorization of a Hermitian indefinite packed matrix.
*
      ELSE IF( LSAMEN( 2, C2, 'HP' ) ) THEN
*
*        CHPTRF
*
         SRNAMT = 'CHPTRF'
         INFOT = 1
         CALL CHPTRF( '/', 0, A, IP, INFO )
         CALL CHKXER( 'CHPTRF', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHPTRF( 'U', -1, A, IP, INFO )
         CALL CHKXER( 'CHPTRF', INFOT, NOUT, LERR, OK )
*
*        CHPTRI
*
         SRNAMT = 'CHPTRI'
         INFOT = 1
         CALL CHPTRI( '/', 0, A, IP, W, INFO )
         CALL CHKXER( 'CHPTRI', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHPTRI( 'U', -1, A, IP, W, INFO )
         CALL CHKXER( 'CHPTRI', INFOT, NOUT, LERR, OK )
*
*        CHPTRS
*
         SRNAMT = 'CHPTRS'
         INFOT = 1
         CALL CHPTRS( '/', 0, 0, A, IP, B, 1, INFO )
         CALL CHKXER( 'CHPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHPTRS( 'U', -1, 0, A, IP, B, 1, INFO )
         CALL CHKXER( 'CHPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL CHPTRS( 'U', 0, -1, A, IP, B, 1, INFO )
         CALL CHKXER( 'CHPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 7
         CALL CHPTRS( 'U', 2, 1, A, IP, B, 1, INFO )
         CALL CHKXER( 'CHPTRS', INFOT, NOUT, LERR, OK )
*
*        CHPRFS
*
         SRNAMT = 'CHPRFS'
         INFOT = 1
         CALL CHPRFS( '/', 0, 0, A, AF, IP, B, 1, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'CHPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHPRFS( 'U', -1, 0, A, AF, IP, B, 1, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'CHPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL CHPRFS( 'U', 0, -1, A, AF, IP, B, 1, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'CHPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL CHPRFS( 'U', 2, 1, A, AF, IP, B, 1, X, 2, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'CHPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 10
         CALL CHPRFS( 'U', 2, 1, A, AF, IP, B, 2, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'CHPRFS', INFOT, NOUT, LERR, OK )
*
*        CHPCON
*
         SRNAMT = 'CHPCON'
         INFOT = 1
         CALL CHPCON( '/', 0, A, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CHPCON', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CHPCON( 'U', -1, A, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CHPCON', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL CHPCON( 'U', 1, A, IP, -ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CHPCON', INFOT, NOUT, LERR, OK )
      END IF
*
*     Print a summary line.
*
      CALL ALAESM( PATH, OK, NOUT )
*
      RETURN
*
*     End of CERRHE
*
      END
