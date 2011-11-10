      SUBROUTINE CERRSY( PATH, NUNIT )
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
*  CERRSY tests the error exits for the COMPLEX routines
*  for symmetric indefinite matrices.
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
      EXTERNAL           ALAESM, CHKXER, CSPCON, CSPRFS, CSPTRF, CSPTRI,
     $                   CSPTRS, CSYCON, CSYRFS, CSYTF2, CSYTRF, CSYTRI,
     $                   CSYTRI2, CSYTRS
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
*     factorization of a symmetric indefinite matrix.
*
      IF( LSAMEN( 2, C2, 'SY' ) ) THEN
*
*        CSYTRF
*
         SRNAMT = 'CSYTRF'
         INFOT = 1
         CALL CSYTRF( '/', 0, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'CSYTRF', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSYTRF( 'U', -1, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'CSYTRF', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL CSYTRF( 'U', 2, A, 1, IP, W, 4, INFO )
         CALL CHKXER( 'CSYTRF', INFOT, NOUT, LERR, OK )
*
*        CSYTF2
*
         SRNAMT = 'CSYTF2'
         INFOT = 1
         CALL CSYTF2( '/', 0, A, 1, IP, INFO )
         CALL CHKXER( 'CSYTF2', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSYTF2( 'U', -1, A, 1, IP, INFO )
         CALL CHKXER( 'CSYTF2', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL CSYTF2( 'U', 2, A, 1, IP, INFO )
         CALL CHKXER( 'CSYTF2', INFOT, NOUT, LERR, OK )
*
*        CSYTRI
*
         SRNAMT = 'CSYTRI'
         INFOT = 1
         CALL CSYTRI( '/', 0, A, 1, IP, W, INFO )
         CALL CHKXER( 'CSYTRI', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSYTRI( 'U', -1, A, 1, IP, W, INFO )
         CALL CHKXER( 'CSYTRI', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL CSYTRI( 'U', 2, A, 1, IP, W, INFO )
         CALL CHKXER( 'CSYTRI', INFOT, NOUT, LERR, OK )
*
*        CSYTRI2
*
         SRNAMT = 'CSYTRI2'
         INFOT = 1
         CALL CSYTRI2( '/', 0, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'CSYTRI2', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSYTRI2( 'U', -1, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'CSYTRI2', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL CSYTRI2( 'U', 2, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'CSYTRI2', INFOT, NOUT, LERR, OK )
*
*        CSYTRS
*
         SRNAMT = 'CSYTRS'
         INFOT = 1
         CALL CSYTRS( '/', 0, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'CSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSYTRS( 'U', -1, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'CSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL CSYTRS( 'U', 0, -1, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'CSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL CSYTRS( 'U', 2, 1, A, 1, IP, B, 2, INFO )
         CALL CHKXER( 'CSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL CSYTRS( 'U', 2, 1, A, 2, IP, B, 1, INFO )
         CALL CHKXER( 'CSYTRS', INFOT, NOUT, LERR, OK )
*
*        CSYRFS
*
         SRNAMT = 'CSYRFS'
         INFOT = 1
         CALL CSYRFS( '/', 0, 0, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'CSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSYRFS( 'U', -1, 0, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2,
     $                W, R, INFO )
         CALL CHKXER( 'CSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL CSYRFS( 'U', 0, -1, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2,
     $                W, R, INFO )
         CALL CHKXER( 'CSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL CSYRFS( 'U', 2, 1, A, 1, AF, 2, IP, B, 2, X, 2, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'CSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 7
         CALL CSYRFS( 'U', 2, 1, A, 2, AF, 1, IP, B, 2, X, 2, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'CSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 10
         CALL CSYRFS( 'U', 2, 1, A, 2, AF, 2, IP, B, 1, X, 2, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'CSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 12
         CALL CSYRFS( 'U', 2, 1, A, 2, AF, 2, IP, B, 2, X, 1, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'CSYRFS', INFOT, NOUT, LERR, OK )
*
*        CSYCON
*
         SRNAMT = 'CSYCON'
         INFOT = 1
         CALL CSYCON( '/', 0, A, 1, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSYCON( 'U', -1, A, 1, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL CSYCON( 'U', 2, A, 1, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 6
         CALL CSYCON( 'U', 1, A, 1, IP, -ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CSYCON', INFOT, NOUT, LERR, OK )
*
*     Test error exits of the routines that use the diagonal pivoting
*     factorization of a symmetric indefinite packed matrix.
*
      ELSE IF( LSAMEN( 2, C2, 'SP' ) ) THEN
*
*        CSPTRF
*
         SRNAMT = 'CSPTRF'
         INFOT = 1
         CALL CSPTRF( '/', 0, A, IP, INFO )
         CALL CHKXER( 'CSPTRF', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSPTRF( 'U', -1, A, IP, INFO )
         CALL CHKXER( 'CSPTRF', INFOT, NOUT, LERR, OK )
*
*        CSPTRI
*
         SRNAMT = 'CSPTRI'
         INFOT = 1
         CALL CSPTRI( '/', 0, A, IP, W, INFO )
         CALL CHKXER( 'CSPTRI', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSPTRI( 'U', -1, A, IP, W, INFO )
         CALL CHKXER( 'CSPTRI', INFOT, NOUT, LERR, OK )
*
*        CSPTRS
*
         SRNAMT = 'CSPTRS'
         INFOT = 1
         CALL CSPTRS( '/', 0, 0, A, IP, B, 1, INFO )
         CALL CHKXER( 'CSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSPTRS( 'U', -1, 0, A, IP, B, 1, INFO )
         CALL CHKXER( 'CSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL CSPTRS( 'U', 0, -1, A, IP, B, 1, INFO )
         CALL CHKXER( 'CSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 7
         CALL CSPTRS( 'U', 2, 1, A, IP, B, 1, INFO )
         CALL CHKXER( 'CSPTRS', INFOT, NOUT, LERR, OK )
*
*        CSPRFS
*
         SRNAMT = 'CSPRFS'
         INFOT = 1
         CALL CSPRFS( '/', 0, 0, A, AF, IP, B, 1, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'CSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSPRFS( 'U', -1, 0, A, AF, IP, B, 1, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'CSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL CSPRFS( 'U', 0, -1, A, AF, IP, B, 1, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'CSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL CSPRFS( 'U', 2, 1, A, AF, IP, B, 1, X, 2, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'CSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 10
         CALL CSPRFS( 'U', 2, 1, A, AF, IP, B, 2, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'CSPRFS', INFOT, NOUT, LERR, OK )
*
*        CSPCON
*
         SRNAMT = 'CSPCON'
         INFOT = 1
         CALL CSPCON( '/', 0, A, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CSPCON', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL CSPCON( 'U', -1, A, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CSPCON', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL CSPCON( 'U', 1, A, IP, -ANRM, RCOND, W, INFO )
         CALL CHKXER( 'CSPCON', INFOT, NOUT, LERR, OK )
      END IF
*
*     Print a summary line.
*
      CALL ALAESM( PATH, OK, NOUT )
*
      RETURN
*
*     End of CERRSY
*
      END
