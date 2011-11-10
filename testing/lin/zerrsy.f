      SUBROUTINE ZERRSY( PATH, NUNIT )
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
*  ZERRSY tests the error exits for the COMPLEX*16 routines
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
      DOUBLE PRECISION   ANRM, RCOND
*     ..
*     .. Local Arrays ..
      INTEGER            IP( NMAX )
      DOUBLE PRECISION   R( NMAX ), R1( NMAX ), R2( NMAX )
      COMPLEX*16         A( NMAX, NMAX ), AF( NMAX, NMAX ), B( NMAX ),
     $                   W( 2*NMAX ), X( NMAX )
*     ..
*     .. External Functions ..
      LOGICAL            LSAMEN
      EXTERNAL           LSAMEN
*     ..
*     .. External Subroutines ..
      EXTERNAL           ALAESM, CHKXER, ZSPCON, ZSPRFS, ZSPTRF, ZSPTRI,
     $                   ZSPTRS, ZSYCON, ZSYRFS, ZSYTF2, ZSYTRF, ZSYTRI,
     $                   ZSYTRI2, ZSYTRS
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
      C2 = PATH( 2: 3 )
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
         R1( J ) = 0.D0
         R2( J ) = 0.D0
         W( J ) = 0.D0
         X( J ) = 0.D0
         IP( J ) = J
   20 CONTINUE
      ANRM = 1.0D0
      OK = .TRUE.
*
*     Test error exits of the routines that use the diagonal pivoting
*     factorization of a symmetric indefinite matrix.
*
      IF( LSAMEN( 2, C2, 'SY' ) ) THEN
*
*        ZSYTRF
*
         SRNAMT = 'ZSYTRF'
         INFOT = 1
         CALL ZSYTRF( '/', 0, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'ZSYTRF', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSYTRF( 'U', -1, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'ZSYTRF', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL ZSYTRF( 'U', 2, A, 1, IP, W, 4, INFO )
         CALL CHKXER( 'ZSYTRF', INFOT, NOUT, LERR, OK )
*
*        ZSYTF2
*
         SRNAMT = 'ZSYTF2'
         INFOT = 1
         CALL ZSYTF2( '/', 0, A, 1, IP, INFO )
         CALL CHKXER( 'ZSYTF2', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSYTF2( 'U', -1, A, 1, IP, INFO )
         CALL CHKXER( 'ZSYTF2', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL ZSYTF2( 'U', 2, A, 1, IP, INFO )
         CALL CHKXER( 'ZSYTF2', INFOT, NOUT, LERR, OK )
*
*        ZSYTRI
*
         SRNAMT = 'ZSYTRI'
         INFOT = 1
         CALL ZSYTRI( '/', 0, A, 1, IP, W, INFO )
         CALL CHKXER( 'ZSYTRI', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSYTRI( 'U', -1, A, 1, IP, W, INFO )
         CALL CHKXER( 'ZSYTRI', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL ZSYTRI( 'U', 2, A, 1, IP, W, INFO )
         CALL CHKXER( 'ZSYTRI', INFOT, NOUT, LERR, OK )
*
*        ZSYTRI2
*
         SRNAMT = 'ZSYTRI2'
         INFOT = 1
         CALL ZSYTRI2( '/', 0, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'ZSYTRI2', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSYTRI2( 'U', -1, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'ZSYTRI2', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL ZSYTRI2( 'U', 2, A, 1, IP, W, 1, INFO )
         CALL CHKXER( 'ZSYTRI2', INFOT, NOUT, LERR, OK )
*
*        ZSYTRS
*
         SRNAMT = 'ZSYTRS'
         INFOT = 1
         CALL ZSYTRS( '/', 0, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'ZSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSYTRS( 'U', -1, 0, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'ZSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL ZSYTRS( 'U', 0, -1, A, 1, IP, B, 1, INFO )
         CALL CHKXER( 'ZSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL ZSYTRS( 'U', 2, 1, A, 1, IP, B, 2, INFO )
         CALL CHKXER( 'ZSYTRS', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL ZSYTRS( 'U', 2, 1, A, 2, IP, B, 1, INFO )
         CALL CHKXER( 'ZSYTRS', INFOT, NOUT, LERR, OK )
*
*        ZSYRFS
*
         SRNAMT = 'ZSYRFS'
         INFOT = 1
         CALL ZSYRFS( '/', 0, 0, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'ZSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSYRFS( 'U', -1, 0, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2,
     $                W, R, INFO )
         CALL CHKXER( 'ZSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL ZSYRFS( 'U', 0, -1, A, 1, AF, 1, IP, B, 1, X, 1, R1, R2,
     $                W, R, INFO )
         CALL CHKXER( 'ZSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL ZSYRFS( 'U', 2, 1, A, 1, AF, 2, IP, B, 2, X, 2, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'ZSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 7
         CALL ZSYRFS( 'U', 2, 1, A, 2, AF, 1, IP, B, 2, X, 2, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'ZSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 10
         CALL ZSYRFS( 'U', 2, 1, A, 2, AF, 2, IP, B, 1, X, 2, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'ZSYRFS', INFOT, NOUT, LERR, OK )
         INFOT = 12
         CALL ZSYRFS( 'U', 2, 1, A, 2, AF, 2, IP, B, 2, X, 1, R1, R2, W,
     $                R, INFO )
         CALL CHKXER( 'ZSYRFS', INFOT, NOUT, LERR, OK )
*
*        ZSYCON
*
         SRNAMT = 'ZSYCON'
         INFOT = 1
         CALL ZSYCON( '/', 0, A, 1, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'ZSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSYCON( 'U', -1, A, 1, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'ZSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 4
         CALL ZSYCON( 'U', 2, A, 1, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'ZSYCON', INFOT, NOUT, LERR, OK )
         INFOT = 6
         CALL ZSYCON( 'U', 1, A, 1, IP, -ANRM, RCOND, W, INFO )
         CALL CHKXER( 'ZSYCON', INFOT, NOUT, LERR, OK )
*
*     Test error exits of the routines that use the diagonal pivoting
*     factorization of a symmetric indefinite packed matrix.
*
      ELSE IF( LSAMEN( 2, C2, 'SP' ) ) THEN
*
*        ZSPTRF
*
         SRNAMT = 'ZSPTRF'
         INFOT = 1
         CALL ZSPTRF( '/', 0, A, IP, INFO )
         CALL CHKXER( 'ZSPTRF', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSPTRF( 'U', -1, A, IP, INFO )
         CALL CHKXER( 'ZSPTRF', INFOT, NOUT, LERR, OK )
*
*        ZSPTRI
*
         SRNAMT = 'ZSPTRI'
         INFOT = 1
         CALL ZSPTRI( '/', 0, A, IP, W, INFO )
         CALL CHKXER( 'ZSPTRI', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSPTRI( 'U', -1, A, IP, W, INFO )
         CALL CHKXER( 'ZSPTRI', INFOT, NOUT, LERR, OK )
*
*        ZSPTRS
*
         SRNAMT = 'ZSPTRS'
         INFOT = 1
         CALL ZSPTRS( '/', 0, 0, A, IP, B, 1, INFO )
         CALL CHKXER( 'ZSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSPTRS( 'U', -1, 0, A, IP, B, 1, INFO )
         CALL CHKXER( 'ZSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL ZSPTRS( 'U', 0, -1, A, IP, B, 1, INFO )
         CALL CHKXER( 'ZSPTRS', INFOT, NOUT, LERR, OK )
         INFOT = 7
         CALL ZSPTRS( 'U', 2, 1, A, IP, B, 1, INFO )
         CALL CHKXER( 'ZSPTRS', INFOT, NOUT, LERR, OK )
*
*        ZSPRFS
*
         SRNAMT = 'ZSPRFS'
         INFOT = 1
         CALL ZSPRFS( '/', 0, 0, A, AF, IP, B, 1, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'ZSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSPRFS( 'U', -1, 0, A, AF, IP, B, 1, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'ZSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 3
         CALL ZSPRFS( 'U', 0, -1, A, AF, IP, B, 1, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'ZSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 8
         CALL ZSPRFS( 'U', 2, 1, A, AF, IP, B, 1, X, 2, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'ZSPRFS', INFOT, NOUT, LERR, OK )
         INFOT = 10
         CALL ZSPRFS( 'U', 2, 1, A, AF, IP, B, 2, X, 1, R1, R2, W, R,
     $                INFO )
         CALL CHKXER( 'ZSPRFS', INFOT, NOUT, LERR, OK )
*
*        ZSPCON
*
         SRNAMT = 'ZSPCON'
         INFOT = 1
         CALL ZSPCON( '/', 0, A, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'ZSPCON', INFOT, NOUT, LERR, OK )
         INFOT = 2
         CALL ZSPCON( 'U', -1, A, IP, ANRM, RCOND, W, INFO )
         CALL CHKXER( 'ZSPCON', INFOT, NOUT, LERR, OK )
         INFOT = 5
         CALL ZSPCON( 'U', 1, A, IP, -ANRM, RCOND, W, INFO )
         CALL CHKXER( 'ZSPCON', INFOT, NOUT, LERR, OK )
      END IF
*
*     Print a summary line.
*
      CALL ALAESM( PATH, OK, NOUT )
*
      RETURN
*
*     End of ZERRSY
*
      END
