      SUBROUTINE ZCHKSY( DOTYPE, NN, NVAL, NNB, NBVAL, NNS, NSVAL,
     $                   THRESH, TSTERR, NMAX, A, AFAC, AINV, B, X,
     $                   XACT, WORK, RWORK, IWORK, NOUT )
*
*  -- LAPACK test routine (version 3.1) --
*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd..
*     June 2010
*
*     .. Scalar Arguments ..
      LOGICAL            TSTERR
      INTEGER            NMAX, NN, NNB, NNS, NOUT
      DOUBLE PRECISION   THRESH
*     ..
*     .. Array Arguments ..
      LOGICAL            DOTYPE( * )
      INTEGER            IWORK( * ), NBVAL( * ), NSVAL( * ), NVAL( * )
      DOUBLE PRECISION   RWORK( * )
      COMPLEX*16         A( * ), AFAC( * ), AINV( * ), B( * ),
     $                   WORK( * ), X( * ), XACT( * )
*     ..
*
*  Purpose
*  =======
*
*  ZCHKSY tests ZSYTRF, -TRI2, -TRS, -TRS2,  -RFS, and -CON.
*
*  Arguments
*  =========
*
*  DOTYPE  (input) LOGICAL array, dimension (NTYPES)
*          The matrix types to be used for testing.  Matrices of type j
*          (for 1 <= j <= NTYPES) are used for testing if DOTYPE(j) =
*          .TRUE.; if DOTYPE(j) = .FALSE., then type j is not used.
*
*  NN      (input) INTEGER
*          The number of values of N contained in the vector NVAL.
*
*  NVAL    (input) INTEGER array, dimension (NN)
*          The values of the matrix dimension N.
*
*  NNB     (input) INTEGER
*          The number of values of NB contained in the vector NBVAL.
*
*  NBVAL   (input) INTEGER array, dimension (NBVAL)
*          The values of the blocksize NB.
*
*  NNS     (input) INTEGER
*          The number of values of NRHS contained in the vector NSVAL.
*
*  NSVAL   (input) INTEGER array, dimension (NNS)
*          The values of the number of right hand sides NRHS.
*
*  THRESH  (input) DOUBLE PRECISION
*          The threshold value for the test ratios.  A result is
*          included in the output file if RESULT >= THRESH.  To have
*          every test ratio printed, use THRESH = 0.
*
*  TSTERR  (input) LOGICAL
*          Flag that indicates whether error exits are to be tested.
*
*  NMAX    (input) INTEGER
*          The maximum value permitted for N, used in dimensioning the
*          work arrays.
*
*  A       (workspace) COMPLEX*16 array, dimension (NMAX*NMAX)
*
*  AFAC    (workspace) COMPLEX*16 array, dimension (NMAX*NMAX)
*
*  AINV    (workspace) COMPLEX*16 array, dimension (NMAX*NMAX)
*
*  B       (workspace) COMPLEX*16 array, dimension (NMAX*NSMAX)
*          where NSMAX is the largest entry in NSVAL.
*
*  X       (workspace) COMPLEX*16 array, dimension (NMAX*NSMAX)
*
*  XACT    (workspace) COMPLEX*16 array, dimension (NMAX*NSMAX)
*
*  WORK    (workspace) COMPLEX*16 array, dimension
*                      (NMAX*max(2,NSMAX))
*
*  RWORK   (workspace) DOUBLE PRECISION array,
*                                 dimension (NMAX+2*NSMAX)
*
*  IWORK   (workspace) INTEGER array, dimension (NMAX)
*
*  NOUT    (input) INTEGER
*          The unit number for output.
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO
      PARAMETER          ( ZERO = 0.0D+0 )
      INTEGER            NTYPES
      PARAMETER          ( NTYPES = 11 )
      INTEGER            NTESTS
      PARAMETER          ( NTESTS = 9 )
*     ..
*     .. Local Scalars ..
      LOGICAL            TRFCON, ZEROT
      CHARACTER          DIST, TYPE, UPLO, XTYPE
      CHARACTER*3        PATH
      INTEGER            I, I1, I2, IMAT, IN, INB, INFO, IOFF, IRHS,
     $                   IUPLO, IZERO, J, K, KL, KU, LDA, LWORK, MODE,
     $                   N, NB, NERRS, NFAIL, NIMAT, NRHS, NRUN, NT
      DOUBLE PRECISION   ANORM, CNDNUM, RCOND, RCONDC
*     ..
*     .. Local Arrays ..
      CHARACTER          UPLOS( 2 )
      INTEGER            ISEED( 4 ), ISEEDY( 4 )
      DOUBLE PRECISION   RESULT( NTESTS )
*     ..
*     .. External Functions ..
      DOUBLE PRECISION   DGET06, ZLANSY
      EXTERNAL           DGET06, ZLANSY
*     ..
*     .. External Subroutines ..
      EXTERNAL           ALAERH, ALAHD, ALASUM, XLAENV, ZERRSY, ZGET04,
     $                   ZLACPY, ZLARHS, ZLATB4, ZLATMS, ZLATSY, ZPOT05,
     $                   ZSYCON, ZSYRFS, ZSYT01, ZSYT02, ZSYT03, ZSYTRF,
     $                   ZSYTRI2, ZSYTRS, ZSYTRS2
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN
*     ..
*     .. Scalars in Common ..
      LOGICAL            LERR, OK
      CHARACTER*32       SRNAMT
      INTEGER            INFOT, NUNIT
*     ..
*     .. Common blocks ..
      COMMON             / INFOC / INFOT, NUNIT, OK, LERR
      COMMON             / SRNAMC / SRNAMT
*     ..
*     .. Data statements ..
      DATA               ISEEDY / 1988, 1989, 1990, 1991 /
      DATA               UPLOS / 'U', 'L' /
*     ..
*     .. Executable Statements ..
*
*     Initialize constants and the random number seed.
*
      PATH( 1: 1 ) = 'Zomplex precision'
      PATH( 2: 3 ) = 'SY'
      NRUN = 0
      NFAIL = 0
      NERRS = 0
      DO 10 I = 1, 4
         ISEED( I ) = ISEEDY( I )
   10 CONTINUE
*
*     Test the error exits
*
      IF( TSTERR )
     $   CALL ZERRSY( PATH, NOUT )
      INFOT = 0
*
*     Do for each value of N in NVAL
*
      DO 180 IN = 1, NN
         N = NVAL( IN )
         LDA = MAX( N, 1 )
         XTYPE = 'N'
         NIMAT = NTYPES
         IF( N.LE.0 )
     $      NIMAT = 1
*
         IZERO = 0
         DO 170 IMAT = 1, NIMAT
*
*           Do the tests only if DOTYPE( IMAT ) is true.
*
            IF( .NOT.DOTYPE( IMAT ) )
     $         GO TO 170
*
*           Skip types 3, 4, 5, or 6 if the matrix size is too small.
*
            ZEROT = IMAT.GE.3 .AND. IMAT.LE.6
            IF( ZEROT .AND. N.LT.IMAT-2 )
     $         GO TO 170
*
*           Do first for UPLO = 'U', then for UPLO = 'L'
*
            DO 160 IUPLO = 1, 2
               UPLO = UPLOS( IUPLO )
*
               IF( IMAT.NE.NTYPES ) THEN
*
*                 Set up parameters with ZLATB4 and generate a test
*                 matrix with ZLATMS.
*
                  CALL ZLATB4( PATH, IMAT, N, N, TYPE, KL, KU, ANORM,
     $                         MODE, CNDNUM, DIST )
*
                  SRNAMT = 'ZLATMS'
                  CALL ZLATMS( N, N, DIST, ISEED, TYPE, RWORK, MODE,
     $                         CNDNUM, ANORM, KL, KU, 'N', A, LDA, WORK,
     $                         INFO )
*
*                 Check error code from ZLATMS.
*
                  IF( INFO.NE.0 ) THEN
                     CALL ALAERH( PATH, 'ZLATMS', INFO, 0, UPLO, N, N,
     $                            -1, -1, -1, IMAT, NFAIL, NERRS, NOUT )
                     GO TO 160
                  END IF
*
*                 For types 3-6, zero one or more rows and columns of
*                 the matrix to test that INFO is returned correctly.
*
                  IF( ZEROT ) THEN
                     IF( IMAT.EQ.3 ) THEN
                        IZERO = 1
                     ELSE IF( IMAT.EQ.4 ) THEN
                        IZERO = N
                     ELSE
                        IZERO = N / 2 + 1
                     END IF
*
                     IF( IMAT.LT.6 ) THEN
*
*                       Set row and column IZERO to zero.
*
                        IF( IUPLO.EQ.1 ) THEN
                           IOFF = ( IZERO-1 )*LDA
                           DO 20 I = 1, IZERO - 1
                              A( IOFF+I ) = ZERO
   20                      CONTINUE
                           IOFF = IOFF + IZERO
                           DO 30 I = IZERO, N
                              A( IOFF ) = ZERO
                              IOFF = IOFF + LDA
   30                      CONTINUE
                        ELSE
                           IOFF = IZERO
                           DO 40 I = 1, IZERO - 1
                              A( IOFF ) = ZERO
                              IOFF = IOFF + LDA
   40                      CONTINUE
                           IOFF = IOFF - IZERO
                           DO 50 I = IZERO, N
                              A( IOFF+I ) = ZERO
   50                      CONTINUE
                        END IF
                     ELSE
                        IF( IUPLO.EQ.1 ) THEN
*
*                          Set the first IZERO rows to zero.
*
                           IOFF = 0
                           DO 70 J = 1, N
                              I2 = MIN( J, IZERO )
                              DO 60 I = 1, I2
                                 A( IOFF+I ) = ZERO
   60                         CONTINUE
                              IOFF = IOFF + LDA
   70                      CONTINUE
                        ELSE
*
*                          Set the last IZERO rows to zero.
*
                           IOFF = 0
                           DO 90 J = 1, N
                              I1 = MAX( J, IZERO )
                              DO 80 I = I1, N
                                 A( IOFF+I ) = ZERO
   80                         CONTINUE
                              IOFF = IOFF + LDA
   90                      CONTINUE
                        END IF
                     END IF
                  ELSE
                     IZERO = 0
                  END IF
               ELSE
*
*                 Use a special block diagonal matrix to test alternate
*                 code for the 2 x 2 blocks.
*
                  CALL ZLATSY( UPLO, N, A, LDA, ISEED )
               END IF
*
*              Do for each value of NB in NBVAL
*
               DO 150 INB = 1, NNB
                  NB = NBVAL( INB )
                  CALL XLAENV( 1, NB )
*
*                 Compute the L*D*L' or U*D*U' factorization of the
*                 matrix.
*
                  CALL ZLACPY( UPLO, N, N, A, LDA, AFAC, LDA )
                  LWORK = MAX( 2, NB )*LDA
                  SRNAMT = 'ZSYTRF'
                  CALL ZSYTRF( UPLO, N, AFAC, LDA, IWORK, AINV, LWORK,
     $                         INFO )
*
*                 Adjust the expected value of INFO to account for
*                 pivoting.
*
                  K = IZERO
                  IF( K.GT.0 ) THEN
  100                CONTINUE
                     IF( IWORK( K ).LT.0 ) THEN
                        IF( IWORK( K ).NE.-K ) THEN
                           K = -IWORK( K )
                           GO TO 100
                        END IF
                     ELSE IF( IWORK( K ).NE.K ) THEN
                        K = IWORK( K )
                        GO TO 100
                     END IF
                  END IF
*
*                 Check error code from ZSYTRF.
*
                  IF( INFO.NE.K )
     $               CALL ALAERH( PATH, 'ZSYTRF', INFO, K, UPLO, N, N,
     $                            -1, -1, NB, IMAT, NFAIL, NERRS, NOUT )
                  IF( INFO.NE.0 ) THEN
                     TRFCON = .TRUE.
                  ELSE
                     TRFCON = .FALSE.
                  END IF
*
*+    TEST 1
*                 Reconstruct matrix from factors and compute residual.
*
                  CALL ZSYT01( UPLO, N, A, LDA, AFAC, LDA, IWORK, AINV,
     $                         LDA, RWORK, RESULT( 1 ) )
                  NT = 1
*
*+    TEST 2
*                 Form the inverse and compute the residual.
*
                  IF( INB.EQ.1 .AND. .NOT.TRFCON ) THEN
                     CALL ZLACPY( UPLO, N, N, AFAC, LDA, AINV, LDA )
                     SRNAMT = 'ZSYTRI2'
                     LWORK = (N+NB+1)*(NB+3)
                     CALL ZSYTRI2( UPLO, N, AINV, LDA, IWORK, WORK,
     $                            LWORK, INFO )
*
*                 Check error code from ZSYTRI2.
*
                     IF( INFO.NE.0 )
     $                  CALL ALAERH( PATH, 'ZSYTRI2', INFO, 0, UPLO, N,
     $                               N, -1, -1, -1, IMAT, NFAIL, NERRS,
     $                               NOUT )
*
                     CALL ZSYT03( UPLO, N, A, LDA, AINV, LDA, WORK, LDA,
     $                            RWORK, RCONDC, RESULT( 2 ) )
                     NT = 2
                  END IF
*
*                 Print information about the tests that did not pass
*                 the threshold.
*
                  DO 110 K = 1, NT
                     IF( RESULT( K ).GE.THRESH ) THEN
                        IF( NFAIL.EQ.0 .AND. NERRS.EQ.0 )
     $                     CALL ALAHD( NOUT, PATH )
                        WRITE( NOUT, FMT = 9999 )UPLO, N, NB, IMAT, K,
     $                     RESULT( K )
                        NFAIL = NFAIL + 1
                     END IF
  110             CONTINUE
                  NRUN = NRUN + NT
*
*                 Skip the other tests if this is not the first block
*                 size.
*
                  IF( INB.GT.1 )
     $               GO TO 150
*
*                 Do only the condition estimate if INFO is not 0.
*
                  IF( TRFCON ) THEN
                     RCONDC = ZERO
                     GO TO 140
                  END IF
*
                  DO 130 IRHS = 1, NNS
                     NRHS = NSVAL( IRHS )
*
*+    TEST 3 (Using ZSYTRS)
*                 Solve and compute residual for  A * X = B.
*
                     SRNAMT = 'ZLARHS'
                     CALL ZLARHS( PATH, XTYPE, UPLO, ' ', N, N, KL, KU,
     $                            NRHS, A, LDA, XACT, LDA, B, LDA,
     $                            ISEED, INFO )
                     CALL ZLACPY( 'Full', N, NRHS, B, LDA, X, LDA )
*
                     SRNAMT = 'ZSYTRS'
                     CALL ZSYTRS( UPLO, N, NRHS, AFAC, LDA, IWORK, X,
     $                            LDA, INFO )
*
*                 Check error code from ZSYTRS.
*
                     IF( INFO.NE.0 )
     $                  CALL ALAERH( PATH, 'ZSYTRS', INFO, 0, UPLO, N,
     $                               N, -1, -1, NRHS, IMAT, NFAIL,
     $                               NERRS, NOUT )
*
                     CALL ZLACPY( 'Full', N, NRHS, B, LDA, WORK, LDA )
                     CALL ZSYT02( UPLO, N, NRHS, A, LDA, X, LDA, WORK,
     $                            LDA, RWORK, RESULT( 3 ) )
*
*+    TEST 4 (Using ZSYTRS2)
*                 Solve and compute residual for  A * X = B.
*
                     SRNAMT = 'ZLARHS'
                     CALL ZLARHS( PATH, XTYPE, UPLO, ' ', N, N, KL, KU,
     $                            NRHS, A, LDA, XACT, LDA, B, LDA,
     $                            ISEED, INFO )
                     CALL ZLACPY( 'Full', N, NRHS, B, LDA, X, LDA )
*
                     SRNAMT = 'ZSYTRS2'
                     CALL ZSYTRS2( UPLO, N, NRHS, AFAC, LDA, IWORK, X,
     $                            LDA, WORK, INFO )
*
*                 Check error code from ZSYTRS.
*
                     IF( INFO.NE.0 )
     $                  CALL ALAERH( PATH, 'ZSYTRS', INFO, 0, UPLO, N,
     $                               N, -1, -1, NRHS, IMAT, NFAIL,
     $                               NERRS, NOUT )
*
                     CALL ZLACPY( 'Full', N, NRHS, B, LDA, WORK, LDA )
                     CALL ZSYT02( UPLO, N, NRHS, A, LDA, X, LDA, WORK,
     $                            LDA, RWORK, RESULT( 4 ) )
*
*
*+    TEST 5
*                 Check solution from generated exact solution.
*
                     CALL ZGET04( N, NRHS, X, LDA, XACT, LDA, RCONDC,
     $                            RESULT( 5 ) )
*
*+    TESTS 6, 7, and 8
*                 Use iterative refinement to improve the solution.
*
                     SRNAMT = 'ZSYRFS'
                     CALL ZSYRFS( UPLO, N, NRHS, A, LDA, AFAC, LDA,
     $                            IWORK, B, LDA, X, LDA, RWORK,
     $                            RWORK( NRHS+1 ), WORK,
     $                            RWORK( 2*NRHS+1 ), INFO )
*
*                 Check error code from ZSYRFS.
*
                     IF( INFO.NE.0 )
     $                  CALL ALAERH( PATH, 'ZSYRFS', INFO, 0, UPLO, N,
     $                               N, -1, -1, NRHS, IMAT, NFAIL,
     $                               NERRS, NOUT )
*
                     CALL ZGET04( N, NRHS, X, LDA, XACT, LDA, RCONDC,
     $                            RESULT( 6 ) )
                     CALL ZPOT05( UPLO, N, NRHS, A, LDA, B, LDA, X, LDA,
     $                            XACT, LDA, RWORK, RWORK( NRHS+1 ),
     $                            RESULT( 7 ) )
*
*                    Print information about the tests that did not pass
*                    the threshold.
*
                     DO 120 K = 3, 8
                        IF( RESULT( K ).GE.THRESH ) THEN
                           IF( NFAIL.EQ.0 .AND. NERRS.EQ.0 )
     $                        CALL ALAHD( NOUT, PATH )
                           WRITE( NOUT, FMT = 9998 )UPLO, N, NRHS,
     $                        IMAT, K, RESULT( K )
                           NFAIL = NFAIL + 1
                        END IF
  120                CONTINUE
                     NRUN = NRUN + 5
  130             CONTINUE
*
*+    TEST 9
*                 Get an estimate of RCOND = 1/CNDNUM.
*
  140             CONTINUE
                  ANORM = ZLANSY( '1', UPLO, N, A, LDA, RWORK )
                  SRNAMT = 'ZSYCON'
                  CALL ZSYCON( UPLO, N, AFAC, LDA, IWORK, ANORM, RCOND,
     $                         WORK, INFO )
*
*                 Check error code from ZSYCON.
*
                  IF( INFO.NE.0 )
     $               CALL ALAERH( PATH, 'ZSYCON', INFO, 0, UPLO, N, N,
     $                            -1, -1, -1, IMAT, NFAIL, NERRS, NOUT )
*
                  RESULT( 9 ) = DGET06( RCOND, RCONDC )
*
*                 Print information about the tests that did not pass
*                 the threshold.
*
                  IF( RESULT( 9 ).GE.THRESH ) THEN
                     IF( NFAIL.EQ.0 .AND. NERRS.EQ.0 )
     $                  CALL ALAHD( NOUT, PATH )
                     WRITE( NOUT, FMT = 9997 )UPLO, N, IMAT, 9,
     $                  RESULT( 9 )
                     NFAIL = NFAIL + 1
                  END IF
                  NRUN = NRUN + 1
  150          CONTINUE
  160       CONTINUE
  170    CONTINUE
  180 CONTINUE
*
*     Print a summary of the results.
*
      CALL ALASUM( PATH, NOUT, NFAIL, NRUN, NERRS )
*
 9999 FORMAT( ' UPLO = ''', A1, ''', N =', I5, ', NB =', I4, ', type ',
     $      I2, ', test ', I2, ', ratio =', G12.5 )
 9998 FORMAT( ' UPLO = ''', A1, ''', N =', I5, ', NRHS=', I3, ', type ',
     $      I2, ', test(', I2, ') =', G12.5 )
 9997 FORMAT( ' UPLO = ''', A1, ''', N =', I5, ',', 10X, ' type ', I2,
     $      ', test(', I2, ') =', G12.5 )
      RETURN
*
*     End of ZCHKSY
*
      END
