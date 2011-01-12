!
!   -- MAGMA (version 1.0) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      November 2010
!

      module magma

      implicit none

      integer, parameter :: sizeof_complex_16 = 16
      integer, parameter :: sizeof_complex    =  8
      integer, parameter :: sizeof_double     =  8
      integer, parameter :: sizeof_real       =  4 

!---- Fortran interfaces to MAGMA subroutines ----
      interface

        subroutine magma_zgetrf_gpu(m, n, A, lda, ipiv, info)
           integer   m, n, A(4), lda, ipiv(*), info
        end subroutine
 
        subroutine magma_zgetrs_gpu(trans, n, nrhs, dA, ldda,
     $                              ipiv, dB, lddb, info)
           character trans
           integer n, nrhs, dA(4), ldda, ipiv(*), dB(4), lddb, info
        end subroutine

      end interface
      end module magma
