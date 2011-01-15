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
  integer, parameter :: sizeof_complex    = 8
  integer, parameter :: sizeof_double     = 8
  integer, parameter :: sizeof_real       = 4
  
  !---- Fortran interfaces to MAGMA subroutines ----
  interface
     
     subroutine magma_sgetrf_gpu(m, n, A, lda, ipiv, info)
       integer   m, n, lda, ipiv(*), info
       integer(kind=16) A
     end subroutine magma_sgetrf_gpu
     
     subroutine magma_sgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)
       character trans
       integer n, nrhs, ldda, ipiv(*), lddb, info
       integer(kind=16) dA, dB
     end subroutine magma_sgetrs_gpu
     
     subroutine magma_dgetrf_gpu(m, n, A, lda, ipiv, info)
       integer   m, n, lda, ipiv(*), info
       integer(kind=16) A
     end subroutine magma_dgetrf_gpu
     
     subroutine magma_dgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)
       character trans
       integer n, nrhs, ldda, ipiv(*), lddb, info
       integer(kind=16) dA, dB
     end subroutine magma_dgetrs_gpu
     
     subroutine magma_cgetrf_gpu(m, n, A, lda, ipiv, info)
       integer   m, n, lda, ipiv(*), info
       integer(kind=16) A
     end subroutine magma_cgetrf_gpu
     
     subroutine magma_cgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)
       character trans
       integer n, nrhs, ldda, ipiv(*), lddb, info
       integer(kind=16) dA, dB
     end subroutine magma_cgetrs_gpu
     
     subroutine magma_zgetrf_gpu(m, n, A, lda, ipiv, info)
       integer   m, n, lda, ipiv(*), info
       integer(kind=16) A
     end subroutine magma_zgetrf_gpu
     
      subroutine magma_zgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)
        character trans
        integer n, nrhs, ldda, ipiv(*), lddb, info
        integer(kind=16) dA, dB
      end subroutine magma_zgetrs_gpu
     
  End interface
end module magma
