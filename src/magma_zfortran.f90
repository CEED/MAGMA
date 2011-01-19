!
!   -- MAGMA (version 1.0) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      November 2010
!
!   @precisions normal z -> c d s
!

module magma_zfortran
  
  implicit none
  
  !---- Fortran interfaces to MAGMA subroutines ----
  interface
     
     subroutine magma_zgetrf_gpu(m, n, A, lda, ipiv, info)
       integer   m, n, lda, ipiv(*), info
       integer(kind=16) A
     end subroutine magma_zgetrf_gpu
     
      subroutine magma_zgetrs_gpu(trans, n, nrhs, dA, ldda, ipiv, dB, lddb, info)
        character trans
        integer n, nrhs, ldda, ipiv(*), lddb, info
        integer(kind=16) dA, dB
      end subroutine magma_zgetrs_gpu
     
   end interface
end module magma_zfortran
