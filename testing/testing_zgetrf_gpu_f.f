!
!   -- MAGMA (version 1.0) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      November 2010
!
!  @precisions normal z -> c d s
!

      program test_zgetrf

      double precision rnumber(2), matnorm, work
      complex*16, allocatable :: h_A(:), h_R(:), h_work(:)
      complex*16, device, allocatable :: d_A(:)
      integer, allocatable :: ipiv(:)

      integer i, n, nb, info, lda, lwork

!      call cublas_init( )

      n = 2048
      nb = 128
      lda = n

!------ Allocate CPU memory
      allocate(h_A(n*n), h_R(n*n), ipiv(n))

!------ Allocate GPU memory
      allocate(d_A(n*n))

!------ Initializa the matrix
      do i=1,n*n
        call random_number(rnumber)
        h_A(i) = rnumber(1)
        h_R(i) = rnumber(1)
      end do

!      d_A = h_A

!------ Call magma -------------------
!      call magma_zgetrf_gpu(N, N, d_A, lda, ipiv, info)
      write(*,*) 'exit magma LU'

!------ Call LAPACK ------------------
      call zgetrf(N, N, h_A, lda, ipiv, info)
      write(*,*) 'exit lapack LU'
!------ Compare the two results ------

!      matnorm = zlange('f', n, n, h_A, n, work)
!      call zaxpy(n*n, -1., h_A, 1, h_R, 1)

!      h_R = d_A
      do i=1,n*n
         h_R(i) = h_R(i) - h_A(i)
      end do

      matnorm = 0
      do i=1,n*n
         if (matnorm < abs(h_R(i))) then
            matnorm = h_R(i)
         end if
      end do

!     write(*,105) 'error = ', zlange('f',n,n,h_R,n,work)/matnorm
!     write(*,105) 'error = ', zlange('f',n,n,h_R,n,work)
      write(*,105) 'error = ', matnorm
 105  format((a10,es10.3))

!      call cublas_shutdown()
      end
