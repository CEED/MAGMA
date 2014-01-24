!
!   -- MAGMA (version 1.1) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      @date
!
!  @precisions normal z -> c d s
!

c==================================================================

      subroutine cudaMallocHost(ptr, n)

      integer, intent(in)        :: n
      real*4,pointer,dimension(:):: ptr

      integer :: sizeof_real = 4
      integer*8 off
      real*4, TARGET :: phony(1)

      call cublasMallochost_f( phony(1), n, off)
      ptr => phony(off/sizeof_real+1:)

      end subroutine

c==================================================================

      subroutine cublasAlloc(n, size, ptr)

      integer, intent(in)        :: n
      integer, intent(in)        :: size
      real*4,pointer,dimension(:):: ptr

      integer*8 off
      real*4, TARGET :: phony(1)

      call cublasAlloc_f(n, size, phony(1), off)
      ptr => phony(off+1:)

      end subroutine

c==================================================================

      module magma
      interface magmaAlloc
         subroutine cublasAlloc(n, size, ptr)
           integer, intent(in)        :: n
           integer, intent(in)        :: size
           real*4,pointer,dimension(:):: ptr
         end subroutine

         subroutine cudaMallocHost(ptr, n)
           integer, intent(in)        :: n
           real*4,pointer,dimension(:):: ptr
         end subroutine
      end interface
      end module

c==================================================================

      program testing_sgetrf_f

      use magma

      real*4 rnumber(2), matnorm
      real*4, allocatable,dimension(:) :: h_A(:)
      integer,allocatable,dimension(:) :: ipiv

      real*4, pointer :: h_R(:)

      integer i, n, nb, info, lda
      integer :: sizeof_real = 4

      n = 2048
      nb = 128
      lda = n

!------ Initialize CUBLAS --------------------------
      call cublas_init( )

!------ Allocate CPU memory ------------------------
      allocate(h_A(n*n), ipiv(n))

!------ Allocate CPU pinned memory -----------------
      call cudaMallocHost(h_R, n*n*sizeof_real)

!------ Initialize the matrix ----------------------
      do i=1,n*n
        call random_number(rnumber)
        h_A(i) = rnumber(1)
        h_R(i) = rnumber(1)
      end do

!------ Call magma ---------------------------------
      call magma_sgetrf(N, N, h_R, lda, ipiv, info)

!------ Call LAPACK --------------------------------
      call sgetrf(N, N, h_A, lda, ipiv, info)

!------ Compare the two results --------------------
      do i=1,n*n
         h_R(i) = h_R(i) - h_A(i)
      end do

      matnorm = 0
      do i=1,n*n
         if (matnorm < abs(h_R(i))) then
            matnorm = h_R(i)
         end if
      end do

      write(*,105) 'error = ', matnorm
 105  format((a10,es10.3))

      call cublas_free(h_A)
      deallocate(h_A, ipiv)

      call cublas_shutdown()

      end
