!
!   -- MAGMA (version 1.0) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      November 2010
!

module magma

  use magma_zf77
  use magma_df77
  use magma_cf77
  use magma_sf77
  
  implicit none

  integer, parameter :: sizeof_complex_16 = 16
  integer, parameter :: sizeof_complex    = 8
  integer, parameter :: sizeof_double     = 8
  integer, parameter :: sizeof_real       = 4
  
end module magma
