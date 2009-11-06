/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/
#include <stdio.h>
double magma_dlamch(char *CMACH){
  /*

  Purpose
  =======

  DLAMCH determines double precision machine parameters.
  Currently it calls the LAPACK routine in CPU 
  it should be rewritten for GPU.

  Arguments
  =========

  CMACH   (input) CHARACTER*1
          Specifies the value to be returned by DLAMCH:
          = 'E' or 'e',   DLAMCH := eps
          = 'S' or 's ,   DLAMCH := sfmin
          = 'B' or 'b',   DLAMCH := base
          = 'P' or 'p',   DLAMCH := eps*base
          = 'N' or 'n',   DLAMCH := t
          = 'R' or 'r',   DLAMCH := rnd
          = 'M' or 'm',   DLAMCH := emin
          = 'U' or 'u',   DLAMCH := rmin
          = 'L' or 'l',   DLAMCH := emax
          = 'O' or 'o',   DLAMCH := rmax

          where

          eps   = relative machine precision
          sfmin = safe minimum, such that 1/sfmin does not overflow
          base  = base of the machine
          prec  = eps*base
          t     = number of (base) digits in the mantissa
          rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise
          emin  = minimum exponent before (gradual) underflow
          rmin  = underflow threshold - base**(emin-1)
          emax  = largest exponent before overflow
          rmax  = overflow threshold  - (base**emax)*(1-eps)
 
  */
  if( *CMACH != 'E' && *CMACH != 'e'){
      printf("Other Functionality isn't provided yet!");
      exit(-1);
  }
  return  dlamch_(CMACH);
}

