/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010
*/

#ifndef _MAGMA_
#define _MAGMA_

#include "auxiliary.h"
#include "magmablas.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on CPU
*/
magma_int_t magma_zpotrf( char *, magma_int_t, double2 *, magma_int_t, double2 *, magma_int_t*);
magma_int_t magma_zpotrf2(char *, magma_int_t, double2 *, magma_int_t,            magma_int_t*);
magma_int_t magma_zpotrf3(char *, magma_int_t, double2 *, magma_int_t, double2 *, magma_int_t*);

magma_int_t magma_zgetrf( magma_int_t, magma_int_t, double2 *, magma_int_t, 
			  magma_int_t *, double2 *, double2 *, magma_int_t*);
magma_int_t magma_zgetrf2(magma_int_t, magma_int_t, double2 *, magma_int_t, 
			  magma_int_t *, double2 *, magma_int_t*);



magma_int_t magma_zlarfb(char, char, magma_int_t, magma_int_t, magma_int_t *, double2 *, magma_int_t *, double2 *,
			 magma_int_t *, double2 *, magma_int_t *, double2 *, magma_int_t *);
magma_int_t magma_zgeqrf(magma_int_t *, magma_int_t *, double2 *, magma_int_t  *,  double2  *,
                 double2 *, magma_int_t *, double2 *, magma_int_t *);


/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions / Data on GPU
*/
magma_int_t magma_zpotrf_gpu(char *, magma_int_t, double2 *, magma_int_t, double2 *, magma_int_t*);

magma_int_t magma_zgetrf_gpu( magma_int_t, magma_int_t, double2 *, magma_int_t, 
			      magma_int_t *, double2 *, magma_int_t *);
magma_int_t magma_zgetrf_gpu2(magma_int_t, magma_int_t, double2 *, magma_int_t, 
			      magma_int_t *, double2 *, double2 *, magma_int_t*);


magma_int_t magma_zgeqrf_gpu(magma_int_t *, magma_int_t *, double2 *, magma_int_t  *, double2  *,
		     double2 *, magma_int_t *, double2 *, magma_int_t *);

#ifdef __cplusplus
}
#endif

#endif

