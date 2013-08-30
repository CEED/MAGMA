#include "magma.h"

#ifndef MAGMA_FORTRAN_NAME
#if defined(ADD_)
#define MAGMA_FORTRAN_NAME(lcname, UCNAME)  magmaf_##lcname##_
#elif defined(NOCHANGE)
#define MAGMA_FORTRAN_NAME(lcname, UCNAME)  magmaf_##lcname
#elif defined(UPCASE)
#define MAGMA_FORTRAN_NAME(lcname, UCNAME)  MAGMAF_##UCNAME
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
*/

#define magmaf_init MAGMA_FORTRAN_NAME( init, INIT )
void magmaf_init( void )
{
    magma_init();
}

#define magmaf_finalize MAGMA_FORTRAN_NAME( finalize, FINALIZE )
void magmaf_finalize( void )
{
    magma_finalize();
}

#ifdef __cplusplus
}
#endif
