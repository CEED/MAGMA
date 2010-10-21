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
#include "magma_lapack.h"

typedef int magma_int_t;

#include "magma_z.h"
#include "magma_c.h"
#include "magma_d.h"
#include "magma_s.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- MAGMA function definitions
*/
void magma_xerbla(char *name , magma_int_t *info);

#ifdef __cplusplus
}
#endif

#endif

