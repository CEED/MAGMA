/*
 *   -- MAGMA (version 0.1) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      November 2011
 *
 * @author Mark Gates
 */

#include <stdlib.h>
#include <stdio.h>

#include "magma.h"

#ifdef HAVE_CUBLAS

// ========================================
// initialization
extern "C"
magma_err_t magma_init()
{
    return MAGMA_SUCCESS;
}

// --------------------
extern "C"
magma_err_t magma_finalize()
{
    return MAGMA_SUCCESS;
}

#endif // HAVE_CUBLAS
