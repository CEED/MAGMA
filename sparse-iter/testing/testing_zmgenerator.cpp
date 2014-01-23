/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Debugging file
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_z_sparse_matrix A;

    magma_int_t n = 10;
    magma_int_t offdiags = 2;
    magma_int_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_imalloc_cpu( &diag_offset, offdiags+1 );
    
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = 4;
    diag_vals[0] = MAGMA_Z_MAKE( 4.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -1.0, 0.0 );
    diag_vals[2] = MAGMA_Z_MAKE( -1.0, 0.0 );
    
    magma_zmgenerator( n, offdiags, diag_offset, diag_vals, &A );

    magma_z_mvisu( A );

    TESTING_FINALIZE();
    return 0;
}
