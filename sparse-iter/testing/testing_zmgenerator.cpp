/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

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
#include "../include/magmasparse.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Debugging file
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_z_sparse_matrix A, B;

    magma_int_t n=10;
    magma_int_t nn = n*n;
    magma_int_t offdiags = 1;
    magma_int_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_imalloc_cpu( &diag_offset, offdiags+1 );
    
    diag_offset[0] = 0;
    diag_offset[1] = 1;
    diag_offset[2] = n;
    diag_vals[0] = MAGMA_Z_MAKE( 4.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -1.0, 0.0 );
    //    diag_vals[2] = MAGMA_Z_MAKE( -1.0, 0.0 );
    
    magma_zmgenerator( nn, offdiags, diag_offset, diag_vals, &A );

    magma_z_mvisu( A );

    printf("nnz:%d\n", A.nnz);

    magma_z_mtransfer( A, &B, Magma_CPU, Magma_DEV );

    magma_z_mvisu( B );

    TESTING_FINALIZE();
    return 0;
}
