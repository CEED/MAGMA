/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- running magma_zbcsrlu
*/
int main( int argc, char** argv)
{

    TESTING_INIT();

    TESTING_INIT();

    magma_solver_parameters solver_par;
    solver_par.maxiter = 1000;
    solver_par.verbose = 0;
    int format = 0;

    magma_z_sparse_matrix A, B, B_d;
    magma_z_vector x, b;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

    B.storage_type = Magma_CSR;
    char filename[256]; 

    for( int i = 1; i < argc; ++i ) {
      if ( strcmp("--matrix", argv[i]) == 0 ) {
            strcpy( filename, argv[++i] );
        }
    }
    printf( "\n    usage: ./run_zbcsrlu --matrix filename \n\n" );

    magma_z_csr_mtx( &A, (const char*) filename  ); 


    printf( "\nmatrix info: %d-by-%d with %d nonzeros\n\n"
                                ,A.num_rows,A.num_cols,A.nnz );

    magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

    magma_zbcsrlu( A, b, &x, &solver_par );

    magma_zsolverinfo( &solver_par );

    magma_zsolverinfo_free( &solver_par );

    magma_z_mfree(&A); 
    magma_z_vfree(&x);
    magma_z_vfree(&b);

    TESTING_FINALIZE();
    return 0;
}
