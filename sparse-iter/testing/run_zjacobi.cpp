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
#include "../include/magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- running magma_zjacobi
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_solver_parameters solver_par;
    solver_par.maxiter = 1000;
    solver_par.verbose;
    int format = 0;

    magma_z_sparse_matrix A, B, B_d;
    magma_z_vector x, b;
    B.blocksize = 8;
    B.alignment = 8;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

    B.storage_type = Magma_CSR;
    char filename[256]; 

    for( int i = 1; i < argc; ++i ) {
     if ( strcmp("--format", argv[i]) == 0 ) {
            format = atoi( argv[++i] );
            switch( format ) {
                case 0: B.storage_type = Magma_CSR; break;
                case 1: B.storage_type = Magma_ELLPACK; break;
                case 2: B.storage_type = Magma_ELLPACKT; break;
                case 3: B.storage_type = Magma_ELLPACKRT; break;
                case 4: B.storage_type = Magma_SELLC; break;
            }
        } else if ( strcmp("--maxiter", argv[i]) == 0 ){ // && i+1< argc ) {
            solver_par.maxiter = atoi( argv[++i] );
        } else if ( strcmp("--matrix", argv[i]) == 0 ) {
            strcpy( filename, argv[++i] );
        }
        // else strcpy( filename, argv[++i] );
    }
    printf( "\n    usage: ./run_zjacobi"
        " [ --format %d (0=CSR, 1=ELLPACK, 2=ELLPACKT, 3=ELLPACKRT, 4=SELLC)"
        " [ --blocksize %d --alignment %d ]"
        " --maxiter %d ]"
        " --matrix filename \n\n", format, B.blocksize, B.alignment,
        solver_par.maxiter);

    magma_z_csr_mtx( &A,  filename  ); 

    printf( "\nmatrix info: %d-by-%d with %d nonzeros\n\n"
                                ,A.num_rows,A.num_cols,A.nnz );

    magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

    magma_z_mconvert( A, &B, Magma_CSR, B.storage_type );
    magma_z_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );


    magma_zjacobi( B_d, b, &x, &solver_par );

    magma_zsolverinfo( &solver_par );

    magma_zsolverinfo_free( &solver_par );

    magma_z_mfree(&B_d);
    magma_z_mfree(&B);
    magma_z_mfree(&A); 
    magma_z_vfree(&x);
    magma_z_vfree(&b);

    TESTING_FINALIZE();
    return 0;
}
