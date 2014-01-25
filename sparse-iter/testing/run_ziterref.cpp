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
   -- running magma_ziterref
*/
int main( int argc, char** argv)
{

    TESTING_INIT();

    TESTING_INIT();

    magma_solver_parameters solver_par;
    solver_par.epsilon = 10e-16;
    solver_par.maxiter = 1000;
    solver_par.verbose = 0;
    int version = 0;
    int format = 0;

    magma_precond_parameters precond_par;
    precond_par.solver = Magma_JACOBI;
    precond_par.epsilon = 1e-8;
    precond_par.maxiter = 100;
    precond_par.restart = 30;

    magma_z_sparse_matrix A, B, B_d;
    magma_z_vector x, b;
    
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
            }
        } else if ( strcmp("--maxiter", argv[i]) == 0 ) {
            solver_par.maxiter = atoi( argv[++i] );
        }else if ( strcmp("--verbose", argv[i]) == 0 ) {
            solver_par.verbose = atoi( argv[++i] );
        } else if ( strcmp("--tol", argv[i]) == 0 ) {
            sscanf( argv[++i], "%lf", &solver_par.epsilon );
        } else if ( strcmp("--preconditioner", argv[i]) == 0 ) {
            version = atoi( argv[++i] );
            switch( version ) {
                case 0: precond_par.solver = Magma_JACOBI; break;
                case 1: precond_par.solver = Magma_CG; break;
                case 2: precond_par.solver = Magma_BICGSTAB; break;
                case 3: precond_par.solver = Magma_GMRES; break;
            }
        } else if ( strcmp("--precond-maxiter", argv[i]) == 0 ) {
            precond_par.maxiter = atoi( argv[++i] );
        }else if ( strcmp("--precond-tol", argv[i]) == 0 ) {
            sscanf( argv[++i], "%lf", &precond_par.epsilon );
        }else if ( strcmp("--precond-restart", argv[i]) == 0 ) {
            precond_par.restart = atoi( argv[++i] );
        }else if ( strcmp("--matrix", argv[i]) == 0 ) {
            strcpy( filename, argv[++i] );
        }
    }
    printf( "\n    usage: ./run_ziterref"
            " [ --format %d (0=CSR, 1=ELLPACK, 2=ELLPACKT, 3=ELLPACKRT)"
            " --verbose %d (0=summary, k=details every k iterations)"
            " --maxiter %d --tol %.2e"
            " --preconditioner %d (0=Jacobi, 1=CG, 2=BiCGStab, 3=GMRES)"
            " [ --precond-maxiter %d --precond-tol %.2e"
            " --precond-restart %d ] ]"
            " --matrix filename \n\n", format, solver_par.verbose,
            solver_par.maxiter, solver_par.epsilon, version,
            precond_par.maxiter, precond_par.epsilon, precond_par.restart );

    magma_z_csr_mtx( &A, (const char*) filename  ); 


    printf( "\nmatrix info: %d-by-%d with %d nonzeros\n\n"
                                ,A.num_rows,A.num_cols,A.nnz );

    magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

    magma_z_mconvert( A, &B, Magma_CSR, B.storage_type );
    magma_z_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );

    magma_ziterref( B_d, b, &x, &solver_par, &precond_par );

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
