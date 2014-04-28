/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

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
   -- running magma_zlobpcg
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_z_solver_par solver_par;
    solver_par.epsilon = 1e-5;
    solver_par.maxiter = 1000;
    solver_par.verbose = 0;
    solver_par.num_eigenvalues = 32;
    magma_z_preconditioner precond_par;
    precond_par.solver = Magma_JACOBI;
    int precond = 0;
    int format = 0;
    int version = 0;
    
    magma_z_sparse_matrix A, B, dA;
    magma_z_vector x, b;
    B.blocksize = 8;
    B.alignment = 8;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

    B.storage_type = Magma_CSR;
    int i;
    for( i = 1; i < argc; ++i ) {
     if ( strcmp("--format", argv[i]) == 0 ) {
            format = atoi( argv[++i] );
            switch( format ) {
                case 0: B.storage_type = Magma_CSR; break;
                case 1: B.storage_type = Magma_ELL; break;
                case 2: B.storage_type = Magma_ELLRT; break;
                case 3: B.storage_type = Magma_SELLP; break;
            }
        }else if ( strcmp("--precond", argv[i]) == 0 ) {
            format = atoi( argv[++i] );
            switch( precond ) {
                case 0: precond_par.solver = Magma_JACOBI; break;
            }

        }else if ( strcmp("--version", argv[i]) == 0 ) {
            version = atoi( argv[++i] );
        }else if ( strcmp("--blocksize", argv[i]) == 0 ) {
            B.blocksize = atoi( argv[++i] );
        }else if ( strcmp("--alignment", argv[i]) == 0 ) {
            B.alignment = atoi( argv[++i] );
        }else if ( strcmp("--verbose", argv[i]) == 0 ) {
            solver_par.verbose = atoi( argv[++i] );
        }  else if ( strcmp("--maxiter", argv[i]) == 0 ) {
            solver_par.maxiter = atoi( argv[++i] );
        } else if ( strcmp("--tol", argv[i]) == 0 ) {
            sscanf( argv[++i], "%lf", &solver_par.epsilon );
        } else if ( strcmp("--eigenvalues", argv[i]) == 0 ) {
            solver_par.num_eigenvalues = atoi( argv[++i] );
        } else
            break;
    }
    printf( "\n    usage: ./run_zlobpcg"
        " [ --format %d (0=CSR, 1=ELL, 2=ELLRT, 4=SELLP)"
        " [ --blocksize %d --alignment %d ]"
        " --verbose %d (0=summary, k=details every k iterations)"
        " --maxiter %d --tol %.2e"
        " --preconditioner %d (0=Jacobi) "
        " --eigenvalues %d ]"
        " matrices \n\n", format, B.blocksize, B.alignment,
        solver_par.verbose,
        solver_par.maxiter, solver_par.epsilon, precond,  
        solver_par.num_eigenvalues);

    while(  i < argc ){

        magma_z_csr_mtx( &A,  argv[i]  ); 

        printf( "\nmatrix info: %d-by-%d with %d nonzeros\n\n"
                                    ,A.num_rows,A.num_cols,A.nnz );
        solver_par.ev_length = A.num_cols;

        magma_z_sparse_matrix A2;
        A2.storage_type = Magma_SELLC;
        A2.blocksize = 8;
        A2.alignment = 4;
        magma_z_mconvert( A, &A2, Magma_CSR, A2.storage_type );

        // copy matrix to GPU                                                     
        magma_z_mtransfer( A2, &dA, Magma_CPU, Magma_DEV);

        magma_zsolverinfo_init( &solver_par, &precond_par ); // inside the loop!
                           // as the matrix size has influence on the EV-length



        real_Double_t  gpu_time;

        // Find the blockSize smallest eigenvalues and corresponding eigen-vectors
        gpu_time = magma_wtime();
        magma_zlobpcg( dA, &solver_par );
        gpu_time = magma_wtime() - gpu_time;

        printf("Time (sec) = %7.2f\n", gpu_time);
        printf("solver runtime (sec) = %7.2f\n", solver_par.runtime );



        magma_zsolverinfo_free( &solver_par, &precond_par );

        magma_z_mfree(     &dA    );
        magma_z_mfree(     &A2    );
        magma_z_mfree(     &A    );

        i++;
    }

    TESTING_FINALIZE();
    return 0;
}
