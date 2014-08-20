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
   -- running magma_zbicgstab magma_zbicgstab_merge magma_zbicgstab_merge2 
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_zopts zopts;
/*

    zopts.solver_par.epsilon = 10e-16;
    zopts.solver_par.maxiter = 1000;
    zopts.solver_par.verbose = 0;
    zopts.solver_par.num_eigenvalues = 0;
    zopts.output_format = Magma_CSR;
    zopts.scaling = Magma_NOSCALE;



    


    B.storage_type = Magma_CSR;
*/
    int i=1;
    magma_zparse_opts( argc, argv, &zopts, &i);


    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magma_z_sparse_matrix A, B, B_d;
    magma_z_vector x, b;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    magma_zsolverinfo_init( &zopts.solver_par, &zopts.precond_par );

    while(  i < argc ){

        magma_z_csr_mtx( &A,  argv[i]  ); 

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale matrix
        magma_zmscale( &A, zopts.scaling );

        magma_z_mconvert( A, &B, Magma_CSR, zopts.output_format );
        magma_z_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );

        // vectors and initial guess
        magma_z_vinit( &b, Magma_DEV, A.num_cols, one );
        magma_z_vinit( &x, Magma_DEV, A.num_cols, one );
        magma_z_spmv( one, B_d, x, zero, b );                 //  b = A x
        magma_z_vfree(&x);
        magma_z_vinit( &x, Magma_DEV, A.num_cols, zero );

        if( zopts.solver_par.version == 0 )  // standard
            magma_zbicgstab( B_d, b, &x, &zopts.solver_par );
        else if ( zopts.solver_par.version == 1 )    // merged with SpMV isolated
            magma_zbicgstab_merge( B_d, b, &x, &zopts.solver_par );
        else if ( zopts.solver_par.version == 2 ) // merged SpMV (works only for CSR)
            magma_zbicgstab_merge2( B_d, b, &x, &zopts.solver_par );

        magma_zsolverinfo( &zopts.solver_par, &zopts.precond_par );

        magma_z_mfree(&B_d);
        magma_z_mfree(&B);
        magma_z_mfree(&A); 
        magma_z_vfree(&x);
        magma_z_vfree(&b);

        i++;
    }

    magma_zsolverinfo_free( &zopts.solver_par, &zopts.precond_par );

    TESTING_FINALIZE();
    return 0;
}
