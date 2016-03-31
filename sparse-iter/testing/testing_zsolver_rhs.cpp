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
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"
#include "magmasparse_internal.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_INIT();

    magma_zopts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    //Chronometry
    real_Double_t tempo1, tempo2, t_transfer = 0.0;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magma_z_matrix A={Magma_CSR}, B={Magma_CSR}, B_d={Magma_CSR};
    magma_z_matrix x={Magma_CSR}, x_h={Magma_CSR}, b_h={Magma_DENSE}, b={Magma_DENSE};
    
    int i=1;
    CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));
    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    // make sure preconditioner is NONE for unpreconditioned systems
    if ( zopts.solver_par.solver != Magma_PCG &&
         zopts.solver_par.solver != Magma_PCGMERGE &&
         zopts.solver_par.solver != Magma_PGMRES &&
         zopts.solver_par.solver != Magma_PBICGSTAB &&
         zopts.solver_par.solver != Magma_ITERREF  &&
         zopts.solver_par.solver != Magma_PIDR  &&
         zopts.solver_par.solver != Magma_PCGS  &&
         zopts.solver_par.solver != Magma_PCGSMERGE &&
         zopts.solver_par.solver != Magma_PTFQMR &&
         zopts.solver_par.solver != Magma_PTFQMRMERGE &&
         zopts.solver_par.solver != Magma_LOBPCG ){
                    zopts.precond_par.solver = Magma_NONE;
         }
    CHECK( magma_zsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue ));
    // more iterations

    
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_zm_5stencil(  laplace_size, &A, queue ));
            CHECK( magma_zvinit( &b_h, Magma_CPU, A.num_cols, 1, one, queue ));
        } else {                        // file-matrix test
            CHECK( magma_z_csr_mtx( &A,  argv[i], queue ));
            CHECK( magma_zvread( &b_h, A.num_cols, argv[i+1], queue ));
            i++;
        }

        printf( "\n%% matrix info: %d-by-%d with %d nonzeros\n\n",
                            int(A.num_rows), int(A.num_cols), int(A.nnz) );
        
        printf("matrixinfo = [ \n");
        printf("%%   size   (m x n)     ||   nonzeros (nnz)   ||   nnz/m \n");
        printf("%%======================================================="
                            "======%%\n");
        printf("  %8d  %8d      %10d        %10d\n",
            int(A.num_rows), int(A.num_cols), int(A.nnz), int(A.nnz/A.num_rows) );
        printf("%%======================================================="
        "======%%\n");
        printf("];\n");
        // for the eigensolver case
        zopts.solver_par.ev_length = A.num_cols;
        CHECK( magma_zeigensolverinfo_init( &zopts.solver_par, queue ));
        fflush(stdout);
        

        t_transfer = 0.0;
        zopts.precond_par.setuptime = 0.0;
        zopts.precond_par.runtime = 0.0;
        //CHECK( magma_zvinit( &b_h, Magma_CPU, A.num_cols, 1, MAGMA_Z_ONE, queue ));

        i++;
        tempo1 = magma_sync_wtime( queue );
        magma_z_vtransfer(b_h, &b, Magma_CPU, Magma_DEV, queue);
        tempo2 = magma_sync_wtime( queue );
        t_transfer += tempo2-tempo1;
        
        // scale matrix
        CHECK( magma_zmscale( &A, zopts.scaling, queue ));
        
        // preconditioner
        if ( zopts.solver_par.solver != Magma_ITERREF ) {
            CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        }
        // make sure alignment is 1 for SELLP
        B.alignment = 1;
        B.blocksize = 256;
        CHECK( magma_zmconvert( A, &B, Magma_CSR, zopts.output_format, queue ));
        tempo1 = magma_sync_wtime( queue );
        CHECK( magma_zmtransfer( B, &B_d, Magma_CPU, Magma_DEV, queue ));
        tempo2 = magma_sync_wtime( queue );
        t_transfer += tempo2-tempo1;
        
        CHECK( magma_zvinit( &x, Magma_DEV, A.num_cols, 1, zero, queue ));
        
        info = magma_z_solver( B_d, b, &x, &zopts, queue );
        if( info != 0 ) {
            printf("%%error: solver returned: %s (%d).\n",
                magma_strerror( info ), int(info) );
        }
        
        magma_zmfree(&x_h, queue );
        tempo1 = magma_sync_wtime( queue );
        magma_z_vtransfer(x, &x_h, Magma_DEV, Magma_CPU, queue);
        tempo2 = magma_sync_wtime( queue );
        t_transfer += tempo2-tempo1;  
        
        printf("data = [\n");
        magma_zsolverinfo( &zopts.solver_par, &zopts.precond_par, queue );
        printf("];\n\n");
        
        printf("precond_info = [\n");
        printf("%%   setup  runtime\n");        
        printf("  %.6f  %.6f\n",
           zopts.precond_par.setuptime, zopts.precond_par.runtime );
        printf("];\n\n");
        
        printf("transfer_time = %.6f;\n\n", t_transfer);
        magma_zmfree(&x, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&B_d, queue );
        magma_zmfree(&B, queue );
        magma_zsolverinfo_free( &zopts.solver_par, &zopts.precond_par, queue );
        fflush(stdout);
        
        magma_zmfree(&B_d, queue );
        magma_zmfree(&B, queue );
        magma_zmfree(&A, queue );
        magma_zmfree(&x, queue );
        magma_zmfree(&x_h, queue );
        magma_zmfree(&b, queue );
        i++;
    }

cleanup:
    magma_zmfree(&B_d, queue );
    magma_zmfree(&B, queue );
    magma_zmfree(&A, queue );
    magma_zmfree(&x, queue );
    magma_zmfree(&x_h, queue );
    magma_zmfree(&b, queue );
    magma_zsolverinfo_free( &zopts.solver_par, &zopts.precond_par, queue );
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
