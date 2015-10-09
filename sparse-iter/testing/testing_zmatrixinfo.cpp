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
#include "common_magmasparse.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_INIT();

    magma_zopts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( &queue );
    
    real_Double_t res;
    magma_z_matrix Z={Magma_CSR}, A={Magma_CSR}, AT={Magma_CSR}, 
    A2={Magma_CSR}, B={Magma_CSR}, B_d={Magma_CSR};
    
    int i=1;
    CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_zm_5stencil(  laplace_size, &Z, queue ));
        } else {                        // file-matrix test
            CHECK( magma_z_csr_mtx( &Z,  argv[i], queue ));
        }
        printf("matrixinfo = [ \n");
        printf("%%   size (n)   ||   nonzeros (nnz)   ||   nnz/n \n");
        printf("%%======================================================="
                        "======%%\n");
        printf("   %4d          %4d          %4d\n",
        (int) Z.num_rows,(int) Z.nnz, (int) Z.nnz/Z.num_rows );
        printf("%%======================================================="
        "======%%\n");
        printf("];\n");
        
        magma_zmfree(&Z, queue );

        i++;
    }

cleanup:
    magma_zmfree(&AT, queue );
    magma_zmfree(&A, queue );
    magma_zmfree(&B, queue );
    magma_zmfree(&B_d, queue );
    magma_zmfree(&A2, queue );
    magma_zmfree(&Z, queue );
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
