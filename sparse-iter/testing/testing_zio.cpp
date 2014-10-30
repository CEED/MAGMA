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
#include <unistd.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver 
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_zopts zopts;

    int i=1;
    magma_zparse_opts( argc, argv, &zopts, &i);


    real_Double_t res;
    magma_z_sparse_matrix A, A2;

    while(  i < argc ){

        if( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ){   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_zm_5stencil(  laplace_size, &A );
        } else {                        // file-matrix test
            magma_z_csr_mtx( &A,  argv[i]  ); 
        }

        printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // filename for temporary matrix storage
        const char *filename = "testmatrix.mtx";

        // write to file
        write_z_csrtomtx( A, filename );

        // read from file
        magma_z_csr_mtx( &A2, filename );

        // delete temporary matrix
        unlink( filename );

        magma_zmdiff( A, A2, &res);
        printf("# ||A-B||_F = %f\n", res);
        if( res < .000001 )
            printf("# tester:  ok\n");
        else
            printf("# tester:  failed\n");

        magma_z_mfree(&A); 
        magma_z_mfree(&A2); 

        i++;
    }

    TESTING_FINALIZE();
    return 0;
}
