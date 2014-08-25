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
   -- testing any solver 
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_zopts zopts;

    int i=1;
    magma_zparse_opts( argc, argv, &zopts, &i);


    real_Double_t res;
    magma_z_sparse_matrix A, B, B_d, A2;

    B.blocksize = zopts.blocksize;
    B.alignment = zopts.alignment;

    while(  i < argc ){

        magma_z_csr_mtx( &A,  argv[i]  ); 

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );

        // scale matrix
        magma_zmscale( &A, zopts.scaling );

        // convert, copy back and forth to check everything works
        magma_z_mconvert( A, &B, Magma_CSR, zopts.output_format );
        magma_z_mtransfer( B, &B_d, Magma_CPU, Magma_DEV );
        magma_z_mfree(&B);
        magma_z_mtransfer( B_d, &B, Magma_DEV, Magma_CPU );
        magma_z_mfree(&B_d);
        magma_z_mconvert( B, &A2, zopts.output_format,Magma_CSR );      
        magma_z_mfree(&B);

        magma_zmdiff( A, A2, &res);
        printf(" ||A-B||_F = %f\n", res);

        magma_z_mfree(&A); 
        magma_z_mfree(&A2); 

        i++;
    }

    TESTING_FINALIZE();
    return 0;
}
