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
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver 
*/
int main(  int argc, char** argv )
{
    /* Initialize */
    TESTING_INIT();
        magma_queue_t queue;
            magma_queue_create( /*devices[ opts->device ],*/ &queue );

    magma_int_t j, n=1000000, FLOPS;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0 );
    magmaDoubleComplex two = MAGMA_Z_MAKE( 2.0, 0.0 );

    magma_z_vector a, ad, bd, cd;
    magma_zvinit( &a, Magma_CPU, n, one, queue );
    magma_zvinit( &bd, Magma_DEV, n, two, queue );
    magma_zvinit( &cd, Magma_DEV, n, one, queue );
    
    magma_zvtransfer( a, &ad, Magma_CPU, Magma_DEV, queue ); 

    real_Double_t start, end, res;
    
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j<100; j++)
        res = magma_dznrm2(n, ad.dval, 1); 
    end = magma_sync_wtime( queue );
    printf( " > MAGMA nrm2: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/100, FLOPS*100/1e9/(end-start) );   
    FLOPS = n;
    start = magma_sync_wtime( queue );
    for (j=0; j<100; j++)
        magma_zscal( n, two, ad.dval, 1 );   
    end = magma_sync_wtime( queue );
    printf( " > MAGMA scal: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/100, FLOPS*100/1e9/(end-start) );   
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j<100; j++)
        magma_zaxpy( n, one, ad.dval, 1, bd.dval, 1 );
    end = magma_sync_wtime( queue );
    printf( " > MAGMA axpy: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/100, FLOPS*100/1e9/(end-start) );   
    FLOPS = n;
    start = magma_sync_wtime( queue );
    for (j=0; j<100; j++)
        magma_zcopy( n, bd.dval, 1, ad.dval, 1 );
    end = magma_sync_wtime( queue );
    printf( " > MAGMA copy: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/100, FLOPS*100/1e9/(end-start) );
    FLOPS = 2*n;
    start = magma_sync_wtime( queue );
    for (j=0; j<100; j++) 
        res = MAGMA_Z_REAL( magma_zdotc(n, ad.dval, 1, bd.dval, 1) );
    end = magma_sync_wtime( queue );
    printf( " > MAGMA dotc: %.2e seconds %.2e GFLOP/s\n",
                                    (end-start)/100, FLOPS*100/1e9/(end-start) );   

    printf("# tester BLAS:  ok\n");


    magma_z_vfree( &a, queue);
    magma_z_vfree(&ad, queue);
    magma_z_vfree(&bd, queue);
    magma_z_vfree(&cd, queue);

    /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
    
    return 0;
}
