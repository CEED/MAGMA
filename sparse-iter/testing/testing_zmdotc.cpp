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
   -- testing zdot
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    // set queue for old dense routines
    magma_queue_t queue=NULL;
    magma_queue_create( &queue );
    magmablasGetKernelStream( &queue );

    const magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
    const magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex alpha;

    TESTING_INIT();

    magma_z_matrix a={Magma_CSR}, b={Magma_CSR}, x={Magma_CSR}, y={Magma_CSR}, skp={Magma_CSR};

    printf("%%================================================================================================================================================\n");
    printf("\n");
    printf("%%            |     runtime            |       GFLOPS\n");
    printf("%% n num_vecs |  CUDOT    MAGMA MDOTC  |  CUDOT    MAGMA MDOTC\n");
    printf("%%------------------------------------------------------------------------------------------------------------------------------------------------\n");
    printf("\n");

    for( magma_int_t num_vecs=2; num_vecs < 9; num_vecs += 2 ) {
        for( magma_int_t n=1000000; n < 5000001; n += 1000000 ) {
            int iters = 10;
            double computations = ( n * iters * num_vecs);

            #define ENABLE_TIMER
            #ifdef ENABLE_TIMER
            real_Double_t mdot1, mdot2, cudot1, cudot2;
            real_Double_t mdot_time, cudot_time;
            #endif

            CHECK( magma_zvinit( &a, Magma_DEV, n, num_vecs, one, queue ));
            CHECK( magma_zvinit( &b, Magma_DEV, num_vecs, 1, one, queue ));
            int min_ten = min(num_vecs, 15);
            CHECK( magma_zvinit( &x, Magma_DEV, min_ten, n, one, queue ));
            CHECK( magma_zvinit( &y, Magma_DEV, min_ten, n, one, queue ));
            CHECK( magma_zvinit( &skp, Magma_DEV, num_vecs, 1, zero, queue ));

            // warm up
            CHECK( magma_zgemvmdot( n, num_vecs, a.dval, b.dval, x.dval, y.dval, skp.dval, queue ));

            // CUDOT
            #ifdef ENABLE_TIMER
            cudot1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h<iters; h++) {
                for( int l=0; l<num_vecs/2; l++)
                    alpha = magma_zdotc(n, a.dval, 1, b.dval, 1);
            }
            #ifdef ENABLE_TIMER
            cudot2 = magma_sync_wtime( queue );
            cudot_time=cudot2-cudot1;
            #endif
          
            // MAGMA MDOTC
            #ifdef ENABLE_TIMER
            mdot1 = magma_sync_wtime( queue );
            #endif
            for( int h=0; h<iters; h++) {
                if( num_vecs == 2 ){
                    magma_zmdotc1( n, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                }
                else if( num_vecs == 4 ){
                    magma_zmdotc1( n, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                    magma_zmdotc1( n, a.dval, b.dval, x.dval, y.dval, skp.dval, queue );
                }
                else if( num_vecs == 6 ){
                    magma_zmdotc3( n, a.dval, b.dval, a.dval+n, b.dval+n, a.dval+2*n, b.dval+2*n, x.dval, y.dval, skp.dval, queue );
                }
                else if( num_vecs == 8 ){
                    magma_zmdotc3( n, a.dval, b.dval, a.dval+n, b.dval+n, a.dval+2*n, b.dval+2*n, x.dval, y.dval, skp.dval, queue );
                }
                else{
                    printf("error: not supported.\n");
                }
            }
            #ifdef ENABLE_TIMER
            mdot2 = magma_sync_wtime( queue );
            mdot_time=mdot2-mdot1;
            #endif
           
            //Chronometry
            #ifdef ENABLE_TIMER
            printf("%d  %d  %e  %e  %e  %e\n",
                    int(n), int(num_vecs),
                    cudot_time/iters,
                    (mdot_time)/iters,
                    computations/(cudot_time*1e9),
                    computations/(mdot_time*1e9));
            #endif

            magma_zmfree(&a, queue );
            magma_zmfree(&b, queue );
            magma_zmfree(&x, queue );
            magma_zmfree(&y, queue );
            magma_zmfree(&skp, queue );
        }

        printf("%%================================================================================================================================================\n");
        printf("\n");
        printf("\n");
    }
    
    // use alpha to silence compiler warnings
    if ( isnan( real( alpha ))) {
        info = -1;
    }

cleanup:
    magma_zmfree(&a, queue );
    magma_zmfree(&b, queue );
    magma_zmfree(&x, queue );
    magma_zmfree(&y, queue );
    magma_zmfree(&skp, queue );
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
