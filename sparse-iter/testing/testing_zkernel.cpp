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
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- testing zdot
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

        printf("#====================================================================\n");
        printf("\n");
        printf("        |         runtime           |          GFLOPS\n");
        printf("#n      |  CUBLAS        MERGE      |      CUBLAS       MERGE      \n");
        printf("#---------------------------------------------------------------------\n");
    printf("\n");

    for( magma_int_t n=2000; n<1000001; n+=2000 ){
           

        magma_z_vector a,b,c, x, y, z, skp;
        int iters = 10;
        double flops = (5.* n * iters); 

        
        magmaDoubleComplex one = MAGMA_Z_ONE;
        magmaDoubleComplex alpha = MAGMA_Z_ONE;
        magmaDoubleComplex beta = MAGMA_Z_ONE;

        #define ENABLE_TIMER
        #ifdef ENABLE_TIMER
        double merge1, merge2, cublas1, cublas2;
        double merge_time, cublas_time;
        #endif


        magma_z_vinit( &a, Magma_DEV, n, one );
        magma_z_vinit( &b, Magma_DEV, n, one );
        magma_z_vinit( &c, Magma_DEV, n, one );
        magma_z_vinit( &x, Magma_DEV, n, one );
        magma_z_vinit( &y, Magma_DEV, n, one );
        magma_z_vinit( &z, Magma_DEV, n, one );
        magma_z_vinit( &skp, Magma_DEV, 8, one );

        // warm up
            magma_zscal( n, beta, a.val, 1 );                                 // p = beta*p

        // CUBLAS
        #ifdef ENABLE_TIMER
        magma_device_sync(); cublas1=magma_wtime();
        #endif
        for( int h=0; h<iters; h++){
            magma_zscal( n, beta, a.val, 1 );                                 // p = beta*p
            magma_zaxpy( n, alpha, b.val, 1 , a.val, 1 );        // p = p-omega*beta*v
            magma_zaxpy( n, one, c.val, 1, a.val, 1 );                        // p = p+r
        }
        #ifdef ENABLE_TIMER
        magma_device_sync(); cublas2=magma_wtime();
        cublas_time=cublas2-cublas1;
        #endif
        

        // MERGE
        #ifdef ENABLE_TIMER
        magma_device_sync(); merge1=magma_wtime();
        #endif
        for( int h=0; h<iters; h++){
            magma_zbicgmerge1( n, skp.val, x.val, y.val, z.val );            // merge: p=r+beta*(p-omega*v) 
        }
        #ifdef ENABLE_TIMER
        magma_device_sync(); merge2=magma_wtime();
        merge_time=merge2-merge1;
        #endif
       

        //Chronometry  
        #ifdef ENABLE_TIMER
        printf("%d  %e  %e  %e  %e\n", 
                n,
                cublas_time/iters, 
                (merge_time)/iters, 
                flops/(cublas_time*(1.e+09)), 
                flops/(merge_time*(1.e+09)));
        #endif

        magma_z_vfree(&a);
        magma_z_vfree(&b);
        magma_z_vfree(&c);
        magma_z_vfree(&x);
        magma_z_vfree(&y);
        magma_z_vfree(&z);
        magma_z_vfree(&skp);
    }
    printf("#====================================================================\n");
    printf("\n");
    printf("\n");

    TESTING_FINALIZE();
    return 0;
}
