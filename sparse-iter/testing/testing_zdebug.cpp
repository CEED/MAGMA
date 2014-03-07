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
#include <omp.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "../include/magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"



/* ////////////////////////////////////////////////////////////////////////////
   -- running magma_zcg magma_zcg_merge 
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    for(int num_vecs=5; num_vecs<6; num_vecs++){
    int n=50;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);

    magma_z_vector x1_h, x2_h, x1_d, x2_d;

    magma_z_vinit( &x1_h, Magma_CPU, n*num_vecs, zero );

    for(int i=0; i<num_vecs; i++)
        x1_h.val[i*n+i] = MAGMA_Z_MAKE(double(i+1), 0.0);
    for(int i=0; i<num_vecs; i++)
        x1_h.val[i*n] = MAGMA_Z_MAKE(double(i+1), 0.0);

    //magma_z_vvisu( x1_h, 0, 20 );

    magma_z_vtransfer( x1_h, &x1_d, Magma_CPU, Magma_DEV );

    magma_zprint_gpu( n, num_vecs, x1_d.val, n );

        real_Double_t  gpu_time;
        real_Double_t FLOPS = 2.0*num_vecs*n/1e9;
        gpu_time = magma_wtime();
    //magma_zlobpcg_shift( n, num_vecs, 2, x1_d.val );

    magma_zorthomgs(n, num_vecs, x1_d.val );
        gpu_time = magma_wtime() - gpu_time;
        printf( "blocksize: %d   GFLOP/s:  %.2e\n",num_vecs, FLOPS/gpu_time );//GFLOPS

    magma_zprint_gpu( n, num_vecs, x1_d.val, n );


    magma_z_vfree(&x1_h);
    magma_z_vfree(&x1_d);
    }



    TESTING_FINALIZE();
    return 0;
}
