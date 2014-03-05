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

    int num_vecs=20;
    int n=1000000;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);

    magma_z_vector x1_h, x2_h, x1_d, x2_d;

    magma_z_vinit( &x1_h, Magma_CPU, n*num_vecs, one );

    for(int i=0; i<num_vecs*n; i++)
        x1_h.val[i] = MAGMA_Z_MAKE(double(i), 0.0);

    magma_z_vvisu( x1_h, 0, 20 );

    magma_z_vtransfer( x1_h, &x1_d, Magma_CPU, Magma_DEV );

    magma_z_vvisu( x1_d, 0, 20 );

    magma_zlobpcg_shift( n, num_vecs, 2, x1_d.val );

    magma_z_vvisu( x1_d, 0, 20 );

    magma_z_vfree(&x1_h);
    magma_z_vfree(&x1_d);



    TESTING_FINALIZE();
    return 0;
}
