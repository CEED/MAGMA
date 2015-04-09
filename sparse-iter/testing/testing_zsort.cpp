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
    magma_queue_create( &queue );
    magmablasSetKernelStream( queue );

    magma_int_t n=100, stat_cpu = 0;

    stat_cpu += magma_index_malloc_cpu( &x, n );

    srand(time(NULL));  
    for(magma_int_t i = 0; i<n; i++ ){
        int r = rand();
        x[i] = r;
        printf("%d  ", x[i]);
    }

    magma_zsort(x, n, 0, n );

    for(magma_int_t i = 0; i<n; i++ ){
        printf("%d  ", x[i]);
    }

    magma_free_cpu( x );

    /* Shutdown */
    magmablasSetKernelStream( NULL );
    magma_queue_destroy( queue );
    magma_finalize();
    
    return 0;
}
