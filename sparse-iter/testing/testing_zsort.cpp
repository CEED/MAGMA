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
#include <time.h>


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

    magma_int_t i=1, n=100, stat_cpu = 0;
    magma_index_t *x;

    stat_cpu += magma_index_malloc_cpu( &x, n );
    if( stat_cpu != 0 ){
        magma_free_cpu( x );
        return MAGMA_ERR_HOST_ALLOC;
    } 
    printf("unsorted:\n");
    srand(time(NULL));  
    for(magma_int_t i = 0; i<n; i++ ){
        int r = rand()%100;
        x[i] = r;
        printf("%d  ", x[i]);
    }
    printf("\n\n"); 
    
    printf("sorting...");
    magma_zindexsort(x, 0, n-1, queue );
    printf("done.\n\n");
    
    printf("sorted:\n");
    for(magma_int_t i = 0; i<n; i++ ){
        printf("%d  ", x[i]);
    }
    printf("\n\n"); 

    magma_free_cpu( x );
    
    while(  i < argc ) {
        magma_z_matrix A;

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_zm_5stencil(  laplace_size, &A, queue );
        } else {                        // file-matrix test
            magma_z_csr_mtx( &A,  argv[i], queue );
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) A.num_rows,(int) A.num_cols,(int) A.nnz );
    
        stat_cpu += magma_index_malloc_cpu( &x, A.num_rows*10 );
        if( stat_cpu != 0 ){
            magma_free_cpu( x );
            return MAGMA_ERR_HOST_ALLOC;
        } 
        magma_int_t num_ind = 0;

        magma_zdomainoverlap( A.num_rows, &num_ind, A.row, A.col, x, queue );
                printf("domain overlap indices:\n");
        for(magma_int_t j = 0; j<num_ind; j++ ){
            printf("%d  ", x[j]);
        }
        printf("\n\n"); 
        magma_free_cpu( x );
        magma_zmfree(&A, queue);
        
        i++;
        
    }
    

    /* Shutdown */
    magmablasSetKernelStream( NULL );
    magma_queue_destroy( queue );
    magma_finalize();
    
    return 0;
}
