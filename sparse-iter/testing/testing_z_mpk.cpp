/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "mkl_spblas.h"

#define PRECISION_z
#if defined(PRECISION_z)
#define MKL_ADDR(a) (MKL_Complex16*)(a)
#elif defined(PRECISION_c)
#define MKL_ADDR(a) (MKL_Complex8*)(a)
#else
#define MKL_ADDR(a) (a)
#endif


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing matrix vector product
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    const char *filename[] =
    {
     "test_matrices/Trefethen_20.mtx",       // 0
     "test_matrices/Trefethen_200.mtx",      // 1
     "test_matrices/Trefethen_2000.mtx",     // 2
     "test_matrices/G3_circuit.mtx",         // 3
     "test_matrices/test3.mtx",               // 4
     "test_matrices/bcsstk01.mtx",           // 5
     "test_matrices/Pres_Poisson.mtx",       // 6
     "test_matrices/bloweybq.mtx",           // 7
     "test_matrices/ecology2.mtx",           // 8
     "test_matrices/apache2.mtx",            // 9
     "test_matrices/crankseg_2.mtx",         // 10
     "test_matrices/bmwcra_1.mtx",           // 11
     "test_matrices/F1.mtx",                 // 12
     "test_matrices/audikw_1.mtx",           // 13
     "test_matrices/boneS10.mtx",            // 14
     "test_matrices/parabolic_fem.mtx",      // 15
     "test_matrices/fv1.mtx",                // 16
    };

    int id = -1, matrix = 0, i, ione = 1, *pntre, num_rblocks, num_cblocks;
    double start, end, work[1];
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--id", argv[i]) == 0 ) {
            id = atoi( argv[++i] );
        }
    }
    if (id > -1) printf( "\n    Usage: ./testing_z_mv --id %d\n\n",id );

    for(matrix=4; matrix<5; matrix++)
    {
        magma_z_sparse_matrix hA, hB, hB1, hB2, hB3, dA, hC;
        magma_z_vector a,b,c;
        magma_int_t num_add_rows, *add_rows;
        magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
  
        // init matrix on CPU
        if (id > -1) matrix = id;
        magma_z_csr_mtx( &hA, filename[matrix] );
        printf( "\nmatrix read: %d-by-%d with %d nonzeros\n\n",hA.num_rows,hA.num_cols,hA.nnz );
        //magma_z_mvisu(hA );
        // possible usage of other formats
        //magma_z_mconvert( hA, &hB, hA.storage_type, Magma_ELLPACK);
        //magma_z_mtransfer( hB, &dA, hB.memory_location, Magma_DEV);


        for(int k=0; k<5; k++){
            printf("\n \n \n -----------------\nk=%d\n",k ); 
            // possibility to print information about additional rows
            //magma_z_mpkinfo( hA, 3, 0, k, &num_add_rows, add_rows );
            magma_z_mpksetup( hA, &hB1, 3, 0, k );
            //magma_z_mvisu( hB1 );
            magma_z_mpksetup( hA, &hB2, 3, 1, k );
            //magma_z_mvisu( hB2 );
            magma_z_mpksetup( hA, &hB3, 3, 2, k );
            //magma_z_mvisu( hB3 );
            

        }

        // free CPU memory
        magma_z_mfree(&hA);

        if (id > -1) break;
    }

    TESTING_FINALIZE();
    return 0;
}
