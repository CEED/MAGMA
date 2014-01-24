/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds
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
   -- Testing magma_zcpir
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    magma_opts opts;
    parse_opts( argc, argv, &opts );

    //Chronometry
    double t_1 = 0.0, t_1s, t_1e, t_2 = 0.0, t_2s, t_2e, t_3 = 0.0, t_3s, t_3e;
    
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
     "test_matrices/lap_25.mtx",             // 17
     "test_matrices/beaflw.mtx",             // 18
     "test_matrices/mbeaflw.mtx",            // 19
    };

    printf( "#=============================================================#\n" );
    printf( "# init of sp matrix | conversion | merging both | difference \n" );
    printf( "#-------------------------------------------------------------#\n" );

    for(magma_int_t matrix=18; matrix<21; matrix++){

        int k, ms = 1000;

        magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
        const char *N="N";

        magma_z_sparse_matrix Ah, Ad;
        magma_c_sparse_matrix Bh, Bd;

        // read in matrix and visualize
        magma_z_csr_mtx( &Ah, filename[matrix] );
        //magma_z_mvisu( Ah );    
        // transfer to DEV  
        magma_z_mtransfer( Ah, &Ad, Magma_CPU, Magma_DEV );

        for( k=0; k<ms; k++){
            //Chronometry 
            magma_device_sync(); t_1s = magma_wtime();
            magma_zlag2c_CSR_DENSE( Ad, &Bd );       
            //Chronometry  
            magma_device_sync(); t_1e = magma_wtime(); t_1 += t_1e - t_1s;
            magma_c_mfree( &Bd );
        }

        for( k=0; k<ms; k++){
            //Chronometry 
            magma_device_sync(); t_2s = magma_wtime();
            magma_zlag2c_CSR_DENSE_alloc( Ad, &Bd );          
            //Chronometry  
            magma_device_sync(); t_2e = magma_wtime(); t_2 += t_2e - t_2s;
            magma_c_mfree( &Bd );
        }

        magma_zlag2c_CSR_DENSE_alloc( Ad, &Bd );      

        for( k=0; k<ms; k++){
            //Chronometry 
            magma_device_sync(); t_3s = magma_wtime();
            magma_zlag2c_CSR_DENSE_convert( Ad, &Bd );             
            //Chronometry  
            magma_device_sync(); t_3e = magma_wtime(); t_3 += t_3e - t_3s;
        }

        printf("%.2lf  |  %.2lf  |  %.2lf  |  %.2lf \n", 
                                   t_2, t_3, t_1, (t_2+t_3)-t_1 );


        // transfer to CPU 
        magma_c_mtransfer( Bd, &Bh, Magma_DEV, Magma_CPU );

        //magma_c_mvisu( Bh );

        magma_z_mfree( &Ah );
        magma_z_mfree( &Ad );
        magma_c_mfree( &Bd );
        magma_c_mfree( &Bh );

    }

    printf( "#=============================================================#\n" );


    TESTING_FINALIZE();
    return 0;
}
