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
     "test_matrices/lap_25.mtx",             // 17

     "test_matrices/Trefethen_2000.mtx",
     "test_matrices/bcsstk01.mtx",
     "test_matrices/Pres_Poisson.mtx",
     "test_matrices/bloweybq.mtx",
     "test_matrices/ecology2.mtx",
     "test_matrices/apache2.mtx",
     "test_matrices/crankseg_2.mtx",
     "test_matrices/bmwcra_1.mtx",
     "test_matrices/F1.mtx",
     "test_matrices/audikw_1.mtx",
     "test_matrices/ldoor.mtx",
     "test_matrices/G3_circuit.mtx",
     "test_matrices/tmt_sym.mtx",
     "test_matrices/offshore.mtx",
     "test_matrices/bmw3_2.mtx",
     "test_matrices/cyl6.mtx",
     "test_matrices/poisson3Da.mtx",
     "test_matrices/stokes64.mtx",
     "test_matrices/circuit_3.mtx",
     "test_matrices/cage10.mtx",
     "test_matrices/boneS01.mtx",
     "test_matrices/boneS10.mtx",
     "test_matrices/bone010.mtx",
     "test_matrices/gr_30_30.mtx",
     "test_matrices/ML_Laplace.mtx",
     "test_matrices/parabolic_fem.mtx",
     "test_matrices/circuit5M.mtx"
    };

    int id = -1, matrix = 0, i, ione = 1, *pntre, num_rblocks, num_cblocks;
    double start, end, work[1];
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--id", argv[i]) == 0 ) {
            id = atoi( argv[++i] );
        }
    }
    if (id > -1) printf( "\n    Usage: ./testing_z_mv --id %d\n\n",id );

    for(matrix=40; matrix<50; matrix++)
    {
        magma_z_sparse_matrix hA, hB1, hB2, hB3, dA, hC;
        magma_int_t num_add_rows, *add_rows;
        magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
        magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);


  
        // init matrix on CPU
        if (id > -1) matrix = id;
        magma_z_csr_mtx( &hA, filename[matrix] );
        printf( "\nmatrix read: %d-by-%d with %d nonzeros\n\n",hA.num_rows,hA.num_cols,hA.nnz );
        //magma_z_mvisu(hA );

        // set the total number of matrix powers - in CA-GMRES this
        // corresponds to the restart parameter
        magma_int_t power_count = 30;
        // array containing the shift for higher numerical stability
        magmaDoubleComplex lambda[power_count];
        for( i=0; i<power_count; i++)
            lambda[ i ] = zero;


        #define ENABLE_TIMER
        #ifdef ENABLE_TIMER
        double t_spmv1, t_spmv2, t_spmv = 0.0;
        double computations = power_count * (2.* hA.nnz);
            printf( "#===================================================================================#\n" );
            printf( "# matrix powers | power kernel | restarts | additional rows > GPU | runtime | GFLOPS\n" );
            printf( "#-----------------------------------------------------------------------------------#\n" );
        #endif

        // GPU allocation and streams
        magma_int_t num_gpus = 2;
        magma_queue_t stream[num_gpus];
        for( int gpu=0; gpu<num_gpus; gpu++ ){
            magma_setdevice(gpu);
            magma_queue_create( &stream[gpu] );
        }
        
        // distribution of the matrix: equal number of rows
        magma_int_t big_blocksize, *blocksize, *offset;
        magma_imalloc_cpu( &(blocksize), num_gpus );
        magma_imalloc_cpu( &(offset), num_gpus );
        big_blocksize = floor( hA.num_rows / num_gpus );
        for( int gpu=0; gpu<num_gpus-1; gpu++ ){
            blocksize[gpu] = big_blocksize;
        }
        blocksize[num_gpus-1] = hA.num_rows -  (num_gpus-1) * big_blocksize;
        offset[0] = 0;
        for( int gpu=0; gpu<num_gpus-1; gpu++ ){
            offset[gpu+1] = offset[gpu] + blocksize[gpu];
        }

        // CPU memory allocation (for max number of GPUs)
        magma_z_sparse_matrix hB[MagmaMaxGPUs];
        magma_z_vector ha;
        magma_z_vinit( &ha, Magma_CPU, hA.num_rows, one );

        // GPU memory allocation
        magma_z_sparse_matrix dB[MagmaMaxGPUs];
        magma_z_vector a[MagmaMaxGPUs][power_count+1];
        for( int gpu=0; gpu<num_gpus; gpu++ ){
            magma_setdevice(gpu);
            magmablasSetKernelStream(stream[gpu]);
            for(int j=0; j<power_count+1; j++)
                magma_z_vinit( &a[gpu][j], Magma_DEV, hA.num_rows, zero );
        }

        for(int sk=1; sk<power_count+1; sk++){
            // set the number of matrix power kernel restarts for
            // for the matrix power number
            magma_int_t restarts = (power_count+sk-1)/sk;
            
            #ifdef ENABLE_TIMER
            // use magma_z_mpkinfo to show number of additional rows
            magma_int_t num_add_rows[MagmaMaxGPUs];
            magma_z_mpkinfo( hA, num_gpus, offset, blocksize, sk, num_add_rows );
            printf("    %d      %d      %d    |  ", power_count, sk, restarts );
            for( int gpu=0; gpu<num_gpus; gpu++ ){           
                printf("%d > %d    ",num_add_rows[gpu], gpu);

            }
            #endif


            // setup for matrix power kernel and initialization of host vector
            magma_z_mpksetup(  hA, hB, num_gpus, offset, blocksize, sk );
            for( i=0; i<hA.num_rows; i++)
                ha.val[i] = one;
            // distribution of the matrices to the GPUs
            for( int gpu=0; gpu<num_gpus; gpu++ ){
                magma_setdevice(gpu);
                magmablasSetKernelStream(stream[gpu]);
                magma_z_mtransfer( hB[gpu], &dB[gpu], Magma_CPU, Magma_DEV);
            }

            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv1=magma_wtime();
            #endif
         
            // use the matrix power kernel with sk restart-1 times
            for( int i=0; i<restarts; i++){
                int j;
                // distribute vector for next sk powers
                for( int gpu=0; gpu<num_gpus; gpu++ ){
                    magma_setdevice(gpu);
                    magmablasSetKernelStream(stream[gpu]);
                    magma_zsetvector_async( hA.num_rows, ha.val, 1, (a[gpu][i*sk]).val, 1, stream[gpu] );
                }

                if( i<restarts-1 ){
                    for(j=0; j<sk; j++){
                        // compute sk spmvs
                        for (int gpu=0; gpu<num_gpus; gpu++) {
                            magma_setdevice(gpu);
                            magmablasSetKernelStream(stream[gpu]);
                            magma_z_spmv_shift( one, dB[gpu], lambda[i*sk+j], a[gpu][i*sk+j], zero, a[gpu][i*sk+j+1] );
                        }
                    }
                }
                else{
                    for(j=0; j< power_count-sk*(restarts-1); j++){
                        // the last matrix power kernel handles less powers
                        for (int gpu=0; gpu<num_gpus; gpu++) {
                            magma_setdevice(gpu);
                            magmablasSetKernelStream(stream[gpu]);
                            magma_z_spmv_shift( one, dB[gpu], lambda[i*sk+j], a[gpu][i*sk+j], zero, a[gpu][i*sk+j+1] );
                        }
                    }
                }
                // collect the blocks in CPU memory
                for( int gpu=0; gpu<num_gpus; gpu++ ){
                    magma_setdevice(gpu);
                    magmablasSetKernelStream(stream[gpu]);
                    magma_zgetvector_async( blocksize[gpu], (a[gpu][i*sk+j]).val+offset[gpu], 1, ha.val+offset[gpu], 1, stream[gpu] );
                }
            }
           
            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv2 = magma_wtime(); t_spmv = t_spmv2 - t_spmv1;
            //printf("sk: %d  runtime: %.2lf  GFLOPS: %.2lf\n", sk, t_spmv, (double) computations/(t_spmv*(1.e+09)) );
            printf("|   %.2lf  %.2lf\n", t_spmv, (double) computations/(t_spmv*(1.e+09)) );
            #endif

            // clean up CPU matrix memory
            for( int gpu=0; gpu<num_gpus; gpu++ )
                magma_z_mfree(&hB[gpu]);
            // clean up GPU matrix memory
            for( int gpu=0; gpu<num_gpus; gpu++ ){
                magma_setdevice(gpu);
                magma_z_mfree(&dB[gpu]);
            }


        }

        #ifdef ENABLE_TIMER
        printf( "#===================================================================================#\n\n" );
        #endif

        // clean up GPU vector memory
        for( int gpu=0; gpu<num_gpus; gpu++ ){
            magma_setdevice(gpu);
            for(int j=0; j<power_count+1; j++)
                magma_z_vfree(&a[gpu][j]);
        }

        // free CPU memory
        magma_z_mfree(&hA);

        if (id > -1) break;
    }

    TESTING_FINALIZE();
    return 0;
}
