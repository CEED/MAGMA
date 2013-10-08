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

     "test_matrices/jj_StocF-1465.mtx",      // 18
     "test_matrices/jj_Geo_1438.mtx",
     "test_matrices/jj_Emilia_923.mtx",
     "test_matrices/jj_ML_Geer.mtx",
     "test_matrices/jj_Flan_1565.mtx",
     "test_matrices/jj_Hook_1498.mtx",
     "test_matrices/jj_Long_Coup_dt0.mtx",
     "test_matrices/jj_Long_Coup_dt6.mtx",
     "test_matrices/jj_Cube_Coup_dt6.mtx",
     "test_matrices/jj_Cube_Coup_dt0.mtx",
     "test_matrices/jj_CoupCons3D.mtx",
     "test_matrices/jj_ML_Laplace.mtx",      // 29

     "test_matrices/vanbody.mtx",           // 30
     "test_matrices/apache2.mtx",           
     "test_matrices/s3dkt3m2.mtx",
     "test_matrices/s3dkq4m2.mtx",
     "test_matrices/oilpan.mtx",
     "test_matrices/minsurfo.mtx",
     "test_matrices/jnlbrng1.mtx",
     "test_matrices/gridgena.mtx",           //37

     "test_matrices/Si41Ge41H72.mtx",           // 38
     "test_matrices/Si87H76.mtx",
     "test_matrices/Si87H76.mtx",
     "test_matrices/Si34H36.mtx",
     "test_matrices/Ge87H76.mtx",
     "test_matrices/GaAsH6.mtx",
     "test_matrices/Ga41As41H72.mtx",          
     "test_matrices/Ga3As3H12.mtx",
     "test_matrices/Ga19As19H42.mtx",
     "test_matrices/Ga10As10H30.mtx",
     "test_matrices/CO.mtx",           //48


     "test_matrices/A050.rsa",   //49
     "test_matrices/A063.rsa",
     "test_matrices/A080.rsa",          
     "test_matrices/A100.rsa",
     "test_matrices/A126.rsa",
     "test_matrices/A159.rsa",
     "test_matrices/A200.rsa",
     "test_matrices/A252.rsa",
     "test_matrices/A318.rsa",           //57

     "test_matrices/cant.mtx",           //58
     "test_matrices/pwtk.mtx",
     "test_matrices/cfd2.mtx",
     "test_matrices/xenon2.mtx",
     "test_matrices/shipsec1.mtx",           //62

     "test_matrices/G3_circuit_rcm.mtx",     //63

     "test_matrices/kkt_power.mtx",           //64
     "test_matrices/G3_circuit.mtx",
     "test_matrices/Hook_1498.mtx",
     "test_matrices/StocF-1465.mtx",
     "test_matrices/dielFilterV2real.mtx",           //68


     "test_matrices/Trefethen_2000.mtx",     // 69
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

    for(matrix=58; matrix<59; matrix++)
   // for(matrix=31; matrix<38; matrix++)
    {
        magma_z_sparse_matrix hA;

        magmaDoubleComplex one  = MAGMA_Z_MAKE(1.0, 0.0);
        magmaDoubleComplex mone  = MAGMA_Z_MAKE(-1.0, 0.0);
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
            lambda[ i ] = mone;


        #define ENABLE_TIMER
        #ifdef ENABLE_TIMER
        double t_spmv1, t_spmv2, t_spmv = 0.0;
        double computations = power_count * (2.* hA.nnz);
            printf( "#==================================================================================================#\n" );
            printf( "# matrix powers | power kernel | restarts | additional rows (components) > GPU | runtime [ms] | GFLOPS\n" );
            printf( "#--------------------------------------------------------------------------------------------------#\n" );
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
        magma_z_sparse_matrix hC[MagmaMaxGPUs];
        magma_z_sparse_matrix hD[MagmaMaxGPUs];
        magma_z_vector ha;
        magma_z_vinit( &ha, Magma_CPU, hA.num_rows, one );

        // GPU memory allocation
        magma_z_sparse_matrix dB[MagmaMaxGPUs];
        magma_z_vector a[MagmaMaxGPUs][power_count+1];
        magma_z_vector spmv_comp[MagmaMaxGPUs];
        for( int gpu=0; gpu<num_gpus; gpu++ ){
            magma_setdevice(gpu);
            magmablasSetKernelStream(stream[gpu]);
            for(int j=0; j<power_count+1; j++)
                magma_z_vinit( &a[gpu][j], Magma_DEV, hA.num_rows, zero );
        }
        for( int sk=1; sk<power_count+1; sk++ ){
            // set the number of matrix power kernel restarts for
            // for the matrix power number
            magma_int_t restarts = (power_count+sk-1)/sk;
            
            #ifdef ENABLE_TIMER
            // use magma_z_mpkinfo to show number of additional rows
            magma_int_t num_add_vecs[ MagmaMaxGPUs ], num_vecs_back[ MagmaMaxGPUs ];
            magma_int_t **num_add_rows = (magma_int_t**)malloc( MagmaMaxGPUs * sizeof(magma_int_t*) );
            magma_int_t **add_rows = (magma_int_t**)malloc( MagmaMaxGPUs * sizeof(magma_int_t*) );
            magma_int_t **add_vecs = (magma_int_t**)malloc( MagmaMaxGPUs * sizeof(magma_int_t*) );
            magma_int_t **vecs_back = (magma_int_t**)malloc( MagmaMaxGPUs * sizeof(magma_int_t*) );
            magma_int_t **add_vecs_gpu = (magma_int_t**)malloc( MagmaMaxGPUs * sizeof(magma_int_t*) );
            magma_int_t **add_rows_gpu = (magma_int_t**)malloc( MagmaMaxGPUs * sizeof(magma_int_t*) );
            magma_int_t **vecs_back_gpu = (magma_int_t**)malloc( MagmaMaxGPUs * sizeof(magma_int_t*) );
            magmaDoubleComplex **compressed = (magmaDoubleComplex**)malloc( MagmaMaxGPUs * sizeof(magmaDoubleComplex*) );
            magmaDoubleComplex **compressed_back = (magmaDoubleComplex**)malloc( MagmaMaxGPUs * sizeof(magmaDoubleComplex*) );
            magmaDoubleComplex **compressed_gpu = (magmaDoubleComplex**)malloc( MagmaMaxGPUs * sizeof(magmaDoubleComplex*) );
            magmaDoubleComplex **compressed_back_gpu = (magmaDoubleComplex**)malloc( MagmaMaxGPUs * sizeof(magmaDoubleComplex*) );
            magma_z_mpkinfo( hA, num_gpus, offset, blocksize, sk, num_add_rows, add_rows, num_add_vecs, add_vecs, num_vecs_back, vecs_back );
           
            printf("    %d      %d      %d    |  ", power_count, sk, restarts );
            for( int gpu=0; gpu<num_gpus; gpu++ ){           
                printf("%d > %d ",num_add_rows[gpu][sk-1], gpu);
                printf("( %d > %d )  ",num_add_vecs[gpu], gpu);
                magma_zmalloc_cpu( &compressed[gpu], num_add_vecs[gpu] );
                magma_zmalloc_cpu( &compressed_back[gpu], num_vecs_back[gpu] );
            }
            #endif


            // setup for matrix power kernel and initialization of host vector
            magma_z_mpksetup(  hA, hB, num_gpus, offset, blocksize, sk );
            for( i=0; i<hA.num_rows; i++)
                ha.val[i] = one;
            // conversion to ELLPACKT + distribution of the matrices to the GPUs + comm-vectors
            for( int gpu=0; gpu<num_gpus; gpu++ ){
                magma_setdevice( gpu );
                magmablasSetKernelStream( stream[gpu] );
                magma_z_vinit( &spmv_comp[gpu], Magma_DEV, blocksize[gpu]+num_add_rows[gpu][sk-1], zero );

                magma_z_mpk_mcompresso( hB[gpu], &hC[gpu], offset[gpu], blocksize[gpu], num_add_rows[gpu][sk-1], add_rows[gpu] );
                magma_z_mconvert( hC[gpu], &hD[gpu], Magma_CSR, Magma_ELLPACKT ); 
                magma_z_mtransfer( hD[gpu], &dB[gpu], Magma_CPU, Magma_DEV );
                        //printf("truncated matrix:\n");magma_z_mvisu(hD[gpu] );
                magma_zmalloc( &compressed_gpu[gpu], num_add_vecs[gpu] );
                magma_zmalloc( &compressed_back_gpu[gpu], num_vecs_back[gpu] );
                magma_imalloc( &vecs_back_gpu[gpu], num_vecs_back[gpu] );
                magma_imalloc( &add_rows_gpu[gpu], num_add_rows[gpu][sk-1] );
                magma_imalloc( &add_vecs_gpu[gpu], num_add_vecs[gpu] );
                cublasSetVector( num_add_rows[gpu][sk-1], sizeof( magma_int_t ), add_rows[gpu], 1, add_rows_gpu[gpu], 1 ); 
                cublasSetVector( num_add_vecs[gpu], sizeof( magma_int_t ), add_vecs[gpu], 1, add_vecs_gpu[gpu], 1 );
                cublasSetVector( num_vecs_back[gpu], sizeof( magma_int_t ), vecs_back[gpu], 1, vecs_back_gpu[gpu], 1 ); 
                
            }

            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv1=magma_wtime();
            #endif
// do it 100 times to remove noise
for( int noise=0; noise<100; noise++){

            magma_z_vinit( &ha, Magma_CPU, hA.num_rows, one );

            for( int gpu=0; gpu<num_gpus; gpu++ ){
                magma_setdevice( gpu );
                magmablasSetKernelStream( stream[gpu] );
                magma_zsetvector_async( blocksize[gpu], ha.val+offset[gpu], 1, (a[gpu][0]).val+offset[gpu], 1, stream[gpu] );
            }
         
            // use the matrix power kernel with sk restart-1 times
            for( int i=0; i<restarts; i++){
                int j;
                // distribute vector for next sk powers
                for( int gpu=0; gpu<num_gpus; gpu++ ){
                    magma_setdevice( gpu );
                    magmablasSetKernelStream( stream[gpu] );
                    magma_z_mpk_compress( num_add_vecs[gpu], add_vecs[gpu], ha.val, compressed[gpu] );
                    magma_zsetvector_async( num_add_vecs[gpu], compressed[gpu], 1, compressed_gpu[gpu], 1, stream[gpu] );
                    magma_z_mpk_uncompress_gpu( num_add_vecs[gpu], add_vecs_gpu[gpu], compressed_gpu[gpu], (a[gpu][i*sk]).val );
                }

                if( i<restarts-1 ){
                    for(j=0; j<sk; j++){
                        // compute sk spmvs
                        for ( int gpu=0; gpu<num_gpus; gpu++ ) {
                            magma_setdevice( gpu );
                            magmablasSetKernelStream( stream[gpu] );
                            //magma_z_spmv_shift( one, dB[gpu], lambda[i*sk+j], a[gpu][i*sk+j], zero, a[gpu][i*sk+j+1] );
                            //magma_z_spmv_shift( one, dB[gpu], lambda[i*sk+j], a[gpu][i*sk+j], zero, a[gpu][i*sk+j+1] );
                            magma_z_spmv_shift( one, dB[gpu], lambda[i*sk+j], a[gpu][i*sk+j], zero, offset[gpu], blocksize[gpu], add_rows_gpu[gpu], spmv_comp[gpu] );
                            //magma_z_spmv( one, dB[gpu], a[gpu][i*sk+j], zero, spmv_comp[gpu] );
                            magma_z_mpk_uncompspmv( offset[gpu], blocksize[gpu], num_add_rows[gpu][sk-1], add_rows_gpu[gpu], (spmv_comp[gpu]).val, (a[gpu][i*sk+j+1]).val );
                        }
                    }
                }
                else{
                    for( j=0; j< power_count-sk*(restarts-1); j++ ){
                        // the last matrix power kernel handles less powers
                        for ( int gpu=0; gpu<num_gpus; gpu++ ) {
                            magma_setdevice( gpu );
                            magmablasSetKernelStream( stream[gpu] );
                            //magma_z_spmv_shift( one, dB[gpu], lambda[i*sk+j], a[gpu][i*sk+j], zero, a[gpu][i*sk+j+1] );
                            //magma_z_spmv_shift( one, dB[gpu], lambda[i*sk+j], a[gpu][i*sk+j], zero, a[gpu][i*sk+j+1] );
                            magma_z_spmv_shift( one, dB[gpu], lambda[i*sk+j], a[gpu][i*sk+j], zero, offset[gpu], blocksize[gpu], add_rows_gpu[gpu], spmv_comp[gpu]  );
                            //magma_z_spmv( one, dB[gpu], a[gpu][i*sk+j], zero, spmv_comp[gpu] );
                            magma_z_mpk_uncompspmv( offset[gpu], blocksize[gpu], num_add_rows[gpu][sk-1], add_rows_gpu[gpu], (spmv_comp[gpu]).val, (a[gpu][i*sk+j+1]).val );
                        }
                    }
                }


                // selective copy back to CPU
                for( int gpu=0; gpu<num_gpus; gpu++ ){
                    magma_setdevice( gpu );
                    magmablasSetKernelStream( stream[gpu] );
                    magma_z_mpk_compress_gpu( num_vecs_back[gpu], vecs_back_gpu[gpu], (a[gpu][i*sk+j]).val, compressed_back_gpu[gpu] );
                    magma_zgetvector_async( num_vecs_back[gpu], compressed_back_gpu[gpu], 1, compressed_back[gpu], 1, stream[gpu] );
                    magma_z_mpk_uncompress( num_vecs_back[gpu], vecs_back[gpu], compressed_back[gpu], ha.val );
                }
            }

}//end 100
            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv2 = magma_wtime(); t_spmv = t_spmv2 - t_spmv1;
            //printf("sk: %d  runtime: %.2lf  GFLOPS: %.2lf\n", sk, t_spmv, (double) computations/(t_spmv*(1.e+09)) );
            printf("|   %.2lf  %.2lf\n", t_spmv*10, (double) computations*100/(t_spmv*(1.e+09)) );
            #endif

           


// check the results
            for( int gpu=0; gpu<num_gpus; gpu++ )
                magma_zgetvector_async( blocksize[gpu], (a[gpu][power_count]).val+offset[gpu], 1, ha.val+offset[gpu], 1, stream[gpu] );


            magma_z_vector du, dv, hw;
            magma_z_vinit( &du, Magma_DEV, hA.num_rows, one );
            magma_z_vinit( &dv, Magma_DEV, hA.num_rows, zero );
            magma_z_vinit( &hw, Magma_CPU, hA.num_rows, zero );
            magma_z_sparse_matrix dA;
            magma_z_mtransfer( hA, &dA, Magma_CPU, Magma_DEV);

            for (i=0; i<power_count; i++){
                magma_z_spmv( one, dA, du, zero, dv);
                magma_zaxpy(hA.num_rows, one, du.val, 1, dv.val, 1);   
                //printf("%d:\n",i);
                //magma_z_vvisu(dv,0,7);
                cudaMemcpy( du.val, dv.val, dA.num_rows*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
            }
            
            //magma_z_vvisu(du, 0, 7);
            cublasSetVector(hA.num_rows, sizeof(magmaDoubleComplex), ha.val, 1, dv.val, 1);
            //magma_z_vvisu(dv, 0, 7);
            magma_zaxpy(hA.num_rows, mone, dv.val, 1, du.val, 1);   
            cublasGetVector(hA.num_rows, sizeof(magmaDoubleComplex), du.val, 1, hw.val, 1);

            // printf("-------------------------------\n difference in components 0 -- 7:\n"); magma_z_vvisu(hw, 0, 7);
            for( i=0; i<hA.num_rows; i++){
                if( MAGMA_Z_REAL(hw.val[i]) > 0.0 )
                        printf("error in component %d: %f\n", i, hw.val[i]);
            }
            magma_z_mfree( &dA );
            // end check results


            // clean up CPU matrix memory
            for( int gpu=0; gpu<num_gpus; gpu++ ){
                magma_z_mfree( &hB[gpu] );
                magma_z_mfree( &hC[gpu] );
                magma_z_mfree( &hD[gpu] );
                //magma_free_cpu( &add_rows[gpu] );
                //magma_free_cpu( &compressed[gpu] );
            }
            // clean up GPU matrix memory
            for( int gpu=0; gpu<num_gpus; gpu++ ){
                magma_setdevice( gpu );
                magma_z_mfree( &dB[gpu] );
            }


        }

        #ifdef ENABLE_TIMER
        printf( "#==================================================================================================#\n\n" );
        #endif

        // clean up GPU vector memory
        for( int gpu=0; gpu<num_gpus; gpu++ ){
            magma_setdevice(gpu);
            for( int j=0; j<power_count+1; j++ )
                magma_z_vfree( &a[gpu][j] );
        }

        // free CPU memory
        magma_z_mfree( &hA );

        if (id > -1) break;
    }

    TESTING_FINALIZE();
    return 0;
}
