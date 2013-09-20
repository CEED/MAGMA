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
    };

    int id = -1, matrix = 0, i, ione = 1, *pntre, num_rblocks, num_cblocks;
    double start, end, work[1];
    for( int i = 1; i < argc; ++i ) {
        if ( strcmp("--id", argv[i]) == 0 ) {
            id = atoi( argv[++i] );
        }
    }
    if (id > -1) printf( "\n    Usage: ./testing_z_mv --id %d\n\n",id );

    for(matrix=17; matrix<18; matrix++)
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
        // possible usage of other formats
        //magma_z_mconvert( hA, &hB, hA.storage_type, Magma_ELLPACK);
        //magma_z_mtransfer( hB, &dA, hB.memory_location, Magma_DEV);

        //MagmaMaxGPUs
        magma_int_t num_gpus = 2;
        magma_int_t num_procs=num_gpus, big_blocksize, *blocksize, *offset;
        magma_imalloc_cpu( &(blocksize), num_procs );
        magma_imalloc_cpu( &(offset), num_procs );
        big_blocksize = floor( hA.num_rows / num_procs );
        for( int gpu=0; gpu<num_procs-1; gpu++ ){
            blocksize[gpu] = big_blocksize;
        }
        blocksize[num_procs-1] = hA.num_rows -  (num_procs-1) * big_blocksize;

        offset[0] = 0;
        for( int gpu=0; gpu<num_procs-1; gpu++ ){
            offset[gpu+1] = offset[gpu] + blocksize[gpu];
        }

        // GPU stream
        magma_queue_t stream[num_procs];
        for( int gpu=0; gpu<num_procs; gpu++ )
            magma_queue_create( &stream[gpu] );

        for(int sk=2; sk<5; sk++){
            printf("k: %d\n", sk);
            magma_z_sparse_matrix hB[MagmaMaxGPUs];
            magma_z_sparse_matrix dB[MagmaMaxGPUs];
            magma_z_vector a[MagmaMaxGPUs][6];
            magma_z_mpksetup(  hA, hB, num_procs, offset, blocksize, sk );
            for( int gpu=0; gpu<num_procs; gpu++ ){
                magma_setdevice(gpu);
                magmablasSetKernelStream(stream[gpu]);
                magma_z_mtransfer( hB[gpu], &dB[gpu], Magma_CPU, Magma_DEV);
                for(int j=0; j<8; j++)
                    magma_z_vinit( &a[gpu][j], Magma_DEV, hA.num_rows, one );
            }

            
/*
            // >>> gather q[k-1] into qt[d] <<<
            for (d=0; d<num_gpus; d++) {
                magma_setdevice(d);
                magmablasSetKernelStream(stream[d][0]);
                for( i=0; i<num_gpus; i++ ) {
                    if( i == d ) {
                        magma_zcopy(dofs(d), qval(d,k+j-1), 1, &(qt[i].val[frow(d)]), 1);
                    } else {
                        magma_zcopymatrix_async(dofs(d),1, qval(d,k+j-1), ld(d), &(qt[i].val[frow(d)]), dofs(d), stream[d][0]);
                        magma_event_record( event[d][i], stream[d][0] );
                    }
                }
            }*/
            for(int j=0; j<sk; j++){
                // compute spmv
                for (int d=0; d<num_gpus; d++) {
                    magma_setdevice(d);
                    magmablasSetKernelStream(stream[d]);
                    // r = A q[k] 
                    //rt[d].val = qval(d,k+j); rt[d].num_rows = rt[d].nnz = dofs(d);
                    magma_z_spmv( one, dB[d], a[d][j], zero, a[d][j+1] );

                    //  r = r - B(j) q[j]
                    //magma_zaxpy( dofs(d),-B[j], qval(d,k+j-1), 1, qval(d,k+j), 1);// shift for newton basis
                }
            }


        }

        // free CPU memory
        magma_z_mfree(&hA);

        if (id > -1) break;
    }

    TESTING_FINALIZE();
    return 0;
}
