/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
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
   -- Debugging file
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    //Chronometry
    struct timeval inicio, fim;
    double tempo1, tempo2;

    magma_opts opts;
    parse_opts( argc, argv, &opts );
    
    printf("If running lapack (option --lapack), MAGMA and CUBLAS error are both computed\n"
           "relative to CPU BLAS result. Else, MAGMA error is computed relative to CUBLAS result.\n\n"
           "transA = %c, transB = %c\n", opts.transA, opts.transB );
    printf("    M     N     K   MAGMA Gflop/s (ms)  CUBLAS Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  CUBLAS error\n");
    printf("=========================================================================================================\n");


    const char *filename[] =
    {
   /*  "test_matrices/tuma1.mtx",
     "test_matrices/msc23052.mtx",
     "test_matrices/bcsstm36.mtx",
     "test_matrices/c-49.mtx",
     "test_matrices/c-60.mtx",
     "test_matrices/smt.mtx",
     "test_matrices/shallow_water1.mtx",*/
     "test_matrices/nasasrb.mtx",
     //"test_matrices/venkat50.mtx",
     "test_matrices/bbmat.mtx",
     "test_matrices/wang4.mtx",
     "test_matrices/wang3.mtx",
     "test_matrices/raefsky4.mtx",
     "test_matrices/raefsky3.mtx",
     "test_matrices/CAG_mat72.mtx",
     "test_matrices/Trefethen_20_b.mtx",
     "test_matrices/Trefethen_200.mtx",
     "test_matrices/Trefethen_2000.mtx",
     "test_matrices/Trefethen_20000.mtx",

     "test_matrices/apache1.mtx",
     "test_matrices/filter3D.mtx",
    // "test_matrices/Baumann.mtx",
     "test_matrices/boneS01.mtx",
     "test_matrices/stomach.mtx",
     "test_matrices/G2_circuit.mtx",

     "test_matrices/Laplace2D_1M.mtx",
     "test_matrices/apache2.mtx",
     "test_matrices/lap_25.mtx",
     "test_matrices/diag.mtx",
     "test_matrices/tridiag_2.mtx",
     "test_matrices/LF10.mtx",
     "test_matrices/Trefethen_20_b.mtx",
     "test_matrices/test.mtx",
     "test_matrices/bcsstk01.mtx",
     "test_matrices/Trefethen_200.mtx",
     "test_matrices/Trefethen_2000.mtx",
     "test_matrices/cant.mtx",
     "test_matrices/parabolic_fem.mtx",
     "test_matrices/shallow_water1.mtx",
     "test_matrices/Pres_Poisson.mtx",
     "test_matrices/bloweybq.mtx",
     "test_matrices/ecology2.mtx",
     "test_matrices/crankseg_2.mtx",
     "test_matrices/bmwcra_1.mtx",
     "test_matrices/F1.mtx",
     "test_matrices/audikw_1.mtx",
     "test_matrices/circuit5M.mtx",
     "test_matrices/boneS10.mtx",
     "test_matrices/parabolic_fem.mtx",
     "test_matrices/inline_1.mtx",
     "test_matrices/ldoor.mtx"
    };
for(magma_int_t matrix=0; matrix<18; matrix++){


    magma_z_sparse_matrix A, B, C, D, E, F, G, H, I, J, K, Z;
    magma_z_vector a,b,c,x, y, z;
    magma_int_t k=1;

    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex mone = MAGMA_Z_MAKE(-1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    const char *N="N";



     #define ENABLE_TIMER
    #ifdef ENABLE_TIMER
    double t_gmres1, t_gmres = 0.0;
    double t_lu1, t_lu = 0.0;
    double t_lusv1, t_lusv = 0.0;
    #endif
    magma_z_csr_mtx( &A, filename[matrix] );
    //printf("A:\n");
    //magma_z_mvisu( A );

    for(int defaultsize=400; defaultsize>=16; defaultsize--){
        if( A.num_rows%defaultsize == 0 )
            B.blocksize = defaultsize;
    }
    printf("  %d  %d  |", A.num_rows, B.blocksize);
    //B.blocksize = 17 ;
    magma_z_mconvert( A, &B, Magma_CSR, Magma_BCSR);

    //printf("B:\n");
    //magma_z_mvisu( B );
    magma_z_mtransfer( B, &C, Magma_CPU, Magma_DEV);

    magma_int_t *ipiv;
    magma_imalloc_cpu( &ipiv, C.blocksize*(ceil( (float)C.num_rows / (float)C.blocksize )+1) );

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_lu1=magma_wtime();
        #endif
    magma_zbcsrlu( C, &D, ipiv);
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_lu+=(magma_wtime()-t_lu1);
        #endif
    

    //printf("D:\n");

    //magma_z_mvisu( D );


    magma_z_vinit( &b, Magma_DEV, A.num_cols, mone );
    magma_z_vinit( &c, Magma_DEV, A.num_cols, mone );
    magma_z_vinit( &x, Magma_DEV, A.num_cols, one );
    magma_z_vinit( &y, Magma_DEV, A.num_cols, one );
    magma_z_vvisu( x, 0, 5 );

    magma_z_solver_par solver_par;
    solver_par.epsilon = 10e-12;
    solver_par.maxiter = 10000;
    solver_par.restart = 30;

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_lusv1=magma_wtime();
        #endif
    magma_zbcsrlusv( D, b, &x, &solver_par, ipiv );
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_lusv+=(magma_wtime()-t_lusv1);
        #endif

    
    magma_z_vvisu( x, 0, 2);
   // magma_z_vvisu( x, 1000, 2);
    //magma_z_vvisu( x, 1500, 2);    
   // magma_z_vvisu( x, 1900, 2);


    magma_z_mtransfer( A, &E, Magma_CPU, Magma_DEV);

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_gmres1=magma_wtime();
        #endif
    //magma_zgmres( E, c, &y, &solver_par );
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_gmres+=(magma_wtime()-t_gmres1);
        #endif

    magma_z_vvisu( y, 0, 2);
   // magma_z_vvisu( y, 1000, 2);
   // magma_z_vvisu( y, 1500, 2);    
   // magma_z_vvisu( y, 1900, 2);

    double res;


    magma_zresidual( E, b, x, &res );
    magma_zresidual( E, b, y, &res );
/*
    for(int z=0; z<A.num_rows; z++)
        printf("%d ",ipiv[z]);
    printf("\n"); 
*/
    magma_z_mfree(&A);
    magma_z_mfree(&B);
    magma_z_mfree(&C);
    magma_z_mfree(&D);
    magma_z_mfree(&E);
    magma_z_mfree(&F);
    printf("D.numblocks:%d\n", D.numblocks);

    #ifdef ENABLE_TIMER
    printf("lu: %.2lf (%.2lf%%), lusv: %.2lf (%.2lf%%) gmres: %.2lf\n", 
            t_lu, 100.0*t_lu/(t_lu+t_lusv), 
            t_lusv, 100.0*t_lusv/(t_lu+t_lusv), 
            t_gmres);
    #endif
}
    



    TESTING_FINALIZE();
    return 0;
}
