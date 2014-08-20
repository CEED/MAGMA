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

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "../include/magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"

#define PRECISION_z


/* ////////////////////////////////////////////////////////////////////////////
   -- running magma_zcg magma_zcg_merge 
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    const char *filename[] =
    {

            "/home/hanzt/sparse_matrices/mtx/thermal2.mtx", //            n:1228045 nnz:8580313 nnz/n:6 max_nnz_row:11 
            "/home/hanzt/sparse_matrices/mtx/ecology2.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/G3_circuit.mtx", //          n:1585478 nnz:7660826 nnz/n:4 max_nnz_row:6 
            "/home/hanzt/sparse_matrices/mtx/parabolic_fem.mtx", //       n:525825 nnz:3674625 nnz/n:6 max_nnz_row:7
            "/home/hanzt/sparse_matrices/mtx/apache2.mtx", //             n:715176 nnz:4817870 nnz/n:6 max_nnz_row:8 
            "/home/hanzt/sparse_matrices/mtx/offshore.mtx", //            n:259789 nnz:4242673 nnz/n:16 max_nnz_row:31 
            "/home/hanzt/sparse_matrices/mtx/af_shell3.mtx", //           n:504855 nnz:17562051 nnz/n:34 max_nnz_row:40     

            "/home/hanzt/sparse_matrices/mtx/Trefethen_20.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/Trefethen_200.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/Trefethen_2000.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/Trefethen_20000.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4


/*
   //         "/home/hanzt/sparse_matrices/mtx/A010.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
    //        "/home/hanzt/sparse_matrices/mtx/A013.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
    //        "/home/hanzt/sparse_matrices/mtx/A016.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
    //        "/home/hanzt/sparse_matrices/mtx/A020.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0      
    //        "/home/hanzt/sparse_matrices/mtx/A025.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0            
    //        "/home/hanzt/sparse_matrices/mtx/A032.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A040.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A050.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A063.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A080.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A100.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0            
            "/home/hanzt/sparse_matrices/mtx/A126.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A159.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
*/
        //    "/home/hanzt/sparse_matrices/mtx/audikw_1.mtx", //            n:943695 nnz:77651847 nnz/n:82 max_nnz_row:345 
     //       "/home/hanzt/sparse_matrices/mtx/af_shell3.mtx", //           n:504855 nnz:17562051 nnz/n:34 max_nnz_row:40                 13 
      //      "/home/hanzt/sparse_matrices/mtx/Trefethen_20.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4

            "/home/hanzt/sparse_matrices/mtx/thermal2.mtx", //            n:1228045 nnz:8580313 nnz/n:6 max_nnz_row:11 

        //    "/home/hanzt/sparse_matrices/mtx/shallow_water2.mtx", //            n:81920 nnz:327680 nnz/n:4 max_nnz_row:8 SELLP-storage:655360 overhead:  5.0000e+01
            "/home/hanzt/sparse_matrices/mtx/minsurfo.mtx", //                  n:40806 nnz:203622 nnz/n:4 max_nnz_row:8 SELLP-storage:326464 overhead:  3.7628e+01
            "/home/hanzt/sparse_matrices/mtx/Pres_Poisson.mtx", //              n:14822 nnz:715804 nnz/n:48 max_nnz_row:56 SELLP-storage:822400 overhead:  1.2962e+01
      //      "/home/hanzt/sparse_matrices/mtx/ecology2.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/G3_circuit.mtx", //          n:1585478 nnz:7660826 nnz/n:4 max_nnz_row:6 
            "/home/hanzt/sparse_matrices/mtx/parabolic_fem.mtx", //       n:525825 nnz:3674625 nnz/n:6 max_nnz_row:7
            "/home/hanzt/sparse_matrices/mtx/apache2.mtx", //             n:715176 nnz:4817870 nnz/n:6 max_nnz_row:8 
        //    "/home/hanzt/sparse_matrices/mtx/offshore.mtx", //            n:259789 nnz:4242673 nnz/n:16 max_nnz_row:31 
            "/home/hanzt/sparse_matrices/mtx/Trefethen_20000.mtx", //     n:20000 nnz:554466 nnz/n:27 max_nnz_row:29 
        //    "/home/hanzt/sparse_matrices/mtx/af_shell3.mtx", //           n:504855 nnz:17562051 nnz/n:34 max_nnz_row:40                 13 

            "/home/hanzt/sparse_matrices/mtx/A010.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A013.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A016.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A020.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0      
            "/home/hanzt/sparse_matrices/mtx/A025.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0            
            "/home/hanzt/sparse_matrices/mtx/A032.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A040.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A050.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A063.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A080.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A100.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0            
            "/home/hanzt/sparse_matrices/mtx/A126.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A159.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0



// some more tests...
 //           "/home/hanzt/sparse_matrices/mtx/nasasrb.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
  //          "/home/hanzt/sparse_matrices/mtx/m_t1.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
   //         "/home/hanzt/sparse_matrices/mtx/consph.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
   //         "/home/hanzt/sparse_matrices/mtx/thread.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
    //        "/home/hanzt/sparse_matrices/mtx/s3dkq4m2.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
      //      "/home/hanzt/sparse_matrices/mtx/pdb1HYS.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
        //    "/home/hanzt/sparse_matrices/mtx/ship_001.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
    //        "/home/hanzt/sparse_matrices/mtx/smt.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0    
    //        "/home/hanzt/sparse_matrices/mtx/oilpan.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
     //       "/home/hanzt/sparse_matrices/mtx/cfd1.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
     //       "/home/hanzt/sparse_matrices/mtx/smt.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
     //       "/home/hanzt/sparse_matrices/mtx/crystm03.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/thermal1.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/apache1.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/shallow_water1.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/shallow_water2.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/minsurfo.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
       //     "/home/hanzt/sparse_matrices/mtx/bloweybq.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/Pres_Poisson.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0

            "/home/hanzt/sparse_matrices/mtx/A010.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A013.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A016.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A020.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0      
            "/home/hanzt/sparse_matrices/mtx/A025.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0            
            "/home/hanzt/sparse_matrices/mtx/A032.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A040.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A050.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A063.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A080.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/A100.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0            
            "/home/hanzt/sparse_matrices/mtx/A126.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0


//---

            "/home/hanzt/sparse_matrices/mtx/ecology2.mtx", //            n:999999 nnz:4995991 nnz/n:4 max_nnz_row:5                    0
            "/home/hanzt/sparse_matrices/mtx/G3_circuit.mtx", //          n:1585478 nnz:7660826 nnz/n:4 max_nnz_row:6 
            "/home/hanzt/sparse_matrices/mtx/parabolic_fem.mtx", //       n:525825 nnz:3674625 nnz/n:6 max_nnz_row:7
            "/home/hanzt/sparse_matrices/mtx/apache2.mtx", //             n:715176 nnz:4817870 nnz/n:6 max_nnz_row:8 

            "/home/hanzt/sparse_matrices/mtx/Trefethen_20.mtx", //        n:19 nnz:147 nnz/n:7 max_nnz_row:9                            4
            "/home/hanzt/sparse_matrices/mtx/thermal2.mtx", //            n:1228045 nnz:8580313 nnz/n:6 max_nnz_row:11 
//            "/home/hanzt/sparse_matrices/mtx/Trefethen_200.mtx", //       n:200 nnz:2890 nnz/n:14 max_nnz_row:16 

//            "/home/hanzt/sparse_matrices/mtx/stomach.mtx", //             n:213360 nnz:3021648 nnz/n:14 max_nnz_row:19                  7
//            "/home/hanzt/sparse_matrices/mtx/Trefethen_2000.mtx", //      n:2000 nnz:41906 nnz/n:20 max_nnz_row:22 
//            "/home/hanzt/sparse_matrices/mtx/cage10.mtx", //              n:11397 nnz:150645 nnz/n:13 max_nnz_row:25
//            "/home/hanzt/sparse_matrices/mtx/xenon2.mtx", //              n:157464 nnz:3866688 nnz/n:24 max_nnz_row:27 
//            "/home/hanzt/sparse_matrices/mtx/Trefethen_20000.mtx", //     n:20000 nnz:554466 nnz/n:27 max_nnz_row:29 
            "/home/hanzt/sparse_matrices/mtx/offshore.mtx", //            n:259789 nnz:4242673 nnz/n:16 max_nnz_row:31 

            "/home/hanzt/sparse_matrices/mtx/af_shell3.mtx", //           n:504855 nnz:17562051 nnz/n:34 max_nnz_row:40                 13 
            "/home/hanzt/sparse_matrices/mtx/bone010.mtx", //             n:986703 nnz:47851783 nnz/n:48 max_nnz_row:63 

            "/home/hanzt/sparse_matrices/mtx/boneS10.mtx", //             n:914898 nnz:40878708 nnz/n:44 max_nnz_row:67                 15
            "/home/hanzt/sparse_matrices/mtx/jj_Cube_Coup_dt0.mtx", //    n:2164760 nnz:124406070 nnz/n:57 max_nnz_row:68 
            "/home/hanzt/sparse_matrices/mtx/shipsec1.mtx", //            n:140874 nnz:3568176 nnz/n:25 max_nnz_row:68 
            "/home/hanzt/sparse_matrices/mtx/jj_ML_Geer.mtx", //          n:1504002 nnz:110686677 nnz/n:73 max_nnz_row:74 
            "/home/hanzt/sparse_matrices/mtx/ldoor.mtx", //               n:952203 nnz:42493817 nnz/n:44 max_nnz_row:77 
            "/home/hanzt/sparse_matrices/mtx/cant.mtx", //                n:62451 nnz:4007383 nnz/n:64 max_nnz_row:78 
            "/home/hanzt/sparse_matrices/mtx/kkt_power.mtx", //           n:2063494 nnz:12771361 nnz/n:6 max_nnz_row:90 
            "/home/hanzt/sparse_matrices/mtx/Hook_1498.mtx", //           n:1498023 nnz:59374451 nnz/n:39 max_nnz_row:93 
            "/home/hanzt/sparse_matrices/mtx/dielFilterV2real.mtx", //    n:1157456 nnz:48538952 nnz/n:41 max_nnz_row:110               

            "/home/hanzt/sparse_matrices/mtx/pwtk.mtx", //                n:217918 nnz:11524432 nnz/n:52 max_nnz_row:180                24
            "/home/hanzt/sparse_matrices/mtx/StocF-1465.mtx", //          n:1465137 nnz:21005389 nnz/n:14 max_nnz_row:189 
            "/home/hanzt/sparse_matrices/mtx/m_t1.mtx", //                n:97578 nnz:9753570 nnz/n:99 max_nnz_row:237 

            "/home/hanzt/sparse_matrices/mtx/Fault_639.mtx", //           n:638802 nnz:27245944 nnz/n:42 max_nnz_row:267                27 
            "/home/hanzt/sparse_matrices/mtx/audikw_1.mtx", //            n:943695 nnz:77651847 nnz/n:82 max_nnz_row:345 
            "/home/hanzt/sparse_matrices/mtx/bmw3_2.mtx", //              n:227362 nnz:11288630 nnz/n:49 max_nnz_row:336 
            "/home/hanzt/sparse_matrices/mtx/bmwcra_1.mtx", //            n:148770 nnz:10641602 nnz/n:71 max_nnz_row:351 
            "/home/hanzt/sparse_matrices/mtx/F1.mtx", //                  n:343791 nnz:26837113 nnz/n:78 max_nnz_row:435 
            "/home/hanzt/sparse_matrices/mtx/pre2.mtx", //                n:659033 nnz:5834044 nnz/n:8 max_nnz_row:627 
            "/home/hanzt/sparse_matrices/mtx/inline_1.mtx", //            n:503712 nnz:36816170 nnz/n:73 max_nnz_row:843 
           // "/home/hanzt/sparse_matrices/mtx/crankseg_2.mtx", //          n:63838 nnz:14148858 nnz/n:221 max_nnz_row:3423 


    };
    
    real_Double_t start, end;

     for(int matrix=6; matrix<7; matrix++){
    //for(int matrix=0; matrix<1; matrix++){
    //for(int matrix=4; matrix<5; matrix++){
    int num_vecs = 10;

    magma_z_sparse_matrix hA, hA2, hA2t, hLU, hA3, hAtmp, dA, hAD, hADD, hADDt, dAD, dADD;


// matrix from UFMC
    

    magma_z_csr_mtx( &hA, filename[matrix] );

    magma_zmscale( &hA, Magma_UNITDIAG );

    magma_zilustruct( &hA, 2);

    magma_z_mconvert( hA, &hAtmp, Magma_CSR, Magma_CSRL );
    magma_z_mtransfer( hAtmp, &dA, Magma_CPU, Magma_DEV );

    real_Double_t FLOPS = 2.0*hA.nnz/1e9;


    magma_z_mconvert( hAtmp, &hAD, Magma_CSR, Magma_CSRCOO );

    magma_z_mtransfer( hAD, &dAD, Magma_CPU, Magma_DEV );

    //magma_zinitguess( hAD, &hADD, &hADDt );


//magma_z_mvisu( hA );


  //  printf("n: %d max_nnz_row: %d\n", hAD.num_rows, hAD.max_nnz_row);



        printf("######################################\n");   
     //   printf("# diag: %.4e\n", diag_vals[0]);
        printf("# iters  sec(MAGMA)  sec(CUSPARSE)  ||G - G*||_F     ||A-GG'||_F\n");   


            real_Double_t t_cusparse, t_chow;

            magmaDoubleComplex alpha = MAGMA_Z_MAKE(1.0, 0.0);
            magmaDoubleComplex beta = MAGMA_Z_MAKE(0.0, 0.0);
            magma_z_vector dx, dy;

            // init DEV vectors
            magma_z_vinit( &dx, Magma_DEV, hA.num_rows, alpha );
            magma_z_vinit( &dy, Magma_DEV, hA.num_rows, beta );

            //magma_z_mvisu( dA);
        
          //  printf("\n\n\n cusparse comparison:\n\n");

            // CUSPARSE context //
            cusparseHandle_t cusparseHandle;
            cusparseStatus_t cusparseStatus;
            cusparseStatus = cusparseCreate(&cusparseHandle);
             if(cusparseStatus != 0)    printf("error in Handle.\n");


            cusparseMatDescr_t descrA;
            cusparseStatus = cusparseCreateMatDescr(&descrA);
             if(cusparseStatus != 0)    printf("error in MatrDescr.\n");

            cusparseStatus =
            cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_SYMMETRIC);
             if(cusparseStatus != 0)    printf("error in MatrType.\n");

            cusparseStatus =
            cusparseSetMatDiagType (descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
             if(cusparseStatus != 0)    printf("error in DiagType.\n");

            cusparseStatus =
            cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
             if(cusparseStatus != 0)    printf("error in IndexBase.\n");

            cusparseStatus =
            cusparseSetMatFillMode(descrA,CUSPARSE_FILL_MODE_LOWER);
             if(cusparseStatus != 0)    printf("error in fillmode.\n");

            cusparseSolveAnalysisInfo_t info;
            cusparseStatus =
            cusparseCreateSolveAnalysisInfo(&info);
             if(cusparseStatus != 0)    printf("error in info.\n");

            // end CUSPARSE context //




        magma_device_sync(); start = magma_wtime(); 
            cusparseStatus =
            cusparseZcsrsv_analysis( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        dA.num_rows, dA.nnz, descrA,
                            dA.val, dA.row, dA.col, info); 
             if(cusparseStatus != 0)    printf("error in analysis.\n");

            cusparseStatus =
            cusparseZcsric0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                              dA.num_rows, descrA, 
                             (magmaDoubleComplex*) dA.val, (const int *) dA.row, (const int *) dA.col, info);
             if(cusparseStatus != 0)    printf("error in ICC:%d.\n", cusparseStatus);




        magma_device_sync(); end = magma_wtime(); 
     //   printf( " > CUSPARSE: %.2e seconds %.2e GFLOP/s    (SELLP).\n",
                         //               (end-start), FLOPS/(end-start) );

            t_cusparse = end-start;


            cusparseDestroySolveAnalysisInfo( info );
             if(cusparseStatus != 0)    printf("error in info-free.\n");


            magma_z_mtransfer( dA, &hA3, Magma_DEV, Magma_CPU );
            magma_z_mfree(&hAtmp);

/*
            magma_z_mconvert( hA3, &hAtmp, Magma_CSRL, Magma_CSR );
            magma_z_mfree(&hA3);
            magma_z_mconvert( hAtmp, &hA3, Magma_CSR, Magma_CSR );
            magma_z_mfree(&hAtmp);*/


    magma_zmscale( &hAD, Magma_UNITROW );


for(int iters=0; iters<31; iters++){


    magma_z_mtransfer( hADD, &dADD, Magma_CPU, Magma_DEV );

   // magma_z_mvisu( hA );
  //  printf("\n");


        magma_device_sync(); start = magma_wtime(); 
    //magma_z_mvisu( dADD);
    //printf("\n");

    for(int i=0; i<iters; i++){
       magma_zaic_csr_s( dAD, dADD );
      //magma_z_mvisu( dADD);
     // printf("\n");
    }

        magma_device_sync(); end = magma_wtime(); 


        printf(" %d    %.2e   ",1* iters, (end-start));


    magma_z_mtransfer( dADD, &hA2, Magma_DEV, Magma_CPU );
    magma_z_mfree(&dADD);



    real_Double_t res = 0.0;
    real_Double_t nonlinres = 0.0;

    magma_zfrobenius( hA3, hA2, &res );

    magma_z_cucsrtranspose(   hA2, &hA2t );

    magma_znonlinres(   hA, hA2, hA2t, &hLU, &nonlinres); 



        printf(" %.2e    %.4e    %.4e   \n", t_cusparse, res, nonlinres);

    magma_z_mfree(&hA2);

    magma_z_mfree(&hA2t);


    }
    magma_z_mfree(&dAD);
    magma_z_mfree(&hA);
    magma_z_mfree(&hAD);
    magma_z_mfree(&hA3);
    magma_z_mfree(&dA);

}


    TESTING_FINALIZE();
    return 0;
}
