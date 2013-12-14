/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/

#include <cuda_runtime_api.h>
#include <cublas_v2.h>  // include before magma.h

#include "magma.h"
#include "magma_lapack.h"
#include <stdio.h>
#include <stdlib.h>


//#include "common_magma.h"
#include "../include/magmasparse.h"

#define PRECISION_z

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16

#define  blockinfo(i,j)  A.blockinfo[(i)*c_blocks   + (j)]
#define  Mblockinfo(i,j)  M->blockinfo[(i)*c_blocks   + (j)]
#define M(i,j) M->val+((Mblockinfo(i,j)-1)*size_b*size_b)
#define A(i,j) A.val+((blockinfo(i,j)-1)*size_b*size_b)
#define x(i) x->val+(i*size_b)

 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })



#define CUBLASBATCHED

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    ILU Preconditioner for a BCSR matrix A. 
    In the first approach we assume all diagonal blocks to be nonzero.

    Arguments
    =========

    magma_z_sparse_matrix A                   input matrix A (on DEV)
    magma_z_sparse_matrix *M                  output matrix M approx. (LU)^{-1} (on DEV)

    =====================================================================  */


magma_int_t
magma_zbcsrlu( magma_z_sparse_matrix A, magma_z_sparse_matrix *M, magma_int_t *ipiv ){


    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;

    magma_int_t i,j,k,l,info;

    cublasHandle_t handle;
    cudaSetDevice( 0 );
    cublasCreate( &handle );


    // GPU stream
    int num_streams = 16;
    magma_queue_t stream[num_streams];
    for( i=0; i<num_streams; i++ )
        magma_queue_create( &stream[i] );

    //cublasStatus stat;
    #define ENABLE_TIMER
    #ifdef ENABLE_TIMER
    double t_blockinfo1, t_blockinfo = 0.0;
    double t_memoryalloc1, t_memoryalloc = 0.0;
    double t_valfill1, t_valfill = 0.0;
    double t_rowcolfill1, t_rowcolfill = 0.0;
    double t_cpucopy1, t_cpucopy = 0.0;
    double t_tr1, t_tr = 0.0;
    double t_getrf1, t_getrf = 0.0;
    double t_rup1, t_rup = 0.0;
    double t_lup1, t_lup = 0.0;
    double t_trsm1, t_trsm = 0.0;
    double t_gemm1, t_gemm = 0.0;
    #endif

    // fill in information for B
    M->storage_type = A.storage_type;
    M->memory_location = Magma_DEV;
    M->num_rows = A.num_rows;
    M->num_cols = A.num_cols;
    M->blocksize = A.blocksize;
    magma_int_t size_b = A.blocksize;
    magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     // max number of blocks per row
    magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     // max number of blocks per column

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_blockinfo1=magma_wtime();
        #endif
    //complete fill-in
    magma_imalloc_cpu( &M->blockinfo, r_blocks * c_blocks );

    for( k=0; k<r_blocks; k++){
        for( j=0; j<c_blocks; j++ ){
            Mblockinfo(k,j) = blockinfo(k,j);
        }
    }
    for( k=0; k<r_blocks; k++){
        for( j=k+1; j<r_blocks; j++ ){
            if( (Mblockinfo(j,k)!=0) ){
                for( i=k+1; i<c_blocks; i++ ){
                    if( (Mblockinfo(j,i)==0) && (Mblockinfo(k,i)!=0) ){
                        Mblockinfo(j,i) = -1;
                    }
                }
            }
        }
    }  
/*
    for( k=0; k<r_blocks; k++){
        for( j=0; j<c_blocks; j++ ){
            printf("%d  ", A.blockinfo[k*c_blocks + j] );
        }
        printf("\n");
    } 

    for( k=0; k<r_blocks; k++){
        for( j=0; j<c_blocks; j++ ){
            printf("%d  ", M->blockinfo[k*c_blocks + j] );
        }
        printf("\n");
    } 


    const char nontrans = 'N';
    const char trans = 'T';
    const float s_one = 1.0;
    const float s_zero = 0.0;

    float *blockinfo_tmp;
    magma_smalloc_cpu( &blockinfo_tmp, r_blocks * c_blocks );

    for(i=0; i<r_blocks*r_blocks; i++)
        blockinfo_tmp[i]; = (float) A.blockinfo[i];

                blasf77_sgemm( &nontrans, &trans, &r_blocks, &r_blocks, &r_blocks,
                               &s_one, (float *) A.blockinfo, &r_blocks,
                                       (float *) A.blockinfo, &r_blocks,
                               &s_zero,  blockinfo_tmp, &r_blocks );



    for( k=0; k<r_blocks; k++){
        for( j=0; j<c_blocks; j++ ){
            printf("%d  ", blockinfo_tmp[k*c_blocks + j] );
        }
        printf("\n");
    } 
*/


    magma_int_t num_blocks_tmp = 0;
    for( magma_int_t i=0; i<r_blocks * c_blocks; i++ ){
        if( M->blockinfo[i]!=0 ){
            num_blocks_tmp++;
            M->blockinfo[i] = num_blocks_tmp;
        }
    }

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_blockinfo+=(magma_wtime()-t_blockinfo1);
        #endif



        #ifdef ENABLE_TIMER
        magma_device_sync(); t_memoryalloc1=magma_wtime();
        #endif
    M->numblocks = num_blocks_tmp;


    magma_zmalloc( &M->val, size_b*size_b*(M->numblocks) );
    magma_imalloc( &M->row, r_blocks+1 );
    magma_imalloc( &M->col, M->numblocks );

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_memoryalloc+=(magma_wtime()-t_memoryalloc1);
        #endif

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_valfill1=magma_wtime();
        #endif


////-------------------------------------------------------

        // Prepare A 
        magmaDoubleComplex **At, **dAt,  **Bt, **dBt, **Bt2, **dBt2;
        int rowt=0, rowt2=0;
        magma_malloc_cpu((void **)& At, A.numblocks*sizeof(magmaDoubleComplex *) );
        magma_malloc(    (void **)&dAt, A.numblocks*sizeof(magmaDoubleComplex *) );
        magma_malloc_cpu((void **)& Bt, A.numblocks*sizeof(magmaDoubleComplex *) );
        magma_malloc(    (void **)&dBt, A.numblocks*sizeof(magmaDoubleComplex *) );
        magma_malloc_cpu((void **)& Bt2, (M->numblocks-A.numblocks)*sizeof(magmaDoubleComplex *) );
        magma_malloc(    (void **)&dBt2, (M->numblocks-A.numblocks)*sizeof(magmaDoubleComplex *) );
        for( i = 0; i< r_blocks; i++){
            for( j = 0; j< r_blocks; j++){
               if ( (Mblockinfo(i, j) != 0) && (blockinfo(i,j)!=0) ){
                  At[rowt] = A(i, j);
                  Bt[rowt] = M(i, j);
                  rowt++;
               }
               else if ( (Mblockinfo(i, j) != 0) && (blockinfo(i, j) == 0) ){
                  Bt2[rowt2] = M(i, j);
                  rowt2++;
               }
            }
        }
        cublasSetVector(  A.numblocks, sizeof(magmaDoubleComplex *), At, 1, dAt, 1 );
        cublasSetVector(  A.numblocks, sizeof(magmaDoubleComplex *), Bt, 1, dBt, 1 );
        cublasSetVector(  (M->numblocks-A.numblocks), sizeof(magmaDoubleComplex *), Bt2, 1, dBt2, 1 );

        magma_zbcsrvalcpy(  size_b, A.numblocks, (M->numblocks-A.numblocks), dAt, dBt, dBt2 );

        magma_free(dAt);
        magma_free(dBt);
        magma_free(dBt2);
        magma_free_cpu(At);
        magma_free_cpu(Bt);
        magma_free_cpu(Bt2);



///////----------------------------------------------------
/*
    // setup a 0 matrix
    magmaDoubleComplex *zero, *zero_d;
    magma_zmalloc_cpu( &zero, size_b*size_b );
    if ( zero == NULL )
        return MAGMA_ERR_HOST_ALLOC;
    for( i=0; i<size_b*size_b; i++){
         zero[i] = c_zero;
    }
    if (MAGMA_SUCCESS != magma_zmalloc( &zero_d, (size_b*size_b) ) ) 
        return MAGMA_ERR_DEVICE_ALLOC;
    magma_zsetvector( size_b*size_b, zero, 1, zero_d , 1 );
    magma_free_cpu(zero);

    // data transfer
    for( magma_int_t i=0; i<r_blocks; i++ ){
        for( magma_int_t j=0; j<c_blocks; j++ ){
             if( blockinfo(i,j)!=0 ){
                cudaMemcpy( M(i,j), A(i,j), size_b*size_b*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
            }
            if( blockinfo(i,j)==0 && Mblockinfo(i,j)!=0 ){
                cudaMemcpy( M(i,j), zero_d, size_b*size_b*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );
            }
        }
    }
*/

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_valfill+=(magma_wtime()-t_valfill1);
        #endif
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_rowcolfill1=magma_wtime();
        #endif

    num_blocks_tmp=0;
    magma_int_t *cpu_row, *cpu_col;
    magma_imalloc_cpu( &cpu_row, r_blocks+1 );
    magma_imalloc_cpu( &cpu_col, M->numblocks );


    num_blocks_tmp=0;
    for( i=0; i<c_blocks * r_blocks; i++ ){
        if( i%c_blocks == 0) {
            magma_int_t tmp = i/c_blocks;
            cpu_row[tmp] = num_blocks_tmp;

        }
        if( M->blockinfo[i] != 0 ){
            magma_int_t tmp = i%c_blocks;
            cpu_col[num_blocks_tmp] = tmp;

            num_blocks_tmp++;
        }
    }
    cpu_row[r_blocks] = num_blocks_tmp;
    M->nnz = num_blocks_tmp;

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_rowcolfill+=(magma_wtime()-t_rowcolfill1);
        #endif

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_cpucopy1=magma_wtime();
        #endif



    cublasSetVector( r_blocks+1, sizeof( magma_int_t ), cpu_row, 1, M->row, 1 );            
    cublasSetVector(  M->numblocks, sizeof( magma_int_t ), cpu_col, 1, M->col, 1 );

    magma_free_cpu( cpu_row );
    magma_free_cpu( cpu_col );


        #ifdef ENABLE_TIMER
        magma_device_sync(); t_cpucopy+=(magma_wtime()-t_cpucopy1);
        #endif


    magma_int_t ldda, lddb, lddc, ldwork, lwork;
    magmaDoubleComplex tmp;

    ldda = size_b;//((size_b+31)/32)*32;
    lddb = size_b;//((size_b+31)/32)*32;
    lddc = size_b;//((size_b+31)/32)*32;

    magmaDoubleComplex *dwork;
    ldwork = size_b * magma_get_zgetri_nb( size_b );
    magma_zmalloc( &dwork, ldwork );
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_tr1=magma_wtime();
        #endif

    //----------------------------------------------------------------
    //  LU factorization
    // kij-version

    for( k=0; k<r_blocks; k++){
        //printf("k=%d\n", k);
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_getrf1=magma_wtime();
        #endif
        magma_zgetrf_gpu( size_b, size_b, M(k,k), ldda, ipiv+k*size_b, &info );
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_getrf+=(magma_wtime()-t_getrf1);
        #endif


        #ifdef ENABLE_TIMER
        magma_device_sync(); t_blockinfo1=magma_wtime();
        #endif

        int num_block_rows = 0, kblocks = 0, klblocks = 0, row = 0;
        magmaDoubleComplex **A, **dA, **B, ** dB, **BL, **dBL, **C, **dC;
        magmaDoubleComplex **AIs, **AIIs, **dAIs, **dAIIs, **BIs, **dBIs;
        magmaDoubleComplex m_one = MAGMA_Z_MAKE( -1.0, 0.0);
        magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0);

        for( i = k+1; i< c_blocks; i++)
           if ( Mblockinfo( k, i ) != 0 )
              kblocks++;

        for( i = 0; i< k; i++)
           if ( Mblockinfo( k, i ) != 0 )
              klblocks++;

        for( i = k+1; i< r_blocks; i++)
           if ( Mblockinfo(i , k ) != 0 )
              num_block_rows++;

        // Prepare A (vertical under diagonal block)
        magma_malloc_cpu((void **)& A, num_block_rows*sizeof(magmaDoubleComplex *) );
        magma_malloc(    (void **)&dA, num_block_rows*sizeof(magmaDoubleComplex *) );
        for( i = k+1; i< r_blocks; i++)
           if ( Mblockinfo(i, k) != 0 ){
              A[row] =M(i, k);
              row++;
           }
        cublasSetVector( num_block_rows, sizeof(magmaDoubleComplex *), A, 1, dA, 1 );
        // Prepare B (horizontal right of diagonal block)
        row = 0;
        magma_malloc_cpu((void **)& B, kblocks*sizeof(magmaDoubleComplex *) );
        magma_malloc(    (void **)&dB, kblocks*sizeof(magmaDoubleComplex *) );
        for( int i = k+1; i< c_blocks; i++)
           if ( Mblockinfo( k, i ) != 0 ){
              B[row] = M( k, i);
              row++;
           }
            cublasSetVector( kblocks, sizeof(magmaDoubleComplex *), B, 1, dB, 1 );
        // Prepare BL (horizontal left of diagonal block)
        row = 0;
        magma_malloc_cpu((void **)& BL, klblocks*sizeof(magmaDoubleComplex *) );
        magma_malloc(    (void **)&dBL, klblocks*sizeof(magmaDoubleComplex *) );
        for( int i = 0; i< k; i++)
           if ( Mblockinfo( k, i ) != 0 ){
              BL[row] = M( k, i);
              row++;
           }
            cublasSetVector( klblocks, sizeof(magmaDoubleComplex *), B, 1, dB, 1 );
        // Prepare C (trailing matrix)
        row = 0;
        magma_malloc_cpu((void **)& C, num_block_rows*kblocks*sizeof(magmaDoubleComplex *) );
        magma_malloc(    (void **)&dC, num_block_rows*kblocks*sizeof(magmaDoubleComplex *) );
            
        for( int i = k+1; i< r_blocks; i++)
           if ( Mblockinfo(i, k) != 0 ){
              for( int j = k+1; j<c_blocks; j++)
                 if ( Mblockinfo(k, j) != 0 ){
                    C[row] = M(i, j);
                    row++;
                 }
           }
        cublasSetVector( kblocks*num_block_rows, sizeof(magmaDoubleComplex *), C, 1, dC, 1 );



        #ifdef CUBLASBATCHED
        // AIs and BIs for the batched GEMMs later
        magma_malloc_cpu((void **)&AIs, num_block_rows*kblocks*sizeof(magmaDoubleComplex *));
        magma_malloc_cpu((void **)&BIs, num_block_rows*kblocks*sizeof(magmaDoubleComplex *));
        magma_malloc((void **)&dAIs, num_block_rows*kblocks*sizeof(magmaDoubleComplex *));
        magma_malloc((void **)&dBIs, num_block_rows*kblocks*sizeof(magmaDoubleComplex *));
        for(i=0; i<num_block_rows; i++){
           for(j=0; j<kblocks; j++){
              AIs[j+i*kblocks] = A[i];
              BIs[j+i*kblocks] = B[j];
            }
        }
        cublasSetVector( kblocks*num_block_rows, sizeof(magmaDoubleComplex *), AIs, 1, dAIs, 1 );
        cublasSetVector( kblocks*num_block_rows, sizeof(magmaDoubleComplex *), BIs, 1, dBIs, 1 );
        #endif

        #ifndef CUBLASBATCHED
        // BIs for the batched GEMMs later
        magma_malloc_cpu((void **)&BIs, kblocks*sizeof(magmaDoubleComplex *));
        magma_malloc((void **)&dBIs, kblocks*sizeof(magmaDoubleComplex *));
        for(j=0; j<kblocks; j++){
          BIs[j] = B[j];
        }
        cublasSetVector( kblocks, sizeof(magmaDoubleComplex *), BIs, 1, dBIs, 1 );
        #endif
    
        // AIIs for the batched TRSMs under the factorized block
        magma_malloc_cpu((void **)&AIIs, max(num_block_rows, kblocks)*sizeof(magmaDoubleComplex *));
        magma_malloc((void **)&dAIIs, max(num_block_rows, kblocks)*sizeof(magmaDoubleComplex *));
        for(i=0; i<max(num_block_rows, kblocks); i++){
           AIIs[i] = M(k,k);
        }
        cublasSetVector( max(num_block_rows, kblocks), sizeof(magmaDoubleComplex *), AIIs, 1, dAIIs, 1 );
        




        #ifdef ENABLE_TIMER
        magma_device_sync(); t_blockinfo+=(magma_wtime()-t_blockinfo1);
        #endif

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_rup1=magma_wtime();
        #endif
        // Swap elements on the right before update
        for( j=0; j<kblocks; j++ ){// updates in row
            magmablasSetKernelStream( stream[j%num_streams] );
            magmablas_zlaswpx( size_b, B[j], 1, size_b,
                      1, size_b, ipiv+k*size_b, 1 );
        }// end updates in row
        magmablasSetKernelStream( stream[0] );
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_rup+=(magma_wtime()-t_rup1);
        #endif

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_trsm1=magma_wtime();
        #endif
        // update blocks right
        cublasZtrsmBatched( handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                            CUBLAS_OP_N, CUBLAS_DIAG_UNIT, size_b, size_b, 
                            &c_one, dAIIs, size_b, dBIs, size_b, kblocks );
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_trsm+=(magma_wtime()-t_trsm1);
        #endif



        #ifdef ENABLE_TIMER
        magma_device_sync(); t_lup1=magma_wtime();
        #endif
        // Swap elements on the left
        for( j=0; j<klblocks; j++ ){// swap elements left
            magmablasSetKernelStream( stream[j%num_streams] );
            magmablas_zlaswpx(size_b, BL[j], 1, size_b,
              1, size_b,
              ipiv+k*size_b, 1);                  
        }// end swap elements left
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_lup+=(magma_wtime()-t_lup1);
        #endif
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_trsm1=magma_wtime();
        #endif
        // update blocks below
        cublasZtrsmBatched( handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, 
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, size_b, size_b, 
                            &c_one, dAIIs, size_b, dA, size_b, num_block_rows );
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_trsm+=(magma_wtime()-t_trsm1);
        #endif



        #ifdef ENABLE_TIMER
        magma_device_sync(); t_gemm1=magma_wtime();
        #endif
        
        #ifndef CUBLASBATCHED

        magma_zbcsrluegemm( size_b, num_block_rows, kblocks, dA, dB, dC ); 

        #endif

        #ifdef CUBLASBATCHED
        //-------------------------------------------------------------------------
        // update trailing matrix using cublas batched GEMM
        cublasZgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, size_b, size_b, size_b,
                          &m_one, (const magmaDoubleComplex **) dAIs, size_b, 
                                  (const magmaDoubleComplex **) dBIs, size_b, 
                          &one, dC , size_b, kblocks*num_block_rows);
        magma_free_cpu( AIs );
        magma_free( dAIs );
        // end update trailing matrix using cublas batched GEMM
        //-------------------------------------------------------------------------
        #endif

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_gemm+=(magma_wtime()-t_gemm1);
        #endif


        magma_free( dBIs );
        magma_free( dAIIs );
        magma_free( dA );
        magma_free( dB );
        magma_free( dBL );
        magma_free( dC );
        magma_free_cpu( A );
        magma_free_cpu( B );
        magma_free_cpu( BL );
        magma_free_cpu( C );
        magma_free_cpu( AIIs );
        magma_free_cpu( BIs );


    
    }// end block k
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_tr+=(magma_wtime()-t_tr1);
        #endif

    #ifdef ENABLE_TIMER
    printf("  %.2lf%% %.2lf%% %.2lf%% %.2lf%% %.2lf%% %.2lf%% %.2lf%% %.2lf%% %.2lf%% %.2lf%%   |", 
            t_blockinfo*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_memoryalloc*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_valfill*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_rowcolfill*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_cpucopy*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_getrf*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_rup*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_lup*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_trsm*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_gemm*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_getrf+t_rup+t_lup+t_trsm+t_gemm) );
    #endif
    



    return MAGMA_SUCCESS;
}   /* magma_zilusetup */






magma_int_t
magma_zbcsrlusv( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
           magma_solver_parameters *solver_par, magma_int_t *ipiv ){

    // some useful variables
    magma_int_t size_b = A.blocksize;
    magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     // max number of blocks per row
    magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     // max number of blocks per column

    // set x = b
    cudaMemcpy( x->val, b.val, A.num_rows*sizeof( magmaDoubleComplex ), cudaMemcpyDeviceToDevice );

    //#define ENABLE_TIMER
    #ifdef ENABLE_TIMER
    double t_swap1, t_swap = 0.0;
    double t_forward1, t_forward = 0.0;
    double t_backward1, t_backward = 0.0;
    #endif

    for( int z=0; z<1; z++){

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_swap1=magma_wtime();
        #endif

        // First pivot the RHS
        magma_zbcsrswp( r_blocks, size_b, ipiv, x(0) );

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_swap+=(magma_wtime()-t_swap1);
        magma_device_sync(); t_forward1=magma_wtime();
        #endif

        // forward solve
        magma_zbcsrtrsv( MagmaLower, r_blocks, c_blocks, size_b, 
                         A.val, A.blockinfo, x->val );

        //magma_zprint_gpu( A.num_rows,  1, x->val, A.num_rows );

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_forward+=(magma_wtime()-t_forward1);
        magma_device_sync(); t_backward1=magma_wtime();
        #endif

        // backward solve
        magma_zbcsrtrsv( MagmaUpper, r_blocks, c_blocks, size_b, 
                         A.val, A.blockinfo, x->val );

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_backward+=(magma_wtime()-t_backward1);
        #endif
    }

    #ifdef ENABLE_TIMER
    printf("swap: %.2lf (%.2lf%%), forward trsv: %.2lf (%.2lf%%) backward trsv: %.2lf (%.2lf%%)\n", 
            t_swap, 100.0*t_swap/(t_swap+t_forward+t_backward), 
            t_forward, 100.0*t_forward/(t_swap+t_forward+t_backward), 
            t_backward, 100.0*t_backward/(t_swap+t_forward+t_backward));
    #endif
    

    return MAGMA_SUCCESS;
}
