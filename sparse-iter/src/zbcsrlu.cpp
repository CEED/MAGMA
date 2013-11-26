/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Hartwig Anzt 

       @precisions normal z -> s d c
*/

#include "common_magma.h"
#include "../include/magmasparse.h"

#define RTOLERANCE     10e-16
#define ATOLERANCE     10e-16

#define  blockinfo(i,j)  A.blockinfo[(i)*c_blocks   + (j)]
#define  Mblockinfo(i,j)  M->blockinfo[(i)*c_blocks   + (j)]
#define M(i,j) M->val+((Mblockinfo(i,j)-1)*size_b*size_b)
#define A(i,j) A.val+((blockinfo(i,j)-1)*size_b*size_b)
#define x(i) x->val+(i*size_b)

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

    // GPU stream
    int num_streams = 16;
    magma_queue_t stream[num_streams];
    for( i=0; i<num_streams; i++ )
        magma_queue_create( &stream[i] );

    cublasStatus stat;
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
         //   Mblockinfo(j,k) = blockinfo(j,k);
            if( (Mblockinfo(j,k)!=0) ){
                for( i=k+1; i<c_blocks; i++ ){
                    if( (Mblockinfo(j,i)==0) && (Mblockinfo(k,i)!=0) ){
                        Mblockinfo(j,i) = -1;
                        //printf("working on row %d -> insert at %d %d\n",k, j,i);
                    }
                }
            }
        }
    }  
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
*/

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_memoryalloc1=magma_wtime();
        #endif
    M->numblocks = num_blocks_tmp;
    // memory allocation
    stat = cublasAlloc( size_b*size_b*(M->numblocks), sizeof( magmaDoubleComplex ), ( void** )&M->val );
    if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
    stat = cublasAlloc(  r_blocks+1 , sizeof( magma_int_t ), ( void** )&M->row );
    if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
    stat = cublasAlloc( M->numblocks, sizeof( magma_int_t ), ( void** )&M->col );
    if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_memoryalloc+=(magma_wtime()-t_memoryalloc1);
        #endif

        #ifdef ENABLE_TIMER
        magma_device_sync(); t_valfill1=magma_wtime();
        #endif


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

/*

    magma_int_t *Ablockinfo, *Mblockinfo;
    magma_imalloc( &Ablockinfo, (r_blocks*c_blocks) );
    magma_imalloc( &Mblockinfo, (r_blocks*c_blocks) );  
    cudaMemcpy( Ablockinfo, A.blockinfo, r_blocks*c_blocks*sizeof( magma_int_t ), cudaMemcpyHostToDevice );
    cudaMemcpy( Mblockinfo, M->blockinfo, r_blocks*c_blocks*sizeof( magma_int_t ), cudaMemcpyHostToDevice );
    magma_zbcsrvalcpy(  r_blocks, c_blocks, size_b, Ablockinfo, Mblockinfo, A.val, M->val ); 

*/
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_valfill+=(magma_wtime()-t_valfill1);
        #endif
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_rowcolfill1=magma_wtime();
        #endif
    num_blocks_tmp=0;
    for( i=0; i<c_blocks * r_blocks; i++ ){
        if( i%c_blocks == 0) {
            magma_int_t tmp = i/c_blocks;
            cudaMemcpy( M->row+tmp, &num_blocks_tmp, sizeof( magma_int_t ), cudaMemcpyHostToDevice );
            //M->row[i/c_blocks] = num_blocks_tmp;
        }
        if( M->blockinfo[i] != 0 ){
            magma_int_t tmp = i%c_blocks;
            cudaMemcpy( M->col+num_blocks_tmp, &tmp, sizeof( magma_int_t ), cudaMemcpyHostToDevice );
            num_blocks_tmp++;
            //B->col[num_blocks_tmp] = i%c_blocks;
        }
    }
    cudaMemcpy( M->row+r_blocks, &num_blocks_tmp, sizeof( magma_int_t ), cudaMemcpyHostToDevice );
    M->nnz = num_blocks_tmp;
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_rowcolfill+=(magma_wtime()-t_rowcolfill1);
        #endif


        #ifdef ENABLE_TIMER
        magma_device_sync(); t_cpucopy1=magma_wtime();
        #endif
    magma_int_t *cpu_row, *cpu_col;
    magma_imalloc_cpu( &cpu_row, r_blocks+1 );
    magma_imalloc_cpu( &cpu_col, M->numblocks );
    cublasGetVector( r_blocks+1, sizeof( magma_int_t ), A.row, 1, cpu_row, 1 );            
    cublasGetVector(  M->numblocks, sizeof( magma_int_t ), A.col, 1, cpu_col, 1 );
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
    // kij-version
    for( k=0; k<r_blocks; k++){
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_getrf1=magma_wtime();
        #endif
        magma_zgetrf_gpu( size_b, size_b, M(k,k), ldda, ipiv+k*size_b, &info );
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_getrf+=(magma_wtime()-t_getrf1);
        #endif
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_rup1=magma_wtime();
        #endif
        for( j=k+1; j<c_blocks; j++ ){
            if( (Mblockinfo(k,j)!=0) ){
                magmablasSetKernelStream( stream[j%num_streams] );
                // Swap elements on the right before update
                magmablas_zlaswpx(size_b, M(k,j), 1, size_b,
                          1, size_b, ipiv+k*size_b, 1);
                // update
                magma_ztrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                size_b, size_b, c_one,
                M(k,k), ldda, M(k,j), size_b );
            }
        }
        magmablasSetKernelStream( stream[0] );
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_rup+=(magma_wtime()-t_rup1);
        #endif
        // ------ in one routine (merged blocks)------
    /*
        magma_int_t count = 0;
        for( j=k+1; j<c_blocks; j++ ){
            if( (blockinfo(k,j)!=0) ){
                   count++;
            }
        }
           // Swap elements on the right before update
        magmablas_zlaswpx(size_b*count, M(k,j), 1, size_b*count,
                  1, size_b, ipiv+k*size_b, 1);
        // update          
        magma_ztrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
        size_b, size_b, c_one,
        M(k,k), ldda, M(k,j), size_b );
    */
        // ------- in one routine --------------------
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_lup1=magma_wtime();
        #endif
        // Swap elements on the left
        for( j=0; j<k; j++ ){
            if( (Mblockinfo(k,j)!=0) ){
                magmablasSetKernelStream( stream[j%num_streams] );
                magmablas_zlaswpx(size_b, M(k,j), 1, size_b,
                  1, size_b,
                  ipiv+k*size_b, 1);                  
            }
        }
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_lup+=(magma_wtime()-t_lup1);
        #endif
        for( i=k+1; i<r_blocks; i++ ){
            if( (Mblockinfo(i,k)!=0) && (i!=k) ){
                // update blocks below
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_trsm1=magma_wtime();
        #endif
                magma_ztrsm(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    size_b, size_b, c_one,
                    M(k,k), ldda, M(i,k), size_b);
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_trsm+=(magma_wtime()-t_trsm1);
        #endif
                // update the blocks in the respective rows
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_gemm1=magma_wtime();
        #endif
                for( j=k+1; j<c_blocks; j++ ){
                    if( (Mblockinfo(k,j)!=0) ){
                        magmablasSetKernelStream( stream[j%num_streams] );
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, size_b, size_b, size_b,
                                         c_mone, M(i,k), size_b,
                                         M(k,j), size_b, c_one,  M(i,j), size_b );
                    }
                }
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_gemm+=(magma_wtime()-t_gemm1);
        #endif

            }
        }
    }
        #ifdef ENABLE_TIMER
        magma_device_sync(); t_tr+=(magma_wtime()-t_tr1);
        #endif
/*
    #ifdef ENABLE_TIMER
    printf("blockinfo: %.2lf (%.2lf%%)  alloc: %.2lf (%.2lf%%)  val: %.2lf (%.2lf%%)  row+col: %.2lf (%.2lf%%)  cpucopy: %.2lf (%.2lf%%)  tr: %.2lf (%.2lf%%)\n", 
            t_blockinfo, t_blockinfo*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_tr), 
            t_memoryalloc, t_memoryalloc*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_tr), 
            t_valfill, t_valfill*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_tr), 
            t_rowcolfill, t_rowcolfill*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_tr), 
            t_cpucopy, t_cpucopy*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_tr), 
            t_tr, t_tr*100.0/(t_blockinfo+t_memoryalloc+t_valfill+t_rowcolfill+t_cpucopy+t_tr) );
    #endif
    #ifdef ENABLE_TIMER
    printf("getrf: %.2lf (%.2lf%%)  rup: %.2lf (%.2lf%%)  lup: %.2lf (%.2lf%%)  trsm: %.2lf (%.2lf%%)  gemm: %.2lf (%.2lf%%)\n", 
            t_getrf, t_getrf*100.0/(t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_rup, t_rup*100.0/(t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_lup, t_lup*100.0/(t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_trsm, t_trsm*100.0/(t_getrf+t_rup+t_lup+t_trsm+t_gemm), 
            t_gemm, t_gemm*100.0/(t_getrf+t_rup+t_lup+t_trsm+t_gemm) );
    #endif
    */
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
    
    //magma_z_mvisu( *M );

    free(cpu_row);
    free(cpu_col);

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
