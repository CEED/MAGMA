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

    cublasStatus stat;


    // fill in information for B
    M->storage_type = A.storage_type;
    M->memory_location = Magma_DEV;
    M->num_rows = A.num_rows;
    M->num_cols = A.num_cols;
    M->blocksize = A.blocksize;
    magma_int_t size_b = A.blocksize;
    magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     // max number of blocks per row
    magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     // max number of blocks per column


    //complete fill-in
    magma_imalloc_cpu( &M->blockinfo, r_blocks * c_blocks );

    for( k=0; k<r_blocks; k++){
        for( j=k+1; j<r_blocks; j++ ){
            Mblockinfo(j,k) = blockinfo(j,k);
            if( (blockinfo(j,k)!=0) ){
                for( i=k+1; i<c_blocks; i++ ){
                    if( (blockinfo(j,i)==0) && (blockinfo(k,i)!=0) )
                        M->blockinfo[j*c_blocks + i] = -1;
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
/*
    for( k=0; k<r_blocks; k++){
        for( j=0; j<c_blocks; j++ ){
            printf("%d  ", M->blockinfo[k*c_blocks + j] );
        }
        printf("\n");
    } 
*/

    M->numblocks = num_blocks_tmp;
    // memory allocation
    stat = cublasAlloc( size_b*size_b*(M->numblocks), sizeof( magmaDoubleComplex ), ( void** )&M->val );
    if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }
    stat = cublasAlloc(  r_blocks+1 , sizeof( magma_int_t ), ( void** )&M->row );
    if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); } 
    stat = cublasAlloc( M->numblocks, sizeof( magma_int_t ), ( void** )&M->col );
    if( ( int )stat != 0 ) {printf("Memory Allocation Error transferring matrix\n"); exit(0); }

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

/*
    //for visu
    magma_int_t *vtmp;
    magma_imalloc_cpu( &vtmp, M->numblocks ); 
    cublasGetVector( M->numblocks , sizeof( magma_int_t ), M->col, 1, vtmp, 1 );
    printf("skp :  ");
    for( magma_int_t i=0; i<M->numblocks; i++)
        printf("%d  ",vtmp[i]);
    printf(" end skp\n");
    free(vtmp);
*/


    magma_int_t *cpu_row, *cpu_col;
    magma_imalloc_cpu( &cpu_row, r_blocks+1 );
    magma_imalloc_cpu( &cpu_col, M->numblocks );
    cublasGetVector( r_blocks+1, sizeof( magma_int_t ), A.row, 1, cpu_row, 1 );            
    cublasGetVector(  M->numblocks, sizeof( magma_int_t ), A.col, 1, cpu_col, 1 );


    magma_int_t ldda, lddb, lddc, ldwork, lwork;
    magmaDoubleComplex tmp;

    ldda = size_b;//((size_b+31)/32)*32;
    lddb = size_b;//((size_b+31)/32)*32;
    lddc = size_b;//((size_b+31)/32)*32;


    magmaDoubleComplex *dwork;
    ldwork = size_b * magma_get_zgetri_nb( size_b );
    magma_zmalloc( &dwork, ldwork );

    // kij-version
    for( k=0; k<r_blocks; k++){

        magma_zgetrf_gpu( size_b, size_b, M(k,k), ldda, ipiv+k*size_b, &info );

        for( j=k+1; j<c_blocks; j++ ){
            if( (blockinfo(k,j)!=0) ){
                // Swap elements on the right before update
                magmablas_zlaswpx(size_b, M(k,j), 1, size_b,
                          1, size_b, ipiv+k*size_b, 1);
                // update
                magma_ztrsm(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
                size_b, size_b, c_one,
                M(k,k), ldda, M(k,j), size_b );
            }
        }
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

        // Swap elements on the left
        for( j=0; j<k; j++ ){
            if( (blockinfo(k,j)!=0) ){
                magmablas_zlaswpx(size_b, M(k,j), 1, size_b,
                  1, size_b,
                  ipiv+k*size_b, 1);                  
            }
        }

        for( i=k+1; i<r_blocks; i++ ){
            if( (blockinfo(i,k)!=0) && (i!=k) ){
                // update blocks below
                magma_ztrsm(MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    size_b, size_b, c_one,
                    M(k,k), ldda, M(i,k), size_b);
                // update the blocks in the respective rows
                for( j=k+1; j<c_blocks; j++ ){
                    if( (blockinfo(k,j)!=0) ){
                        magmablas_zgemm( MagmaNoTrans, MagmaNoTrans, size_b, size_b, size_b,
                                         c_mone, M(i,k), size_b,
                                         M(k,j), size_b, c_one,  M(i,j), size_b );
                    }
                }
            }
        }
    }
    
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
