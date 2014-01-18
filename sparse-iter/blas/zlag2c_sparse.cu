/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions mixed zc -> ds

*/
#include "common_magma.h"
#include "../include/magmasparse_z.h"
#include "../include/magmasparse_zc.h"
#include "../../include/magma.h"
#include "../include/mmio.h"
#include "common_magma.h"

#define PRECISION_z
#define BLOCKSIZE 512

#define min(a, b) ((a) < (b) ? (a) : (b))

// TODO get rid of global variable!
__device__ int flag = 0; 

__global__ void 
magmaint_zlag2c_sparse(  int M, int N, 
                  const magmaDoubleComplex *A,
                  magmaFloatComplex *SA ){

    int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; 
                                    // global thread index

    if( thread_id < M ){
        for( int i=0; i<N; i++ ){

            SA[i*M+thread_id] = cuComplexDoubleToFloat( A[i*M+thread_id] );

        }
    } 
}


extern "C" void 
magmablas_zlag2c_sparse( magma_int_t M, magma_int_t N , 
                  const magmaDoubleComplex *A, magma_int_t lda, 
                  magmaFloatComplex *SA,       magma_int_t ldsa, 
                  magma_int_t *info ) 
{    
/*
    Note
    ====
          - We have to provide INFO at the end that zlag2c isn't doable now. 
          - Transfer a single value TO/FROM CPU/GPU
          - SLAMCH that's needed is called from underlying BLAS
          - Only used in iterative refinement
          - Do we want to provide this in the release?
    
    Purpose
    =======
    ZLAG2C converts a COMPLEX_16 matrix A to a COMPLEX
    matrix SA.
    
    RMAX is the overflow for the COMPLEX arithmetic.
    ZLAG2C checks that all the entries of A are between -RMAX and
    RMAX. If not the convertion is aborted and a flag is raised.
        
    Arguments
    =========
    M       (input) INTEGER
            The number of lines of the matrix A.  M >= 0.
    
    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.
    
    A       (input) COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N coefficient matrix A.
    
    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    SA      (output) COMPLEX array, dimension (LDSA,N)
            On exit, if INFO=0, the M-by-N coefficient matrix SA; if
            INFO>0, the content of SA is unspecified.
    
    LDSA    (input) INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,M).
    
    INFO    (output) INTEGER
            = 0:  successful exit.
            < 0:  if INFO = -i, the i-th argument had an illegal value
            = 1:  an entry of the matrix A is greater than the COMPLEX
                  overflow threshold, in this case, the content
                  of SA in exit is unspecified.
    ======================================================================    */

    *info = 0;
    if ( M < 0 )
        *info = -1;
    else if ( N < 0 )
        *info = -2;
    else if ( lda < max(1,M) )
        *info = -4;
    else if ( ldsa < max(1,M) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        //return *info;
    }


    dim3 grid( (M+BLOCKSIZE-1)/BLOCKSIZE, 1, 1);


    cudaMemcpyToSymbol( flag, info, sizeof(flag) );    // flag = 0
    magmaint_zlag2c_sparse<<< grid, BLOCKSIZE, 0, magma_stream >>>
                                        ( M, N, A, SA ) ; 
    cudaMemcpyFromSymbol( info, flag, sizeof(flag) );  // info = flag
}



__global__ void 
magma_zlag2c_CSR_DENSE_kernel( int num_rows, int num_cols, 
                               magmaDoubleComplex *Aval, int *Arow, int *Acol, 
                               magmaFloatComplex *Bval ){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        for( j=0; j<num_cols; j++)
            Bval[ j ] = MAGMA_C_MAKE(0.0, 0.0);
        int start = Arow[ row ];
        int end = Arow[ row+1 ];
        for( j=start; j<end; j++ )
            Bval[ row*num_rows+Acol[j] ] = cuComplexDoubleToFloat( Aval[ j] );
    }
}

__global__ void 
magma_zlag2c_CSR_DENSE_kernel_1( int num_rows, int num_cols, 
                                 magmaFloatComplex *Bval ){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        for( j=0; j<num_cols; j++)
            Bval[ j ] = MAGMA_C_MAKE(0.0, 0.0);
    }
}

__global__ void 
magma_zlag2c_CSR_DENSE_kernel_2( int num_rows, int num_cols, 
                               magmaDoubleComplex *Aval, int *Arow, int *Acol, 
                               magmaFloatComplex *Bval ){

    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        int start = Arow[ row ];
        int end = Arow[ row+1 ];
        for( j=start; j<end; j++ )
            Bval[ row*num_rows+Acol[j] ] = cuComplexDoubleToFloat( Aval[ j] );
    }
}




extern "C" void 
magma_zlag2c_CSR_DENSE(       magma_z_sparse_matrix A, 
                              magma_c_sparse_matrix *B ){

    cublasStatus stat;

    if( A.memory_location == Magma_DEV && A.storage_type == Magma_CSR){
        B->storage_type = Magma_DENSE;
        B->memory_location = A.memory_location;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
        B->nnz = A.nnz;
        stat = cublasAlloc( A.num_rows* A.num_cols, 
                sizeof( magmaFloatComplex ), ( void** )&B->val );
        if( ( int )stat != 0 ) 
        {printf("Memory Allocation Error converting matrix\n"); exit(0); }
        
        dim3 Bs( BLOCKSIZE );
        dim3 Gs( (A.num_rows+BLOCKSIZE-1)/BLOCKSIZE );

        magma_zlag2c_CSR_DENSE_kernel<<< Bs, Gs, 0, magma_stream >>>
        ( A.num_rows, A.num_cols, A.val, A.row, A.col, B->val );
    }
}



extern "C" void 
magma_zlag2c_CSR_DENSE_alloc( magma_z_sparse_matrix A, 
                              magma_c_sparse_matrix *B ){

    cublasStatus stat;

    if( A.memory_location == Magma_DEV && A.storage_type == Magma_CSR){
        B->storage_type = Magma_DENSE;
        B->memory_location = A.memory_location;
        B->num_rows = A.num_rows;
        B->num_cols = A.num_cols;
        B->nnz = A.nnz;
        stat = cublasAlloc( A.num_rows* A.num_cols, 
                sizeof( magmaFloatComplex ), ( void** )&B->val );
        if( ( int )stat != 0 ) 
        {printf("Memory Allocation Error converting matrix\n"); exit(0); }
        
        dim3 Bs( BLOCKSIZE );
        dim3 Gs( (A.num_rows+BLOCKSIZE-1)/BLOCKSIZE );

        magma_zlag2c_CSR_DENSE_kernel_1<<< Bs, Gs, 0, magma_stream >>>
        ( A.num_rows, A.num_cols, B->val );
    }
}


extern "C" void 
magma_zlag2c_CSR_DENSE_convert( magma_z_sparse_matrix A, 
                                magma_c_sparse_matrix *B ){


    if( B->memory_location == Magma_DEV && B->storage_type == Magma_DENSE){
        dim3 Bs( BLOCKSIZE );
        dim3 Gs( (A.num_rows+BLOCKSIZE-1)/BLOCKSIZE );

        magma_zlag2c_CSR_DENSE_kernel_2<<< Bs, Gs, 0, magma_stream >>>
        ( A.num_rows, A.num_cols, A.val, A.row, A.col, B->val );
    }
}
