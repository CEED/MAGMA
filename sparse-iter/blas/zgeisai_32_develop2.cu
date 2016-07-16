/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "common_magmasparse.h"
#include <cuda_profiler_api.h>

#define PRECISION_z
#define COMPLEX
#define BLOCKSIZE 32
#define WARP_SIZE 32
#define WRP 32
#define WRQ 4

#include <cuda.h>  // for CUDA_VERSION

#if (CUDA_VERSION <= 6000) // this won't work, just to have something...
// CUDA 6.5 adds Double precision version; here's an implementation for CUDA 6.0 and earlier.
// from https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
__device__ inline
real_Double_t __shfl(real_Double_t var, unsigned int srcLane, int width=32) {
  int2 a = *reinterpret_cast<int2*>(&var);
  a.x = __shfl(a.x, srcLane, width);
  a.y = __shfl(a.y, srcLane, width);
  return *reinterpret_cast<double*>(&a);
}
#endif
         


__global__ void                                                                     
magma_zlowerisai_a32_kernel(                                                         
magma_int_t num_rows,                                                                    
const magma_index_t * __restrict__ Arow,                                            
const magma_index_t * __restrict__ Acol,                                            
const magmaDoubleComplex * __restrict__ Aval,                                       
magma_index_t *Mrow,                                                                
magma_index_t *Mcol,                                                                
magmaDoubleComplex *Mval )                                                          
{                                                                                   
#ifdef REAL                                                                         
    int tid = threadIdx.x;                                                          
    int row = blockIdx.y * gridDim.x + blockIdx.x; 
    //int row = blockDim.x * blockIdx.x + threadIdx.y; 
    
    if( row > num_rows )
        return;
    
    // only if within the size                                                      
    int mstart = Mrow[ row ];                                                       
    int mlim = Mrow[ row+1 ];  
    
    if( tid >= mlim-mstart )
        return;
                                                                                    
    magmaDoubleComplex rB;      // registers for trsv                               
    magmaDoubleComplex rA[ 32 ];  // registers for trisystem 
    //magmaDoubleComplex rAT[ 32 ];  // registers for trisystem 
    //extern __shared__ magmaDoubleComplex dA[ 32 ];
                                                                                    
    // set rA to 0                                                                  
    for( int j = 0; j < 32; j++ ){                                                  
        rA[ j ] = MAGMA_Z_ZERO;                                                     
    }                    
    __syncthreads();
                                                                                    
    // generate the triangular systems                                              
    #pragma unroll                                                                  
    for( int j = 0; j < mlim-mstart; j++ ){                                                  
        int t = Mcol[ mstart + j ];                                                 
        int k = Arow[ t ];                                                          
        int alim = Arow[ t+1 ];                                                     
        int l = mstart;                                                             
        int idx = 0;                                                                
        while( k < alim && l < mlim ){ // stop once this column is done             
            int mcol =  Mcol[ l ];                                                  
            int acol = Acol[k];                                                     
            if( mcol == acol ){ //match                                             
                if( idx == tid ){
                    rA[ j ] = Aval[ k ];        
                }                                         
                k++;                                                                
                l++;                                                                
                idx++;                                                              
            } else if( acol < mcol ){// need to check next element                  
                k++;                                                                
            } else { // element does not exist, i.e. l < LC.col[k]                  
                l++; // check next elment in the sparsity pattern                   
                idx++; // leave this element equal zero                             
            }                                                                       
        }                                                                           
    }                   
   // __syncthreads();
   // for( int z=0; z<mlim-mstart; z++){
   //     dA[ tid ] = rA[ z ];
   //     __syncthreads();
   //     if( tid == z ){
   //         for(int k=0; k<mlim-mstart; k++){
   //             rAT[ k ] = dA[ k ];
   //         }
   //     }
   //     __syncthreads();
   // }
    
    // second: solve the triangular systems - in registers                          
    // we know how RHS looks like                                                   
    rB = ( tid == 0 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO; 
    __syncthreads();
                                                                                    
    // Triangular solve in regs.                                                    
    #pragma unroll                                                                  
    for( int k = 0; k < mlim-mstart; k++ )                                                   
    {                                                                               
        if( k%WARP_SIZE == tid )                                                   
            rB /= rA[k];                                                            
        magmaDoubleComplex top = __shfl(rB, k%WARP_SIZE );                         
        if ( tid > k)                                                               
            rB -= (top*rA[k]);                                                      
    }          
    __syncthreads();
    // Drop B to dev memory - in ISAI preconditioner M                              
    Mval[ mstart + tid ] = rB;                                                      
                                                                                    
#endif                                                                              
                                                                                    
}// kernel for size                                                                                     
                                                                                         
                                                                                    
__global__ void                                                                     
magma_zupperisai_a32_kernel(                                                         
magma_int_t num_rows,                                                                    
const magma_index_t * __restrict__ Arow,                                            
const magma_index_t * __restrict__ Acol,                                            
const magmaDoubleComplex * __restrict__ Aval,                                       
magma_index_t *Mrow,                                                                
magma_index_t *Mcol,                                                                
magmaDoubleComplex *Mval )                                                          
{          
#ifdef REAL                                                                         
    int tid = threadIdx.x;  
    int row = blockIdx.y * gridDim.x + blockIdx.x; 
    //int row = blockDim.x * blockIdx.x + threadIdx.y; 
    

    
    if( row >= num_rows )
        return;
    
    // only if within the size                                                      
    int mstart = Mrow[ row ];                                                       
    int mlim = Mrow[ row+1 ];  

    if( tid >= mlim-mstart )
        return;
    
    magmaDoubleComplex rB;      // registers for trsv                               
    magmaDoubleComplex dA[ 32 ];  // registers for trisystem     
    magmaDoubleComplex rA;
     
    // set dA to 0        
    #pragma unroll 
     for( int j = 0; j < 32; j++ ){                                                  
         dA[ j ] = MAGMA_Z_ZERO;    
     } 
    
    // generate the triangular systems                                              
    int t = Mcol[ mstart + tid ]; 
    int k = Arow[ t ];                                                          
    int alim = Arow[ t+1 ];                                                     
    int l = mstart;                                                             
    int idx = 0;                                                                
    while( k < alim && l < mlim ){ // stop once this column is done             
        int mcol =  Mcol[ l ];                                                  
        int acol = Acol[k];                                                     
        if( mcol == acol ){ //match   
             // if( idx == tid ){
             //     dA[ j ] = Aval[ k ];        
             // }
            dA[ idx ] = Aval[ k ];      
            // just to check whether this would give speedup
            // dA[ idx ] = Aval[ k ];       
            k++;                                                                
            l++;                                                                
            idx++;                                                              
        } else if( acol < mcol ){// need to check next element                  
            k++;                                                                
        } else { // element does not exist, i.e. l < LC.col[k]                  
            l++; // check next elment in the sparsity pattern                   
            idx++; // leave this element equal zero                             
        }                                                                       
    }                                                                           
    
                                                                                  
    // second: solve the triangular systems - in registers                          
    // we know how RHS looks like                                                   
    rB = ( tid == mlim-mstart-1 ) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;                              

    
        // Triangular solve in regs.                                                                    
    #pragma unroll                                                                                  
    for (int k = mlim-mstart-1; k >-1; k--)                                                                  
    {                                                                                               
        rA = dA[ k ];                                                                   
         if (k%WARP_SIZE == tid)                                                                     
             rB /= rA;                                                                               
         magmaDoubleComplex bottom = __shfl(rB, k%(mlim-mstart));                                       
         if ( tid < k)                                                                               
             rB -= (bottom*rA);                                                                      
    }   
    
    // Drop B to dev memory - in ISAI preconditioner M                              
    Mval[ mstart + tid ] = rB;    
    
#endif                                                                              
                                                                                    
}// kernel for size                                                                                     
                                                                                    
                                                                                               
                                                                                                                                                                                                                

/**
    Purpose
    -------
    This routine is designet to combine all kernels into one.

    Arguments
    ---------
    

    @param[in]
    uplotype    magma_uplo_t
                lower or upper triangular
                
    @param[in]
    transtype   magma_trans_t
                possibility for transposed matrix
                
    @param[in]
    diagtype    magma_diag_t
                unit diagonal or not
                
    @param[in]
    L           magma_z_matrix
                triangular factor for which the ISAI matrix is computed.
                Col-Major CSR storage.
                
    @param[in,out]
    M           magma_z_matrix*
                SPAI preconditioner CSR col-major
                
    @param[out]
    sizes       magma_int_t*
                Number of Elements that are replaced.
                
    @param[out]
    locations   magma_int_t*
                Array indicating the locations.
                
    @param[out]
    trisystems  magmaDoubleComplex*
                trisystems
                
    @param[out]
    rhs         magmaDoubleComplex*
                right-hand sides

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zisai_generator_regs2(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix L,
    magma_z_matrix *M,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,    
    magma_queue_t queue )
{
    magma_int_t info = 0;  
    
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
    
    
    // routine 1
    int r1bs1 = 1;
    int r1bs2 = 1;
    int r1dg1 = min( int( sqrt( double( M->num_rows ))), 65535 );
    int r1dg2 = min(magma_ceildiv( M->num_rows, r1dg1 ), 65535);
    int r1dg3 = magma_ceildiv( M->num_rows, r1dg1*r1dg2 );
    //printf(" grid: %d x %d x %d\n", r1dg1, r1dg2, r1dg3 );
    dim3 r1block( r1bs1, r1bs2, 1 );
    dim3 r1grid( r1dg1, r1dg2, r1dg3 );
    
    int r2bs1 = 256;
    int r2bs2 = 1;
    int r2dg1 = magma_ceildiv( L.num_rows, r2bs1 );
    int r2dg2 = 1;
    int r2dg3 = 1;
    dim3 r2block( r2bs1, r2bs2, 1 );
    dim3 r2grid( r2dg1, r2dg2, r2dg3 );
    
    int r3bs1 = 32;
    int r3bs2 = 1;
    int r3dg1 = min( int( sqrt( double( M->num_rows ))), 65535 );
    int r3dg2 = min(magma_ceildiv( M->num_rows, r1dg1 ), 65535);
    int r3dg3 = magma_ceildiv( M->num_rows, r1dg1*r1dg2 );
    //printf(" grid: %d x %d x %d\n", r1dg1, r1dg2, r1dg3 );
    dim3 r3block( r3bs1, r3bs2, 1 );
    dim3 r3grid( r3dg1, r3dg2, r3dg3 );
    
    int r4bs1 = 32;
    int r4bs2 = 256;
    int r4dg1 = magma_ceildiv( L.num_rows, r4bs2 );
    int r4dg2 = 1;
    int r4dg3 = 1;
    //printf(" grid: %d x %d x %d\n", r1dg1, r1dg2, r1dg3 );
    dim3 r4block( r4bs1, 1, 1 );
    dim3 r4grid( r4dg1, r4dg2, r4dg3 );
    
    
    if( uplotype == MagmaLower ){//printf("in here lower new kernel\n");
        cudaProfilerStart();
        magma_zlowerisai_a32_kernel<<< r3grid, r3block, 0, queue->cuda_stream() >>>(                                                         
            L.num_rows,                                                                                   
            L.row,
            L.col,
            L.val,
            M->row,
            M->col,
            M->val ); 
         cudaProfilerStop(); 
         //exit(-1);
    } else {// printf("in here upper new kernel\n");
        magma_zupperisai_a32_kernel<<< r3grid, r3block, 0, queue->cuda_stream() >>>(                                                         
            L.num_rows,                                                                                   
            L.row,
            L.col,
            L.val,
            M->row,
            M->col,
            M->val ); 
    }
        
    
    return info;
}

