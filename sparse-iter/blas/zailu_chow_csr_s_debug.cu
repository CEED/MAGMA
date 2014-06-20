/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/

#include "common_magma.h"
#include "../include/magmasparse_z.h"
#include "../../include/magma.h"
#include <curand.h>
#include <curand_kernel.h>


#define PRECISION_z



// every row is handled by one threadblock
__global__ void 
magma_zailu_csr_s_debug_kernel(   int *blockidx,
                            magma_int_t Lnum_rows, 
                            magma_int_t Lnnz,  
                            const magmaDoubleComplex * __restrict__ AL, 
                            magmaDoubleComplex *valL, 
                            magma_index_t *rowptrL, 
                            magma_index_t *rowidxL, 
                            magma_index_t *colidxL,
                            magma_int_t Unum_rows, 
                            magma_int_t Unnz,  
                            const magmaDoubleComplex * __restrict__ AU, 
                            magmaDoubleComplex *valU, 
                            magma_index_t *rowptrU, 
                            magma_index_t *rowidxU, 
                            magma_index_t *colidxU ){

    int i, j;
    int k = blockDim.x * blockIdx.x + threadIdx.x ;

    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex s, sp;
    int il, iu, jl, ju;

    //curandState_t state;
    //curand_init(&state);
    //int blockidx = curand(&state) % (gridDim.x);
    k = blockidx[blockIdx.x+gridDim.x * blockIdx.z] * blockDim.x + threadIdx.x ;

    if (k < Lnnz )
    {     

        i = (blockIdx.y%2 == 0 ) ? rowidxL[k] : rowidxU[k]  ;
        j = (blockIdx.y%2 == 0 ) ? colidxL[k] : colidxU[k]  ;

#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        s = (blockIdx.y%2 == 0 ) ? __ldg( AL+k ) : __ldg( AU+k );
#else
        s = (blockIdx.y%2 == 0 ) ? AL[k] : AU[k] ;
#endif

        il = rowptrL[i];
        iu = rowptrU[j];

        while (il < rowptrL[i+1] && iu < rowptrU[j+1])
        {
            sp = zero; 
            jl = colidxL[il];
            ju =  rowidxU[iu];


            // avoid branching
            sp = ( jl == ju ) ? valL[il] * valU[iu] : sp;
            s = ( jl == ju ) ? s-sp : s;
            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;

/*
            if (jl < ju)
                il++;
            else if (ju < jl)
                iu++;
            else
            {
                // we are going to modify this u entry
                sp = valL[il] * valU[iu];
                s -= sp;
                il++;
                iu++;
            }

*/
        }
        // undo the last operation (it must be the last)
        s += sp;
        __syncthreads();
        // modify u entry
        if (blockIdx.y%2 == 0)
            valL[k] =  s / valU[rowptrU[j+1]-1];
        else{
            valU[k] = s;
        }

    }

}// kernel 













/**
    Purpose
    -------
    
    This routine computes the ILU approximation of a matrix iteratively. 
    The idea is according to Edmond Chow's presentation at SIAM 2014.
    The input format of the matrix is Magma_CSRCOO for the upper and lower 
    triangular parts. Note however, that we flip col and rowidx for the 
    U-part.
    Every component of L and U is handled by one thread. 

    Arguments
    ---------

    @param
    A_L         magma_z_sparse_matrix
                input matrix L

    @param
    A_U         magma_z_sparse_matrix
                input matrix U

    @param
    L           magma_z_sparse_matrix
                input/output matrix L

    @param
    U           magma_z_sparse_matrix
                input/output matrix U

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zailu_csr_s_debug( magma_z_sparse_matrix A_L,
                   magma_z_sparse_matrix A_U,
                   magma_z_sparse_matrix L,
                   magma_z_sparse_matrix U ){
    
    int blocksize1 = 256;
    int blocksize2 = 1;

    int dimgrid1 = ( A_L.nnz + blocksize1 -1 ) / blocksize1;
    int dimgrid2 = 2;
    int dimgrid3 = 1;

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );

    // backward engineering
    int limit = dimgrid1*dimgrid3;
    int *blockidx, *d_blockidx;
    magma_imalloc_cpu( &blockidx, limit );
    magma_imalloc( &d_blockidx, limit );
    for(int i=0; i< limit; i++)
     //   blockidx[i] = i%dimgrid1;
        blockidx[i] = rand()%dimgrid1;    //rand
        //blockidx[i] = dimgrid1-1-i%dimgrid1;       //backward
/*

    // random without duplicates: create range then shuffle
    for(int i=0; i< limit; i++)
        blockidx[i] = i%dimgrid1;
    for(int i=0; i< limit; i++){
        int idx1 = rand()%limit;
        int idx2 = rand()%limit;
        int tmp = blockidx[idx1];
        blockidx[idx1] = blockidx[idx2];
        blockidx[idx2] = tmp;
    }*/

    magma_setvector( limit , blockidx, 1, d_blockidx, 1 );
 //   magma_getvector( dimgrid1 , d_blockidx, 1, blockidx, 1 );
   // printf(" %d %d %d\n", blockidx[0], blockidx[5], blockidx[7]);
    // backward engineering

    cudaFuncSetCacheConfig(magma_zailu_csr_s_debug_kernel, cudaFuncCachePreferL1);

    magma_zailu_csr_s_debug_kernel<<< grid, block, 0, magma_stream >>>
        ( d_blockidx, A_L.num_rows, A_L.nnz,  A_L.val, L.val, L.row, L.rowidx, L.col, 
          A_U.num_rows, A_U.nnz,  A_U.val, U.val, U.row, U.col, U.rowidx );

    // backward engineering
    magma_free_cpu( blockidx );
    magma_free( d_blockidx );
    // backward engineering

    return MAGMA_SUCCESS;
}



