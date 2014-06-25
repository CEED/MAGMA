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


#define PRECISION_z



// every row is handled by one threadblock
__global__ void 
magma_zailu_csr_s_kernel(   magma_int_t Lnum_rows, 
                            magma_int_t Lnnz,  
                            magma_index_t *rowidxAL, 
                            magma_index_t *colidxAL,
                            const magmaDoubleComplex * __restrict__ AL, 
                            magma_index_t *rowptrL, 
                            magma_index_t *colidxL, 
                            magmaDoubleComplex *valL, 
                            magma_index_t *rowidxAU, 
                            magma_index_t *colidxAU,
                            const magmaDoubleComplex * __restrict__ AU, 
                            magma_index_t *rowptrU, 
                            magma_index_t *rowidxU, 
                            magmaDoubleComplex *valU ){

    int i, j;
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex s, sp;
    int il, iu, jl, ju;


    if (k < Lnnz)
    {     

        i = (blockIdx.y == 0 ) ? rowidxAL[k] : rowidxAU[k]  ;
        j = (blockIdx.y == 0 ) ? colidxAL[k] : colidxAU[k]  ;

#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        s = (blockIdx.y == 0 ) ? __ldg( AL+k ) : __ldg( AU+k );
#else
        s = (blockIdx.y == 0 ) ? AL[k] : AU[k] ;
#endif

        il = rowptrL[i];
        iu = rowptrU[j];

        while (il < rowptrL[i+1] && iu < rowptrU[j+1])
        {
            sp = zero;
            jl = colidxL[il];
            ju = rowidxU[iu];

            // avoid branching
            sp = ( jl == ju ) ? valL[il] * valU[iu] : sp;
            s = ( jl == ju ) ? s-sp : s;
            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;

        }
        // undo the last operation (it must be the last)
        s += sp;
        __syncthreads();
        // modify u entry
        if (blockIdx.y == 0)
            valL[il-1] =  s / valU[rowptrU[j+1]-1];
        else{
            valU[il-1] = s;
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
                input/output matrix L containing the ILU approximation

    @param
    U           magma_z_sparse_matrix
                input/output matrix U containing the ILU approximation

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zailu_csr_a( magma_z_sparse_matrix A_L,
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
    magma_zailu_csr_s_kernel<<< grid, block, 0, magma_stream >>>
        ( A_L.num_rows, A_L.nnz, 
          A_L.rowidx, A_L.col, A_L.val, 
          L.row, L.col, L.val, 
          A_U.rowidx, A_U.col, A_U.val, 
          U.row, U.col, U.val );


    return MAGMA_SUCCESS;
}
