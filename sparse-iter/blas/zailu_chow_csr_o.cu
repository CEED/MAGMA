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
magma_zailu_csr_o_kernel(   magma_int_t num_rows, 
                            magma_int_t nnz,  
                            const magmaDoubleComplex * __restrict__ A, 
                            magmaDoubleComplex *val, 
                            magma_index_t *rowptr, 
                            magma_index_t *rowidx, 
                            magma_index_t *colidx ){

    int i, j;
    int k = blockDim.x * blockIdx.x + threadIdx.x ;

    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex s, sp;
    int il, iu, jl, ju;


    if (k < nnz )
    {     

        i = rowidx[k];
        j = colidx[k];

#if (__CUDA_ARCH__ >= 350) && (defined(PRECISION_d) || defined(PRECISION_s))
        s = __ldg( A+k );
#else
        s = A[k];
#endif

        il = rowptr[i];
        iu = rowptr[j];

        while (il < rowptr[i+1] && iu < rowptr[j+1])
        {
            sp = zero;
            jl = colidx[il];
            ju =  rowidx[iu];

            if( jl==i || ju==j )
                break;
            else if (jl < ju)
                il++;
            else if (ju < jl)
                iu++;
            else
            {
                // we are going to modify this u entry
                sp = val[il] * val[iu];
                s -= sp;
                il++;
                iu++;
            }
        }
        // undo the last operation (it must be the last)
        s += sp;
        __syncthreads();
        // modify u entry
        if ( i<j )
            val[k] =  s / val[iu-1];
        else{
            val[k] = s;
        }

    }

}// kernel 













/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    
    This routine computes the ILU approximation of a matrix iteratively. 
    The idea is according to Edmond Chow's presentation at SIAM 2014.
    The input format of the matrix is Magma_CSRCOO for the upper and lower 
    triangular parts. Note however, that we flip col and rowidx for the 
    U-part.
    Every component of L and U is handled by one thread. 

    Arguments
    =========

    magma_z_sparse_matrix A_L               input matrix L
    magma_z_sparse_matrix A_U               input matrix U
    magma_z_sparse_matrix L                 input/output matrix L
    magma_z_sparse_matrix U                 input/output matrix U

    ======================================================================    */

extern "C" magma_int_t
magma_zailu_csr_o( magma_z_sparse_matrix A,
                   magma_z_sparse_matrix LU ){
    
    int blocksize1 = 256;
    int blocksize2 = 1;

    int dimgrid1 = ( A.nnz + blocksize1 -1 ) / blocksize1;
    int dimgrid2 = 2;
    int dimgrid3 = 5;

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );
    magma_zailu_csr_o_kernel<<< grid, block, 0, magma_stream >>>
        ( A.num_rows, A.nnz,  A.val, LU.val, A.row, A.rowidx, A.col );


    return MAGMA_SUCCESS;
}



