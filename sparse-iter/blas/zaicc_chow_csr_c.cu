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

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE  64


#define PRECISION_z



// every row is handled by one threadblock
__global__ void 
magma_zaic_csr_c_kernel( magma_int_t num_rows, 
                         magma_int_t nnz,  
                         magmaDoubleComplex *A_val, 
                         magmaDoubleComplex *val,
                         magma_index_t *rowptr, 
                         magma_index_t *rowidx, 
                         magma_index_t *colidx,
                         magmaDoubleComplex *val_n ){

    int i, j;
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    magmaDoubleComplex s;
    int il, iu, jl, ju;

    if (k < nnz)
    {     


        i = rowidx[k];
        j = colidx[k];
        s = A_val[k];



        il = rowptr[i];
        iu = rowptr[j];

        while (il < rowptr[i+1] && iu < rowptr[j+1])
        {
            jl = colidx[il];
            ju = colidx[iu];

            if (jl < ju)
                il++;
            else if (ju < jl)
                iu++;
            else
            {
                // we are going to modify this u entry
                s -= val[il] * val[iu];
                il++;
                iu++;
            }
        }

        // undo the last operation (it must be the last)
        s += val[il-1]*val[iu-1];

        // modify u entry
        if (i == j){
            val[k] = MAGMA_Z_MAKE(sqrt(abs(MAGMA_Z_REAL(s))), 0.0);// MAGMA_Z_MAKE((double)i, 0.0);//MAGMA_Z_MAKE(sqrtf(abs(MAGMA_Z_REAL(s))), 0.0);
       // printf("process element %d at %d , %d , sqrt(%.2e -%.2e) = %.2e\n", k, i, j, A_val[k],  val[k] );
//if( MAGMA_Z_REAL( val_n[k])> 10e+30)
    //    printf("error:large element:%.2e \n",  val_n[k]);
}
        else{
            val[k] =  s / val[iu-1]; //MAGMA_Z_MAKE((double)k, 0.0);//s / val[iu-1];
//if( MAGMA_Z_REAL(val[iu-1])< 0.0000000000000001 && MAGMA_Z_REAL(val[iu-1])> -0.0000000000000000000001)
   //     printf("error: division by zero!  %.2e\n",val[iu-1]);
}
    }

}// kernel 













/**
    Purpose
    -------
    
    This routine computes the ILU approximation of a matrix iteratively. 
    The idea is according to Edmond Chow's presentation at SIAM 2014.
    The input format of the matrix is Magma_ELLDD. In the same matrix, the
    ILU approximation will be returned. 
    The approach is to store matrix L and U as ELL but(!) L as row major, 
    U as col major.
    an additional array is needed to store infomation (0,1) whether this entry
    is within the uper/lower triangle.

    Arguments
    ---------

    @param
    num_rows    magma_int_t
                number of rows

    @param
    num_vecs    magma_int_t
                number of vectors

    @param
    shift       magma_int_t
                shift number

    @param
    x           magmaDoubleComplex*
                input/output vector x


    @ingroup magmasparse_zsgpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zaic_csr_c( magma_z_sparse_matrix A,
                 magma_z_sparse_matrix A_CSR ){



    
    int blocksize1 = 256;
    int blocksize2 = 1;

    int dimgrid1 = ( A.nnz + blocksize1 -1 ) / blocksize1;
    int dimgrid2 = 1;
    int dimgrid3 = 1;

    magmaDoubleComplex *val_n;
    magma_zmalloc( &val_n, A.nnz);

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );
    magma_zaic_csr_c_kernel<<< grid, block, 0, magma_stream >>>
            ( A.num_rows, A.nnz,  A.val, A_CSR.val, A_CSR.row, A_CSR.blockinfo,  A_CSR.col, val_n );

    //magma_zcopyvector( A.nnz, val_n, 1, A_CSR.val, 1 );
    magma_free( val_n );
    return MAGMA_SUCCESS;
}



