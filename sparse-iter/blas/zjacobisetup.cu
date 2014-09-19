/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "magmasparse.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif



__global__ void 
zvjacobisetup_gpu(  int num_rows, 
                    int num_vecs,
                    magmaDoubleComplex *b, 
                    magmaDoubleComplex *d, 
                    magmaDoubleComplex *c,
                    magmaDoubleComplex *x){

    int row = blockDim.x * blockIdx.x + threadIdx.x ;

    if(row < num_rows ){
        for( int i=0; i<num_vecs; i++ ){
            c[row+i*num_rows] = b[row+i*num_rows] / d[row];
            x[row+i*num_rows] = c[row+i*num_rows];
        }
    }
}





/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c. It calls a GPU kernel

    Arguments
    ---------

    @param
    num_rows    magma_int_t
                number of rows
                
    @param
    b           magma_z_vector
                RHS b

    @param
    d           magma_z_vector
                vector with diagonal entries

    @param
    c           magma_z_vector*
                c = D^(-1) * b

    @param
    x           magma_z_vector*
                iteration vector

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup_vector_gpu(  int num_rows, 
                                magma_z_vector b, 
                                magma_z_vector d, 
                                magma_z_vector c,
                                magma_z_vector *x ){


   dim3 grid( (num_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
   int num_vecs = b.num_rows / num_rows;

   zvjacobisetup_gpu<<< grid, BLOCK_SIZE, 0 >>>
                ( num_rows, num_vecs, b.val, d.val, c.val, x->val );

   return MAGMA_SUCCESS;
}






__global__ void 
zjacobidiagscal_kernel(  int num_rows,
                         int num_vecs, 
                    magmaDoubleComplex *b, 
                    magmaDoubleComplex *d, 
                    magmaDoubleComplex *c){

    int row = blockDim.x * blockIdx.x + threadIdx.x ;

    if(row < num_rows ){
        for( int i=0; i<num_vecs; i++)
            c[row+i*num_rows] = b[row+i*num_rows] * d[row];
    }
}





/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c. It calls a GPU kernel

    Arguments
    ---------

    @param
    num_rows    magma_int_t
                number of rows
                
    @param
    b           magma_z_vector
                RHS b

    @param
    d           magma_z_vector
                vector with diagonal entries

    @param
    c           magma_z_vector*
                c = D^(-1) * b

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobi_diagscal(         int num_rows, 
                                magma_z_vector d, 
                                magma_z_vector b, 
                                magma_z_vector *c){


   dim3 grid( (num_rows+BLOCK_SIZE-1)/BLOCK_SIZE, 1, 1);
   int num_vecs = b.num_rows/num_rows;

   zjacobidiagscal_kernel<<< grid, BLOCK_SIZE, 0 >>>( num_rows, num_vecs, b.val, d.val, c->val );

   return MAGMA_SUCCESS;
}



