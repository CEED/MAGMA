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

    @param[in]
    num_rows    magma_int_t
                number of rows
                
    @param[in]
    b           magma_z_vector
                RHS b

    @param[in]
    d           magma_z_vector
                vector with diagonal entries

    @param[out]
    c           magma_z_vector*
                c = D^(-1) * b

    @param[out]
    x           magma_z_vector*
                iteration vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup_vector_gpu(
    int num_rows, 
    magma_z_vector b, 
    magma_z_vector d, 
    magma_z_vector c,
    magma_z_vector *x,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    int num_vecs = b.num_rows / num_rows;
    magma_int_t threads = BLOCK_SIZE;
    zvjacobisetup_gpu<<< grid, threads, 0 >>>
                ( num_rows, num_vecs, b.dval, d.dval, c.dval, x->val );

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

    @param[in]
    num_rows    magma_int_t
                number of rows
                
    @param[in]
    b           magma_z_vector
                RHS b

    @param[in]
    d           magma_z_vector
                vector with diagonal entries

    @param[out]
    c           magma_z_vector*
                c = D^(-1) * b
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobi_diagscal(
    int num_rows, 
    magma_z_vector d, 
    magma_z_vector b, 
    magma_z_vector *c,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( num_rows, BLOCK_SIZE ));
    int num_vecs = b.num_rows*b.num_cols/num_rows;
    magma_int_t threads = BLOCK_SIZE;
    zjacobidiagscal_kernel<<< grid, threads, 0 >>>( num_rows, num_vecs, b.dval, d.dval, c->val );

    return MAGMA_SUCCESS;
}



