/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @precisions normal s

       @author Tingxing Dong
       @author Azzam Haidar

*/
#include "common_magma.h"
#include "commonblas_z.h"
#include "magma_templates.h"

#define PRECISION_z

#include "gemv_template_kernel_batched.cuh"
#include "gemv_config/gemvn_param.h"
#include "gemv_config/gemvt_param.h"
#define version(s,v) s ## _V_ ## v




/**
    Purpose
    -------
    ZGEMV performs one of the matrix-vector operations
    
        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,
    
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ----------
    @param[in]
    trans   magma_trans_t
            On entry, TRANS specifies the operation to be performed as
            follows:
      -     = MagmaNoTrans:    y := alpha*A  *x + beta*y
      -     = MagmaTrans:      y := alpha*A^T*x + beta*y
      -     = MagmaConjTrans:  y := alpha*A^H*x + beta*y

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of the matrix A
 
    @param[in]
    alpha   COMPLEX_16
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    dA      COMPLEX_16 array of dimension ( LDDA, n ) on the GPU.
   
    @param[in]
    ldda    INTEGER
            LDDA specifies the leading dimension of A.

    @param[in]
    dx      COMPLEX_16 array of dimension
            n if trans == MagmaNoTrans
            m if trans == MagmaTrans or MagmaConjTrans
     
    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.
  
    @param[in]
    beta    DOUBLE REAL
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy      DOUBLE PRECISION array of dimension
            m if trans == MagmaNoTrans
            n if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incy    Specifies the increment for the elements of Y.
            INCY must not be zero.

    @ingroup magma_dblas2
    ********************************************************************/

extern "C" void
magmablas_zgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA_array[], magma_int_t ldda, 
    magmaDoubleComplex_ptr dx_array[], magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy_array[], magma_int_t incy, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -6;
    else if ( incx == 0 )
        info = -8;
    else if ( incy == 0 )
        info = -11;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }    
/*
    gemvn_template_batched<magmaDoubleComplex, version(N, 70)>             
                 ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );

     return;
*/
    if ( trans == MagmaNoTrans ) {                                                   
        if(max(m, n) <= 96){ // small size                         
            if(m <n){ // Fat matrix
                if ( m<=8) 
                {    
                    gemvn_template_batched<magmaDoubleComplex, version(N, 72)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );

                }
                else if( m<=32) 
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 100)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else if( m<=64)            
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 121)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 132)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }  
            }else{   // Tall or square matrix
                if ( n<=16) 
                {    
                    gemvn_template_batched<magmaDoubleComplex, version(N, 129)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else if( n<=64)            
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 131)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 132)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }  
            }
        }
        else{ // big size
            if(m < n){ // Fat matrix
                if(m <= 16)
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 72)>              
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else if(m <= 32)
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 100)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else if(m <= 64)
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 116)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 133)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
            }
            else{ // Tall or square matrix
                if(m <= 256)
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 137)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else
                {
                    gemvn_template_batched<magmaDoubleComplex, version(N, 140)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
            }
        }// big size        
     }else {

        magma_int_t CONJA = -1;

        if ( trans == MagmaConjTrans ) {
            CONJA = 1;
        }else if( trans == MagmaTrans ) {
            CONJA = 0;
        }
        else{
            return ;
        }      
                                                
        if(max(m, n) <= 96){// small size                 
            if(n <= 16)
            {
                gemvc_template_batched<magmaDoubleComplex, version(T, 42)>             
                        (m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, CONJA, batchCount, queue );
            }
            else
            {
                gemvc_template_batched<magmaDoubleComplex, version(T, 46)>             
                        (m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, CONJA, batchCount, queue );
            }  
        }
        else{ // big size
            if(m <= n) // Fat or square matrix
            {    
                if(m <=64)
                {
                    gemvc_template_batched<magmaDoubleComplex, version(T, 47)>             
                        (m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, CONJA, batchCount, queue );
                }
                else
                {
                    gemvc_template_batched<magmaDoubleComplex, version(T, 46)>             
                        (m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, CONJA, batchCount, queue );
                }

            }                           
            else// (m > n) Tall matrix
            {
            
                if(n <= 8)
                {
                    gemvc_template_batched<magmaDoubleComplex, version(T, 130)>             
                        (m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, CONJA, batchCount, queue );
                }
                else
                {
                    gemvc_template_batched<magmaDoubleComplex, version(T, 46)>             
                        (m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, CONJA, batchCount, queue );
                }
            }
        }        
     }                   

}
///////////////////////////////////////////////////////////////////////////////////////////////////
