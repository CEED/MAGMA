/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c


       @author Adrien REMY
*/
#include "common_magma.h"


#define block_height  32
#define block_width  4
#define block_length 256
#define NB 64

__global__ void 
magmablas_zelementary_multiplication(
    magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t ldda, 
    magmaDoubleComplex *du, 
    magmaDoubleComplex *dv)
{    
    magma_int_t idx, idy;

    idx = blockIdx.x * blockDim.x + threadIdx.x;
    idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((idx < n/2)&&(idy < n/2)){

        dA += idx + idy * ldda;

        magmaDoubleComplex a00, a10, a01, a11, b1, b2, b3, b4;
        __shared__ magmaDoubleComplex u1[block_height], u2[block_height], v1[block_width], v2[block_width];

        du += idx;
        dv += idy;

        u1[threadIdx.x]=du[0];
        u2[threadIdx.x]=du[n/2];
        v1[threadIdx.y]=dv[0];
        v2[threadIdx.y]=dv[n/2];

        __syncthreads();

        a00 = dA[0];
        a01 = dA[ldda*n/2];
        a10 = dA[n/2];
        a11 = dA[ldda*n/2+n/2];

        b1 = a00 + a01;
        b2 = a10 + a11;
        b3 = a00 - a01;
        b4 = a10 - a11;

        dA[0] = u1[threadIdx.x] * v1[threadIdx.y] * (b1 + b2);
        dA[ldda*n/2] = u1[threadIdx.x] * v2[threadIdx.y] * (b3 + b4);
        dA[n/2] = u2[threadIdx.x] * v1[threadIdx.y] * (b1 - b2);
        dA[ldda*n/2+n/2] = u2[threadIdx.x] * v2[threadIdx.y] *(b3 - b4);
    }
}


__global__ void 
magmablas_zapply_vector(
    magma_int_t n,
    magmaDoubleComplex *du, magmaDoubleComplex *db)
{
    magma_int_t idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n/2){

        du += idx;
        db += idx;

        magmaDoubleComplex a1,a2;

        a1 = du[0]*db[0];
        a2 = du[n/2]*db[n/2];

        db[0] = a1 + a2;
        db[n/2] = a1 -a2;
    }
}


__global__ void 
magmablas_zapply_transpose_vector(
    magma_int_t n,
    magmaDoubleComplex *du,magmaDoubleComplex *db )
{
    magma_int_t idx;

    idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n/2){

        du += idx;
        db += idx;

        magmaDoubleComplex a1,a2;

        a1 = db[0] + db[n/2];
        a2 = db[0] - db[n/2];

        db[0] = du[0]*a1;
        db[n/2] = du[n/2]*a2;
    }
}


/**
    Purpose
    -------
    ZPRBT_MVT compute B = UTB to randomize B
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of values of db.  n >= 0.

    @param[in]
    du     COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V
    
    @param[in,out]
    db     COMPLEX_16 array, dimension (n)
            The n vector db computed by ZGESV_NOPIV_GPU
            On exit db = du*db
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    ********************************************************************/
extern "C" void
magmablas_zprbt_mtv_q(
    magma_int_t n, 
    magmaDoubleComplex *du, magmaDoubleComplex *db,
    magma_queue_t queue)
{
    /*

     */
    magma_int_t threads = block_length;
    magma_int_t grid = n/(4*block_length) + ((n%(4*block_length))!=0);

    magmablas_zapply_transpose_vector<<< grid, threads, 0, queue >>>(n/2, du+n, db);
    magmablas_zapply_transpose_vector<<< grid, threads, 0, queue >>>(n/2, du+n+n/2, db+n/2);

    threads = block_length;
    grid = n/(2*block_length) + ((n%(2*block_length))!=0);
    magmablas_zapply_transpose_vector<<< grid, threads, 0, queue >>>(n, du, db);
}


/**
    @see magmablas_zprbt_mtv_q
    ********************************************************************/
extern "C" void
magmablas_zprbt_mtv(
    magma_int_t n, 
    magmaDoubleComplex *du, magmaDoubleComplex *db)
{
    magmablas_zprbt_mtv_q(n, du, db, magma_stream);
}



/**
    Purpose
    -------
    ZPRBT_MV compute B = VB to obtain the non randomized solution
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of values of db.  n >= 0.
    
    @param[in,out]
    db      COMPLEX_16 array, dimension (n)
            The n vector db computed by ZGESV_NOPIV_GPU
            On exit db = dv*db
    
    @param[in]
    dv      COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    ********************************************************************/
extern "C" void
magmablas_zprbt_mv_q(
    magma_int_t n, 
    magmaDoubleComplex *dv, magmaDoubleComplex *db,
    magma_queue_t queue)
{

    magma_int_t threads = block_length;
    magma_int_t grid = n/(2*block_length) + ((n%(2*block_length))!=0);

    magmablas_zapply_vector<<< grid, threads, 0, queue >>>(n, dv, db);


    threads = block_length;
    grid = n/(4*block_length) + ((n%(4*block_length))!=0);

    magmablas_zapply_vector<<< grid, threads, 0, queue >>>(n/2, dv+n, db);
    magmablas_zapply_vector<<< grid, threads, 0, queue >>>(n/2, dv+n+n/2, db+n/2);


}



/**
    @see magmablas_zprbt_mtv_q
    ********************************************************************/
extern "C" void
magmablas_zprbt_mv(
    magma_int_t n, 
    magmaDoubleComplex *dv, magmaDoubleComplex *db)
{
    magmablas_zprbt_mv_q(n, dv, db, magma_stream);
}


/**
    Purpose
    -------
    ZPRBT randomize a square general matrix using partial randomized transformation
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of columns and rows of the matrix dA.  n >= 0.
    
    @param[in,out]
    dA      COMPLEX_16 array, dimension (n,ldda)
            The n-by-n matrix dA
            On exit dA = duT*dA*d_V
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDA >= max(1,n).
    
    @param[in]
    du      COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix U
    
    @param[in]
    dv      COMPLEX_16 array, dimension (n,2)
            The 2*n vector representing the random butterfly matrix V
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    ********************************************************************/
extern "C" void 
magmablas_zprbt_q(
    magma_int_t n, 
    magmaDoubleComplex *dA, magma_int_t ldda, 
    magmaDoubleComplex *du, magmaDoubleComplex *dv,
    magma_queue_t queue)
{
    du += ldda;
    dv += ldda;

    dim3 threads(block_height, block_width);
    dim3 grid(n/(4*block_height) + ((n%(4*block_height))!=0), 
            n/(4*block_width)  + ((n%(4*block_width))!=0));

    magmablas_zelementary_multiplication<<< grid, threads, 0, queue >>>(n/2, dA, ldda, du, dv);
    magmablas_zelementary_multiplication<<< grid, threads, 0, queue >>>(n/2, dA+ldda*n/2, ldda, du, dv+n/2);
    magmablas_zelementary_multiplication<<< grid, threads, 0, queue >>>(n/2, dA+n/2, ldda, du+n/2, dv);
    magmablas_zelementary_multiplication<<< grid, threads, 0, queue >>>(n/2, dA+ldda*n/2+n/2, ldda, du+n/2, dv+n/2);

    dim3 threads2(block_height, block_width);
    dim3 grid2(n/(2*block_height) + ((n%(2*block_height))!=0), 
            n/(2*block_width)  + ((n%(2*block_width))!=0));
    magmablas_zelementary_multiplication<<< grid2, threads2, 0, queue >>>(n, dA, ldda, du-ldda, dv-ldda);
}


/**
    @see magmablas_zprbt_q
    ********************************************************************/
extern "C" void 
magmablas_zprbt(
    magma_int_t n, 
    magmaDoubleComplex *dA, magma_int_t ldda, 
    magmaDoubleComplex *du, magmaDoubleComplex *dv)
{
    magmablas_zprbt_q(n, dA, ldda, du, dv, magma_stream);
}



// adds   x += r  --and--
// copies r = b
// each thread does one index, x[i] and r[i]
__global__ void
zaxpycp2_kernel(
    int m, magmaDoubleComplex *r, magmaDoubleComplex *x,
    const magmaDoubleComplex *b)
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_Z_ADD( x[i], r[i] );
        r[i] = b[i];
    }
}


// ----------------------------------------------------------------------
// adds   x += r  --and--
// copies r = b
extern "C" void
magmablas_zaxpycp2_q(
    magma_int_t m, magmaDoubleComplex *r, magmaDoubleComplex *x,
    const magmaDoubleComplex *b,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( (m + NB - 1)/NB );
    zaxpycp2_kernel <<< grid, threads, 0, queue >>> ( m, r, x, b );
}


extern "C" void
magmablas_zaxpycp2(
    magma_int_t m, magmaDoubleComplex *r, magmaDoubleComplex *x,
    const magmaDoubleComplex *b)
{
    magmablas_zaxpycp2_q( m, r, x, b, magma_stream );
}
