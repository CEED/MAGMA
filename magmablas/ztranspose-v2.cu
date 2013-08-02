/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

       @author Stan Tomov
*/
#include "common_magma.h"
#define PRECISION_z
#include "commonblas.h"

__global__ void ztranspose3_32( magmaDoubleComplex       *B, int ldb, 
                                const magmaDoubleComplex *A, int lda,
                                int m, int m32, int n, int n32)
{
    __shared__ magmaDoubleComplex sA[32][ZSIZE_1SHARED+1];

    int inx = threadIdx.x;
    int iny = threadIdx.y;
    int ibx = blockIdx.x*32;
    int iby = blockIdx.y*32;

    A += ibx + inx + __mul24( iby + iny, lda );
    B += iby + inx + __mul24( ibx + iny, ldb );

    int t2 = iby+iny;
    if (ibx+inx < m) {
        if (t2   < n) {
            sA[iny+0][inx] = A[0*lda];
            if (t2+ 8 < n) {
                sA[iny+8][inx] = A[8*lda];
                if (t2 + 16 < n) {
                    sA[iny+16][inx] = A[16*lda];
                    if (t2 + 24 < n) {
                        sA[iny+24][inx] = A[24*lda];
                    }
                }
            }
        }
    }
    __syncthreads();

#if defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c)
    if (iby + inx < n) {
        if (ibx + iny < m) {
            B[0*ldb] = sA[inx][iny+0];
            if (ibx + iny + 8 < m) {
                B[8*ldb] = sA[inx][iny+8];
                if (ibx + iny +16 < m) {
                    B[16*ldb] = sA[inx][iny+16];
                    if (ibx + iny + 24 < m) {
                        B[24*ldb] = sA[inx][iny+24];
                    }
                }
            }
        }
    }
#else /* defined(PRECISION_z) */
    if (iby + inx < n) {
        if (ibx + iny < m) {
            B[0*ldb] = sA[inx][iny+0];
            if (ibx + iny + 8 < m) {
                B[8*ldb] = sA[inx][iny+8];
            }
        }                
        if (iby + inx + 16 < n) {
            if (ibx + iny < m) {
                B[0*ldb+16] = sA[inx+16][iny+0];
                if (ibx + iny + 8 < m) {
                    B[8*ldb+16] = sA[inx+16][iny+8];
                }
            }
        }
    }
    
    __syncthreads();
    A += ZSIZE_1SHARED;
    B += __mul24( 16, ldb );

    sA[iny+ 0][inx] = A[ 0*lda];
    sA[iny+ 8][inx] = A[ 8*lda];
    sA[iny+16][inx] = A[16*lda];
    sA[iny+24][inx] = A[24*lda];

    __syncthreads();

    if (iby + inx < n) {
        if (ibx + iny + 16 < m) {
            B[0*ldb] = sA[inx][iny+0];
            if (ibx + iny + 24 < m) {
                B[8*ldb] = sA[inx][iny+8];
            }
        }                
        if (iby + inx + 16 < n) {
            if (ibx + iny + 16 < m) {
                B[0*ldb+16] = sA[inx+16][iny+0];
                if (ibx + iny + 24 < m) {
                    B[8*ldb+16] = sA[inx+16][iny+8];
                }
            }
        }
    }
#endif

}



__global__ void ztranspose2_32( magmaDoubleComplex       *B, int ldb, 
                                const magmaDoubleComplex *A, int lda, 
                                int m, int m32, int n, int n32)
{        
    __shared__ magmaDoubleComplex sA[32][ZSIZE_1SHARED+1];
    
    int inx = threadIdx.x;
    int iny = threadIdx.y;
    int ibx = blockIdx.x*32;
    int iby = blockIdx.y*32;
    
    int dx, dy;
    if (ibx+32 < m)
       dx = 0;
    else
       dx = m32;

    if (iby+32 < n)
       dy = 0;
    else
       dy = n32;

    A += ibx + inx - dx + __mul24( iby + iny - dy, lda );
    B += iby + inx - dy + __mul24( ibx + iny - dx, ldb );
    
    sA[iny+0][inx] = A[0*lda];
    sA[iny+8][inx] = A[8*lda];
    sA[iny+16][inx] = A[16*lda];
    sA[iny+24][inx] = A[24*lda];
    
    __syncthreads();
        
#if defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c)
    B[ 0*ldb] = sA[inx][iny+0];
    B[ 8*ldb] = sA[inx][iny+8];
    B[16*ldb] = sA[inx][iny+16];
    B[24*ldb] = sA[inx][iny+24];
#else /* defined(PRECISION_z) */
    B[0*ldb]    = sA[inx][iny+0];
    B[8*ldb]    = sA[inx][iny+8];
    B[0*ldb+16] = sA[inx+16][iny+0];
    B[8*ldb+16] = sA[inx+16][iny+8];
    
    __syncthreads();
    A += ZSIZE_1SHARED;
    B += __mul24( 16, ldb );
    
    sA[iny+ 0][inx] = A[ 0*lda];
    sA[iny+ 8][inx] = A[ 8*lda];
    sA[iny+16][inx] = A[16*lda];
    sA[iny+24][inx] = A[24*lda];
    
    __syncthreads();
    
    B[0*ldb]    = sA[inx   ][iny+0];
    B[8*ldb]    = sA[inx   ][iny+8];
    B[0*ldb+16] = sA[inx+16][iny+0];
    B[8*ldb+16] = sA[inx+16][iny+8];
#endif
} 

//
//        m, n - dimensions in the source (input) matrix
//             This version transposes for general m, n .
//             Note that ldi >= m and ldo >= n.
//
extern "C" void 
magmablas_ztranspose2(magmaDoubleComplex       *odata, magma_int_t ldo, 
                      const magmaDoubleComplex *idata, magma_int_t ldi, 
                      magma_int_t m, magma_int_t n )
{
    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    dim3 threads( ZSIZE_1SHARED, 8, 1 );
    dim3 grid( (m+31)/32, (n+31)/32, 1 );
    ztranspose3_32<<< grid, threads, 0, magma_stream >>>(
        odata, ldo, idata, ldi, m, (32-m%32)%32, n, (32-n%32)%32 );
}

extern "C" void
magmablas_ztranspose2s(magmaDoubleComplex       *odata, magma_int_t ldo,
                       const magmaDoubleComplex *idata, magma_int_t ldi,
                       magma_int_t m, magma_int_t n, magma_queue_t stream )
{
    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    dim3 threads( ZSIZE_1SHARED, 8, 1 );
    dim3 grid( (m+31)/32, (n+31)/32, 1 );
    ztranspose3_32<<< grid, threads, 0, stream >>>(
        odata, ldo, idata, ldi, m, (32-m%32)%32, n, (32-n%32)%32 );
}
