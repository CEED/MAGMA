/*
    -- MAGMA (version 1.5) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/
#include "common_magma.h"
#include "batched_kernel_param.h"

/**
    Purpose
    -------
    ZGETRI computes the inverse of a matrix using the LU factorization
    computed by ZGETRF. This method inverts U and then computes inv(A) by
    solving the system inv(A)*L = inv(U) for inv(A).
    
    Note that it is generally both faster and more accurate to use ZGESV,
    or ZGETRF and ZGETRS, to solve the system AX = B, rather than inverting
    the matrix and multiplying to form X = inv(A)*B. Only in special
    instances should an explicit inverse be computed with this routine.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the factors L and U from the factorization
            A = P*L*U as computed by ZGETRF_GPU.
            On exit, if INFO = 0, the inverse of the original matrix A.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[in]
    ipiv    INTEGER array, dimension (N)
            The pivot indices from ZGETRF; for 1 <= i <= N, row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    dwork   (workspace) COMPLEX_16 array on the GPU, dimension (MAX(1,LWORK))
  
    @param[in]
    lwork   INTEGER
            The dimension of the array DWORK.  LWORK >= N*NB, where NB is
            the optimal blocksize returned by magma_get_zgetri_nb(n).
    \n
            Unlike LAPACK, this version does not currently support a
            workspace query, because the workspace is on the GPU.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, U(i,i) is exactly zero; the matrix is
                  singular and its cannot be computed.

    @ingroup magma_zgesv_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zgetri_batched( magma_int_t n, 
                  magmaDoubleComplex **dA_array, magma_int_t ldda,
                  magma_int_t **dipiv_array, 
                  magmaDoubleComplex **dinvA_array, magma_int_t lddia,
                  magma_int_t *info_array,
                  magma_int_t batchCount)
       
{
    /* Local variables */
    /*
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;


    magmaDoubleComplex *dL = dwork;
    magma_int_t lddl    = n;
    magma_int_t trinb   = magma_get_zgetri_nb(n);
    magma_int_t j, jmax, jb, jp;
    */    
    magma_int_t info = 0;
    if (n < 0)
        info = -1;
    else if (ldda < max(1,n))
        info = -3;
    else if (lddia < max(1,n))
        info = -6;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }

    /* Quick return if possible */
    if ( n == 0 )
        return info;




    magma_int_t ib, j;
    magma_int_t nb = 256;//256;// BATRF_NB;



    magmaDoubleComplex **dA_displ   = NULL;
    magmaDoubleComplex **dW0_displ  = NULL;
    magmaDoubleComplex **dW1_displ  = NULL;
    magmaDoubleComplex **dW2_displ  = NULL;
    magmaDoubleComplex **dW3_displ  = NULL;
    magmaDoubleComplex **dW4_displ  = NULL;
    magmaDoubleComplex **dinvdiagA_array = NULL;
    magmaDoubleComplex **work_array = NULL;
    magmaDoubleComplex **dW_array   = NULL;

    magma_malloc((void**)&dA_displ,   batchCount * sizeof(*dA_displ));
    magma_malloc((void**)&dW0_displ,  batchCount * sizeof(*dW0_displ));
    magma_malloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
    magma_malloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));
    magma_malloc((void**)&dW3_displ,  batchCount * sizeof(*dW3_displ));
    magma_malloc((void**)&dW4_displ,  batchCount * sizeof(*dW4_displ));
    magma_malloc((void**)&dinvdiagA_array, batchCount * sizeof(*dinvdiagA_array));
    magma_malloc((void**)&work_array, batchCount * sizeof(*work_array));
    magma_malloc((void**)&dW_array,  batchCount * sizeof(*dW_array));

    magmaDoubleComplex* dinvdiagA;
    magmaDoubleComplex* work;// dinvdiagA and work are workspace in ztrsm

    //magma_int_t invdiagA_msize =  BATRI_NB*((nb/BATRI_NB)+(nb % BATRI_NB != 0))* BATRI_NB ;
    magma_int_t invdiagA_msize = ((n+TRI_NB-1)/TRI_NB)*TRI_NB*TRI_NB;
    magma_int_t work_msize = n*nb;
    magma_zmalloc( &dinvdiagA, invdiagA_msize * batchCount);
    magma_zmalloc( &work, work_msize * batchCount );
    zset_pointer(work_array, work, n, 0, 0, work_msize, batchCount);
    zset_pointer(dinvdiagA_array, dinvdiagA, ((n+TRI_NB-1)/TRI_NB)*TRI_NB, 0, 0, invdiagA_msize, batchCount);
    cudaMemset( dinvdiagA, 0, batchCount * ((n+TRI_NB-1)/TRI_NB)*TRI_NB*TRI_NB * sizeof(magmaDoubleComplex) );

    magma_zdisplace_pointers(dA_displ, dA_array, ldda, 0, 0, batchCount);

    magma_queue_t cstream;
    magmablasGetKernelStream(&cstream);

    //printf(" I am after malloc getri\n");

#if 0
    gbinfo = magmablas_zcheckdiag(n, n, dA_array, ldda, batchCount, info_array); 
    if ( gbinfo != 0 ){
        magma_int_t **cpuarray = (magma_int_t**) malloc(batchCount*sizeof(magma_int_t*));
        magma_getvector( batchCount, sizeof(magma_int_t*), info_array, 1, cpuarray, 1);
        for(int k=0; k<batchCount; k++)
            if(cpu_array[k] != 0) 
                printf("error on matrix %d info equal %d meaning U(%d,%d) is zero matrix is singular NO inverse\n",
                        k,cpu_array[k],cpu_array[k],cpu_array[k]);
            
        return info;
    }
#endif

    for(j = 0; j < n; j+=nb) {
        ib = min(nb, n-j);
        // dinvdiagA * Piv' = I * U^-1 * L^-1 = U^-1 * L^-1 * I
        // Azzam : optimization can be done:
        //          1- swapping identity first
        //          2- compute invdiagL invdiagU only one time

        //magma_queue_sync(NULL);
        //printf(" @ step %d calling set identity \n",j);
        // set dinvdiagA to identity at position j,j
        magma_zdisplace_pointers(dW0_displ, dinvA_array, lddia, j, j, batchCount);
        magmablas_zlaset_batched(MagmaUpperLower, n-j, ib, MAGMA_Z_ZERO, MAGMA_Z_ONE, dW0_displ, lddia, batchCount);

        //magma_queue_sync(NULL);
        //printf(" @ step %d calling solve 1 \n",j);
        // solve work = L^-1 * I
        magma_zdisplace_pointers(dW0_displ, dinvA_array, lddia, 0, j, batchCount);
        magmablas_ztrsm_outofplace_batched(MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 1,
                n, ib,
                MAGMA_Z_ONE,
                dA_displ,       ldda, // dA
                dW0_displ,   lddia, // dB
                work_array,        n, // dX //output
                dinvdiagA_array,  invdiagA_msize, 
                dW1_displ,   dW2_displ, 
                dW3_displ,   dW4_displ,
                1, batchCount);

        //magma_queue_sync(NULL);
        //printf(" @ step %d calling solve 2 \n",j);
        // solve dinvdiagA = U^-1 * work
        magmablas_ztrsm_outofplace_batched(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 1,
                n, ib,
                MAGMA_Z_ONE,
                dA_displ,       ldda, // dA
                work_array,        n, // dB 
                dW0_displ,   lddia, // dX //output
                dinvdiagA_array,  invdiagA_msize, 
                dW1_displ,   dW2_displ, 
                dW3_displ,   dW4_displ,
                1, batchCount);
    }

    // Apply column interchanges
    /*
    for( j = n-2; j >= 0; --j ) {
        jp = ipiv[j] - 1;
        if ( jp != j ) {
            magmablas_zswap( n, dA(0,j), 1, dA(0,jp), 1 );
            magma_zswap(n, dA_array, lda, gbj, ipiv_array, batchCount);
            
        }
    }
    */
    magma_zlaswp_columnserial_batched( n, dinvA_array, lddia, min(1,n-1), 1, dipiv_array, batchCount);
    
    
     //TODO TODO Azzam 
#if 0    
    /* Invert the triangular factor U */
    magmablas_ztrtri_batched( uplo, diag, m, dA_array, ldda, dinvA_array, batchCount );
    
    magma_ztrtri_gpu( MagmaUpper, MagmaNonUnit, n, dA, ldda, info );
    if ( info != 0 )
        return info;
    
    jmax = ((n-1) / nb)*nb;
    for( j = jmax; j >= 0; j -= nb ) {
        jb = min( nb, n-j );
        
        // copy current block column of A to work space dL
        // (only needs lower trapezoid, but we also copy upper triangle),
        // then zero the strictly lower trapezoid block column of A.
        magmablas_zlacpy( MagmaFull, n-j, jb,
                          dA(j,j), ldda,
                          dL(j,0), lddl );
        magmablas_zlaset( MagmaLower, n-j-1, jb, c_zero, c_zero, dA(j+1,j), ldda );
        
        // compute current block column of Ainv
        // Ainv(:, j:j+jb-1)
        //   = ( U(:, j:j+jb-1) - Ainv(:, j+jb:n) L(j+jb:n, j:j+jb-1) )
        //   * L(j:j+jb-1, j:j+jb-1)^{-1}
        // where L(:, j:j+jb-1) is stored in dL.
        if ( j+jb < n ) {
            magma_zgemm( MagmaNoTrans, MagmaNoTrans, n, jb, n-j-jb,
                         c_neg_one, dA(0,j+jb), ldda,
                                    dL(j+jb,0), lddl,
                         c_one,     dA(0,j),    ldda );
        }
        // TODO use magmablas work interface
        magma_ztrsm( MagmaRight, MagmaLower, MagmaNoTrans, MagmaUnit,
                     n, jb, c_one,
                     dL(j,0), lddl,
                     dA(0,j), ldda );
    }

    // Apply column interchanges
    for( j = n-2; j >= 0; --j ) {
        jp = ipiv[j] - 1;
        if ( jp != j ) {
            magmablas_zswap( n, dA(0,j), 1, dA(0,jp), 1 );
        }
    }
#endif    


    magma_queue_sync(cstream);

    magma_free(dA_displ);
    magma_free(dW1_displ);
    magma_free(dW2_displ);
    magma_free(dW3_displ);
    magma_free(dW4_displ);
    magma_free(dinvdiagA_array);
    magma_free(work_array);
    magma_free(dW_array);

    magma_free( dinvdiagA );
    magma_free( work );

    
    return info;
}
