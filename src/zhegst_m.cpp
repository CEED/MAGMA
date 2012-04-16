/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Raffaele Solca

       @precisions normal z -> s d c
*/
#define N_MAX_GPU 8

#include "common_magma.h"
#include "cblas.h"

extern "C"
magma_int_t magma_get_zhegst_m_nb() { return 256;}

#define A(i, j) (a+(j)*nb*lda + (i)*nb)
#define B(i, j) (b+(j)*nb*ldb + (i)*nb)

#define dA(gpui, i, j) (dw[gpui] + (j)*nb*ldda + (i)*nb)
#define dB_c(gpui, i, j) (dw[gpui] + dima*ldda + (i)*nb + (j)*nb*lddbc)
#define dB_r(gpui, i, j) (dw[gpui] + dima*ldda + (i)*nb + (j)*nb*lddbr)

extern "C" magma_int_t
magma_zhegst_m(magma_int_t nrgpu, magma_int_t itype, char uplo, magma_int_t n,
               cuDoubleComplex *a, magma_int_t lda,
               cuDoubleComplex *b, magma_int_t ldb, magma_int_t *info)
{
/*
  -- MAGMA (version 1.1) --
     Univ. of Tennessee, Knoxville
     Univ. of California, Berkeley
     Univ. of Colorado, Denver
     November 2011


   Purpose
   =======

   ZHEGST reduces a complex Hermitian-definite generalized
   eigenproblem to standard form.

   If ITYPE = 1, the problem is A*x = lambda*B*x,
   and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H)

   If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
   B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H*A*L.

   B must have been previously factorized as U**H*U or L*L**H by ZPOTRF.

   Arguments
   =========

   ITYPE   (input) INTEGER
           = 1: compute inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H);
           = 2 or 3: compute U*A*U**H or L**H*A*L.

   UPLO    (input) CHARACTER*1
           = 'U':  Upper triangle of A is stored and B is factored as
                   U**H*U;
           = 'L':  Lower triangle of A is stored and B is factored as
                   L*L**H.

   N       (input) INTEGER
           The order of the matrices A and B.  N >= 0.

   A       (input/output) COMPLEX*16 array, dimension (LDA,N)
           On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
           N-by-N upper triangular part of A contains the upper
           triangular part of the matrix A, and the strictly lower
           triangular part of A is not referenced.  If UPLO = 'L', the
           leading N-by-N lower triangular part of A contains the lower
           triangular part of the matrix A, and the strictly upper
           triangular part of A is not referenced.

           On exit, if INFO = 0, the transformed matrix, stored in the
           same format as A.

   LDA     (input) INTEGER
           The leading dimension of the array A.  LDA >= max(1,N).

   B       (input) COMPLEX*16 array, dimension (LDB,N)
           The triangular factor from the Cholesky factorization of B,
           as returned by ZPOTRF.

   LDB     (input) INTEGER
           The leading dimension of the array B.  LDB >= max(1,N).

   INFO    (output) INTEGER
           = 0:  successful exit
           < 0:  if INFO = -i, the i-th argument had an illegal value

   =====================================================================*/

    char uplo_[2] = {uplo, 0};
    magma_int_t        k, kb, j, jb, kb2;
    magma_int_t        ldda, dima ,lddbr, lddbc;
    cuDoubleComplex    zone  = MAGMA_Z_ONE;
    cuDoubleComplex    mzone  = MAGMA_Z_NEG_ONE;
    cuDoubleComplex    zhalf  = MAGMA_Z_HALF;
    cuDoubleComplex    mzhalf  = MAGMA_Z_NEG_HALF;
    cuDoubleComplex* dw[N_MAX_GPU];
    cudaStream_t stream [N_MAX_GPU][3];
    magma_int_t igpu = 0;

    int gpu_b;
    cudaGetDevice(&gpu_b);

    double             done  = (double) 1.0;
    long int           upper = lapackf77_lsame(uplo_, "U");

    magma_int_t nb = magma_get_zhegst_m_nb();

    /* Test the input parameters. */
    *info = 0;
    if (itype<1 || itype>3){
        *info = -1;
    }else if ((! upper) && (! lapackf77_lsame(uplo_, "L"))) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < max(1,n)) {
        *info = -5;
    }else if (ldb < max(1,n)) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return */
    if ( n == 0 )
        return *info;

    magma_int_t nbl = (n-1)/nb+1; // number of blocks

    if ( (itype==1 && upper) || (itype!=1 && !upper) ){
        ldda = ((nbl-1)/nrgpu+1)*nb;
        dima = n;
    } else {
        ldda = n;
        dima = ((nbl-1)/nrgpu+1)*nb;
    }
    lddbr = 2 * nb;
    lddbc = n;
    for (igpu = 0; igpu < nrgpu; ++igpu){
        cudaSetDevice(igpu);
        if (cudaSuccess != cudaMalloc( (void**)&dw[igpu], (dima*ldda + lddbc*lddbr)*sizeof(cuDoubleComplex) ) ) {
            *info = MAGMA_ERR_CUBLASALLOC;
            return *info;
        }
        cudaStreamCreate(&stream[igpu][0]);
        cudaStreamCreate(&stream[igpu][1]);
        cudaStreamCreate(&stream[igpu][2]);
    }

    /* Use hybrid blocked code */

    if (itype==1) {
        if (upper) {

            /* Compute inv(U')*A*inv(U) */

            //copy A to mgpus
            for (k = 0; k < nbl; ++k){
                igpu = k%nrgpu;
                cudaSetDevice(igpu);
                kb = min(nb, n-k*nb);
                cudaMemcpy2DAsync(dA(igpu, k/nrgpu, k), ldda * sizeof(cuDoubleComplex),
                                  A(k, k), lda  * sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*kb, n-k*nb,
                                  cudaMemcpyHostToDevice, stream[igpu][0]);
            }
            kb= min(n,nb);
            igpu = 0;
            cudaSetDevice(igpu);
            // dB_r(0,0) is used to store B(k,k)
            cudaMemcpy2DAsync(dB_r(igpu, 0, 0), lddbr * sizeof(cuDoubleComplex),
                              B(0, 0), ldb * sizeof(cuDoubleComplex),
                              sizeof(cuDoubleComplex)*kb, kb,
                              cudaMemcpyHostToDevice, stream[igpu][1]);

            for(k = 0; k<nbl; ++k){
                kb= min(n-k*nb,nb);
                kb2= min(n-(k+1)*nb,nb);

                if(k+1<nbl){
                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cudaStreamSynchronize(stream[igpu][0]);
                        cudaMemcpy2DAsync(dB_r(igpu, 0, k+1), lddbr * sizeof(cuDoubleComplex),
                                          B(k, k+1), ldb * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*kb, n-(k+1)*nb,
                                          cudaMemcpyHostToDevice, stream[igpu][0]);
                    }
                }

                igpu = k%nrgpu;
                cudaSetDevice(igpu);

                cudaStreamSynchronize(stream[igpu][1]); // Needed, otherwise conflicts reading B(k,k) between hegs2 and cudaMemcpy2D
                cudaStreamSynchronize(stream[igpu][2]);

                if(k+1<nbl){
                    cublasSetKernelStream(stream[igpu][1]);
                    // dB_r(0,0) stores B(k,k)
                    cublasZtrsm(MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                kb, n-(k+1)*nb,
                                zone, dB_r(igpu, 0, 0), lddbr,
                                dA(igpu, k/nrgpu, k+1), ldda);
                }

                lapackf77_zhegs2( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);
printf("hegs2%d\n", k);
                if (k+1<nbl) {
                    cudaMemcpy2DAsync(dA(igpu, k/nrgpu, k), ldda * sizeof(cuDoubleComplex),
                                      A(k, k), lda * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*kb, kb,
                                      cudaMemcpyHostToDevice, stream[igpu][0]);

                    cudaStreamSynchronize(stream[igpu][1]);
                    cublasSetKernelStream(stream[igpu][0]);

                    cublasZhemm(MagmaLeft, MagmaUpper,
                                kb, n-(k+1)*nb,
                                mzhalf, dA(igpu, k/nrgpu, k), ldda,
                                dB_r(igpu, 0, k+1), lddbr,
                                zone, dA(igpu, k/nrgpu, k+1), ldda);

                    cudaStreamSynchronize(stream[igpu][0]);

                    cublasGetMatrix(kb, n-(k+1)*nb, sizeof(cuDoubleComplex), dA(igpu, k/nrgpu, k+1), ldda, A(k, k+1), lda);

                    // send the partially updated panel of dA to each gpu in the second dB block
                    // to overlap hemm computation

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cudaMemcpy2DAsync(dB_r(igpu, 1, k+1), lddbr * sizeof(cuDoubleComplex),
                                          A(k, k+1), lda * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*kb, n-(k+1)*nb,
                                          cudaMemcpyHostToDevice, stream[igpu][0]);
                    }

                    igpu = k%nrgpu;
                    cudaSetDevice(igpu);
                    cublasSetKernelStream(stream[igpu][1]);

                    cublasZhemm(MagmaLeft, MagmaUpper,
                                kb, n-(k+1)*nb,
                                mzhalf, dA(igpu, k/nrgpu, k), ldda,
                                dB_r(igpu, 0, k+1), lddbr,
                                zone, dA(igpu, k/nrgpu, k+1), ldda);

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][0]);
                    }

                    for (j = k+1; j < nbl; ++j){
                        jb = min(nb, n-j*nb);
                        igpu = j%nrgpu;
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][(j/nrgpu)%3]);
                        cublasZher2k(MagmaUpper, MagmaConjTrans,
                                     jb, nb,
                                     mzone, dB_r(igpu, 1, j), lddbr,
                                     dB_r(igpu, 0, j), lddbr,
                                     done, dA(igpu, j/nrgpu, j), ldda);
                        cudaStreamSynchronize(stream[igpu][((j)/nrgpu)%3]); // Needed for correctness. Why?
                        if (j == k+1){
                            cudaStreamSynchronize(stream[igpu][(j/nrgpu)%3]);
                            cudaMemcpy2DAsync( A(k+1, k+1), lda *sizeof(cuDoubleComplex),
                                              dA(igpu, (k+1)/nrgpu, k+1), ldda*sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*kb2, kb2,
                                              cudaMemcpyDeviceToHost, stream[igpu][2]);
                            // dB_r(0,0) is used to store B(k,k)
                            cudaMemcpy2DAsync(dB_r(igpu, 0, 0), lddbr * sizeof(cuDoubleComplex),
                                              B(k+1, k+1), ldb * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*kb2, kb2,
                                              cudaMemcpyHostToDevice, stream[igpu][1]);
                        }
                    }
                    for (j = k+1; j < nbl-1; ++j){
                        igpu = j%nrgpu;
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][0]);
                        cublasZgemm('C', 'N', nb, n-(j+1)*nb, nb, mzone, dB_r(igpu, 0, j), lddbr,
                                    dB_r(igpu, 1, j+1), lddbr, zone, dA(igpu, j/nrgpu, j+1), ldda );

                        cublasZgemm('C', 'N', nb, n-(j+1)*nb, nb, mzone, dB_r(igpu, 1, j), lddbr,
                                    dB_r(igpu, 0, j+1), lddbr, zone, dA(igpu, j/nrgpu, j+1), ldda );
                    }
                }
            }

            for (igpu = 0; igpu < nrgpu; ++igpu){
                cudaStreamSynchronize(stream[igpu][0]);
                cudaStreamSynchronize(stream[igpu][1]);
            }

            if (n > nb){

                magma_int_t nloc[N_MAX_GPU];

                jb = min(nb, n-nb);
                for (igpu = 0; igpu < nrgpu; ++igpu){
                    nloc[igpu]=0;
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dB_r(igpu, 1, 1), lddbr * sizeof(cuDoubleComplex),
                                      B(1, 1), ldb * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*jb, n-nb,
                                      cudaMemcpyHostToDevice, stream[igpu][1]);
                }
                for (j = 1; j < nbl; ++j){
                    if ((j+1)*nb < n){
                        jb = min(nb, n-(j+1)*nb);
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dB_r(igpu, (j+1)%2, j+1), lddbr * sizeof(cuDoubleComplex),
                                              B(j+1, j+1), ldb * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*jb, n-(j+1)*nb,
                                              cudaMemcpyHostToDevice, stream[igpu][(j+1)%2]);
                        }
                    }
                    jb = min(nb, n-j*nb);
                    nloc[(j-1)%nrgpu] += nb;

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][j%2]);
                        cublasZtrsm('R', uplo, 'N', 'N', nloc[igpu], jb, zone, dB_r(igpu, j%2, j), lddbr,
                                    dA(igpu, 0, j), ldda );
                    }

                    if ( j < nbl-1 ){

                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cublasSetKernelStream(stream[igpu][j%2]);
                            cublasZgemm('N', 'N', nloc[igpu], n-(j+1)*nb, nb, mzone, dA(igpu, 0, j), ldda,
                                        dB_r(igpu, j%2, j+1), lddbr, zone, dA(igpu, 0, j+1), ldda );
                        }
                    }

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][j%2]);
                    }

                    for (k = 0; k < j; ++k){
                        igpu = k%nrgpu;
                        cudaSetDevice(igpu);
                        kb = min(nb, n-k*nb);
                        cudaMemcpy2DAsync(A(k, j), lda * sizeof(cuDoubleComplex),
                                          dA(igpu, k/nrgpu, j), ldda  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*kb, jb,
                                          cudaMemcpyDeviceToHost, stream[igpu][2]);
                    }
                }
            }


        } else {
            /* Compute inv(L)*A*inv(L') */
            //copy A to mgpus
            for (k = 0; k < nbl; ++k){
                igpu = k%nrgpu;
                cudaSetDevice(igpu);
                kb = min(nb, n-k*nb);
                cudaMemcpy2DAsync(dA(igpu, k, k/nrgpu), ldda * sizeof(cuDoubleComplex),
                                  A(k, k), lda  * sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*(n-k*nb), kb,
                                  cudaMemcpyHostToDevice, stream[igpu][0]);
            }
            kb= min(n,nb);
            igpu = 0;
            cudaSetDevice(igpu);
            // dB_c(0,0) is used to store B(k,k)
            cudaMemcpy2DAsync(dB_c(igpu, 0, 0), lddbc * sizeof(cuDoubleComplex),
                              B(0, 0), ldb * sizeof(cuDoubleComplex),
                              sizeof(cuDoubleComplex)*kb, kb,
                              cudaMemcpyHostToDevice, stream[igpu][1]);

            for(k = 0; k<nbl; ++k){
                kb= min(n-k*nb,nb);
                kb2= min(n-(k+1)*nb,nb);

                if(k+1<nbl){
                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cudaStreamSynchronize(stream[igpu][0]);
                        cudaMemcpy2DAsync(dB_c(igpu, k+1, 0), lddbc * sizeof(cuDoubleComplex),
                                          B(k+1, k), ldb * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*(n-(k+1)*nb), kb,
                                          cudaMemcpyHostToDevice, stream[igpu][0]);
                    }
                }

                igpu = k%nrgpu;
                cudaSetDevice(igpu);

                cudaStreamSynchronize(stream[igpu][1]); // Needed, otherwise conflicts reading B(k,k) between hegs2 and cudaMemcpy2D
                cudaStreamSynchronize(stream[igpu][2]);

                if(k+1<nbl){
                    cublasSetKernelStream(stream[igpu][1]);
                    // dB_c(0,0) stores B(k,k)
                    cublasZtrsm(MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                n-(k+1)*nb, kb,
                                zone, dB_c(igpu, 0, 0), lddbc,
                                dA(igpu, k+1, k/nrgpu), ldda);
                }

                lapackf77_zhegs2( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);

                if (k+1<nbl) {
                    cudaMemcpy2DAsync(dA(igpu, k , k/nrgpu), ldda * sizeof(cuDoubleComplex),
                                      A(k, k), lda * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*kb, kb,
                                      cudaMemcpyHostToDevice, stream[igpu][0]);

                    cudaStreamSynchronize(stream[igpu][1]);
                    cublasSetKernelStream(stream[igpu][0]);

                    cublasZhemm(MagmaRight, MagmaLower,
                                n-(k+1)*nb, kb,
                                mzhalf, dA(igpu, k, k/nrgpu), ldda,
                                dB_c(igpu, k+1, 0), lddbc,
                                zone, dA(igpu, k+1, k/nrgpu), ldda);

                    cudaStreamSynchronize(stream[igpu][0]);

                    cublasGetMatrix(n-(k+1)*nb, kb, sizeof(cuDoubleComplex), dA(igpu, k+1, k/nrgpu), ldda, A(k+1, k), lda);

                    // send the partially updated panel of dA to each gpu in the second dB block
                    // to overlap hemm computation

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cudaMemcpy2DAsync(dB_c(igpu, k+1, 1), lddbc * sizeof(cuDoubleComplex),
                                          A(k+1, k), lda * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*(n-(k+1)*nb), kb,
                                          cudaMemcpyHostToDevice, stream[igpu][0]);
                    }

                    igpu = k%nrgpu;
                    cudaSetDevice(igpu);
                    cublasSetKernelStream(stream[igpu][1]);

                    cublasZhemm(MagmaRight, MagmaLower,
                                n-(k+1)*nb, kb,
                                mzhalf, dA(igpu, k, k/nrgpu), ldda,
                                dB_c(igpu, k+1, 0), lddbc,
                                zone, dA(igpu, k+1, k/nrgpu), ldda);

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][0]);
                    }

                    for (j = k+1; j < nbl; ++j){
                        jb = min(nb, n-j*nb);
                        igpu = j%nrgpu;
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][(j/nrgpu)%3]);
                        cublasZher2k(MagmaLower, MagmaNoTrans,
                                     jb, nb,
                                     mzone, dB_c(igpu, j, 1), lddbc,
                                     dB_c(igpu, j, 0), lddbc,
                                     done, dA(igpu, j, j/nrgpu), ldda);
                        cudaStreamSynchronize(stream[igpu][((j)/nrgpu)%3]); // Needed for correctness. Why?
                        if (j == k+1){
                            cudaStreamSynchronize(stream[igpu][(j/nrgpu)%3]);
                            cudaMemcpy2DAsync( A(k+1, k+1), lda *sizeof(cuDoubleComplex),
                                              dA(igpu, k+1, (k+1)/nrgpu), ldda*sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*kb2, kb2,
                                              cudaMemcpyDeviceToHost, stream[igpu][2]);
                            // dB_c(0,0) is used to store B(k,k)
                            cudaMemcpy2DAsync(dB_c(igpu, 0, 0), lddbc * sizeof(cuDoubleComplex),
                                              B(k+1, k+1), ldb * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*kb2, kb2,
                                              cudaMemcpyHostToDevice, stream[igpu][1]);
                        }
                    }
                    for (j = k+1; j < nbl-1; ++j){
                        igpu = j%nrgpu;
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][0]);
                        cublasZgemm('N', 'C', n-(j+1)*nb, nb, nb, mzone, dB_c(igpu, j+1, 1), lddbc,
                                    dB_c(igpu, j, 0), lddbc, zone, dA(igpu, j+1, j/nrgpu), ldda );

                        cublasZgemm('N', 'C', n-(j+1)*nb, nb, nb, mzone, dB_c(igpu, j+1, 0), lddbc,
                                    dB_c(igpu, j, 1), lddbc, zone, dA(igpu, j+1, j/nrgpu), ldda );
                    }
                }
            }

            for (igpu = 0; igpu < nrgpu; ++igpu){
                cudaStreamSynchronize(stream[igpu][0]);
                cudaStreamSynchronize(stream[igpu][1]);
            }

            if (n > nb){

                magma_int_t nloc[N_MAX_GPU];

                jb = min(nb, n-nb);
                for (igpu = 0; igpu < nrgpu; ++igpu){
                    nloc[igpu]=0;
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dB_c(igpu, 1, 1), lddbc * sizeof(cuDoubleComplex),
                                      B(1, 1), ldb * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*(n-nb), jb,
                                      cudaMemcpyHostToDevice, stream[igpu][1]);
                }
                for (j = 1; j < nbl; ++j){
                    if ((j+1)*nb < n){
                        jb = min(nb, n-(j+1)*nb);
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dB_c(igpu, j+1, (j+1)%2), lddbc * sizeof(cuDoubleComplex),
                                              B(j+1, j+1), ldb * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*(n-(j+1)*nb), jb,
                                              cudaMemcpyHostToDevice, stream[igpu][(j+1)%2]);
                        }
                    }
                    jb = min(nb, n-j*nb);
                    nloc[(j-1)%nrgpu] += nb;

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][j%2]);
                        cublasZtrsm('L', uplo, 'N', 'N', jb, nloc[igpu], zone, dB_c(igpu, j, j%2), lddbc,
                                    dA(igpu, j, 0), ldda );
                    }

                    if ( j < nbl-1 ){

                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cublasSetKernelStream(stream[igpu][j%2]);
                            cublasZgemm('N', 'N', n-(j+1)*nb, nloc[igpu], nb, mzone, dB_c(igpu, j+1, j%2), lddbc,
                                        dA(igpu, j, 0), ldda, zone, dA(igpu, j+1, 0), ldda );
                        }
                    }

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][j%2]);
                    }

                    for (k = 0; k < j; ++k){
                        igpu = k%nrgpu;
                        cudaSetDevice(igpu);
                        kb = min(nb, n-k*nb);
                        cudaMemcpy2DAsync(A(j, k), lda * sizeof(cuDoubleComplex),
                                          dA(igpu, j, k/nrgpu), ldda  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*jb, kb,
                                          cudaMemcpyDeviceToHost, stream[igpu][2]);
                    }
                }
            }

        }

    } else {

        if (upper) {

            printf("zhegst_m: type2 upper not implemented\n");
            exit(-1);

            /* Compute U*A*U' */

/*            for(k = 0; k<n; k+=nb){
                kb= min(n-k,nb);

                cudaMemcpy2DAsync( A(k, k), lda *sizeof(cuDoubleComplex),
                                  dA(k, k), ldda*sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*kb, kb,
                                  cudaMemcpyDeviceToHost, stream[0]);

                // Update the upper triangle of A(1:k+kb-1,1:k+kb-1)
                if(k>0){

                    cublasZtrmm(MagmaLeft, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                                k, kb,
                                zone ,dB(0,0), lddb,
                                dA(0,k), ldda);

                    cublasZhemm(MagmaRight, MagmaUpper,
                                k, kb,
                                zhalf, dA(k,k), ldda,
                                dB(0,k), lddb,
                                zone, dA(0, k), ldda);

                    cudaStreamSynchronize(stream[1]);

                    cublasZher2k(MagmaUpper, MagmaNoTrans,
                                 k, kb,
                                 zone, dA(0,k), ldda,
                                 dB(0,k), lddb,
                                 done, dA(0,0), ldda);

                    cublasZhemm(MagmaRight, MagmaUpper,
                                k, kb,
                                zhalf, dA(k,k), ldda,
                                dB(0,k), lddb,
                                zone, dA(0, k), ldda);

                    cublasZtrmm(MagmaRight, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                k, kb,
                                zone, dB(k,k), lddb,
                                dA(0,k), ldda);

                }

                cudaStreamSynchronize(stream[0]);

                lapackf77_zhegs2( &itype, uplo_, &kb, A(k, k), &lda, B(k, k), &ldb, info);

                cudaMemcpy2DAsync(dA(k, k), ldda * sizeof(cuDoubleComplex),
                                  A(k, k), lda  * sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*kb, kb,
                                  cudaMemcpyHostToDevice, stream[1]);

            }

            cudaStreamSynchronize(stream[1]);
*/
        } else {

            /* Compute L'*A*L */

            printf("zhegst_m: type2 lower not implemented\n");
            exit(-1);

/*                        if (n > nb){

            magma_int_t nloc[N_MAX_GPU];
            for(igpu = 0; igpu < nrgpu; ++igpu)
                nloc[igpu] = 0;

            kb = min(nb, n);
            for (j = 0; j < nbl; ++j){
                igpu = j%nrgpu;
                cudaSetDevice(igpu);
                jb = min(nb, n-j*nb);
                nloc[igpu] += jb;
                cudaMemcpy2DAsync(dA(igpu, j/nrgpu, 0), ldda * sizeof(cuDoubleComplex),
                                  A(j, 0), lda  * sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*jb, kb,
                                  cudaMemcpyHostToDevice, stream[igpu][0]);
            }
            for (igpu = 0; igpu < nrgpu; ++igpu){
                cudaSetDevice(igpu);
                cudaMemcpy2DAsync(dB_r(igpu, 0, 0), lddbr * sizeof(cuDoubleComplex),
                                  B(0, 0), ldb * sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*kb, kb,
                                  cudaMemcpyHostToDevice, stream[igpu][0]);
            }
            for (k = 0; k < nbl-1; ++k){
                nloc[k%nrgpu] -= nb;
                if (k < nbl-2){
                    kb = min(nb, n-(k+1)*nb);
                    for (j = k; j < nbl; ++j){
                        igpu = j%nrgpu;
                        cudaSetDevice(igpu);
                        jb = min(nb, n-j*nb);
                        cudaMemcpy2DAsync(dA(igpu, j/nrgpu, k+1), ldda * sizeof(cuDoubleComplex),
                                          A(j, k+1), lda  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*jb, kb,
                                          cudaMemcpyHostToDevice, stream[igpu][(k+1)%2]);
                    }
                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cudaMemcpy2DAsync(dB_r(igpu, (k+1)%2, 0), lddbr * sizeof(cuDoubleComplex),
                                          B(k+1, 0), ldb * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*kb, (k+1)*nb + kb,
                                          cudaMemcpyHostToDevice, stream[igpu][(k+1)%2]);
                    }
                }

                kb = min(nb, n-k*nb);

                if (k > 0){
                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][k%2]);
                        cublasZgemm('N', 'N', nloc[igpu], k*nb, kb, zone, dA(igpu, n-nloc[igpu], k), ldda,
                                    dB_r(igpu, k%2, 0), lddbr, zone, dA(igpu, n-nloc[igpu], 0), ldda );
                    }
                }

                for (igpu = 0; igpu < nrgpu; ++igpu){
                    cudaSetDevice(igpu);
                    cublasSetKernelStream(stream[igpu][k%2]);
                    cublasZtrmm('R', uplo, 'N', 'N', nloc[igpu], kb, zone, dB_r(igpu, k%2, k), lddbr,
                                dA(igpu, n-nloc[igpu], k), ldda );
                }

                for (igpu = 0; igpu < nrgpu; ++igpu){
                    cudaStreamSynchronize(stream[igpu][k%2]);
                }

            }

                        }

            /////////
            // put for loop!
            // put copies!

            cublasSetKernelStream(stream[igpu][0]);

            cublasZhemm(MagmaRight, MagmaLower,
                        kb, k*nb,
                        zhalf, dA(igpu, k/nrgpu, 0), ldda,
                        dB_r(igpu, 0, 0), lddbr,
                        zone, dA(igpu, k/nrgpu, 0), ldda);

            cudaStreamSynchronize(stream[igpu][0]);

            cublasGetMatrix(kb, k*nb, sizeof(cuDoubleComplex), dA(igpu, k/nrgpu, 0), ldda, A(k, 0), lda);

            // send the partially updated panel of dA to each gpu in the second dB block
            // to overlap hemm computation

            for (igpu = 0; igpu < nrgpu; ++igpu){
                cudaSetDevice(igpu);
                cudaMemcpy2DAsync(dB_r(igpu, 1, 0), lddbr * sizeof(cuDoubleComplex),
                                  A(k, 0), lda * sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*kb,
                                  cudaMemcpyHostToDevice, stream[igpu][0]);
            }

            igpu = k%nrgpu;
            cudaSetDevice(igpu);
            cublasSetKernelStream(stream[igpu][1]);

            cublasZhemm(MagmaRight, MagmaLower,
                        n-(k+1)*nb, kb,
                        mzhalf, dA(igpu, k, k/nrgpu), ldda,
                        dB_c(igpu, k+1, 0), lddbc,
                        zone, dA(igpu, k+1, k/nrgpu), ldda);

            for (igpu = 0; igpu < nrgpu; ++igpu){
                cudaStreamSynchronize(stream[igpu][0]);
            }


            //copy B from mgpus
            for (j = 0; j < nbl; ++j){
                igpu = j%nrgpu;
                cudaSetDevice(igpu);
                jb = min(nb, n-j*nb);
                cudaMemcpy2DAsync(A(j, 0), lda  * sizeof(cuDoubleComplex),
                                  dA(igpu, j/nrgpu, 0), ldda * sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*jb, n,
                                  cudaMemcpyDeviceToHost, stream[igpu][0]);
            }

*//*            for(k = 0; k<n; k+=nb){
                kb= min(n-k,nb);

                cudaMemcpy2DAsync( A(k, k), lda *sizeof(cuDoubleComplex),
                                  dA(k, k), ldda*sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*kb, kb,
                                  cudaMemcpyDeviceToHost, stream[0]);

                // Update the lower triangle of A(1:k+kb-1,1:k+kb-1)
                if(k>0){

                    cublasZtrmm(MagmaRight, MagmaLower, MagmaNoTrans, MagmaNonUnit,
                                kb, k,
                                zone ,dB(0,0), lddb,
                                dA(k,0), ldda);

                    cublasZhemm(MagmaLeft, MagmaLower,
                                kb, k,
                                zhalf, dA(k,k), ldda,
                                dB(k,0), lddb,
                                zone, dA(k, 0), ldda);

                    cudaStreamSynchronize(stream[1]);

                    cublasZher2k(MagmaLower, MagmaConjTrans,
                                 k, kb,
                                 zone, dA(k,0), ldda,
                                 dB(k,0), lddb,
                                 done, dA(0,0), ldda);

                    cublasZhemm(MagmaLeft, MagmaLower,
                                kb, k,
                                zhalf, dA(k,k), ldda,
                                dB(k,0), lddb,
                                zone, dA(k, 0), ldda);

                    cublasZtrmm(MagmaLeft, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                kb, k,
                                zone, dB(k,k), lddb,
                                dA(k,0), ldda);
                }

                cudaStreamSynchronize(stream[0]);

                lapackf77_zhegs2( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);

                cudaMemcpy2DAsync(dA(k, k), ldda * sizeof(cuDoubleComplex),
                                  A(k, k), lda  * sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*kb, kb,
                                  cudaMemcpyHostToDevice, stream[1]);
            }

            cudaStreamSynchronize(stream[1]);
 */
        }
    }

    for (igpu = 0; igpu < nrgpu; ++igpu){
        cudaSetDevice(igpu);
        cublasSetKernelStream(NULL);
        cudaStreamSynchronize(stream[igpu][2]);
        cudaStreamDestroy(stream[igpu][0]);
        cudaStreamDestroy(stream[igpu][1]);
        cudaStreamDestroy(stream[igpu][2]);
        cudaFree(dw[igpu]);
    }

    cudaSetDevice(gpu_b);

    return *info;
} /* magma_zhegst_gpu */
