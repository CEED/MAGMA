/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver

       @date
            
       @author Stan Tomov

       @precisions normal z -> s d c
*/

#include <sys/time.h>
#include <time.h>

#include "common_magma.h"
#include "../include/magmasparse.h"
#include "trace.h"
#include "magmablas.h"     

//==================================================================================
extern "C" magma_int_t
magma_zcompact(magma_int_t m, magma_int_t n,
               magmaDoubleComplex *dA, magma_int_t ldda,
               double *dnorms, double tol, 
               magma_int_t *activeMask, magma_int_t *cBlockSize);

extern "C" magma_int_t
magma_zcompactActive(magma_int_t m, magma_int_t n,
                     magmaDoubleComplex *dA, magma_int_t ldda, 
                     magma_int_t *active);
//==================================================================================

#define PRECISION_z

extern "C" magma_int_t
magma_zlobpcg( magma_int_t m, magma_int_t n, magma_z_sparse_matrix A, 
               magmaDoubleComplex *blockX, double *evalues,
               magmaDoubleComplex *dwork, magma_int_t ldwork, 
               magmaDoubleComplex *hwork, magma_int_t lwork,
               magma_z_solver_par *solver_par,
               magma_int_t *info )
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    Solves an eigenvalue problem

       A * X = evalues X

    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory.

    This is a GPU implementation of the LOBPCG method.

    Arguments
    =========

    =====================================================================  */

#define  residualNorms(i,iter)  ( residualNorms + (i) + (iter)*n )
#define hresidualNorms(i,iter)  (hresidualNorms + (i) + (iter)*n )
#define magma_z_bspmv(m, n, alpha, A, X, beta, AX)       {              \
        for(int k = 0; k<n; k++) {                                      \
            magma_z_vector x, ax;                                       \
            x.memory_location = Magma_DEV;  x.num_rows = m;  x.nnz = m;  x.val =  X+(k)*(m); \
            ax.memory_location= Magma_DEV; ax.num_rows = m; ax.nnz = m; ax.val = AX+(k)*(m); \
            magma_z_spmv(alpha, A, x, beta, ax );                       \
        }                                                               \
}
#define magma_z_bspmv_tuned(m, n, alpha, A, X, beta, AX)       {        \
            magma_z_vector x, ax;                                       \
            x.memory_location = Magma_DEV;  x.num_rows = m*n;  x.nnz = m*n;  x.val = X; \
            ax.memory_location= Magma_DEV; ax.num_rows = m*n; ax.nnz = m*n; ax.val = AX; \
            magma_z_spmv(alpha, A, x, beta, ax );                       \
}

#define gramA(    m, n)   (gramA     + (m) + (n)*ldgram)
#define gramB(    m, n)   (gramB     + (m) + (n)*ldgram)
#define gevectors(m, n)   (gevectors + (m) + (n)*ldgram) 
#define h_gramB(  m, n)   (h_gramB   + (m) + (n)*ldgram)

    magma_int_t verbosity = 1;

    magmaDoubleComplex *blockP, *blockAP, *blockR, *blockAR, *blockAX, *blockW;
    magmaDoubleComplex *gramA, *gramB, *gramM;
    magmaDoubleComplex *gevectors, *h_gramB;

    magma_int_t *iwork, liwork = 15*n+9;

    // Set solver parameters
    double residualTolerance  = solver_par->epsilon;
    magma_int_t maxIterations = solver_par->maxiter;

    // Set some constants & defaults
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;

    double *residualNorms, *condestGhistory, condestG;
    double *gevalues;
    magma_int_t *activeMask;

    // Allocate GPU memory for the residual norms' history
    magma_dmalloc(&residualNorms, (maxIterations+1) * n);
    magma_malloc( (void **)&activeMask, (n+1) * sizeof(magma_int_t) );

    // Allocate CPU work space
    magma_dmalloc_cpu(&condestGhistory, maxIterations+1);
    magma_dmalloc_cpu(&gevalues, 3 * n);
    magma_malloc_cpu((void **)&iwork, liwork * sizeof(magma_int_t));

    magmaDoubleComplex *hW;
    magma_zmalloc_pinned(&hW, n*n);
    magma_zmalloc_pinned(&gevectors, 9*n*n); 
    magma_zmalloc_pinned(&h_gramB  , 9*n*n);

    // Allocate GPU workspace
    magma_zmalloc(&gramM, n * n);
    magma_zmalloc(&gramA, 9 * n * n);
    magma_zmalloc(&gramB, 9 * n * n);

    #if defined(PRECISION_z) || defined(PRECISION_c)
    double *rwork;
    magma_int_t lrwork = 1 + 5*(3*n) + 2*(3*n)*(3*n);

    magma_dmalloc_cpu(&rwork, lrwork);
    #endif

    // Rearange the work space
    blockAX = dwork;
    blockAR = blockAX +   m * n;
    blockAP = blockAR +   m * n;
    blockR  = blockAP +   m * n;
    blockP  = blockR  +   m * n;
    blockW  = blockP  +   m * n;
    dwork   = blockW  +   m * n;
    
    *info = 0;
    if (m < 2)
        *info = -1;
    else if (n > m)
        *info = -2; 

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    
    // === Set activemask to 0ne
    for(int k =0; k<n; k++)
        iwork[k]=1;
    magma_setmatrix(n, 1, sizeof(magma_int_t), iwork, n ,activeMask, n);

    magma_int_t gramDim, ldgram  = 3*n;
     
    // Make the initial vectors orthonormal
    magma_zgegqr_gpu( m, n, blockX, m, dwork, hwork, info );
    magma_z_bspmv(m, n, c_one, A, blockX, c_zero, blockAX );

    // Compute the Gram matrix = (X, AX) & its eigenstates
    magma_zgemm(MagmaConjTrans, MagmaNoTrans, n, n, m,
                c_one,  blockX, m, blockAX, m, c_zero, gramM, n);

    magma_zheevd_gpu( MagmaVec, MagmaUpper,
                      n, gramM, n, evalues, hW, n, hwork, lwork,
                      #if defined(PRECISION_z) || defined(PRECISION_c)
                      rwork, lrwork,
                      #endif
                      iwork, liwork, info );

    // Update  X =  X * evectors
    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                c_one,  blockX, m, gramM, n, c_zero, blockW, m);
    magmablas_zlacpy( MagmaUpperLower, m, n, blockW, m, blockX, m);

    // Update AX = AX * evectors
    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                c_one,  blockAX, m, gramM, n, c_zero, blockW, m);
    magmablas_zlacpy( MagmaUpperLower, m, n, blockW, m, blockAX, m);

    condestGhistory[1] = 7.82;
    magma_int_t iterationNumber, cBlockSize, restart = 1, iter;

    // === Main LOBPCG loop ============================================================
    for(iterationNumber = 1; iterationNumber < maxIterations; iterationNumber++)
        { 
            // === compute the residuals (R = Ax - x evalues )
            magmablas_zlacpy( MagmaUpperLower, m, n, blockAX, m, blockR, m);
            for(int i=0; i<n; i++){
                magma_zaxpy(m, MAGMA_Z_MAKE(-evalues[i],0), blockX+i*m, 1, blockR+i*m, 1);
            }
            magmablas_dznrm2_cols(m, n, blockR, m, residualNorms(0, iterationNumber));

            // === remove the residuals corresponding to already converged evectors
            magma_zcompact(m, n, blockR, m,
                           residualNorms(0, iterationNumber), residualTolerance, 
                           activeMask, &cBlockSize);
            
            if (cBlockSize == 0)
                break;
        
            // === apply a preconditioner P to the active residulas: R_new = P R_old
            // === for now set P to be identity (no preconditioner => nothing to be done )
            // magmablas_zlacpy( MagmaUpperLower, m, cBlockSize, blockR, m, blockW, m);

            // === make the active preconditioned residuals orthonormal
            magma_zgegqr_gpu( m, cBlockSize, blockR, m, dwork, hwork, info );
            
            // === compute AR
            magma_z_bspmv(m, cBlockSize, c_one, A, blockR, c_zero, blockAR );

            if (!restart) {
                // === compact P & AP as well
                magma_zcompactActive(m, n, blockP,  m, activeMask);
                magma_zcompactActive(m, n, blockAP, m, activeMask);
          
                // === Make P orthonormal & properly change AP (without multiplication by A)
                magma_zgegqr_gpu( m, cBlockSize, blockP, m, dwork, hwork, info );
                magma_zsetmatrix( cBlockSize, cBlockSize, hwork, cBlockSize, dwork, cBlockSize);
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
                             m, cBlockSize, c_one, dwork, cBlockSize, blockAP, m);
            }

            iter = max(1,iterationNumber-10- (int)(log(1.*cBlockSize)));
            double condestGmean = 0.;
            for(int i = 0; i<iterationNumber-iter+1; i++)
                condestGmean += condestGhistory[i];
            condestGmean = condestGmean / (iterationNumber-iter+1);

            if (restart)
                gramDim = n+cBlockSize;
            else
                gramDim = n+2*cBlockSize;

            /* --- The Raileight-Ritz method for [X R P] -----------------------
               [ X R P ]'  [AX  AR  AP] y = evalues [ X R P ]' [ X R P ], i.e.,
       
                      GramA                                 GramB
                / X'AX  X'AR  X'AP \                 / X'X  X'R  X'P \
               |  R'AX  R'AR  R'AP  | y   = evalues |  R'X  R'R  R'P  |
                \ P'AX  P'AR  P'AP /                 \ P'X  P'R  P'P /       
               -----------------------------------------------------------------   */

            // === assemble GramB; first, set it to I
            magmablas_zlaset_identity(ldgram, ldgram, gramB, ldgram);

            if (!restart) {
                magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, n, m,
                            c_one, blockP, m, blockX, m, c_zero, gramB(n+cBlockSize,0), ldgram);
                magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, cBlockSize, m,
                            c_one, blockP, m, blockR, m, c_zero, gramB(n+cBlockSize,n), ldgram);
            }
            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, n, m,
                        c_one, blockR, m, blockX, m, c_zero, gramB(n,0), ldgram);

            // === get GramB from the GPU to the CPU and compute its eigenvalues only
            magma_zgetmatrix(gramDim, gramDim, gramB, ldgram, h_gramB, ldgram);
            lapackf77_zheev("N", "L", &gramDim, h_gramB, &ldgram, gevalues, 
                            hwork, &lwork,
                            #if defined(PRECISION_z) || defined(PRECISION_c)
                            rwork, 
                            #endif
                            info);

            // === check stability criteria if we need to restart
            condestG = log10( gevalues[gramDim-1]/gevalues[0] ) + 1.;
            if ((condestG/condestGmean>2 && condestG>2) || condestG>8) {
                // Steepest descent restart for stability
                restart=1;
                printf("restart at step #%d\n", iterationNumber);
            }

            // === assemble GramA; first, set it to I
            magmablas_zlaset_identity(ldgram, ldgram, gramA, ldgram);

            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, n, m,
                        c_one, blockR, m, blockAX, m, c_zero, gramA(n,0), ldgram);
            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, cBlockSize, m,
                        c_one, blockR, m, blockAR, m, c_zero, gramA(n,n), ldgram);

            if (!restart) {
                magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, n, m, 
                            c_one, blockP, m, blockAX, m, c_zero, 
                            gramA(n+cBlockSize,0), ldgram);
                magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, cBlockSize, m, 
                            c_one, blockP, m, blockAR, m, c_zero, 
                            gramA(n+cBlockSize,n), ldgram);
                magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, cBlockSize, m, 
                            c_one, blockP, m, blockAP, m, c_zero, 
                            gramA(n+cBlockSize,n+cBlockSize), ldgram);
            }

            if (restart==0) {
                magma_zgetmatrix(gramDim, gramDim, gramA, ldgram, gevectors, ldgram);
            }
            else {
                gramDim = n+cBlockSize;
                magma_zgetmatrix(gramDim, gramDim, gramA, ldgram, gevectors, ldgram);
            }

            for(int k=0; k<n; k++)
                *gevectors(k,k) = MAGMA_Z_MAKE(evalues[k], 0);

            // === the previous eigensolver destroyed what is in h_gramB => must copy it again
            magma_zgetmatrix(gramDim, gramDim, gramB, ldgram, h_gramB, ldgram);

            magma_int_t itype = 1;
            lapackf77_zhegvd(&itype, "V", "L", &gramDim, 
                             gevectors, &ldgram, h_gramB, &ldgram,
                             gevalues, hwork, &lwork, 
                             #if defined(PRECISION_z) || defined(PRECISION_c)
                             rwork, &lrwork,
                             #endif
                             iwork, &liwork, info);
 
            for(int k =0; k<n; k++)
                evalues[k] = gevalues[k];
            
            // === copy back the result to gramA on the GPU and use it for the updates
            magma_zsetmatrix(gramDim, gramDim, gevectors, ldgram, gramA, ldgram);

            if (restart == 0) {
                // === contribution from P to the new X (in new search direction P)
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, cBlockSize, 
                            c_one, blockP, m, gramA(n+cBlockSize,0), ldgram, c_zero, dwork, m);
                magmablas_zlacpy( MagmaUpperLower, m, n, dwork, m, blockP, m);
 
                // === contribution from R to the new X (in new search direction P)
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, cBlockSize,
                            c_one, blockR, m, gramA(n,0), ldgram, c_one, blockP, m);

                // === corresponding contribution from AP to the new AX (in AP)
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, cBlockSize,
                            c_one, blockAP, m, gramA(n+cBlockSize,0), ldgram, c_zero, dwork, m);
                magmablas_zlacpy( MagmaUpperLower, m, n, dwork, m, blockAP, m);

                // === corresponding contribution from AR to the new AX (in AP)
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, cBlockSize,
                            c_one, blockAR, m, gramA(n,0), ldgram, c_one, blockAP, m);
            }
            else {
                // === contribution from R (only) to the new X
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, cBlockSize,
                            c_one, blockR, m, gramA(n,0), ldgram, c_zero, blockP, m);

                // === corresponding contribution from AR (only) to the new AX
                magma_zgemm(MagmaNoTrans, MagmaNoTrans,m, n, cBlockSize,
                            c_one, blockAR, m, gramA(n,0), ldgram, c_zero, blockAP, m);
            }
            
            // === contribution from old X to the new X + the new search direction P
            magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                        c_one, blockX, m, gramA, ldgram, c_zero, dwork, m);
            magmablas_zlacpy( MagmaUpperLower, m, n, dwork, m, blockX, m);
            magma_zaxpy(m*n, c_one, blockP, 1, blockX, 1);
            
            // === corresponding contribution from old AX to new AX + AP
            magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                        c_one, blockAX, m, gramA, ldgram, c_zero, dwork, m);
            magmablas_zlacpy( MagmaUpperLower, m, n, dwork, m, blockAX, m);
            magma_zaxpy(m*n, c_one, blockAP, 1, blockAX, 1);
            
            condestGhistory[iterationNumber+1]=condestG;
            if (verbosity==1) {

                // double res;
                // magma_zgetmatrix(1, 1, 
                //                  (magmaDoubleComplex*)residualNorms(0, iterationNumber), 1, 
                //                  (magmaDoubleComplex*)&res, 1);
                // 
                //  printf("Iteration %4d, CBS %4d, Residual: %10.7f\n",
                //         iterationNumber, cBlockSize, res);
                printf("%4d-%2d ", iterationNumber, cBlockSize); 
                magma_dprint_gpu(1, n, residualNorms(0, iterationNumber), 1);
            }

            restart = 0;
        }   // === end for iterationNumber = 1,maxIterations =======================

    
    // =============================================================================
    // === postprocessing;
    // =============================================================================

    // === compute the real AX and corresponding eigenvalues
    magma_z_bspmv(m, n, c_one, A, blockX, c_zero, blockAX );
    magma_zgemm(MagmaConjTrans, MagmaNoTrans, n, n, m,
                c_one,  blockX, m, blockAX, m, c_zero, gramM, n);

    magma_zheevd_gpu( MagmaVec, MagmaUpper,
                      n, gramM, n, gevalues, dwork, n, hwork, lwork, 
                      #if defined(PRECISION_z) || defined(PRECISION_c)
                      rwork, lrwork,
                      #endif
                      iwork, liwork, info );
   
    for(int k =0; k<n; k++)
        evalues[k] = gevalues[k];

    // === update X = X * evectors
    magmablas_zlacpy( MagmaUpperLower, m, n, blockX, m, dwork, m);
    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                c_one, dwork, m, gramM, n, c_zero, blockX, m);

    // === update AX = AX * evectors to compute the final residual
    magmablas_zlacpy( MagmaUpperLower, m, n, blockAX, m, dwork, m);
    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                c_one, dwork, m, gramM, n, c_zero, blockAX, m);

    // === compute R = AX - evalues X
    magmablas_zlacpy( MagmaUpperLower, m, n, blockAX, m, blockR, m);
    for(int i=0; i<n; i++)
        magma_zaxpy(m, MAGMA_Z_MAKE(-evalues[i], 0), blockX, 1, blockR, 1);

    // === residualNorms[iterationNumber] = || R ||    
    magmablas_dznrm2_cols(m, n, blockR, m, residualNorms(0, iterationNumber));

    printf("Eigenvalues:\n");
    for(int i =0; i<n; i++)
        printf("%e  ", evalues[i]);
    printf("\n\n");

    printf("Final residuals:\n");
    magma_dprint_gpu(n, 1, residualNorms(0, iterationNumber), n);
    printf("\n\n");

    //=== Print residual history in a file for plotting ====
    double *hresidualNorms;
    magma_dmalloc_cpu(&hresidualNorms, (iterationNumber+1) * n);
    magma_zgetmatrix(n, iterationNumber, 
                     (magmaDoubleComplex*)residualNorms, n, 
                     (magmaDoubleComplex*)hresidualNorms, n);

    printf("Residuals are stored in file residualNorms\n");
    printf("Plot the residuals using: myplot \n");
    
    FILE *residuals_file;
    residuals_file = fopen("residualNorms", "w");
    for(int i =1; i<iterationNumber; i++) {
        for(int j = 0; j<n; j++)
            fprintf(residuals_file, "%f ", *hresidualNorms(j,i));
        fprintf(residuals_file, "\n");
    }
    fclose(residuals_file);
    magma_free_cpu(hresidualNorms);

    // === free work space
    magma_free(     residualNorms   );
    magma_free_cpu( condestGhistory );
    magma_free_cpu( gevalues        );
    magma_free_cpu( iwork           );

    magma_free_pinned( hW           );
    magma_free_pinned( gevectors    );
    magma_free_pinned( h_gramB      );

    magma_free(     gramM           );
    magma_free(     gramA           );
    magma_free(     gramB           );
    magma_free(  activeMask         );

    #if defined(PRECISION_z) || defined(PRECISION_c)
    magma_free_cpu( rwork           );
    #endif

    return MAGMA_SUCCESS;
}
