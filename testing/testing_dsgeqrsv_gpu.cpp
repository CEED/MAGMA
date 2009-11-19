/*
    -- MAGMA (version 0.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2009
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing dgeqrs
*/
int main( int argc, char** argv) 
{

 FILE *fp ;
 fp = fopen("results_dsgeqrsv_gpu.txt","w");
 if( fp == NULL ) return 1;
 printf("Iterative Refinement- QR \n");
 fprintf(fp, "Iterative Refinement- QR \n");
    printf("\n");


    cuInit( 0 );
    cublasInit( );
    printout_devices( );

    double *h_A, *h_R, *h_work_d, *tau_d;
    double *d_A, *d_work_d, *d_x;
    double gpu_perf, cpu_perf;
    float *tau , *h_work  , *d_work ; 
    double *x, *b, *rr;
    double *d_b;

    TimeStruct start, end;

    /* Matrix size */
    int M, N=0, n2, lda;
    int size[10] = {1024,2048,3072,4032,5184,6016,7040,8000, 9088,10112};
    
    cublasStatus status;
    int i, j, info[1];

    if (argc != 1){
      for(i = 1; i<argc; i++){	
	if (strcmp("-N", argv[i])==0)
	  N = atoi(argv[++i]);
      }
      if (N>0) size[0] = size[5] = N;
      else exit(1);
    }
    else {
      printf("\nUsage: \n");
      printf("  testing_dsgeqrsv_gpu -N %d\n\n", 1024);
    }

    /* Initialize CUBLAS */
    status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! CUBLAS initialization error\n");
	exit(1);;
    }
    int size5 = size[7];
    lda = N;
    n2 = size5 * size5;
    float *  h_AA ;
    /* Allocate host memory for the matrix */
    h_AA = (float*)malloc(n2 * sizeof(h_AA[0]));
    if (h_AA == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
	exit(1);;
    }
      
    h_A = (double*)malloc(n2 * sizeof(h_A[0]));
    if (h_A == 0) {
        fprintf (stderr, "!!!! host memory allocation error (A)\n");
	exit(1);;
    }

    tau_d = (double*)malloc(size5 * sizeof(double));
    if (tau_d == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau_d)\n");
	exit(1);;
    }
    tau = (float*)malloc(size5 * sizeof(float));
    if (tau == 0) {
      fprintf (stderr, "!!!! host memory allocation error (tau)\n");
	exit(1);;
    }
  
    x = (double*)malloc(size5 * sizeof(double));
    b = (double*)malloc(size5 * sizeof(double));
    rr = (double*)malloc(size5 * sizeof(double));

    cudaMallocHost( (void**)&h_R,  n2*sizeof(double) );
    if (h_R == 0) {
        fprintf (stderr, "!!!! host memory allocation error (R)\n");
	exit(1);;
    }

    int nb = magma_get_dgeqrf_nb(size5);
    int nb_s = magma_get_sgeqrf_nb(size5);
    int lwork_d = (3*size5+nb)*nb;
    int lwork = (3*size5+nb)*nb_s;
    float *SWORK ;

    status = cublasAlloc(n2, sizeof(double), (void**)&d_A);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_A)\n");
	exit(1);;
    }

    status = cublasAlloc(n2, sizeof(float), (void**)&SWORK);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (SWORK)\n");
	exit(1);;
    }

   double *X , *WORK ; 
    status = cublasAlloc(size5, sizeof(double), (void**)&WORK);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! device memory allocation error (WORK)\n");
	exit(1);;
    }
    status = cublasAlloc(size5, sizeof(double), (void**)&X);
    if (status != CUBLAS_STATUS_SUCCESS) {
       fprintf (stderr, "!!!! device memory allocation error (X)\n");
       exit(1);;
    }
    int ITER[1] ; 
    status = cublasAlloc(size5, sizeof(double), (void**)&d_b);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_b)\n");
	exit(1);;
    }

    status = cublasAlloc(lwork_d, sizeof(double), (void**)&d_work_d);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_work_d)\n");
	exit(1);;
    }
    status = cublasAlloc(lwork, sizeof(float), (void**)&d_work);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_work)\n");
	exit(1);;
    }

    //status = cublasAlloc(nb, sizeof(double), (void**)&d_x);
    status = cublasAlloc(size5, sizeof(double), (void**)&d_x);
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! device memory allocation error (d_x)\n");
	exit(1);;
    }

    cudaMallocHost( (void**)&h_work_d, lwork_d*sizeof(double) );
    if (h_work_d == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
	exit(1);;
    }
    cudaMallocHost( (void**)&h_work, lwork_d*sizeof(float) );
    if (h_work == 0) {
      fprintf (stderr, "!!!! host memory allocation error (work)\n");
	exit(1);;
    }

    printf("\n\n");
    printf("           CPU GFlop/s                 GPU GFlop/s   \n");
    printf("  N          Doule           Double\tSingle\t Mixed    || b-Ax || / ||A||\n");
    printf("=========================================================================================\n");
    fprintf(fp,"\n\n");
    fprintf(fp,"           CPU GFlop/s                 GPU GFlop/s   \n");
    fprintf(fp,"  N          Doule           Double\tSingle\t Mixed    || b-Ax || / ||A||\n");
    fprintf(fp,"=========================================================================================\n");
    for(i=0; i<8; i++){
      M = N = lda = size[i]  ;
      n2 = N*N;

      for(j = 0; j < n2; j++)
	h_A[j] = rand() / (double)RAND_MAX;

      for(j=0; j<N; j++)
	rr[j] = b[j] = rand() / (double)RAND_MAX;
//      dlag2s_( &N , &N , h_A , &N, h_AA, &N , info ) ; 
//`      cublasSetVector(n2, sizeof(float), h_AA, 1, SWORK, 1);
      cublasSetVector(n2, sizeof(double), h_A, 1, d_A, 1);
      cublasSetVector(N, sizeof(double), b, 1, d_b, 1);
    //=====================================================================
    //              Mixed Precision Iterative Refinement - GPU 
    //=====================================================================
      int nrhs = 1; 
      //printf("%d %d %d \n", M , N , nrhs);
      start = get_current_time();
      magma_dsgeqrsv_gpu(M, N, nrhs, d_A,N, d_b,N, X,N, WORK, SWORK, ITER,
			 info, tau, lwork, h_work, d_work, tau_d, lwork_d, 
			 h_work_d, d_work_d);
      end = get_current_time();
      double mperf = gpu_perf = 
	(4.*N*N*N/3.+2.*N*N)/(1000000.*GetTimerValue(start,end));

    //=====================================================================
    //                 Error Computation 
    //=====================================================================
      cublasGetVector(N, sizeof(double), X, 1, x, 1);
      double work[1], fone = 1.0, mone = -1., matnorm;
      int one = 1;
      dgemv_("n", &N, &N, &mone, h_A, &N, x, &one, &fone, rr, &one);
      matnorm = dlange_("f", &N, &N, h_A, &N, work);
   
    //=====================================================================
    //                 Double Precision Solve 
    //=====================================================================

      start = get_current_time();
      magma_dgeqrf_gpu2(&M, &N, d_A, &N, tau_d, h_work_d, &lwork_d, d_work_d, info);
      magma_dgeqrs_gpu(&M, &N, &nrhs, d_A, &N, tau_d, 
		       d_b, &M, h_work_d, &lwork_d, d_work_d, info);
      end = get_current_time();
      double dperf = gpu_perf = (4.*N*N*N/3.+2.*N*N)/(1000000.*GetTimerValue(start,end));

    //=====================================================================
    //                 Single Precision Solve 
    //=====================================================================

     
      start = get_current_time();
      magma_sgeqrf_gpu2(&M, &N, SWORK, &N, tau, h_work, &lwork, d_work, info);
      magma_sgeqrs_gpu(&M, &N, &nrhs, SWORK, &N, tau, 
		       SWORK + M * N , &M, h_work, &lwork, d_work, info);
      end = get_current_time();
      double sperf = gpu_perf = (4.*N*N*N/3.+2.*N*N)/(1000000.*GetTimerValue(start,end));

 


      /* =====================================================================
         Performs operation using LAPACK 
	 =================================================================== */
      start = get_current_time();
      dgeqrf_(&M, &N, h_A, &lda, tau_d, h_work_d, &lwork_d, info);
      if (info[0] < 0)  
	printf("Argument %d of sgeqrf had an illegal value.\n", -info[0]);

      // Solve the least-squares problem: min || A * X - B ||
      dormqr_("l", "t", &M, &nrhs, &M, h_A, &lda,
	      tau_d, b, &M, h_work_d, &lwork_d, info);
      // B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS)
      dtrsm_("l", "u", "n", "n", &M, &nrhs, &fone, h_A, &lda, b, &M);
      end = get_current_time();
      cpu_perf = (4.*N*N*N/3.+2.*N*N)/(1000000.*GetTimerValue(start,end));
      printf("%5d \t%8.2f\t%9.2f\t%6.2f\t%6.2f  \t %e",
             size[i], cpu_perf, dperf, sperf, mperf , 
             dlange_("f", &N, &nrhs, rr, &N, work)/matnorm );
      printf(" %2d \n", ITER[0]);
      fprintf(fp,"%5d \t%8.2f\t%9.2f\t%6.2f\t%6.2f  \t%e",
             size[i], cpu_perf, dperf, sperf, mperf , 
             dlange_("f", &N, &nrhs, rr, &N, work)/matnorm );
      fprintf(fp, " %2d \n", ITER[0]);

      if (argc != 1)
	break;
    }

    fclose(fp);
    /* Memory clean up */
    free(h_A);
    free(h_AA);
    free(tau_d);
    free(tau);
    free(x);
    free(b);
    free(rr);
    cublasFree(h_work_d);
    cublasFree(d_work_d);
    cublasFree(h_work);
    cublasFree(d_work);
    cublasFree(d_x);
    cublasFree(h_R);
    cublasFree(d_A);
    cublasFree(d_b);
    cublasFree(WORK);
    cublasFree(SWORK);
    cublasFree(X);

    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf (stderr, "!!!! shutdown error (A)\n");
    }
}
