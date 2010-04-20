#include <stdlib.h>
#include "cublas.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cublas.h"
#include "magma.h"
#include "magmablas.h"

#define max(a,b) (a>=b)?a:b
#define BLOCK_SIZE 32 

void error( char *message )
{
	fprintf( stderr, "ERROR: %s\n", message );
	exit (1);
}
#define assert( condition, ... ) { if( !( condition ) ) error( __VA_ARGS__ ); }
inline void Q( cudaError_t status ) { assert( status == cudaSuccess, "CUDA Runtime fails" ); }
inline void Q( cublasStatus status ){ assert( status == CUBLAS_STATUS_SUCCESS, "CUBLAS fails" ); }
inline void Q( CUresult status ) { assert( status == CUDA_SUCCESS, "CUDA Driver fails" ); }

/* 
 * verify_answer
 */
extern "C" void verify_answer (int m, int n, float *d_b1, float *d_b2, double *resid)
{
	long i;
	float * h_btmp1, *h_btmp2;

	h_btmp1 = (float*)malloc(n*m*sizeof(float));
	h_btmp2 = (float*)malloc(n*m*sizeof(float));

	cudaMemcpy (h_btmp1, d_b1, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy (h_btmp2, d_b2, m*n*sizeof(float), cudaMemcpyDeviceToHost);
	
	for (i=0; i<n*m; i++)
		h_btmp1[i] -= h_btmp2[i];

	*resid = (double)slange_ ("F", &m, &n, h_btmp1, &m, NULL)/(double)slange_ ("F", &m, &n, h_btmp2, &m, NULL);

	free (h_btmp1);
	free (h_btmp2);
}


int main(int argc, char** argv)
{
	int m2, n2;
	double flops;
	int n=BLOCK_SIZE, m=BLOCK_SIZE, i, j;
	double resid;
	double t1;
	float t2;
	float t3;
	unsigned int timer = 0;
	float pone = 2.0f;
	float *h_A, *h_b;
	float *d_A, *d_b1, *d_b2;
	
	int from, step ,to;
	char side, uplo, transa, diag; 
	int lda, ldb;
	TimeStruct start, end;

	from = 500;
	step = 233;
	to = 5000;
	
	side = 'L';
	uplo = 'L';
	transa = 'N';
	diag = 'N';
	
	printf ("magma_strsm testing\n");
	printf ("\nusage: \n\t./testing_strsm -m (matrix size) -r from step to -p SIDE, UPLO, TRANSA, DIAG \n");		
	printf ("for example:\n\t./testing_strsm -r 1024 1024 4096 -p L L N U\n\n");

	/* read command line parameter */
	for( i = 1; i < argc; i++ )
	{
		/*
		if( strcmp( argv[i], "-tb" ) == 0 )
		{	tb = atoi(argv[i+1]);		i++;		}
		*/
	
		if( strcmp( argv[i], "-m" ) == 0 )
		{	m = atoi(argv[i+1]);		
			
			from = step = to = m;		
			i++;		
		}
		
		if( strcmp( argv[i], "-r" ) == 0 )
		{	
			from = atoi(argv[i+1]);		
			step = atoi(argv[i+2]);		
			to = atoi(argv[i+3]);		
			i+=3;			
		}
		
		if( strcmp( argv[i], "-p" ) == 0 )
		{	
			side = *(argv[i+1]);
			uplo = *(argv[i+2]);
			transa = *(argv[i+3]);
			diag = *(argv[i+4]);
			i+=4;			
		}
	}
	
	CUdevice dev;
	CUcontext context;

	Q( cuInit( 0 ) );
	Q( cuDeviceGet( &dev, 0 ) );
	Q( cuCtxCreate( &context, 0, dev ) );
	Q( cublasInit( ) );

	printout_devices( );


	/* set the testing case */
	printf ("\ntesting case: SIDE=%c, UPLO=%c, TRANSA=%c, DIAG=%c\n\n", side, uplo, transa, diag); 

	printf ("M\tN\t\tcublas\t\tourblas\t\tresid\n"); 

	if (side=='r' || side=='R')	//testing for side='R'
	{
#ifdef tallskinny
		for (n=from; n<=to; n+=step)
		{
			m = 2*BLOCK_SIZE+13;
#else		
		for (m=from; m<=to; m+=step)
		{
			n = 2*BLOCK_SIZE+13;
#endif
			n2 = n*n;
			srand (n);

			ldb = m;
			lda = n;
			/* Allocate host memory for the matrices */
			h_A = (float*)malloc(n2 * sizeof(float));
			if (h_A == NULL)
			{	printf ("!!!! host memory allocation error in GetData\n"); exit(0);	}
			h_b = (float*)malloc(n * m * sizeof(float));
			if (h_b == NULL)
			{	printf ("!!!! host memory allocation error in GetData\n"); exit(0);	}

			for (j = 0; j < n2; j++) 
				h_A[j] = rand() / (float)RAND_MAX / 10;

			for (i=0; i<n; i++)
				h_A[i*n+i] += 10;

			for (i=0; i<n*m; i++)
				h_b[i] = rand() / (float)RAND_MAX / 10;

			cudaMalloc((void**)&d_A, n2*sizeof(float));
			cudaMalloc((void**)&d_b1, m*n*sizeof(float));
			cudaMalloc((void**)&d_b2, m*n*sizeof(float));
			cudaMemcpy (d_A, h_A, n2*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy (d_b1, h_b, m*n*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy (d_b2, h_b, m*n*sizeof(float), cudaMemcpyHostToDevice);

			//----- do cublas-----//
			start = get_current_time();
			cublasStrsm (side, uplo, transa, diag, m, n, pone, d_A, lda, d_b1, ldb);
			end = get_current_time();
			double cublas_time = GetTimerValue(start,end) ;

			//----- do it ourselves -----//
			start = get_current_time();

			magmablas_strsm (side, uplo, transa, diag, m, n, pone, d_A, lda, d_b2, ldb);//			<----------------------------

			end = get_current_time();
			double magma_time = GetTimerValue(start,end) ;
			/************************/

			verify_answer (m, n, d_b1, d_b2, &resid);

			flops = 1/1e9*(double)m*(double)n*(double)n;
			printf ("%d\t%d\t%f\t%f\t%e\n", m, n, flops/cublas_time, flops/magma_time, resid); 

			free (h_A);
			free (h_b);
			cudaFree(d_A);
			cudaFree(d_b1);
			cudaFree(d_b2);
		}
	}
	else	//testing for side='L'
	{
#ifndef tallskinny
		for (m=from; m<=to; m+=step)
		{
			n = 73; 
#else		
		for (n=from; n<=to; n+=step)
		{
			m = 2*BLOCK_SIZE+13;
#endif
			m2 = m*m;
			srand (m*m);
			/* Allocate host memory for the matrices */
			h_A = (float*)malloc(m2 * sizeof(float));
			if (h_A == NULL)
			{	printf ("!!!! host memory allocation error in GetData\n"); exit(0);	}
			h_b = (float*)malloc(n * m * sizeof(float));
			if (h_b == NULL)
			{	printf ("!!!! host memory allocation error in GetData\n"); exit(0);	}

			for (j = 0; j < m2; j++) 
				h_A[j] = rand() / (float)RAND_MAX / 50;

			for (i=0; i<m; i++)
				h_A[i*m+i] += 10;

			for (i=0; i<n*m; i++)
				h_b[i] = rand() / (float)RAND_MAX / 50;

			cudaMalloc((void**)&d_A, m2*sizeof(float));
			cudaMalloc((void**)&d_b1, m*n*sizeof(float));
			cudaMalloc((void**)&d_b2, m*n*sizeof(float));
			cudaMemcpy (d_A, h_A, m2*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy (d_b1, h_b, m*n*sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy (d_b2, h_b, m*n*sizeof(float), cudaMemcpyHostToDevice);

			//----- do cublas-----//
			start = get_current_time();
			cublasStrsm (side, uplo, transa, diag, m, n, pone, d_A, m, d_b1, m);
			end = get_current_time();
			double cublas_time = GetTimerValue(start,end) ;

			//----- do it ourselves -----//
			start = get_current_time();

			magmablas_strsm (side, uplo, transa, diag, m, n, pone, d_A, m, d_b2, m);//			<----------------------------

			end = get_current_time();
			double magma_time = GetTimerValue(start,end) ;
			/************************/

			verify_answer (m, n, d_b1, d_b2, &resid);

			flops = 1/1e6*(double)n*(double)m*(double)m;
			printf ("%d\t%d\t\t%f\t%f\t%e\n", m, n, flops/cublas_time, flops/magma_time, resid); 

			free (h_A);
			free (h_b);
			cudaFree(d_A);
			cudaFree(d_b1);
			cudaFree(d_b2);
		}
	}
}



