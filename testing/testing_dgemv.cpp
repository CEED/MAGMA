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



void error( char *message )
{
	fprintf( stderr, "ERROR: %s\n", message );
	exit (1);
}

#define assert( condition, ... ) { if( !( condition ) ) error( __VA_ARGS__ ); }
inline void Q( cudaError_t status ) { assert( status == cudaSuccess, "CUDA Runtime fails" ); }
inline void Q( cublasStatus status ){ assert( status == CUBLAS_STATUS_SUCCESS, "CUBLAS fails" ); }
inline void Q( CUresult status ) { assert( status == CUDA_SUCCESS, "CUDA Driver fails" ); }
int fone = 0 ; 
void fill( double *A, int n, int maxi )
{	
	for( int j = 0; j < n; j++ ){
		A[j] =    double( (rand()%(maxi*2+1)) - maxi ) / ( maxi + 1.f );
        }
}	

double diff( int m, int n, double *A, int lda, double *B, int ldb )
{
	double err = 0;
	for( int j = 0; j < m; j++ )
		for( int i = 0; i < n; i++ )
		{
			double e =fabs ( A[i+j*lda] - B[i+j*ldb] );
			if( e > err ){
				err = e;
			}
		}
	return err;
}

extern "C" void mdgemv (char side , char uplo , int m , double alpha ,  double *A , int lda ,
 double *B , int ldb , double beta , double *C , int ldc );


int main(int argc, char **argv)
{	
        FILE *fp ; 
        fp = fopen ("results_dgemv.txt", "w") ;
        if( fp == NULL ){ printf("Couldn't open output file\n"); exit(1);}

        printf("GEMV Double Precision\n");
        fprintf(fp, "GEMV Double Precision\n");
	printf( "\n");
	fprintf(fp,  "\n");

        printf("Usage\n\t\t testing_dgemv N\n");
        fprintf(fp, "Usage\n\t\t testing_dgemv N\n");
	printf( "\n");
	fprintf(fp,  "\n");
	CUdevice dev;
	CUcontext context;

	Q( cuInit( 0 ) );
	Q( cuDeviceGet( &dev, 0 ) );
	Q( cuCtxCreate( &context, 0, dev ) );
	Q( cublasInit( ) );
	
	printout_devices( );
	
        TimeStruct start, end;
	const int N =12*1024+64;
	
	double *A = (double*)malloc( N*N*sizeof( double ) );
	double *C = (double*)malloc( N*N*sizeof( double ) );
	
	assert( A != NULL && C != NULL, "memory allocation error" );
	
	fone = 0 ; 
	fill( A, N*N, 31 );
	fone = 0 ; 
	fill( C, N*N, 31 );
	
	double *cublas_result = (double*)malloc( N*N*sizeof( double ) );
	double *our_result = (double*)malloc( N*N*sizeof( double ) );
	
	assert( cublas_result != NULL && our_result != NULL, "memory allocation error" );
	
	printf( "\n");
	fprintf(fp,  "\n");
	const int nb = 64;
	printf( "   n   CUBLAS,Gflop/s   MAGMABLAS0.2,Gflop/s   \"error\"\n" );
	fprintf(fp, "   n   CUBLAS,Gflop/s   MAGMABLAS0.2,Gflop/s   \"error\"\n" );
	printf("==============================================================\n");
	fprintf(fp, "==============================================================\n");
	for( int idim = 1; idim <= N/nb; idim = int((idim+1)*1.1) )
	{
		int dim =nb*idim;
                if( argc == 2 ) {
			dim = atoi( argv[1] ); 
			idim = N ; 
                }	
	
		const int m = dim;
		const int n = dim;
		const int k = dim;
		const int lda = m;
		const int ldc = m;
		const double alpha = 1;
		const double beta = 1;
		
		printf( "%5d ", n  );
		fprintf( fp, "%5d ", n  );
      /* =====================================================================
         Performs operation using CUDA-BLAS
         =================================================================== */

		{
			double *dA, *dC, *dB;
			Q( cublasAlloc( m*m, sizeof(double), (void**)&dA ) );
			Q( cublasAlloc( m*1, sizeof(double), (void**)&dC ) );
			Q( cublasAlloc( m*1, sizeof(double), (void**)&dB ) );
			
			Q( cublasSetMatrix( m, m, sizeof( double ), A, lda, dA, lda ) );
			Q( cublasGetError( ) );
			Q( cublasSetMatrix( m, 1, sizeof( double ), C, ldc, dC, ldc ) );
			Q( cublasGetError( ) );
			Q( cublasSetMatrix( m, 1, sizeof( double ), C+m, ldc, dB, ldc ) );
			Q( cublasGetError( ) );
			cublasDgemv('n', m, m, 1.0, dA, lda, dB, 1, 0., dC, 1);

			Q( cublasGetError( ) );
			Q( cublasGetMatrix( m, 1, sizeof( double ), dC, ldc, cublas_result, ldc ) );
			Q( cublasGetError( ) );
		     
	
			double cublas_time;
                        start = get_current_time();
			cublasDgemv('n', m, m, 1.0, dA, lda, dB, 1, 0., dC, 1);

			end = get_current_time();
		        cublas_time = GetTimerValue(start,end) ; 
			
			cublasFree( dA );
			cublasFree( dC );
			cublasFree( dB ) ; 
			double cublas_gflops = double(2.*dim*dim)/cublas_time/1e6;
			printf( "%11.2f", cublas_gflops );
			fprintf(fp,  "%11.2f", cublas_gflops );
		}
      /* =====================================================================
         Performs operation using MAGMA-BLAS
         =================================================================== */

		{
			double *dA, *dC, *dB;
			Q( cublasAlloc( m*m, sizeof(double), (void**)&dA ) );
			Q( cublasAlloc( m*1, sizeof(double), (void**)&dC ) );
			Q( cublasAlloc( m*1, sizeof(double), (void**)&dB ) );
			
			Q( cublasSetMatrix( m, m, sizeof( double ), A, lda, dA, lda ) );
			Q( cublasGetError( ) );
			Q( cublasSetMatrix( m, 1, sizeof( double ), C, ldc, dC, ldc ) );
			Q( cublasGetError( ) );
			Q( cublasSetMatrix( m, 1, sizeof( double ), C+m, ldc, dB, ldc ) );
			Q( cublasGetError( ) );

			magmablas_dgemv(m, m, dA, lda, dB, dC);

			Q( cublasGetError( ) );
			Q( cublasGetMatrix( m, 1, sizeof( double ), dC, ldc, our_result, ldc ) );
		
	
			double cublas_time;
                        start = get_current_time();
			magmablas_dgemv(m, m, dA, lda, dB, dC);

			end = get_current_time();
		        cublas_time = GetTimerValue(start,end) ; 
			
			cublasFree( dA );
			cublasFree( dC );
			cublasFree( dB ) ; 
		        	
			double cublas_gflops = double(2.*dim*dim)/cublas_time/1e6;
			printf( "\t%11.2f", cublas_gflops );
			fprintf(fp,  "\t%11.2f", cublas_gflops );
		}
      /* =====================================================================
         Computing the Difference 
         =================================================================== */

		{ 
		double difference = diff( 1, m, cublas_result, ldc, our_result, ldc );
		printf( "\t\t %8g\n", difference );
		fprintf( fp, "\t\t %8g\n", difference );
		}
	}
	
	free( A );
	free( C );
	
	free( cublas_result );
	free( our_result );
		
        fclose( fp ) ; 
	Q( cuCtxDetach( context ) );
	Q( cublasShutdown( ) );
	
	return 0;
}	
