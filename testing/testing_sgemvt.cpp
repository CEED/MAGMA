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
void fill( float *A, int n, int maxi )
{	
	for( int j = 0; j < n; j++ ){
		A[j] =    float( (rand()%(maxi*2+1)) - maxi ) / ( maxi + 1.f );
        }
}	

float diff( int m, int n, float *A, int lda, float *B, int ldb )
{
	float err = 0;
	for( int j = 0; j < m; j++ )
		for( int i = 0; i < n; i++ )
		{
			float e =fabs ( A[i+j*lda] - B[i+j*ldb] );
			if( e > err ){
				err = e;
			}
		}
	return err;
}

extern "C" void mdsymv (char side , char uplo , int m , float alpha ,  float *A , int lda ,
 float *B , int ldb , float beta , float *C , int ldc );


int main(int argc, char **argv)
{	
        FILE *fp ; 
        fp = fopen ("results_sgemvt.txt", "w") ;
        if( fp == NULL ){ printf("Couldn't open output file\n"); exit(1);}

        printf("GEMV Transpose Sinlge Precision\n");
        fprintf(fp, "GEMV Transpose Single Precision\n");
	printf( "\n");
	fprintf(fp,  "\n");

        printf("Usage\n\t\t testing_sgemvt N\n");
        fprintf(fp, "Usage\n\t\t testing_sgemvt N\n");
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
	const int N =8*1024+64;
	
	float *A = (float*)malloc( 2 * N*N*sizeof( float ) );
	float *C = (float*)malloc( 2 * N*N*sizeof( float ) );
	
	assert( A != NULL && C != NULL, "memory allocation error" );
	
	fone = 0 ; 
	fill( A, N*N, 31 );
	fone = 0 ; 
	fill( C, N*N, 31 );
	
	float *cublas_result = (float*)malloc(2 *  N*N*sizeof( float ) );
	float *our_result = (float*)malloc(2*  N*N*sizeof( float ) );
	
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
		const float alpha = 1;
		const float beta = 1;
	        int stride = 1 ; 
		int stridec = 1; 	
		printf( "%5d ", n  );
		fprintf( fp, "%5d ", n  );
      /* =====================================================================
         Performs operation using CUDA-BLAS
         =================================================================== */

		{
			float *dA, *dC, *dB;
			Q( cublasAlloc( m*m, sizeof(float), (void**)&dA ) );
			Q( cublasAlloc( stridec*m*1, sizeof(float), (void**)&dC ) );
			Q( cublasAlloc( stride*m*1, sizeof(float), (void**)&dB ) );
			float cublas_time;
			Q( cublasSetMatrix( m, m, sizeof( float ), A, lda, dA, lda ) );
			Q( cublasGetError( ) );
			Q( cublasSetMatrix( m*stridec, 1, sizeof( float ), C, ldc*stridec, dC, stridec*ldc ) );
			Q( cublasGetError( ) );
			Q( cublasSetMatrix( m*stride, 1, sizeof( float ), C+m*stride, stride* ldc, dB,stride* ldc ) );
			Q( cublasGetError( ) );
			cublasSgemv('t', m, m, 1.0, dA, lda, dB, 1, 0., dC, 1);
			Q( cublasGetError( ) );
			Q( cublasGetMatrix( m*stridec, 1, sizeof( float ), dC, stridec*ldc, cublas_result, stridec*ldc ) );
			Q( cublasGetError( ) );
		     
	
                        start = get_current_time();
			cublasSgemv('t', m, m, 1.0, dA, lda, dB, 1, 0., dC, 1);
			end = get_current_time();
		        cublas_time = GetTimerValue(start,end) ; 
			cublasFree( dA );
			cublasFree( dC );
			cublasFree( dB ) ; 
			float cublas_gflops = float(2.*dim*dim)/cublas_time/1e6;
			printf( "%11.2f", cublas_gflops );
			fprintf(fp,  "%11.2f", cublas_gflops );
		}
      /* =====================================================================
         Performs operation using MAGMA-BLAS
         =================================================================== */
		int pp = 1 ; 
		if( pp == 1 ) 
		{
			float *dA, *dC, *dB;
			Q( cublasAlloc( m*m, sizeof(float), (void**)&dA ) );
			Q( cublasAlloc( stridec*m*1, sizeof(float), (void**)&dC ) );
			Q( cublasAlloc( stride*m*1, sizeof(float), (void**)&dB ) );
			float cublas_time;
			Q( cublasSetMatrix( m, m, sizeof( float ), A, lda, dA, lda ) );
			Q( cublasGetError( ) );
			Q( cublasSetMatrix( m*stridec, 1, sizeof( float ), C, ldc*stridec, dC, stridec*ldc ) );
			Q( cublasGetError( ) );
			Q( cublasSetMatrix( m*stride, 1, sizeof( float ), C+m*stride, stride* ldc, dB,stride* ldc ) );
			Q( cublasGetError( ) );
			magmablas_sgemvt(m, m, 1., dA, lda, dB, dC);

			Q( cublasGetError( ) );
			Q( cublasGetMatrix( m*stridec, 1, sizeof( float ), dC,stridec* ldc, our_result,stridec* ldc ) );
			Q( cublasGetError( ) );
		     
	
                        start = get_current_time();
                        magmablas_sgemvt(m, m, 1., dA, lda, dB, dC);
			end = get_current_time();
		        cublas_time = GetTimerValue(start,end) ; 
			cublasFree( dA );
			cublasFree( dC );
			cublasFree( dB ) ; 
			float cublas_gflops = float(2.*dim*dim)/cublas_time/1e6;
			printf( "%11.2f", cublas_gflops );
			fprintf(fp,  "%11.2f", cublas_gflops );
		}
      /* =====================================================================
         Computing the Difference 
         =================================================================== */

		{ 
		float difference = diff( 1, m*stridec, cublas_result, ldc*stridec, our_result, ldc*stridec );
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
