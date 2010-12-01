#include <stdlib.h>
#include "cublas.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime_api.h"
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
void fill( double2 *A, int n, int maxi )
{	
	double f;
	for( int j = 0; j < n; j++ )
	{
		for(int i=0;i< n; i++)
		{
			f =    double( (rand()%(maxi*2+1)) - maxi ) / ( maxi + 1.f );        
			if(i==j)
			{
				A[j*n+i].x = f;
				A[j*n+i].y = 0.0f;
			}
			else
			{
				A[j*n+i].x = double( (rand()%(maxi*2+1)) - maxi ) / ( maxi + 1.f );        
				A[j*n+i].y = double( (rand()%(maxi*2+1)) - maxi ) / ( maxi + 1.f );        
			}
		}
	}
}	


void Print(double2 *A, int n)
{
	for( int i = 0; i < n; i++ )
	{
		printf("our_result[%d]=(%f,%f)\n",i, A[i].x, A[i].y);
	}
}

double diff( int m, int n, double2 *A, int lda, double2 *B, int ldb )
{
	double err = 0;
	for( int j = 0; j < m; j++ )
		for( int i = 0; i < n; i++ )
		{
			double e =fabs ( A[i+j*lda].x - B[i+j*ldb].x) + fabs(A[i+j*lda].y - B[i+j*ldb].y );
			if( e > err ){
				err = e;
			}
		}
	return err;
}

extern "C" void mzhemv (char side , char uplo , int m , double2 alpha ,  double2 *A , int lda ,
 double2 *B , int ldb , double2 beta , double2 *C , int ldc );


int main(int argc, char **argv)
{	
        FILE *fp ; 
        fp = fopen ("results_zhemv.txt", "w") ;
        if( fp == NULL ){ printf("Couldn't open output file\n"); exit(1);}

        printf("HEMV Double Complex Precision\n");
        fprintf(fp, "HEMV Double Complex Precision\n");
	printf( "\n");
	fprintf(fp,  "\n");

        printf("Usage\n\t\t testing_zhemv N\n");
        fprintf(fp, "Usage\n\t\t testing_zhemv N\n");
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
	
	double2 *A = (double2*)malloc( 2 * N*N*sizeof( double2 ) );
	double2 *C = (double2*)malloc( 2 * N*N*sizeof( double2 ) );
	
	assert( A != NULL && C != NULL, "memory allocation error" );
	
	fone = 0 ; 
	fill( A, N, 31 );
	fone = 0 ; 
	fill( C, N, 31 );
	
	double2 *cublas_result = (double2*)malloc(2 *  N*N*sizeof( double2 ) );
	double2 *our_result = (double2*)malloc(2*  N*N*sizeof( double2 ) );
	
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
		const double2 alpha = {1.0f, 0.0f};
		const double2 beta = {0.0f,0.0f};
	        int stride = 1 ; 
		int stridec = 1; 	
		printf( "%5d ", n  );
		fprintf( fp, "%5d ", n  );
      /* =====================================================================
         Performs operation using CUDA-BLAS
         =================================================================== */

		{   

	        double2 *AA = (double2*)malloc( 2 * m * m *sizeof( double2 ) );
            fill(AA,m,31);     
			double2 *dA, *dC, *dB;
			Q( cublasAlloc( m*m, sizeof(double2), (void**)&dA ) );
			Q( cublasAlloc( stridec*m*1, sizeof(double2), (void**)&dC ) );
			Q( cublasAlloc( stride*m*1, sizeof(double2), (void**)&dB ) );
			float cublas_time;
			//Q( cublasSetMatrix( m, m, sizeof( double2 ), A, lda, dA, lda ) );
			Q( cublasSetMatrix( m, m, sizeof( double2 ), AA, lda, dA, lda ) );
			Q( cublasGetError( ) );
			Q( cublasSetMatrix( m*stridec, 1, sizeof( double2 ), C, ldc*stridec, dC, stridec*ldc ) );
			Q( cublasGetError( ) );
			Q( cublasSetMatrix( m*stride, 1, sizeof( double2 ), C+m*stride, stride* ldc, dB,stride* ldc ) );
			Q( cublasGetError( ) );
                        cublasZhemv('L' , m , alpha,  dA , lda , dB , stride ,beta,  dC , stridec );
 
			Q( cublasGetError( ) );
			Q( cublasGetMatrix( m*stridec, 1, sizeof( double2 ), dC, stridec*ldc, cublas_result, stridec*ldc ) );
			Q( cublasGetError( ) );
		     
	
                        start = get_current_time();
                        cublasZhemv('L' , m , alpha,  dA , lda , dB , stride ,beta,  dC , stridec );
			end = get_current_time();
		        cublas_time = GetTimerValue(start,end) ; 
			
				float cublas_gflops = float(2.*dim*dim)/cublas_time/1e6;
			printf( "%11.2f", cublas_gflops );
			fprintf(fp,  "%11.2f", cublas_gflops );


			Q( cublasGetError( ) );
                        magmablas_zhemv( 'L' , m , alpha,  dA , lda , dB , stride ,beta,  dC , stridec );
 
			Q( cublasGetError( ) );
			Q( cublasGetMatrix( m*stridec, 1, sizeof( double2 ), dC,stridec* ldc, our_result,stridec* ldc ) );
			Q( cublasGetError( ) );
		     
	
                        start = get_current_time();
                        magmablas_zhemv( 'L' , m , alpha,  dA , lda , dB , stride ,beta,  dC , stridec );
			end = get_current_time();
		        cublas_time = GetTimerValue(start,end) ; 
			cublasFree( dA );
			cublasFree( dC );
			cublasFree( dB ) ;
		    free(AA);	
			cublas_gflops = float(2.*dim*dim)/cublas_time/1e6;
			printf( "%11.2f", cublas_gflops );
			fprintf(fp,  "%11.2f", cublas_gflops );
		}
      /* =====================================================================
         Computing the Difference 
         =================================================================== */

		{ 
		double difference = diff( 1, m*stridec, cublas_result, ldc*stridec, our_result, ldc*stridec );
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
