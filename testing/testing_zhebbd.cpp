/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"

#if defined(USEMKL)
#include <mkl_service.h>
#endif

// Flops formula
#define PRECISION_z
#if defined(PRECISION_z) || defined(PRECISION_c)
#define FLOPS(n) ( 6. * FMULS_HETRD(n) + 2. * FADDS_HETRD(n))
#else
#define FLOPS(n) (      FMULS_HETRD(n) +      FADDS_HETRD(n))
#endif

extern "C" magma_int_t
magma_zhebbd(char uplo, magma_int_t n,
             cuDoubleComplex *a, magma_int_t lda,
             cuDoubleComplex *tau,
             cuDoubleComplex *work, magma_int_t lwork,
             magma_int_t *info);

extern "C" magma_int_t
magma_zhebbd2(char uplo, magma_int_t n, magma_int_t nb,
             cuDoubleComplex *a, magma_int_t lda,
             cuDoubleComplex *tau,
             cuDoubleComplex *work, magma_int_t lwork,
             cuDoubleComplex *dT,  
             magma_int_t *info);

extern "C" magma_int_t
magma_zhetrd_bhe2trc( int THREADS, int WANTZ, char uplo, int NE, int n, int nb, 
                   cuDoubleComplex *A, int LDA, double *D, double *E, cuDoubleComplex *dT1, int ldt1);


#if defined(PRECISION_z) || defined(PRECISION_d)
extern "C" void cmp_vals(int n, double *wr1, double *wr2, double *nrmI, double *nrm1, double *nrm2);
extern "C" void zcheck_eig_(char *JOBZ, int  *MATYPE, int  *N, int  *NB,
                       cuDoubleComplex* A, int  *LDA, double *AD, double *AE, double *D1, double *EIG,
                    cuDoubleComplex *Z, int  *LDZ, cuDoubleComplex *WORK, double *RWORK, double *RESU);
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhebbd
*/
int main( int argc, char** argv)
{
    TESTING_CUDA_INIT();

    magma_timestr_t       start, end;
    double           eps, flops, gpu_perf, gpu_time;
    cuDoubleComplex *h_A, *h_R, *h_work, *dT1;
    cuDoubleComplex *tau;
    double *D, *E;

    /* Matrix size */
    magma_int_t N = 0, n2, lda, lwork,ldt;
    magma_int_t size[10] = {1024,2048,3072,4032,5184,6016,7040,8064,9088,10112};

    magma_int_t i, j, k, info, nb, THREADS, checkres, once = 0;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    char *uplo = (char *)MagmaLowerStr;

    int WANTZ=0;
    THREADS=1;
    int NE=N;
    if (argc != 1){
        for(i = 1; i<argc; i++){
            if (strcmp("-N", argv[i])==0) {
                N = atoi(argv[++i]);
                once = 1;
            }
            else if (strcmp("-threads", argv[i])==0) {
                THREADS = atoi(argv[++i]);
            }
            else if (strcmp("-wantz", argv[i])==0) {
                WANTZ = atoi(argv[++i]);
            }
            else if (strcmp("-NE", argv[i])==0) {
                NE = atoi(argv[++i]);
            }
            else if (strcmp("-U", argv[i])==0)
                uplo = (char *)MagmaUpperStr;
            else if (strcmp("-L", argv[i])==0)
                uplo = (char *)MagmaLowerStr;
        }
        if ( N > 0 )
            printf("  testing_zhebbd -L|U -N %d\n\n", N);
        else
        {
            printf("\nUsage: \n");
            printf("  testing_zhebbd -L|U -N %d\n\n", 1024);
            exit(1);
        }
    }
    else {
        printf("\nUsage: \n");
        printf("  testing_zhebbd -L|U -N %d\n\n", 1024);
        N = size[9];
    }
        
    checkres  = 0; //getenv("MAGMA_TESTINGS_CHECK") != NULL;
 
    eps = lapackf77_dlamch( "E" );
    lda = N;
    ldt = N;
    n2  = lda * N; 
    nb  = 64; //64; //magma_get_zhebbd_nb(N);
    /* We suppose the magma nb is bigger than lapack nb */
    lwork = N*nb; 

    /* Allocate host memory for the matrix */
    TESTING_MALLOC(    h_A,    cuDoubleComplex, lda*N );
    TESTING_HOSTALLOC( h_R,    cuDoubleComplex, lda*N );
    TESTING_HOSTALLOC( h_work, cuDoubleComplex, lwork );
    TESTING_MALLOC(    tau,    cuDoubleComplex, N-1   );
    TESTING_HOSTALLOC( D,    double, N );
    TESTING_HOSTALLOC( E,    double, N );
    //TESTING_DEVALLOC( dT1,  cuDoubleComplex, (2*min(N,N)+(N+31)/32*32)*nb );
    TESTING_DEVALLOC( dT1,  cuDoubleComplex, (N*nb) );

    printf("\n\n");
    printf("  N    GPU GFlop/s   \n");
    printf("=====================\n");
    for(i=0; i<10; i++){
        if ( !once ) {
            N = size[i];
        }
        lda  = N;
        n2   = N*lda;
        flops = FLOPS( (double)N ) / 1e6;
        if(WANTZ) flops = 2.0*flops;

        /* ====================================================================
           Initialize the matrix
           =================================================================== */
        lapackf77_zlarnv( &ione, ISEED, &n2, h_A );

        // Make the matrix hermitian 
        {
            magma_int_t i, j;
            for(i=0; i<N; i++) {
                MAGMA_Z_SET2REAL( h_A[i*lda+i], ( MAGMA_Z_GET_X(h_A[i*lda+i]) ) );
                for(j=0; j<i; j++)
                    h_A[i*lda+j] = cuConj(h_A[j*lda+i]);
            }
        }
/*
            for(i=0; i<N; i++){ 
                for(j=0; j<N; j++){
                MAGMA_Z_SET2REAL( h_A[i*lda+j], ( MAGMA_Z_GET_X(h_A[i*lda+j]) ) );
                }
            }
*/

/*
    FILE *trace_file;
    trace_file = fopen("AJETE/Ainit", "w");
    for (j = 0; j < N ; j++) 
          for (i = 0; i < N ; i++) 
                         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+1,j+1,MAGMA_Z_REAL(h_A[j*lda+i]) ,  MAGMA_Z_IMAG(h_A[j*lda+i])  );
    fclose(trace_file);
*/



        lapackf77_zlacpy( MagmaUpperLowerStr, &N, &N, h_A, &lda, h_R, &lda );

/*
lapackf77_zlarnv( &ione, ISEED, &N, D );
lapackf77_zlarnv( &ione, ISEED, &N, E );
i= min(12,THREADS);
mkl_set_num_threads( i );
start = get_current_time();
dstedc_withZ('V', N, D, E, h_R, lda);
end = get_current_time();
printf("  Finish EIGEN   timing= %lf  threads %d ---> 0000000 \n" ,GetTimerValue(start,end) / 1000., i);
mkl_set_num_threads( 1 );
return 0;
*/
/*

    FILE *trace_file;
    trace_file = fopen("AJETE/Ainit", "w");
    for (j = 0; j < N ; j++) 
          for (i = 0; i < N ; i++) 
                         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+1,j+1,MAGMA_Z_REAL(h_R[j*lda+i]) ,  MAGMA_Z_IMAG(h_R[j*lda+i])  );
    fclose(trace_file);
*/
    /*
    int pm,pn,indi,indj,n=N;
    i=1;
                      indi = i+nb;
                  indj = i;
                  pm   = n - i - nb + 1;
                  pn   = min(i+nb-1, n-nb) -i + 1;
*/
                  /*
                  printf("voici pm pn %d %d \n",pm,pn);
              lapackf77_zgeqrf(&pm, &pn, &h_R[nb], &lda, 
                             tau, h_work, &lwork, &info);
              printf("TOTOTOTO INFO %d\n",info);
              memset(h_work, 0, lwork*sizeof(cuDoubleComplex));
            lapackf77_zlarft( "F", "C",
                              &pm, &pn, &h_R[nb], &lda,
                              tau, h_work, &pn);

*/


        /* ====================================================================
           Performs operation using MAGMA
           =================================================================== */
        start = get_current_time();
       //magma_zhebbd(uplo[0], N, h_R, lda, tau, h_work, lwork, &info);
        magma_zhebbd2(uplo[0], N, nb, h_R, lda, tau, h_work, lwork, dT1, &info);
        end = get_current_time();
        printf("  Finish BAND    timing= %lf \n" ,GetTimerValue(start,end) / 1000.);

/*        
    int  NB=nb, Vblksiz=-1, blkcnt=-1, LDV=-1, LDT =-1, INgrsiz=1, LDE=-1, BAND=6;
    Vblksiz = NB; //min(NB,64);
    LDT     = Vblksiz;
    findVTsiz(N, NB, Vblksiz, &blkcnt, &LDV);
    cuDoubleComplex *dVV2, *dTT2, dV3;
    int dVVs;
           dVVs = max(N*N,blkcnt*LDV*Vblksiz);
           printf("dvsize %lf \n",(16.0*(real_Double_t)dVVs)*1e-9);
           if( CUBLAS_STATUS_SUCCESS != cublasAlloc(dVVs, sizeof(cuDoubleComplex), (void**)&dVV2) ) { 
               printf ("!!!! -------> cublasAlloc failed for: dVV2\n" );       
               exit(-1);                                                           
           }
    
           if( CUBLAS_STATUS_SUCCESS != cublasAlloc( dVVs, sizeof(cuDoubleComplex), (void**)&dTT2) ) { 
              printf ("!!!! ---------> cublasAlloc failed for: dTT2\n" );       
              exit(-1);                                                           
           }
    
           if( CUBLAS_STATUS_SUCCESS != cublasAlloc( dVVs, sizeof(cuDoubleComplex), (void**)&dV3) ) { 
              printf ("!!!! ---------> cublasAlloc failed for: dV3\n" );       
              exit(-1);                                                           
           }

           printf("done from alloc exit\n");
           */

            /*        
    trace_file = fopen("AJETE/Aafter", "w");
    for (j = 0; j < N ; j++) 
          for (i = 0; i < N ; i++) 
                         fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,h_R[j*lda+i]);
    fclose(trace_file);
*/
 /*
        memset(h_work, 0, lwork*sizeof(cuDoubleComplex));
        cublasGetMatrix( pn, pn, sizeof(cuDoubleComplex), dT1, ldt, h_work, pn);
   trace_file = fopen("AJETE/T", "w");
    for (j = 0; j < pn ; j++) 
          for (i = 0; i < pn ; i++) 
                         fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,h_work[j*pn+i]);
    fclose(trace_file);
*/        
        
        //        dsytrd_bsy2trc(THREADS, uplo[0], N, nb, h_R, lda, D, E);
        magma_zhetrd_bhe2trc(THREADS, WANTZ, uplo[0], NE, N, nb, h_R, lda, D, E, dT1, ldt);
        end = get_current_time();
        if ( info < 0 )
            printf("Argument %d of magma_zhebbd had an illegal value\n", -info);

        gpu_perf = flops / GetTimerValue(start,end);
        gpu_time = GetTimerValue(start,end) / 1000.;

        /* =====================================================================
           Check the factorization
           =================================================================== */
        /*
        if ( checkres ) {
            FILE        *fp ;

            printf("Writing input matrix in matlab_i_mat.txt ...\n");
            fp = fopen ("matlab_i_mat.txt", "w") ;
            if( fp == NULL ){ printf("Couldn't open output file\n"); exit(1);}

            for(j=0; j<N; j++)
                for(k=0; k<N; k++)
                    {
                        #if defined(PRECISION_z) || defined(PRECISION_c)
                        fprintf(fp, "%5d %5d %11.8f %11.8f\n", k+1, j+1, 
                                h_A[k+j*lda].x, h_A[k+j*lda].y);
                        #else
                        fprintf(fp, "%5d %5d %11.8f\n", k+1, j+1, h_A[k+j*lda]);
                        #endif
                    }
            fclose( fp ) ;

          printf("Writing output matrix in matlab_o_mat.txt ...\n");
          fp = fopen ("matlab_o_mat.txt", "w") ;
          if( fp == NULL ){ printf("Couldn't open output file\n"); exit(1);}

          for(j=0; j<N; j++)
            for(k=0; k<N; k++)
              {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                fprintf(fp, "%5d %5d %11.8f %11.8f\n", k+1, j+1,
                        h_R[k+j*lda].x, h_R[k+j*lda].y);
                #else
                fprintf(fp, "%5d %5d %11.8f\n", k+1, j+1, h_R[k+j*lda]);
                #endif
              } 
          fclose( fp ) ;

        }*/



        /* =====================================================================
           Print performance and error.
           =================================================================== */
#if defined(PRECISION_z)  || defined(PRECISION_d)
        if ( checkres ) {
            printf("  Total N %5d  flops %6.2f  timing %6.2f seconds\n", N, gpu_perf, gpu_time );
            char JOBZ;
            if(WANTZ==0) 
                    JOBZ='N';
            else
                    JOBZ = 'V';
            double nrmI=0.0, nrm1=0.0, nrm2=0.0;
            int    lwork2 = 256*N;
            cuDoubleComplex *work2     = (cuDoubleComplex *) malloc (lwork2*sizeof(cuDoubleComplex));
            double *rwork2     = (double *) malloc (N*sizeof(double));
            double *D2          = (double *) malloc (N*sizeof(double));
            cuDoubleComplex *AINIT    = (cuDoubleComplex *) malloc (N*lda*sizeof(cuDoubleComplex));
            memcpy(AINIT, h_A, N*lda*sizeof(cuDoubleComplex));
            /* compute the eigenvalues using lapack routine to be able to compare to it and used as ref */
            start = get_current_time();
            #if defined(USEMKL)
            i= min(12,THREADS);
            mkl_set_num_threads( i );
            #endif

#if defined(PRECISION_z) || defined (PRECISION_c)
            lapackf77_zheev( "N", "L", &N, h_A, &lda, D2, work2, &lwork2, rwork2, &info );
#else
            lapackf77_dsyev( "N", "L", &N, h_A, &lda, D2, work2, &lwork2, &info );
#endif
            ///* call eigensolver for our resulting tridiag [D E] and for Q */
            //dstedc_withZ('V', N, D, E, h_R, lda);
            ////dsterf_( &N, D, E, &info); 
            ////
            end = get_current_time();
            printf("  Finish CHECK - EIGEN   timing= %lf  threads %d \n" ,GetTimerValue(start,end) / 1000., i);
            #if defined(USEMKL)
            mkl_set_num_threads( 1 );
            #endif

            /*
        for(i=0;i<10;i++)
                printf(" voici lpk D[%d] %e\n",i,D2[i]);
            */

            //cuDoubleComplex mydz=0.0,mydo=1.0;
            //cuDoubleComplex *Z = (cuDoubleComplex *) malloc(N*lda*sizeof(cuDoubleComplex));
           // dgemm_("N","N",&N,&N,&N,&mydo,h_R,&lda,h_A,&lda,&mydz,Z,&lda);


            /* compare result */
            cmp_vals(N, D2, D, &nrmI, &nrm1, &nrm2);


           cuDoubleComplex *WORKAJETER;
           double *RWORKAJETER, *RESU;
           WORKAJETER  = (cuDoubleComplex *) malloc( (2* N * N + N) * sizeof(cuDoubleComplex) );
           RWORKAJETER = (double *) malloc( N * sizeof(double) );
           RESU        = (double *) malloc(10*sizeof(double));
           int MATYPE;
           memset(RESU,0,10*sizeof(double));

 
           MATYPE=3;
           double NOTHING=0.0;
           start = get_current_time();
           // check results
           printf("---------1--------\n");
           zcheck_eig_(&JOBZ, &MATYPE, &N, &nb, AINIT, &lda, &NOTHING, &NOTHING, D2 , D, h_R, &lda, WORKAJETER, RWORKAJETER, RESU );
           printf("---------2--------\n");
           end = get_current_time();
           printf("  Finish CHECK - results timing= %lf \n" ,GetTimerValue(start,end) / 1000.);

           printf("\n");
           printf(" ================================================================================================================\n");
           printf("   ==> INFO voici  threads=%d    N=%d    NB=%d   WANTZ=%d\n",THREADS,N, nb, WANTZ);
           printf(" ================================================================================================================\n");
           printf("            DSBTRD                : %15s \n", "STATblgv9withQ    ");
           printf(" ================================================================================================================\n");
           if(WANTZ>0)
              printf(" | A - U S U' | / ( |A| n ulp )   : %15.3E   \n",RESU[0]); 
           if(WANTZ>0)
              printf(" | I - U U' | / ( n ulp )         : %15.3E   \n", RESU[1]);
           printf(" | D1 - EVEIGS | / (|D| ulp)      : %15.3E   \n",  RESU[2]);
           printf(" max | D1 - EVEIGS |              : %15.3E   \n",  RESU[6]);
           printf(" ================================================================================================================\n\n\n");
       
           printf(" ****************************************************************************************************************\n");
           printf(" * Hello here are the norm  Infinite (max)=%e  norm one (sum)=%e   norm2(sqrt)=%e *\n",nrmI, nrm1, nrm2);
           printf(" ****************************************************************************************************************\n\n");

        } 
#endif         
        printf("  Total N %5d  flops %6.2f        timing %6.2f seconds\n", N, 0.0, gpu_time );
        printf("============================================================================\n\n\n");

        if ( once )
            break;
    }

    /* Memory clean up */
    TESTING_FREE( h_A );
    TESTING_FREE( tau ); 
    TESTING_HOSTFREE( h_R ); 
    TESTING_HOSTFREE( h_work ); 

    /* Shutdown */
    TESTING_CUDA_FINALIZE();
    return EXIT_SUCCESS;
}
