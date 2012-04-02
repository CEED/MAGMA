/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Azzam Haidar
       @author Stan Tomov

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "magma_zbulgeinc.h"
#include "checkdiag.h"
#if defined(USEMKL)
#include <mkl_service.h>
#endif
// === Define what BLAS to use ============================================
#define PRECISION_z
#if (defined(PRECISION_s) || defined(PRECISION_d))
  #define cublasZgemm magmablas_zgemm
#endif
// === End defining what BLAS to use =======================================




#define CHECK 0
#define LOGQ 0
#define LOG 0

#if (defined(PRECISION_d))
extern "C" void cmp_vals(int n, double *wr1, double *wr2, double *nrmI, double *nrm1, double *nrm2);
extern "C" void checkdiag_(char *JOBZ, int  *MATYPE, int  *N, int  *NB,
                       double* A, int  *LDA, double *AD, double *AE, double *D1, double *EIG,
                double *Z, int  *LDZ, double *WORK, double *RESU);
#endif


extern "C"  magma_int_t plasma_ceildiv(magma_int_t a, magma_int_t b);
extern "C"  void tile_bulge_applyQ_cpu(int WANTZ, char SIDE, int N, int NB, int Vblksiz, cuDoubleComplex *E, int LDE, cuDoubleComplex *V, cuDoubleComplex *TAU, cuDoubleComplex *T, int *INFO);

static void barrier(magma_int_t my_core_id, magma_int_t cores_num);
static void  *parallel_section(void *thread_id);
static void *applyQ_parallel_section(void *thread_id);
static void tile_bulge_parallel(int my_core_id);
static void tile_bulge_computeT_parallel(int my_core_id);
static void tile_bulge_applyQ_parallel(int my_core_id);
static void tile_bulge_applyQ_parallel2(int my_core_id);

/*
extern "C" void TRD_hbcelr_v62sym_withQ(int N, int NB, cuDoubleComplex *A, int LDA, cuDoubleComplex *V, cuDoubleComplex *TAU, int st, int ed, int myid, int sweep, int Vblksiz);
extern "C" void TRD_type1cHLsym_withQ(int N, int NB, cuDoubleComplex *A, int LDA, cuDoubleComplex *V, cuDoubleComplex *TAU, int st, int ed, int sweep, int Vblksiz);
extern "C" void TRD_type2cHLsym_withQ(int N, int NB, cuDoubleComplex *A, int LDA, cuDoubleComplex *V, cuDoubleComplex *TAU, int st, int ed, int sweep, int Vblksiz);
extern "C" void TRD_type3cHLsym_withQ(int N, int NB, cuDoubleComplex *A, int LDA, cuDoubleComplex *V, cuDoubleComplex *TAU, int st, int ed, int sweep, int Vblksiz);
*/
/*
#define magma_ztrdtype1cbHLsym_withQ                 magma_dtrdtype1cbHLsym_withQ
#define magma_ztrdtype2cbHLsym_withQ                 magma_dtrdtype2cbHLsym_withQ
#define magma_ztrdtype3cbHLsym_withQ                magma_dtrdtype3cbHLsym_withQ
#define magma_zbulge_applyQ                           magma_dbulge_applyQ
#define magma_zstedc_withZ                        magma_dstedc_withZ
#define cublasZgemm                                 magmablas_dgemm
*/

extern "C" void magma_ztrdtype1cbHLsym_withQ(int N, int NB, cuDoubleComplex *A, int LDA, cuDoubleComplex *V, cuDoubleComplex *TAU, int st, int ed, int sweep, int Vblksiz);
extern "C" void magma_ztrdtype2cbHLsym_withQ(int N, int NB, cuDoubleComplex *A, int LDA, cuDoubleComplex *V, cuDoubleComplex *TAU, int st, int ed, int sweep, int Vblksiz);
extern "C" void magma_ztrdtype3cbHLsym_withQ(int N, int NB, cuDoubleComplex *A, int LDA, cuDoubleComplex *V, cuDoubleComplex *TAU, int st, int ed, int sweep, int Vblksiz);
extern "C" void magma_zbulge_applyQ(int WANTZ, char SIDE, int N, int NB, int Vblksiz, cuDoubleComplex *E, int LDE, cuDoubleComplex *V, cuDoubleComplex *TAU, cuDoubleComplex *T, int *INFO, cuDoubleComplex *dV, cuDoubleComplex *dT, cuDoubleComplex *dE, int copytype );
extern "C" void  magma_zstedc_withZ(char JOBZ, int N, double *D, double * E, cuDoubleComplex *Z, int LDZ);


extern "C" magma_int_t
magma_zungqr_2stage_gpu(magma_int_t m, magma_int_t n, magma_int_t k,
                 cuDoubleComplex *da, magma_int_t ldda,
                 cuDoubleComplex *tau, cuDoubleComplex *dT,
                 magma_int_t nb, magma_int_t *info);

         

////////////////////////////////////////////////////////////////////////////////////////////////////          

 volatile magma_int_t barrier_in[MAX_THREADS_BLG];
 volatile magma_int_t barrier_out[MAX_THREADS_BLG];
 volatile magma_int_t *ss_prog;

static void barrier(magma_int_t my_core_id, magma_int_t cores_num);
static void barrier(magma_int_t my_core_id, magma_int_t cores_num)
{
    magma_int_t core;
    
    if (my_core_id == 0)    {
        for (core = 1; core < cores_num; core++)
            while (barrier_in[core] == 0);

        for (core = 1; core < cores_num; core++)
            barrier_in[core] = 0;

        for (core = 1; core < cores_num; core++)
            barrier_out[core] = 1;
    }
    else
    {
        barrier_in[my_core_id] = 1;
        while (barrier_out[my_core_id] == 0);
        barrier_out[my_core_id] = 0;
    }
}
////////////////////////////////////////////////////////////////////////////////////////////////////          

/* START CODE */
extern "C" magma_int_t magma_zhetrd_bhe2trc( int THREADS, int WANTZ, char uplo, int N, int NB, cuDoubleComplex *A1, int LDA1, double *D2, double *E2, cuDoubleComplex *dT1, int LDT1)
{
    char uplo_[2] = {uplo, 0};
    real_Double_t timelpk=0.0,tconvert=0.0,timeaplQ1=0.0,timeaplQ2=0.0,timeblg=0.0, timeaplQ=0.0, timeeigen=0.0, timegemm=0.0;
    double nrmI=0.0, nrm1=0.0, nrm2=0.0;
    FILE *trace_file;
    cuDoubleComplex c_zero    = MAGMA_Z_ZERO;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;

    
    int mklth, thread, INFO;
    int Vblksiz=-1, blkcnt=-1, LDV=-1, LDT =-1, INgrsiz=1, LDE=-1, BAND=6;
    Vblksiz = NB; //min(NB,64);
    LDT     = Vblksiz;
    findVTsiz(N, NB, Vblksiz, &blkcnt, &LDV);

    int i,j;
    int NBTILES, LDA2;
    i = N/NB;
    NBTILES = i*NB==N ? i:i+1;

    /************************************************* 
     *     INITIALIZING MATRICES 
     * ***********************************************/
    /* A1 is equal to A2 */
    LDA2 = NB+1+(NB-1);
    if(LDA2>N){
            printf("error matrix too small N=%d NB=%d\n",N,NB);
            return -14;
    }
    if((N<=0)||(NB<=0)){
            printf("error N or NB <0 N=%d NB=%d\n",N,NB);
            return -14;
    }


/*

       trace_file = fopen("AJETE/Abefore", "w");
    for (j = 0; j < (N-NB); j++)
    {
        for (i = 0; i < NB+1; i++)
        {
               fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+j+1,j+1,MAGMA_Z_REAL(A1[i+j*LDA1+j]) ,  MAGMA_Z_IMAG(A1[i+j*LDA1+j])  );
        }
    }        
    for (j = 0; j < NB; j++)
    {
        for (i = 0; i < NB-j; i++)
        {
               fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+(N-NB+j)+1,N-NB+j+1,MAGMA_Z_REAL(A1[i+(N-NB+j)*LDA1+(N-NB+j)]) ,  MAGMA_Z_IMAG(A1[i+(N-NB+j)*LDA1+(N-NB+j)])  );
        }
    }


       fclose(trace_file);

*/




    timelpk = get_time_azz();
    /* copy the input matrix into a matrix A2 with band storage */
    cuDoubleComplex *A2    = (cuDoubleComplex *) malloc (N*LDA2*sizeof(cuDoubleComplex));
    memset(A2 , 0, N*LDA2*sizeof(cuDoubleComplex));        
    for (j = 0; j < (N-NB); j++)
    {
        for (i = 0; i < NB+1; i++)
        {
            A2[i+j*LDA2]   = A1[i+j*LDA1+j];
            A1[i+j*LDA1+j] = c_zero;
        }
    }        
    for (j = 0; j < NB; j++)
    {
        for (i = 0; i < NB-j; i++)
        {
            A2[i+(N-NB+j)*LDA2] = A1[i+(N-NB+j)*LDA1+(N-NB+j)]; 
            A1[i+(N-NB+j)*LDA1+(N-NB+j)] =c_zero;
        }
    }

    for (j = 0; j < N-NB; j++)
       A1[NB+j*LDA1+j] = c_one;

    timelpk = get_time_azz() - timelpk;
    printf("  Finish CONVERT timing= %lf \n" ,timelpk); 

    

    /************************************************* 
     *     compute Q1 
     * ***********************************************/
    cuDoubleComplex *da, *NOTUSED;
    if((WANTZ>0)&(WANTZ!=4)){
       cublasStatus status;
       status = cublasAlloc(N*LDA1, sizeof(cuDoubleComplex), (void**)&da);
       if (status != CUBLAS_STATUS_SUCCESS) {
         fprintf (stderr, "!!!! device memory allocation error (magma_dsytrd_bsy2trc)\n");
         return 0;
       }
    
       
       



       cudaDeviceSynchronize();
       timeaplQ1 = get_time_azz();
       cublasSetMatrix( N, LDA1, sizeof(cuDoubleComplex), A1, LDA1, da, LDA1);
       magma_zungqr_2stage_gpu(N, N, N, da, LDA1, NOTUSED, dT1, NB, &INFO);
       cudaDeviceSynchronize();
       cublasGetMatrix( N, LDA1, sizeof(cuDoubleComplex), da, LDA1, A1, LDA1);
       timeaplQ1 = get_time_azz()-timeaplQ1;
       printf("  Finish applyQ1 timing= %lf \n" ,timeaplQ1); 
       
       /*
       trace_file = fopen("AJETE/Q1", "w");
       for (j = 0; j < N ; j++) 
             for (i = 0; i < N ; i++) 
                        //fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,A1[j*LDA1+i]);
                         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+1,j+1,MAGMA_Z_REAL(A1[j*LDA1+i]) ,  MAGMA_Z_IMAG(A1[j*LDA1+i])  );
       fclose(trace_file);*/
       
    }
    /***********************************************/
    

    /************************************************* 
     *    For local check use
     * ***********************************************/
    char JOBZ;
    double *D1;
    cuDoubleComplex *AINIT;
    int MM,NN,LDAINIT;
    if(CHECK)
    {
        MM=NB+1;
        NN=N;
        LDAINIT=NB+1;
        AINIT = (cuDoubleComplex *) malloc(MM*NN*sizeof(cuDoubleComplex) );
        memset(AINIT , 0, MM*NN*sizeof(cuDoubleComplex));        
        lapackf77_zlacpy("A", &MM, &NN, A2, &LDA2, AINIT, &LDAINIT );
    }
    /***********************************************/


    cuDoubleComplex *T     = (cuDoubleComplex *) malloc (blkcnt*LDT*Vblksiz*sizeof(cuDoubleComplex));
    cuDoubleComplex *TAU   = (cuDoubleComplex *) malloc (blkcnt*Vblksiz*sizeof(cuDoubleComplex));
    cuDoubleComplex *V     = (cuDoubleComplex *) malloc (blkcnt*LDV*Vblksiz*sizeof(cuDoubleComplex));
    memset(T,   0, blkcnt*LDT*Vblksiz*sizeof(cuDoubleComplex));        
    memset(TAU, 0, blkcnt*Vblksiz*sizeof(cuDoubleComplex));        
    memset(V,   0, blkcnt*LDV*Vblksiz*sizeof(cuDoubleComplex));        





    // Set input parameters
    //    ssched_init(THREADS);
    ss_prog = (int*) malloc((2*NBTILES+THREADS+10)*sizeof(int));
    int   iamdone[MAX_THREADS_BLG]; 
    int   thread_num[MAX_THREADS_BLG];
    pthread_t      thread_id[MAX_THREADS_BLG];
    pthread_attr_t thread_attr;

    //goto ed;
    
#if defined(LIBMKL)
  //  mkl_set_num_threads( 1 );
#endif
    core_in_all.cores_num = THREADS;
    core_in_all.T         = T;
    core_in_all.A         = A2;
    core_in_all.TAU       = TAU;
    core_in_all.V         = V;
    core_in_all.NB        = NB;
    core_in_all.NBTILES   = NBTILES;
    core_in_all.N         = N;
    core_in_all.LDA       = LDA2;
    core_in_all.BAND      = BAND;
    core_in_all.grsiz     = INgrsiz;
    core_in_all.timeblg   = &timeblg;//0.0;
    core_in_all.timeaplQ  = &timeaplQ;//0.0;
    core_in_all.ss_prog   = ss_prog;
    core_in_all.WANTZ     = WANTZ;
    core_in_all.SIDE      = 'R';
    core_in_all.E         = A1;
    core_in_all.LDE       = LDA1;
    core_in_all.Vblksiz   = Vblksiz;


    // Set one thread per core
    pthread_attr_init(&thread_attr);
    pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
    pthread_setconcurrency(THREADS);

    // Initializations
    for (thread = 0; thread < THREADS; thread++)
    {
        barrier_in[thread] = 0;
        barrier_out[thread] = 0;
        event_numblg[thread] = 0;
    }

   for (i = 0; i < 2*NBTILES+THREADS+10; i++)
       ss_prog[i] = 0;


    for (i = 0; i < THREADS; i++)
       iamdone[i] = 0;



    // Launch threads
    for (thread = 1; thread < THREADS; thread++)
    {
        thread_num[thread] = thread;
        pthread_create(&thread_id[thread], &thread_attr, parallel_section, &thread_num[thread]);
    }
    thread_num[0] = 0;
    parallel_section(&thread_num[0]);

    // Wait for completion
    for (thread = 1; thread < THREADS; thread++)
    {
        void *exitcodep;
        pthread_join(thread_id[thread], &exitcodep);
    }
    printf("  Finish BULGE+T timing= %lf \n" ,*(core_in_all.timeblg));

    /*================================================
     *  store resulting diag and lower diag D2 and E2
     *  note that D and E are always real
     *================================================*/  
    /*
     * STORE THE RESULTING diagonal/off-diagonal in D AND E
     */
     memset(D2, 0,  N   *sizeof(double));
     memset(E2, 0, (N-1)*sizeof(double));
    /* Make diagonal and superdiagonal elements real,
     * storing them in D and E
     */
    /* In complex case, the off diagonal element are
     * not necessary real. we have to make off-diagonal
     * elements real and copy them to E.
     * When using HouseHolder elimination,
     * the ZLARFG give us a real as output so, all the
     * diagonal/off-diagonal element except the last one are already
     * real and thus we need only to take the abs of the last
     * one.
     *  */


/*
       trace_file = fopen("AJETE/T", "w");
       for (j = 0; j < N-1 ; j++) {
         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",j+1,j+1,MAGMA_Z_REAL(A2[j*LDA2]) ,  MAGMA_Z_IMAG(A2[j*LDA2])  );
         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",j+2,j+1,MAGMA_Z_REAL(A2[j*LDA2+1]) ,  MAGMA_Z_IMAG(A2[j*LDA2+1])  );
       }
         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",N,N,MAGMA_Z_REAL(A2[(N-1)*LDA2]) ,  MAGMA_Z_IMAG(A2[(N-1)*LDA2])  );

       fclose(trace_file);
*/





#if defined(PRECISION_z) || defined(PRECISION_c)  
    if(uplo==MagmaLower){
       for (i=0; i < N-1 ; i++)
       {
          D2[i] = MAGMA_Z_REAL( A2[i*LDA2]);               
          /*
           * for Householder case, all off-diag
           * are real except the last off-diag, where we
           * have to take the abs
           */
          if(i<(N-2)){
              E2[i] = MAGMA_Z_REAL(A2[i*LDA2+1]);
          }else{
              E2[i] = MAGMA_Z_ABS(A2[i*LDA2+1]);
          }
       }
       D2[N-1] = MAGMA_Z_REAL(A2[(N-1)*LDA2]);
    } else { /* MagmaUpper not tested yet */
        for (i=0; i<N-1; i++)
        {
            D2[i]  =  MAGMA_Z_REAL(A2[i*LDA2+NB]);               
            /*
             * for Householder case, all off-diag
             * are real except the last off-diag, where we
             * have to take the abs
             */
            if( i < (N-2) ){
                E2[i] = MAGMA_Z_REAL(A2[i*LDA2+NB-1]);
            }else{
                E2[i] = MAGMA_Z_ABS(A2[i*LDA2+NB-1]);
            }
        }
        D2[N-1] = MAGMA_Z_REAL(A2[(N-1)*LDA2+NB]);
    } /* end MagmaUpper */
#else
    if( uplo == MagmaLower ){
        for (i=0; i < N-1; i++) {
            D2[i] = A2[i*LDA2];   // diag
            E2[i] = A2[i*LDA2+1]; //lower diag  
        }
        D2[N-1] = A2[(N-1)*LDA2];
    } else {
        for (i=0; i < N-1; i++) {
            D2[i] = A2[i*LDA2+NB];   // diag
            E2[i] = A2[i*LDA2+NB-1]; //lower diag  
        }
        D2[N-1] = A2[(N-1)*LDA2+NB];
    }
#endif






    /* APPLY V2 generated by the bulge to the matrix Q1 of the first stage Q2=Q2*(I-V_2*T_2*V_2') */
    cuDoubleComplex *dV2, *dQ2, *dT2, *dZ, *Q2, *Z;
    int dVsize, LDQ2=N,LDZ=N;
    int parallel=0;

    //========================
    //  WANTZ =1 : compute Q1 and Q2 and the eigenvectors Z 
    //             then make 2 GEMM's Q1*Q2*Z 
    //  WANTZ =2 : compute Q1 then apply V2 to Q1 from right 
    //             generating the global Q, then compute
    //             eigenvectors Z and make GEMM with Z==> Q*Z on GPU
    //             WANTZ=5 is similar, but the GEMM is done 
    //             implicitly during the eigenvectors computation. 
    //             assumed to be bad in perf.
    //  WANTZ =3 : compute Q1, then compute the eigenvectors Z,
    //             then apply V2 to the Z from Left, then make 
    //             GEMM with Q1 ==> Q1 * resulting 
    //  WANTZ =4 : So compute the eigenvectors Z, 
    //             then apply V2, then apply V1 to the result. 
    //             similar to WANTZ=3 but does not compute Q1 
    //             instead it apply V1 (I guess to be good for 
    //             multicore because it avoid computing Q1, while 
    //             on GPU's Q1 is computed overlaped with the 
    //             bulgechasing so for free so it is better to do
    //             a GEMM with Q1 (WANTZ=3) then to do the apply).
    //  WANTZ =5 : compute Q1 then apply V2 to Q1 from right 
    //             generating the global Q, then compute
    //             eigenvectors Z with option "V" that make 
    //             GEMM with Z==> Q*Z implicitly during the 
    //             eigenvectors computation assumed to be bad in perf
    //             because a GEMM on GPU or optimized 
    //             for multithread will be faster. 
    //             Similar to WANTZ=2 where the GEMM is done out of the eigensolver
    //                 
    //========================
    
    timeeigen = get_time_azz();
    if(WANTZ==0){
        timelpk = get_time_azz();
        // compute the eigenvalues using lapack routine to be able to compare to it and used as ref 
#if defined(USEMKL)
        mklth=THREADS; //;
        mkl_set_num_threads(mklth);
#endif
        // call eigensolver for our resulting tridiag [D E] and form E=Q*Z
        magma_zstedc_withZ('N', N, D2, E2, Z, LDZ);
#if defined(USEMKL)
        mkl_set_num_threads( 1 );
#endif
        timelpk = get_time_azz()-timelpk;
        printf("  Finish WANTZ %d  eigensolver 'N'    timing= %lf  threads %d \n" ,WANTZ, timelpk, i);
        /*
        for(i=0;i<10;i++)
                printf(" voici D[%d] %e\n",i,D2[i]);*/
    }
    if(WANTZ>0){
        if(WANTZ==1){
            // compute Q2 by applying V2 on the Identity        
            LDZ=LDA1;        
            Z    = (cuDoubleComplex *) malloc (N*N*sizeof(cuDoubleComplex));
            memset(Z , 0, N*N*sizeof(cuDoubleComplex));
            if(parallel==1){
                Q2    = (cuDoubleComplex *) malloc (N*N*sizeof(cuDoubleComplex));
                // no need to set the matrix to identity if we are using the GPU because it is done inside bulge_applyQ
                memset(Q2 , 0, N*N*sizeof(cuDoubleComplex));        
                for (j = 0; j < N; j++)
                    Q2[j+j*LDQ2]=c_one;
            }
            core_in_all.SIDE      = 'L';
            core_in_all.E         = Q2;
            core_in_all.LDE       = LDQ2;
        }else if(WANTZ==2){
            // compute the Q by applying V2 to Q1 that has been computed 
            // from the Right. Q= Q1 * ( I-V2*T2*V2') = Q1*Q2        
            LDZ=LDA1;        
            Z    = (cuDoubleComplex *) malloc (N*N*sizeof(cuDoubleComplex));
            memset(Z , 0, N*N*sizeof(cuDoubleComplex));
            core_in_all.SIDE      = 'R';
            core_in_all.E         = A1;
            core_in_all.LDE       = LDA1;
        }else if((WANTZ==3)||(WANTZ==4)){
            // wait till the Eigenvector are computed and the apply V2 from the Left, 
            // and then make gemm with Q1 (WANTZ=3) or apply V1 to the result (WANTZ=4). 
            LDZ=LDA1;        
            Z    = (cuDoubleComplex *) malloc (N*N*sizeof(cuDoubleComplex));
            memset(Z , 0, N*N*sizeof(cuDoubleComplex));
            // variable for Q2
            core_in_all.SIDE      = 'L';
            core_in_all.E         = Z;
            core_in_all.LDE       = LDZ;
            if(WANTZ==4) {
                    printf("the implementation is not finished yet. code will exit\n");
                    exit(-1);
            }
        }


        if((WANTZ==1)||(WANTZ==2)||(WANTZ==3)||(WANTZ==4)){
            timelpk = get_time_azz();
            // compute the eigenvalues using lapack routine to be able to compare to it and used as ref 
#if defined(USEMKL)
            mklth=THREADS; //;
            mkl_set_num_threads(mklth);
#endif
            // call eigensolver for our resulting tridiag [D E] and form E=Q*Z
            magma_zstedc_withZ('I', N, D2, E2, Z, LDZ);
#if defined(USEMKL)
            mkl_set_num_threads( 1 );
#endif
            timelpk = get_time_azz()-timelpk;
        }

       /*
        trace_file = fopen("AJETE/Z", "w");
       for (j = 0; j < N ; j++) 
             for (i = 0; i < N ; i++) 
                        //fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,A1[j*LDA1+i]);
                         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+1,j+1,MAGMA_Z_REAL(Z[j*LDA1+i]) ,  MAGMA_Z_IMAG(Z[j*LDA1+i])  );
       fclose(trace_file);

*/


        // ************************************************
        // use GPU code to apply V2 or compute Q2
        // ************************************************
        if(parallel==0){
           // allocate space on GPU for dV2 and dT2        
           dVsize = max(N*N,blkcnt*LDV*Vblksiz);
           //printf("dvsize %lf \n",(16.0*(real_Double_t)dVsize)*1e-9);
           if( CUBLAS_STATUS_SUCCESS != cublasAlloc(dVsize, sizeof(cuDoubleComplex), (void**)&dV2) ) { 
               printf ("!!!! cublasAlloc failed for: dV2\n" );       
               exit(-1);                                                           
           }
    
           if( CUBLAS_STATUS_SUCCESS != cublasAlloc( blkcnt*LDT*Vblksiz, sizeof(cuDoubleComplex), (void**)&dT2) ) { 
              printf ("!!!! cublasAlloc failed for: dT2\n" );       
              exit(-1);                                                           
           }

           //========================
           //  WANTZ =1 
           //  compute Q1  done
           //  compute Q2  here
           //  compute Z   done
           //  make 2 GEMM here
           //========================
           if(WANTZ==1){
               // to avoid allocating a 4 matrix on GPU there are 2 way:
               //   way 1: make Q = Q1*Q2 ==> dV2 = da * dZ
               //          then copy Z to GPU put it on dZ
               //          make Q*Z ==> da = dV2 * dZ
               //          copy the result to CPU, da-->A1                 
               //
               //   way 2: in case we need only 20% of Z so it is better
               //          to start multiplying by Q2*Z, thus to avoid a matrix on GPU
               //          I could copy Q1 (from da) back to CPU and later re-copy to GPU
               //          copy Q1 to CPU, so copy da --> A1
               //          copy Z to GPU, Z --> dZ
               //          make Q = Q2*Z ==> dV2 = da * dZ
               //          copy Q1 to GPU, A1 --> dZ
               //          make Q1*Q ==> da = dZ *dV2
               //          copy the result to CPU, da-->A1  
               //
               // way 2 is implemented because of Raffaele
               timeaplQ2 = get_time_azz();
               cublasFree(dT1);
               // copy Q1 to CPU
               cublasGetMatrix(N, LDA1, sizeof(cuDoubleComplex), da, N, A1, LDA1);
               // compute Q2 by applying V2 to Identity and put it into da           
               magma_zbulge_applyQ(WANTZ, 'L', N, NB, Vblksiz, Q2, N, V, TAU, T, &INFO, dV2, dT2, da, 2);
               // free dT2 and allocate dZ and copy Z to dZ
               cudaDeviceSynchronize();
               timeaplQ2 = get_time_azz()-timeaplQ2;
               cublasFree(dT2);
               if( CUBLAS_STATUS_SUCCESS != cublasAlloc(N*N, sizeof(cuDoubleComplex), (void**)&dZ) ) { 
                  printf ("!!!! cublasAlloc failed for: dZ\n" );       
                  exit(-1);                                                           
               }

        /*
                Q2    = (cuDoubleComplex *) malloc (N*N*sizeof(cuDoubleComplex));
                memset(Q2 , 0, N*N*sizeof(cuDoubleComplex));       
                cublasGetMatrix( N, LDA1, sizeof(cuDoubleComplex), da, N, Q2, N);
                trace_file = fopen("AJETE/Q2", "w");
                for (j = 0; j < N ; j++) 
                      for (i = 0; i < N ; i++) 
                                  fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+1,j+1,MAGMA_Z_REAL(Q2[j*N+i]) ,  MAGMA_Z_IMAG(Q2[j*N+i])  );
                fclose(trace_file);
*/

               cudaDeviceSynchronize();
              // timegemm = get_time_azz();
               // copy the eigenvectors to GPU
               cublasSetMatrix(N, LDZ, sizeof(cuDoubleComplex), Z, LDZ, dZ, N);
               // make GEMM Q2 * Z --> dV2 = da * dZ
               timegemm = get_time_azz();
               cublasZgemm( MagmaNoTrans, MagmaNoTrans, N, N, N, c_one, da, N, dZ, N, c_zero, dV2, N);
               cudaDeviceSynchronize();
               timegemm = get_time_azz()-timegemm;
               // copy Q1 to GPU --> dZ
               cublasSetMatrix(N, LDA1, sizeof(cuDoubleComplex), A1, LDA1, dZ, N);
               // make GEMM Q1 * (Q2 * Z) --> da = dZ * dV2
               cublasZgemm( MagmaNoTrans, MagmaNoTrans, N, N, N, c_one, dZ, N, dV2, N, c_zero, da, N);
               cublasGetMatrix(N, LDA1, sizeof(cuDoubleComplex), da, N, A1, LDA1);
               //timegemm = get_time_azz()-timegemm;
           }
           if(WANTZ==2){
               cublasFree(dT1);
               if( CUBLAS_STATUS_SUCCESS != cublasAlloc(N*N, sizeof(cuDoubleComplex), (void**)&dZ) ) { 
                  printf ("!!!! cublasAlloc failed for: dZ\n" );       
                  exit(-1);                                                           
               }
               // apply V2 from Right to Q1. da = da*(I-V2*T2*V2')
               timeaplQ2 = get_time_azz();
               magma_zbulge_applyQ(WANTZ, 'R', N, NB, Vblksiz, A1, LDA1, V, TAU, T, &INFO, dV2, dT2, da, 2);
               cudaDeviceSynchronize();
               cublasFree(dT2);
               timeaplQ2 = get_time_azz()-timeaplQ2;
               timegemm = get_time_azz();
               // copy the eigenvectors to GPU
               cublasSetMatrix(N, LDZ, sizeof(cuDoubleComplex), Z, LDZ, dZ, N);
               //make a gemm of (Q1 * Q2) * Z = da * dZ --> dV2
               cublasZgemm( MagmaNoTrans, MagmaNoTrans, N, N, N, c_one, da, N, dZ, N, c_zero, dV2, N);
               cublasGetMatrix(N, LDA1, sizeof(cuDoubleComplex), dV2, N, A1, LDA1);
               timegemm = get_time_azz()-timegemm;
           }

           if(WANTZ==3){
               timeaplQ2 = get_time_azz();
               cublasFree(dT1);
               if( CUBLAS_STATUS_SUCCESS != cublasAlloc(N*N, sizeof(cuDoubleComplex), (void**)&dZ) ) { 
                  printf ("!!!! cublasAlloc failed for: dZ\n" );       
                  exit(-1);                                                           
               }
               // apply V2 from left to the eigenvectors Z. dZ = (I-V2*T2*V2')*Z
               magma_zbulge_applyQ(WANTZ, 'L', N, NB, Vblksiz, Z, LDZ, V, TAU, T, &INFO, dV2, dT2, dZ, 3);
               cublasFree(dT2);
               cudaDeviceSynchronize();
               timeaplQ2 = get_time_azz()-timeaplQ2;
               timegemm = get_time_azz();
               //make a gemm of Q1 * (Q2 * Z) = Q1 * ((I-V2T2V2')*Z) = da * dZ --> dV2
               cublasZgemm( MagmaNoTrans, MagmaNoTrans, N, N, N, c_one, da, N, dZ, N, c_zero, dV2, N);
               cublasGetMatrix(N, LDA1, sizeof(cuDoubleComplex), dV2, N, A1, LDA1);
               timegemm = get_time_azz()-timegemm;
           }

           if(WANTZ==5){
               timeaplQ2 = get_time_azz();
               magma_zbulge_applyQ(WANTZ, 'R', N, NB, Vblksiz, A1, LDA1, V, TAU, T, &INFO, dV2, dT2, da, 2);
               cublasGetMatrix(N, LDA1, sizeof(cuDoubleComplex), da, N, A1, LDA1);
               timeaplQ2 = get_time_azz()-timeaplQ2;
               timelpk = get_time_azz();
               // compute the eigenvalues using lapack routine to be able to compare to it and used as ref 
               #if defined(USEMKL)
               mklth=THREADS; //;
               mkl_set_num_threads(mklth);
               #endif
               // call eigensolver for our resulting tridiag [D E] and form E=Q*Z
               magma_zstedc_withZ('V', N, D2, E2, A1, LDA1);
               #if defined(USEMKL)
               mkl_set_num_threads( 1 );
               #endif
               timelpk = get_time_azz()-timelpk;
           }

        }
        // ************************************************


        // ************************************************
        // use parallel cpu code to apply V2 or compute Q2
        // need to be continued for Z
        // ************************************************
        if(parallel==1){
            // copy the matrix Q1 generated up from GPU to CPU because we are using the CPU's.        
            if(WANTZ==2)cublasGetMatrix( N, LDA1, sizeof(cuDoubleComplex), da, LDA1, A1, LDA1);
           // ===============================
           // relaunch thread to apply Q
           // ===============================
           // Set one thread per core
           pthread_attr_init(&thread_attr);
           pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
           pthread_setconcurrency(THREADS);
        
           // Initializations
           for (thread = 0; thread < THREADS; thread++)
           {
               barrier_in[thread] = 0;
               barrier_out[thread] = 0;
               event_numblg[thread] = 0;
           }
           // Launch threads
           for (thread = 1; thread < THREADS; thread++)
           {
               thread_num[thread] = thread;
               pthread_create(&thread_id[thread], &thread_attr, applyQ_parallel_section, &thread_num[thread]);
           }
           thread_num[0] = 0;
           applyQ_parallel_section(&thread_num[0]);
          
           // Wait for completion
           for (thread = 1; thread < THREADS; thread++)
           {
               void *exitcodep;
               pthread_join(thread_id[thread], &exitcodep);
           }
        }
        // ************************************************

       timeeigen = get_time_azz()-timeeigen;
       printf("============================================================================\n");
       printf("  Finish WANTZ %d  computing Q2       timing= %lf \n" ,WANTZ, timeaplQ2);
       if(WANTZ!=5){
           printf("  Finish WANTZ %d  making gemm        timing= %lf \n" ,WANTZ, timegemm);
           printf("  Finish WANTZ %d  eigensolver 'I'    timing= %lf  threads %d \n" ,WANTZ, timelpk, mklth);
       }else{
           printf("  Finish WANTZ %d  eigensolver 'V'    timing= %lf  threads %d \n" ,WANTZ, timelpk, mklth);
       }
       printf("  Finish WANTZ %d  full Eigenvectros  timing= %lf  \n",WANTZ, timeeigen);
       printf("============================================================================\n");
    }

             


/*

       trace_file = fopen("AJETE/U", "w");
       for (j = 0; j < N ; j++) 
             for (i = 0; i < N ; i++) 
                        //fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,A1[j*LDA1+i]);
                         fprintf(trace_file,"%10d %10d %25.15e %25.15e\n",i+1,j+1,MAGMA_Z_REAL(A1[j*LDA1+i]) ,  MAGMA_Z_IMAG(A1[j*LDA1+i])  );
       fclose(trace_file);

       trace_file = fopen("AJETE/D", "w");
       for (j = 0; j < N ; j++) 
                fprintf(trace_file,"%10d%10d%40.30e\n",j+1,1,D2[j]);
       fclose(trace_file);
*/



/*
            cuDoubleComplex mydz=c_zero,mydo=c_one;
            cuDoubleComplex *Z = (cuDoubleComplex *) malloc(N*N*sizeof(cuDoubleComplex));
   dgemm_("N","N",&N,&N,&N,&mydo,A1,&LDA1,Q1,&LDQ1,&mydz,Z,&LDA1);
   memcpy(A1,Z,N*N*sizeof(cuDoubleComplex));
   memcpy(Q1,Z,N*N*sizeof(cuDoubleComplex));
*/
    /*
       trace_file = fopen("AJETE/Qf", "w");
       for (j = 0; j < N ; j++) 
             for (i = 0; i < N ; i++) 
                            fprintf(trace_file,"%10d%10d%40.30e\n",i+1,j+1,A1[j*LDA1+i]);
       fclose(trace_file);
*/

    if(CHECK)
    {
       if(WANTZ!=1) JOBZ='N';
       if(WANTZ==1) JOBZ='V';
       magma_zstedc_withZ(JOBZ, N, D2, E2, Q2, LDQ2);            
       //magma_zstedc_withZ('N', N, D2, E2, Q1, LDQ1);            
    }
    



#if defined(PRECISION_d)  
    if(CHECK)
    {
        /************************************************* 
        *     CALLING LAPACK on original matrix
        * ***********************************************/
        int    LWORK=256*N;
        cuDoubleComplex *WORK    = (cuDoubleComplex *) malloc( LWORK * sizeof(cuDoubleComplex) );
        D1      = (double *) malloc( N * sizeof(double) );
        double *E1      = (double *) malloc( N * sizeof(double) );
        cuDoubleComplex *ALPK    = (cuDoubleComplex *) malloc (N*LDAINIT*sizeof(cuDoubleComplex));
        memcpy(ALPK, AINIT, N*LDAINIT*sizeof(cuDoubleComplex));

        timelpk = get_time_azz();
        lapackf77_zhbtrd("N", "L", &N, &NB, ALPK, &LDAINIT, D1, E1, WORK, &N, WORK, &INFO); 
        timelpk = get_time_azz() - timelpk;
        printf("\n");                
        printf("  Time DSBTRD-MKL-LAPACK                      :   %lf    N : %10d    NB : %10d \n\n\n",timelpk, N, NB );
        /* call eigensolver */
        lapackf77_dsterf(&N, D1, E1, &INFO);             
       /* ***********************************************/ 
       // call eigensolver 
       //dsterf_( &N, D2, E2, &INFO); 
       //magma_zstedc_withZ(JOBZ, N, D2, E2, Q1, LDQ1);
       // compare result 
       cmp_vals(N, D1, D2, &nrmI, &nrm1, &nrm2);
       cuDoubleComplex *WORKAJETER, *RESU;
       WORKAJETER = (cuDoubleComplex *) malloc( (2* N * N + 256 * N) * sizeof(cuDoubleComplex) );
       RESU       = (cuDoubleComplex *) malloc(10*sizeof(cuDoubleComplex));
       int MATYPE;
       memset(RESU,0,10*sizeof(cuDoubleComplex));

 
       MATYPE=2;
       cuDoubleComplex NOTHING=c_zero;
       checkdiag_(&JOBZ, &MATYPE, &N, &NB, AINIT, &LDAINIT, &NOTHING, &NOTHING, D2 , D1, Q2, &LDQ2, WORKAJETER, RESU );
       //checkdiag_(&JOBZ, &MATYPE, &N, &NB, AINIT, &LDAINIT, &NOTHING, &NOTHING, D2 , D1, A1, &LDA1, WORKAJETER, RESU );

        printf("\n");
        printf(" ================================================================================================================================== \n");
        printf("   ==> INFO voici  threads=%d    N=%d    NB=%d   BAND=%d WANTZ=%d\n",thread,N, NB, BAND, WANTZ);
        printf(" ================================================================================================================================== \n");
        printf("            DSBTRD                : %15s \n", "STATblgv9withQ    ");
        printf(" ================================================================================================================================== \n");
        if(WANTZ==1)
           printf(" | A - U S U' | / ( |A| n ulp )   : %15.3E   \n",        RESU[0]); 
        if(WANTZ==1)
           printf(" | I - U U' | / ( n ulp )         : %15.3E   \n", RESU[1]);
        printf(" | D1 - EVEIGS | / (|D| ulp)      : %15.3E   \n",  RESU[2]);
        printf(" max | D1 - EVEIGS |              : %15.3E   \n",  RESU[6]);
        printf(" ================================================================================================================================== \n\n\n");

        printf(" ***********************************************************************************************************************************\n");
        printf(" Hello here are the norm  Infinite (max)=%e  norm one (sum)=%e   norm2(sqrt)=%e \n",nrmI, nrm1, nrm2);
        printf(" ***********************************************************************************************************************************\n\n");


    }
#endif


    cublasFree(dV2);
    cublasFree(da);
    free(A2);
    free(TAU);
    free(V);
    free(T);

    return 0;
} /* END ZHETRD_BHE2TRC */







//##################################################################################################
static void *parallel_section(void *thread_id)
{
    int my_core_id = *((int*)thread_id);
    int cores_num = core_in_all.cores_num;
    int WANTZ = core_in_all.WANTZ;
    real_Double_t timeB=0.0, timeT=0.0;
    static volatile int sys_corenbr = 1;
    int i;


#if defined(SETAFFINITY)    
    // bind threads 
    cpu_set_t set;
    // bind threads 
    CPU_ZERO( &set );
    CPU_SET( my_core_id, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif

    log_eventsblg = 1;
    core_event_startblg(my_core_id);
    barrier(my_core_id, cores_num);
    core_event_endblg(my_core_id);
    core_log_eventblg(0x000000, my_core_id);



    // timing
    if (my_core_id == 0)
       timeB = get_time_azz();

    // bulge chasing
    tile_bulge_parallel(my_core_id);
    barrier(my_core_id, cores_num);
    // timing
    if (my_core_id == 0){
       timeB = get_time_azz()-timeB;
       printf("  Finish BULGE   timing= %lf \n" ,timeB);
    }

    // compute the T's to be used when applying Q2
    if(WANTZ>0){
       if(my_core_id == 0)timeT = get_time_azz();
       tile_bulge_computeT_parallel(my_core_id);
       barrier(my_core_id, cores_num);
       // timing
       if (my_core_id == 0){
          timeT = get_time_azz()-timeT;
          printf("  Finish T's     timing= %lf \n" ,timeT);
       }
    }

    // timing
    if (my_core_id == 0){
        *(core_in_all.timeblg) = timeT+timeB;     
    }

#if defined(SETAFFINITY)    
    // unbind threads 
    sys_corenbr = sysconf(_SC_NPROCESSORS_ONLN);
    CPU_ZERO( &set );
    for(i=0; i<sys_corenbr; i++)
            CPU_SET( i, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif

    return 0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////

//##################################################################################################
static void *applyQ_parallel_section(void *thread_id)
{
    int my_core_id = *((int*)thread_id);
    int cores_num = core_in_all.cores_num;
    int WANTZ = core_in_all.WANTZ;
    real_Double_t *timeaplQ = core_in_all.timeaplQ;
    static volatile int sys_corenbr = 1;
    int i;

#if defined(SETAFFINITY)    
    cpu_set_t set;
    CPU_ZERO( &set );
    CPU_SET( my_core_id, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif


    log_eventsblg = 1;
    core_event_startblg(my_core_id);
    barrier(my_core_id, cores_num);
    core_event_endblg(my_core_id);
    core_log_eventblg(0x000000, my_core_id);



    if (my_core_id == 0)
       *timeaplQ = get_time_azz();
    /* apply Q*/
    if(WANTZ>0){
       tile_bulge_applyQ_parallel(my_core_id);
       barrier(my_core_id, cores_num);
    }
    if (my_core_id == 0)
        *timeaplQ = get_time_azz()-(*timeaplQ);

#if defined(SETAFFINITY)    
    // unbind threads 
    sys_corenbr = sysconf(_SC_NPROCESSORS_ONLN);
    CPU_ZERO( &set );
    for(i=0; i<sys_corenbr; i++)
            CPU_SET( i, &set );
    sched_setaffinity( 0, sizeof(set), &set) ;
#endif

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////////////////////////////
static void tile_bulge_parallel(int my_core_id)
{
    int core;
    int INFO;

    int cores_num = core_in_all.cores_num;
    cuDoubleComplex *A = core_in_all.A;
    cuDoubleComplex *V = core_in_all.V;
    cuDoubleComplex *TAU = core_in_all.TAU;
    int N = core_in_all.N;
    int NB = core_in_all.NB;
    int NBTILES = core_in_all.NBTILES;
    int LDA= core_in_all.LDA;
    int BAND= core_in_all.BAND;
    int grsiz= core_in_all.grsiz;
    int Vblksiz = core_in_all.Vblksiz;
    volatile int *prog = core_in_all.ss_prog;
    //%===========================
    //%   local variables
    //%===========================
    int sweepid, myid, shift, stt, st, ed, stind, edind;
    int blklastind, colpt, algtype;
    int stepercol,mylastid,grnb,grid;
    int i,j,m,k;
    int thgrsiz, thgrnb, thgrid, thed;
    int coreinit,loopfirsttask,coreid;
    int colblktile,maxrequiredcores,colpercore,allcoresnb;
    int fin;


    //printf("=================> my core id %d of %d \n",my_core_id, cores_num);

    if((BAND!=0) && (BAND!=6) && (BAND!=62) && (BAND!=63)){
       if(my_core_id==0)printf(" ===============================================================================\n");
       if(my_core_id==0)printf("         ATTENTION ========> BAND is required to be 0 6 62 63, program will exit\n");
       if(my_core_id==0)printf(" ===============================================================================\n");
       return;
    }
  /* As I store V in the V vector there are overlap between 
   * tasks so shift is now 4 where group need to be always 
   * multiple of 2, because as example if grs=1 task 2 from 
   * sweep 2 can run with task 6 sweep 1., but task 2 sweep 2 
   * will overwrite the V of tasks 5 sweep 1 which are used by 
   * task 6, so keep in mind that group need to be multiple of 2, 
   * and thus tasks 2 sweep 2 will never run with task 6 sweep 1.
   * However, when storing V in A, shift could be back to 3.
   * */ 

    allcoresnb = cores_num;
    //grsiz   = 2;
    shift   = 5;
    if(grsiz==1) 
        colblktile=1;
    else
        colblktile=grsiz/2;

    maxrequiredcores = NBTILES/colblktile;
    if(maxrequiredcores<1)maxrequiredcores=1;
    colpercore  = colblktile*NB;
    if(allcoresnb > maxrequiredcores)
    {
        if(my_core_id==0)printf("==================================================================================\n");
        if(my_core_id==0)printf("  WARNING only %3d threads are required to run this test optimizing cache reuse\n",maxrequiredcores);
        if(my_core_id==0)printf("==================================================================================\n");
        allcoresnb = maxrequiredcores;
    }
    thgrsiz = N;//allcoresnb;

  if(my_core_id==0) printf("  Static bulgechasing version v9_9col threads  %4d      N %5d      NB %5d    grs %4d thgrsiz %4d  BAND %4d\n",cores_num, N, NB, grsiz,thgrsiz,BAND);





  i = shift/grsiz;
  stepercol =  i*grsiz == shift ? i:i+1;

  i       = (N-2)/thgrsiz;
  thgrnb  = i*thgrsiz == (N-2) ? i:i+1;

  for (thgrid = 1; thgrid<=thgrnb; thgrid++){
     stt  = (thgrid-1)*thgrsiz+1;
     thed = min( (stt + thgrsiz -1), (N-2));
     for (i = stt; i <= N-2; i++){
        ed=min(i,thed);
        if(stt>ed)break;
        for (m = 1; m <=stepercol; m++){ 
            st=stt;
            for (sweepid = st; sweepid <=ed; sweepid++){

                for (k = 1; k <=grsiz; k++){ 
                    myid = (i-sweepid)*(stepercol*grsiz) +(m-1)*grsiz + k;        
                    if(myid%2 ==0){
                         colpt      = (myid/2)*NB+1+sweepid-1;
                         stind      = colpt-NB+1;
                         edind      = min(colpt,N);
                         blklastind = colpt;
                         if(stind>=edind){
                             printf("ERROR---------> st>=ed  %d  %d \n\n",stind, edind);
                             exit(-10);
                         }
                    }else{
                         colpt      = ((myid+1)/2)*NB + 1 +sweepid -1 ;
                         stind      = colpt-NB+1;
                         edind      = min(colpt,N);
                         if( (stind>=edind-1) && (edind==N) )
                             blklastind=N;
                         else
                             blklastind=0;
                         if(stind>=edind){
                             printf("ERROR---------> st>=ed  %d  %d \n\n",stind, edind);
                             exit(-10);
                         }
                    }

                    coreid = (stind/colpercore)%allcoresnb;


   //printf("    current col %3d sweep %3d myid %3d  coreid %7d my_core_id %3d ---------------------- st %2d  ed %2d \n",i,sweepid, myid, coreid,my_core_id, stind, edind); 
 //printf("MYID %2d prog  %3d %3d %3d %3d %3d %3d %3d \n",my_core_id,prog[0],prog[1],prog[2],prog[3],prog[4],prog[5],prog[6]);
                    
                    if(my_core_id==coreid)
                    {
                        //printf("--> current col %3d sweep %3d myid %3d  my_core_id %3d   prog[myid-1] %3d    prog[myid+shiftd] %3d\n",i,sweepid, myid,my_core_id,prog[myid-1], prog[myid+shift]);
                        //__sync_synchronize();
                        //fflush(stdout);


                        fin=0;
                        while(fin==0)
                        { 
                            if(myid==1)
                            {
                            if( (prog[myid+shift-1]== (sweepid-1)) )
                                {
                                    
                                    if(LOG) core_event_startblg(my_core_id);
                                    /*
                                    if(BAND==0)
                                            TRD_type1cHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);
                                    else if(BAND==6)*/
                                            magma_ztrdtype1cbHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);/*
                                    else if(BAND==62)
                                            TRD_hbcelr_v62sym_withQ(N, NB, A, LDA, V, TAU, stind, edind, myid, sweepid, Vblksiz);*/
                                    if(LOG) {
                                        core_event_endblg(my_core_id);
                                        core_log_eventblg(0x006680, my_core_id);
                                    }
                                    
                                    fin=1;
                                    prog[myid]= sweepid;
                                    if(blklastind >= (N-1))
                                    {
                                        for (j = 1; j <= shift; j++) 
                                            prog[myid+j]=sweepid;
                                    }                                
                                    } // END progress condition
                            }else{
                                if( (prog[myid-1]==sweepid) && (prog[myid+shift-1]== (sweepid-1)) )
                                    {
                                        
                                    if(LOG) core_event_startblg(my_core_id);
                                    /*
                                    if(BAND==0){
                                        if(myid%2 == 0)
                                            TRD_type2cHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);
                                        else
                                            TRD_type3cHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);
                                    }else if(BAND==6){*/
                                        if(myid%2 == 0)
                                            magma_ztrdtype2cbHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);
                                        else
                                            magma_ztrdtype3cbHLsym_withQ(N, NB, A, LDA, V, TAU, stind, edind, sweepid, Vblksiz);/*
                                    }else if(BAND==62){
                                            TRD_hbcelr_v62sym_withQ(N, NB, A, LDA, V, TAU, stind, edind, myid, sweepid, Vblksiz);
                                    }*/
                                    if(LOG) {
                                        core_event_endblg(my_core_id);
                                        core_log_eventblg(0xff0000, my_core_id);
                                    }        
                                    
                                    fin=1;
                                    prog[myid]= sweepid;
                                    if(blklastind >= (N-1))
                                    {
                                        for (j = 1; j <= shift+allcoresnb; j++) 
                                            prog[myid+j]=sweepid;
                                    }                                
                                    } // END progress condition
                            } // END if myid==1
                        } // END while loop
                        
                    } // END if my_core_id==coreid
                    
                    if(blklastind >= (N-1))
                    {
                          stt=stt+1;
                          break;
                    }
                }   // END for k=1:grsiz
               } // END for sweepid=st:ed
        } // END for m=1:stepercol
     } // END for i=1:N-2
//barrier(my_core_id, cores_num);   
  } // END for thgrid=1:thgrnb

// momentary solution for complex version when A(N,N-1) is complex and to avoid making it real out of this function 
// which will require to multiply the col/row of the Eigenvectors or Q by the scalar that make A(N,N-1) to real.
//magma_ztrdtype1cbHLsym_withQ(N, NB, A, LDA, V, TAU, N, N, N-1, Vblksiz);

} // END FUNCTION
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)   &(T[(m)])
static void tile_bulge_computeT_parallel(int my_core_id)
{
    int INFO;
    int cores_num = core_in_all.cores_num;
    cuDoubleComplex *T     = core_in_all.T;
    cuDoubleComplex *V     = core_in_all.V;
    cuDoubleComplex *TAU   = core_in_all.TAU;
    int  N         = core_in_all.N;
    int  NB        = core_in_all.NB;
    int  Vblksiz   = core_in_all.Vblksiz;

    //%===========================
    //%   local variables
    //%===========================
    int LDT, LDV,blklen,firstcolj;
    int bg, nbGblk,rownbm, k, m, n;
    int st,ed,fst,vlen,vnb,colj;
    int blkid,vpos,taupos,tpos;
    int cur_blksiz,avai_blksiz, ncolinvolvd;
    int nbgr, colst, coled, version;
    int blkpercore,blkcnt, myid;


    findVTsiz(N, NB, Vblksiz, &blkcnt, &LDV);
    blkpercore = blkcnt/cores_num;

    LDT     = Vblksiz;    
    LDV     = NB+Vblksiz-1;
    blklen  = LDV*Vblksiz;
    nbGblk  = plasma_ceildiv((N-2),Vblksiz);

    if(my_core_id==0) printf("  COMPUTE T parallel threads %d with  N %d   NB %d   Vblksiz %d version %d \n",cores_num,N,NB,Vblksiz,version);
    
        for (bg = nbGblk; bg>0; bg--)
        {
           firstcolj = (bg-1)*Vblksiz + 1;
           rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
           for (m = rownbm; m>0; m--)
           {
               vlen = 0;
               vnb  = 0;
               colj      = (bg-1)*Vblksiz; // for k=0;I compute the fst and then can remove it from the loop
               fst       = (rownbm -m)*NB+colj +1;
               for (k=0; k<Vblksiz; k++)
               {
                   colj     = (bg-1)*Vblksiz + k;
                   st       = (rownbm -m)*NB+colj +1;
                   ed       = min(st+NB-1,N-1);
                   if(st==ed)break;
                   vlen=ed-fst+1;
                   vnb=k+1;
               }        
               colj     = (bg-1)*Vblksiz;
               findVTpos(N,NB,Vblksiz,colj,fst, &vpos, &taupos, &tpos, &blkid);
               myid = blkid/blkpercore;
               if(my_core_id==(myid%cores_num)){
                  if((vlen>0)&&(vnb>0))
                     lapackf77_zlarft( "F", "C", &vlen, &vnb, V(vpos), &LDV, TAU(taupos), T(tpos), &LDT);
               }
           }
        }
}
#undef V
#undef TAU
#undef T
////////////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////////////////////////
#define E(m,n)   &(E[(m) + LDE*(n)])
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)     &(T[(m)])
static void tile_bulge_applyQ_parallel(int my_core_id)
{
    int INFO;
    int cores_num  = core_in_all.cores_num;
    cuDoubleComplex *T      = core_in_all.T;
    cuDoubleComplex *E      = core_in_all.E;
    cuDoubleComplex *V      = core_in_all.V;
    cuDoubleComplex *TAU    = core_in_all.TAU;
    int  N         = core_in_all.N;
    int  NB        = core_in_all.NB;
    int  Vblksiz   = core_in_all.Vblksiz;
    int  LDE       = core_in_all.LDE;
    char SIDE      = core_in_all.SIDE;
    int  WANTZ     = core_in_all.WANTZ;

    //%===========================
    //%   local variables
    //%===========================
    int LDT,LDV,blklen,firstcolj;
    int bg, nbGblk,rownbm, k, m, n;
    int st,ed,fst,vlen,vnb,colj,len;
    int blkid, vpos,taupos,tpos;
    cuDoubleComplex *WORK;
    int LWORK;
    int  cur_blksiz,avai_blksiz, ncolinvolvd;
    int  nbgr, colst, coled, version;
    int chunkid,nbchunk,colpercore,corest,corelen;
    int coreid,allcoresnb,maxrequiredcores;

    INFO=0;    
    version = 113;
    LDT     = Vblksiz;
    LDV     = NB+Vblksiz-1;    
    blklen  = LDV*Vblksiz;
    nbGblk  = plasma_ceildiv((N-2),Vblksiz);
    //LWORK   = 2*N*max(Vblksiz,64);
    //WORK    = (cuDoubleComplex *) malloc (LWORK*sizeof(cuDoubleComplex));


    /* use colpercore =  N/cores_num; :if i want to split E into 
     * cores_num chunk so I will have large chunk for each core.
     * However I prefer to split E into chunk of small size where 
     * I guarantee that blas3 occur and the size of chunk remain into 
     * cache, I think it is better. than I will have different chunk of 
     * small sizeper coreand i will do each chunk till the end and 
     * then move tothe second one for data locality
     *
     * version v1: for each chunck it apply all the V's then move to 
     *                    the other chunck. the locality here inside each 
     *                    chunck meaning that thread t apply V_k then move 
     *                    to V_k+1 which overlap with V_k meaning that the 
     *                    E_k+1 overlap with E_k. so here is the 
     *                    locality however thread t had to read V_k+1 and 
     *                    T_k+1 at each apply. note that all thread if they 
     *                    run at same speed they might reading the same V_k 
     *                    and T_k at the same time.
     *
     * version v2: for each V_k and T_k thread t apply those V_k and 
     *                    T_k to E_k for all its chunck, then move to V_k+1 
     *                    T_k+1 and apply them to E_k+1 on all the chunck, 
     *                    and so on. the difference is that, thread keep V_k
     *                    and T_K while move over the E_k.
     *                    both version look like similar in perf.                   
     * */
    colpercore = min(NB,120); //colpercore = N make the code sequential running on thread=0;
    //colpercore =  N/cores_num; 

    colpercore =  plasma_ceildiv(N,cores_num);
    if(colpercore>1000)
        colpercore =  plasma_ceildiv(colpercore,10);
    else if(colpercore>800)
        colpercore =  plasma_ceildiv(colpercore,8);
    else if(colpercore>600)
        colpercore =  plasma_ceildiv(colpercore,6);
    else if(colpercore>400)
        colpercore =  plasma_ceildiv(colpercore,4);
    else if(colpercore>200)
        colpercore =  plasma_ceildiv(colpercore,2);
    if(colpercore>200)colpercore=120;

    LWORK   = 2*colpercore*max(Vblksiz,64);
    //LWORK   = 2*N*max(Vblksiz,64);
    WORK    = (cuDoubleComplex *) malloc (LWORK*sizeof(cuDoubleComplex));


    nbchunk    =  plasma_ceildiv(N,colpercore);

    allcoresnb = cores_num;
    maxrequiredcores = nbchunk;
    if(maxrequiredcores<1)maxrequiredcores=1;
    if(allcoresnb > maxrequiredcores)
    {
        if(my_core_id==0)printf("==================================================================================\n");
        if(my_core_id==0)printf("  WARNING only %3d threads are required to run this test optimizing cache reuse\n",maxrequiredcores);
        if(my_core_id==0)printf("==================================================================================\n");
        allcoresnb = maxrequiredcores;
    }


    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            Also E is splitten by row meaning each apply consist in a block of row (horizontal block) */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n 
     *            Also E is splitten by col meaning each apply consist in a block of col (vertical block) */

     /* WANTZ = 1 meaning E is IDENTITY so form Q using optimized update. 
      *         So we use the reverse order from small q to large one, 
      *         so from q_n to q_1 so Left update to Identity.
      *         Use version 113 because in 114 we need to update the whole matrix and not in icreasing order.
      * WANTZ = 2 meaning E is a full matrix and need to be updated from Left or Right so use normal update
      * */
    if(WANTZ==1) 
    {
        version=113;
        SIDE='L';
    }

    //printf("  ENTERING FUNCTION APPLY Q_v115: same as 113(L) or 114(L) or 93(R)"\n);
    if(my_core_id==0)printf("  APPLY Q_v1   parallel with threads %d   nbchunk %d  colpercore %d  N %d   NB %d   Vblksiz %d SIDE %c version %d WANTZ %d \n",cores_num,nbchunk,colpercore,N,NB,Vblksiz,SIDE,version, WANTZ);

    for (chunkid = 0; chunkid<nbchunk; chunkid++)
    {
        coreid  = chunkid%allcoresnb;
        corest  = chunkid*colpercore;
        corelen = min(colpercore, (N-(chunkid*colpercore)));

        if(my_core_id==coreid)
        {
            //printf("mycore id %d voici nbchunk %d  chunkid %d  coreid %d, corest %d, corelen %d\n",my_core_id,nbchunk, chunkid, coreid, corest, corelen);
            if(SIDE=='L'){
            if(version==113){            
                for (bg = nbGblk; bg>0; bg--)
                {
                   firstcolj = (bg-1)*Vblksiz + 1;
                   rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
                   for (m = rownbm; m>0; m--)
                   {
                       vlen = 0;
                       vnb  = 0;
                       colj      = (bg-1)*Vblksiz; // for k=0;I compute the fst and then can remove it from the loop
                       fst       = (rownbm -m)*NB+colj +1;
                       for (k=0; k<Vblksiz; k++)
                       {
                           colj     = (bg-1)*Vblksiz + k;
                           st       = (rownbm -m)*NB+colj +1;
                           ed       = min(st+NB-1,N-1);
                           if(st==ed)break;
                           vlen=ed-fst+1;
                           vnb=k+1;
                       }        
                       colst     = (bg-1)*Vblksiz;
                       findVTpos(N,NB,Vblksiz,colst,fst, &vpos, &taupos, &tpos, &blkid);
                       //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n",bg,m,vlen, vnb,colst,vpos+1,taupos+1);
                       
                       if(LOGQ) core_event_startblg(my_core_id);
                       if((vlen>0)&&(vnb>0)){
                           //lapackf77_zungqr( "L", "N", &vlen, &corelen, &vnb, V(vpos), &LDV, TAU(taupos), E(fst,corest), &LDE,  WORK, &LWORK, &INFO );
                           //DORMQR_BLG( "L", "N", &vlen, &corelen, &vnb, &NB, V(vpos), &LDV, TAU(taupos), E(fst,corest), &LDE,  WORK, &LWORK, &INFO );
                           if(WANTZ==1){ 
                               st  = max(colst,corest);           
                               len = corelen - (st-corest);    
                               lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &len, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,st), &LDE,  WORK, &len); 
                           }else{
                               lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &corelen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,corest), &LDE,  WORK, &corelen); // corelen for WORK was colpercore before 
                           }
                       }
                       if(INFO!=0) 
                               printf("ERROR DORMQR INFO %d \n",INFO);
                       if(LOGQ) {
                           core_event_endblg(my_core_id);
                           core_log_eventblg(0xff0000, my_core_id);
                       }                
                   }
                }
            }else if(version==114){
                rownbm    = plasma_ceildiv((N-2),NB);
                for (m = rownbm; m>0; m--)
                {
                   ncolinvolvd = min(N-2, m*NB);
                   avai_blksiz=min(Vblksiz,ncolinvolvd);
                   nbgr = plasma_ceildiv(ncolinvolvd,avai_blksiz);
                   for (n = nbgr; n>0; n--)
                   {
                       vlen = 0;
                       vnb  = 0;
                       cur_blksiz = min(ncolinvolvd-(n-1)*avai_blksiz, avai_blksiz);
                       colst = (n-1)*avai_blksiz;
                       coled = colst + cur_blksiz -1;
                       fst   = (rownbm -m)*NB+colst +1;
                       for (colj=colst; colj<=coled; colj++)
                       {
                           st       = (rownbm -m)*NB+colj +1;
                           ed       = min(st+NB-1,N-1);
                           if(st>=ed)break;
                           vlen=ed-fst+1;
                           vnb=vnb+1;
                       }        
                       findVTpos(N,NB,Vblksiz,colst,fst, &vpos, &taupos, &tpos, &blkid);
                       //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n",bg,m,vlen, vnb,colj,vpos+1,taupos+1);
                       if((vlen>0)&&(vnb>0))
                           //lapackf77_zungqr( "L", "N", &vlen, &corelen, &vnb, V(vpos), &LDV, TAU(taupos), E(fst,corest), &LDE,  WORK, &LWORK, &INFO );
                           lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &corelen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,corest), &LDE,  WORK, &colpercore);       
                       if(INFO!=0) 
                               printf("ERROR DORMQR INFO %d \n",INFO);
                   }
                }
            }
            }else if (SIDE=='R'){
                for (bg =1; bg<=nbGblk; bg++)
                {
                   firstcolj = (bg-1)*Vblksiz + 1;
                   rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
                   for (m = 1; m<=rownbm; m++)
                   {
                       vlen = 0;
                       vnb  = 0;
                       // for k=0;I compute the fst and then can remove it from the loop
                       colj     = (bg-1)*Vblksiz; 
                       fst      = (rownbm -m)*NB+colj +1;
                       for (k=0; k<Vblksiz; k++)
                       {
                           colj     = (bg-1)*Vblksiz + k;
                           st       = (rownbm -m)*NB+colj +1;
                           ed       = min(st+NB-1,N-1);
                           if(st==ed)break;
                           vlen=ed-fst+1;
                           vnb=k+1;
                       }        
                       colj     = (bg-1)*Vblksiz;
                       findVTpos(N,NB,Vblksiz,colj,fst, &vpos, &taupos, &tpos, &blkid);
                       //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n",bg,m,vlen, vnb,colj,vpos,taupos);
                       if((vlen>0)&&(vnb>0))
                           //lapackf77_zungqr( "R", "N", &corelen, &vlen, &vnb, V(vpos), &LDV, TAU(taupos), E(corest,fst), &LDE,  WORK, &LWORK, &INFO );
                           lapackf77_zlarfb( "R", "N", "F", "C", &corelen, &vlen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(corest,fst), &LDE,  WORK, &corelen); //colpercore);       
                       if(INFO!=0) 
                               printf("Right ERROR DORMQR INFO %d \n",INFO);
                   }
                }
            }else{
                    printf("ERROR SIDE %d \n",SIDE);
            }
        } // END my_core_id=coreid
    } // END loop over the chunk


}
#undef E
#undef V
#undef TAU
#undef T
////////////////////////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////////////////////////////
#define E(m,n)   &(E[(m) + LDE*(n)])
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)     &(T[(m)])
void tile_bulge_applyQ_parallel2(int my_core_id)
{
    int INFO;
    int cores_num  = core_in_all.cores_num;
    cuDoubleComplex *T      = core_in_all.T;
    cuDoubleComplex *E      = core_in_all.E;
    cuDoubleComplex *V      = core_in_all.V;
    cuDoubleComplex *TAU    = core_in_all.TAU;
    int  N         = core_in_all.N;
    int  NB        = core_in_all.NB;
    int  Vblksiz   = core_in_all.Vblksiz;
    int  LDE       = core_in_all.LDE;
    char SIDE      = core_in_all.SIDE;
    int  WANTZ     = core_in_all.WANTZ;

    //%===========================
    //%   local variables
    //%===========================
    int LDT,LDV,blklen,firstcolj;
    int bg, nbGblk,rownbm, k, m, n;
    int st,ed,fst,vlen,vnb,colj,len;
    int blkid, vpos,taupos,tpos;
    cuDoubleComplex *WORK;
    int LWORK;
    int  cur_blksiz,avai_blksiz, ncolinvolvd;
    int  nbgr, colst, coled, version;
    int chunkid,nbchunk,colpercore,corest,corelen;
    int coreid,allcoresnb,maxrequiredcores;

    INFO=0;    
    version = 113;
    LDT     = Vblksiz;
    LDV     = NB+Vblksiz-1;    
    blklen  = LDV*Vblksiz;
    nbGblk  = plasma_ceildiv((N-2),Vblksiz);
    //LWORK   = 2*N*max(Vblksiz,64);
    //WORK    = (cuDoubleComplex *) malloc (LWORK*sizeof(cuDoubleComplex));


    /* use colpercore =  N/cores_num; :if i want to split E into 
     * cores_num chunk so I will have large chunk for each core.
     * However I prefer to split E into chunk of small size where 
     * I guarantee that blas3 occur and the size of chunk remain into 
     * cache, I think it is better. than I will have different chunk of 
     * small sizeper coreand i will do each chunk till the end and 
     * then move tothe second one for data locality
     *
     * version v1: for each chunck it apply all the V's then move to 
     *                    the other chunck. the locality here inside each 
     *                    chunck meaning that thread t apply V_k then move 
     *                    to V_k+1 which overlap with V_k meaning that the 
     *                    E_k+1 overlap with E_k. so here is the 
     *                    locality however thread t had to read V_k+1 and 
     *                    T_k+1 at each apply. note that all thread if they 
     *                    run at same speed they might reading the same V_k 
     *                    and T_k at the same time.
     *
     * version v2: for each V_k and T_k thread t apply those V_k and 
     *                    T_k to E_k for all its chunck, then move to V_k+1 
     *                    T_k+1 and apply them to E_k+1 on all the chunck, 
     *                    and so on. the difference is that, thread keep V_k
     *                    and T_K while move over the E_k.
     *                    both version look like similar in perf.                   
     * */
    colpercore = min(NB,120); //colpercore = N make the code sequential running on thread=0;
    //colpercore =  N/cores_num; 

    colpercore =  plasma_ceildiv(N,cores_num);
    if(colpercore>1000)
        colpercore =  plasma_ceildiv(colpercore,10);
    else if(colpercore>800)
        colpercore =  plasma_ceildiv(colpercore,8);
    else if(colpercore>600)
        colpercore =  plasma_ceildiv(colpercore,6);
    else if(colpercore>400)
        colpercore =  plasma_ceildiv(colpercore,4);
    else if(colpercore>200)
        colpercore =  plasma_ceildiv(colpercore,2);
    if(colpercore>200)colpercore=120;





    LWORK   = 2*colpercore*max(Vblksiz,64);
    //LWORK   = 2*N*max(Vblksiz,64);
    WORK    = (cuDoubleComplex *) malloc (LWORK*sizeof(cuDoubleComplex));


    nbchunk    =  plasma_ceildiv(N,colpercore);

    allcoresnb = cores_num;
    maxrequiredcores = nbchunk;
    if(maxrequiredcores<1)maxrequiredcores=1;
    if(allcoresnb > maxrequiredcores)
    {
        if(my_core_id==0)printf("==================================================================================\n");
        if(my_core_id==0)printf("  WARNING only %3d threads are required to run this test optimizing cache reuse\n",maxrequiredcores);
        if(my_core_id==0)printf("==================================================================================\n");
        allcoresnb = maxrequiredcores;
    }

    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            Also E is splitten by row meaning each apply consist in a block of row (horizontal block) */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n 
     *            Also E is splitten by col meaning each apply consist in a block of col (vertical block) */

     /* WANTZ = 1 meaning E is IDENTITY so form Q using optimized update. 
      *         So we use the reverse order from small q to large one, 
      *         so from q_n to q_1 so Left update to Identity.
      *         Use version 113 because in 114 we need to update the whole matrix and not in icreasing order.
      * WANTZ = 2 meaning E is a full matrix and need to be updated from Left or Right so use normal update
      * */
    if(WANTZ==1) 
    {
        version=113;
        SIDE='L';
    }

    //printf("  ENTERING FUNCTION APPLY Q_v115: same as 113(L) or 114(L) or 93(R)"\n);
    if(my_core_id==0)printf("  APPLY Q_v2   parallel with threads %d   nbchunk %d  colpercore %d  N %d   NB %d   Vblksiz %d SIDE %c version %d \n",cores_num,nbchunk,colpercore,N,NB,Vblksiz,SIDE,version);

    //printf("mycore id %d voici nbchunk %d  chunkid %d  coreid %d, corest %d, corelen %d\n",my_core_id,nbchunk, chunkid, coreid, corest, corelen);
    if(SIDE=='L'){
        if(version==113){            
            for (bg = nbGblk; bg>0; bg--)
            {
               firstcolj = (bg-1)*Vblksiz + 1;
               rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
               for (m = rownbm; m>0; m--)
               {
                   vlen = 0;
                   vnb  = 0;
                   colj      = (bg-1)*Vblksiz; // for k=0;I compute the fst and then can remove it from the loop
                   fst       = (rownbm -m)*NB+colj +1;
                   for (k=0; k<Vblksiz; k++)
                   {
                       colj     = (bg-1)*Vblksiz + k;
                       st       = (rownbm -m)*NB+colj +1;
                       ed       = min(st+NB-1,N-1);
                       if(st==ed)break;
                       vlen=ed-fst+1;
                       vnb=k+1;
                   }        
                   colst     = (bg-1)*Vblksiz;
                   findVTpos(N,NB,Vblksiz,colst,fst, &vpos, &taupos, &tpos, &blkid);
                   //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n",bg,m,vlen, vnb,colst,vpos+1,taupos+1);
       
                   if((vlen>0)&&(vnb>0)){
                       if(LOGQ) core_event_startblg(my_core_id);
                       for (chunkid = 0; chunkid<nbchunk; chunkid++)
                       {
                           coreid  = chunkid%allcoresnb;
                           corest  = chunkid*colpercore;
                           corelen = min(colpercore, (N-(chunkid*colpercore)));
                           if(my_core_id==coreid)
                           {   
                               if(WANTZ==1){ 
                                       st  = max(colst,corest);           
                                       len = corelen - (st-corest);    
                                       lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &len, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,st), &LDE,  WORK, &len); 
                                   }else{
                                       lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &corelen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,corest), &LDE,  WORK, &corelen); // corelen for WORK was colpercore before 
                                   } // end WANTZ
                           } // END my_core_id=coreid
                       } // END loop over the chunk
                       if(LOGQ) {
                           core_event_endblg(my_core_id);
                           core_log_eventblg(0xff0000, my_core_id);
                       }
                   } // end if vlen and vnb
               } // end for m:rowmnb
            } // end for bg 
        }else if(version==114){
            rownbm    = plasma_ceildiv((N-2),NB);
            for (m = rownbm; m>0; m--)
            {
                ncolinvolvd = min(N-2, m*NB);
                avai_blksiz=min(Vblksiz,ncolinvolvd);
                nbgr = plasma_ceildiv(ncolinvolvd,avai_blksiz);
                for (n = nbgr; n>0; n--)
                {
                    vlen = 0;
                    vnb  = 0;
                    cur_blksiz = min(ncolinvolvd-(n-1)*avai_blksiz, avai_blksiz);
                    colst = (n-1)*avai_blksiz;
                    coled = colst + cur_blksiz -1;
                    fst   = (rownbm -m)*NB+colst +1;
                    for (colj=colst; colj<=coled; colj++)
                    {
                        st       = (rownbm -m)*NB+colj +1;
                        ed       = min(st+NB-1,N-1);
                        if(st>=ed)break;
                        vlen=ed-fst+1;
                        vnb=vnb+1;
                    }        
                    findVTpos(N,NB,Vblksiz,colst,fst, &vpos, &taupos, &tpos, &blkid);
                    //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n",bg,m,vlen, vnb,colj,vpos+1,taupos+1);
                    if((vlen>0)&&(vnb>0)){
                        for (chunkid = 0; chunkid<nbchunk; chunkid++)
                        {
                            coreid  = chunkid%allcoresnb;
                            corest  = chunkid*colpercore;
                            corelen = min(colpercore, (N-(chunkid*colpercore)));
                            if(my_core_id==coreid)
                            {                 
                                lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &corelen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(fst,corest), &LDE,  WORK, &corelen);    
                            } // END my_core_id=coreid
                        } // END loop over the chunk
                    } // end if vlen and vnb
                } // end for n = nbgr
            } // end for m=rowmnb
        }
    }else if (SIDE=='R'){
        for (bg =1; bg<=nbGblk; bg++)
        {
           firstcolj = (bg-1)*Vblksiz + 1;
           rownbm    = plasma_ceildiv((N-(firstcolj+1)),NB);
           for (m = 1; m<=rownbm; m++)
           {
               vlen = 0;
               vnb  = 0;
               // for k=0;I compute the fst and then can remove it from the loop
               colj     = (bg-1)*Vblksiz; 
               fst      = (rownbm -m)*NB+colj +1;
               for (k=0; k<Vblksiz; k++)
               {
                   colj     = (bg-1)*Vblksiz + k;
                   st       = (rownbm -m)*NB+colj +1;
                   ed       = min(st+NB-1,N-1);
                   if(st==ed)break;
                   vlen=ed-fst+1;
                   vnb=k+1;
               }        
               colj     = (bg-1)*Vblksiz;
               findVTpos(N,NB,Vblksiz,colj,fst, &vpos, &taupos, &tpos, &blkid);
               //printf("voici bg %d m %d  vlen %d  vnb %d fcolj %d vpos %d taupos %d \n",bg,m,vlen, vnb,colj,vpos,taupos);
                if((vlen>0)&&(vnb>0)){
                    for (chunkid = 0; chunkid<nbchunk; chunkid++)
                    {
                        coreid  = chunkid%allcoresnb;
                        corest  = chunkid*colpercore;
                        corelen = min(colpercore, (N-(chunkid*colpercore)));
                        if(my_core_id==coreid)
                        {            
                            lapackf77_zlarfb( "R", "N", "F", "C", &corelen, &vlen, &vnb, V(vpos), &LDV, T(tpos), &LDT, E(corest,fst), &LDE,  WORK, &corelen);       
                        } // END my_core_id=coreid
                    } // END loop over the chunk
                } // end if vlen and vnb
           } // end for m=rowmnb
        } // end for bg
    }else{
        printf("ERROR SIDE %d \n",SIDE);
    }


}
#undef E
#undef V
#undef TAU
#undef T
////////////////////////////////////////////////////////////////////////////////////////////////////








