/* 
    -- MAGMA (version 1.3) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       Sept 2013 
 
       @author: Simplice Donfack 
*/ 

#include <math.h>
#include "common_magma.h"

#include "magma_amc.h"

#include "schedule.h" 

#include "core_d.h"


#include "magma_task_dev_d.h"
// === Define what BLAS to use ============================================ 
/* 
#define PRECISION_d 
#if (GPUSHMEM <= 200) && (defined(PRECISION_s) || defined(PRECISION_d)) 
  #define magma_dgemm magmablas_dgemm 
  #define magma_dtrsm magmablas_dtrsm 
#endif 
*/ 
// === End defining what BLAS to use ======================================= 


/*Each column block belong to which GPU*/
#define GID(J) ((J)%num_gpus)


//#define dlA(id,i,j) (dlA[(id)] + (j)*nb*dAl_LD + (i)*nb) 
//#define dlAT(id,i,j) (dlAT[(id)] + (i)*nb*dAlT_LD + (j)*nb) 

/*Automatically set the pointer to the GPU which own column J after a cyclic repartition*/
#define dlAT(I,J) (dlAT[GID((J))] + (I)*nb*dlAT_LD + (((J)-GID((J)))/num_gpus)*nb) 

#define dlA(I,J) (dlA[GID((J))] + (((J)-GID((J)))/num_gpus)*nb*dlA_LD + (I)*nb)
///#define A(I,J) (A + ((J)%A_N)*nb*A_LD + (I)*nb) 
///#define colptr(J) A(0,(J)) 

//&(dlpanel[dd][dlpanel_LD*nb*K])
#define dlpanel(dev, K) (dlpanel[(dev)]+(K)*nb*dlpanel_LD)
 
#define A(I,J) (A + ((J)%A_NMAX)*nb*A_LD + (I)*nb)
#define colptr(J) A(0,(J))

#ifdef PARALLEL_AWORK //obsolete
#undef A
#undef colptr
#define A(I,J) &(A[((J)%A_SIZE)][(I)*nb])
//#define colptr(J) A[((J)%A_SIZE)] 
#define colptr(J) (void *)(intptr_t)(((J)%A_SIZE)+1) //A[((J)%A_SIZE)]
#endif


//#define A_event(I) (A_event+((I)%A_SIZE))

#define ipiv(I) (ipiv+(I)*nb) 
 

//extern "C" 
//void CORE_zgetrf_reclap_init();
/* number of blocs for device dev in cyclic distribution */
int loc2p(int dev, int N, int P)
{
/*
dev:    device id
N: Number of blocks
P: Number of GPUS
*/
    int rest,nblocs;
    rest = N % P;
    nblocs = (N-rest)/P;

    if(dev<rest) nblocs+=1;

    return nblocs;
}

/* number of columns on device dev in a cyclic distribution of the data*/
static int numcols2p(int dev, int n, int b, int P)
{
/*
dev:    device id
n: Number of blocks
b: block size
P: Number of GPUS
*/
    int N, rest,ncols;
    N = (int) ((n - (n % b))/b);

    rest = N % P;
    ncols = (N-rest)/P;

    if(dev<rest) 
        ncols+=1;

    ncols = ncols * b;
    //if(n%b!=0 && dev==P-1) 
    if(n%b!=0 && dev == (N%P) )
        ncols = ncols + (n%b);

    return ncols;
}

/* number of columns on device dev in a cyclic distribution of the data*/
static int numcols2p_v2(int dev, int n, int b, int P)
{
/*
dev:    device id
n: Number of blocks
b: block size
P: Number of GPUS
*/
    int N, rest,ncols;
    N = (int) ((n - (n % b))/b);

    rest = N % P;
    ncols = (N-rest)/P;

    if(dev<rest) ncols+=1;

    ncols = ncols * b;
    if(n%b!=0 && dev==P-1) 
        ncols = ncols -b + (n%b);

    return ncols;
}



/*
*
*  ASYNC_DGETRF_REC_MGPU:
*
*  This function allocate the workspace and call ASYNC_DGETRF_REC_ASYNC_WORK_MGPU
*  REC: recursif Algorithm
*  ASYNC: asynchronous algorithm
*/

extern "C" magma_int_t magma_dgetrf_mgpu_amc_v3(magma_int_t num_gpus, magma_int_t m, magma_int_t n,  
                 double **dlA, magma_int_t dlA_LD, 
                 magma_int_t *ipiv, magma_int_t *info) 
{ 
/*  -- MAGMA (version 1.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       November 2011 
 
    Purpose 
    ======= 
 
    DGETRF_ASYNC_GPU computes an LU factorization of a general M-by-N matrix A 
    using partial pivoting with row interchanges. The technique used for the panel factorization
    is the parallel recursif LU (see lawn 259).
 
    The factorization has the form 
       A = P * L * U 
    where P is a permutation matrix, L is lower triangular with unit 
    diagonal elements (lower trapezoidal if m > n), and U is upper 
    triangular (upper trapezoidal if m < n). 
 
    This is the right-looking Level 3 BLAS version of the algorithm. 
 
    Arguments 
    ========= 
    NUM_GPUS
            (input) INTEGER
            The number of GPUS to be used for the factorization.

    M       (input) INTEGER 
            The number of rows of the matrix A.  M >= 0. 
 
    N       (input) INTEGER 
            The number of columns of the matrix A.  N >= 0. 
 
    A       (input/output) DOUBLE_PRECISION array on the GPU, dimension (LDDA,N). 
            On entry, the M-by-N matrix to be factored. 
            On exit, the factors L and U from the factorization 
            A = P*L*U; the unit diagonal elements of L are not stored. 
 
    LDDA     (input) INTEGER 
            The leading dimension of the array A.  LDDA >= max(1,M). 
 
    IPIV    (output) INTEGER array, dimension (min(M,N)) 
            The pivot indices; for 1 <= i <= min(M,N), row i of the 
            matrix was interchanged with row IPIV(i). 
 
    INFO    (output) INTEGER 
            = 0:  successful exit 
            < 0:  if INFO = -i, the i-th argument had an illegal value 
                  or another error occured, such as memory allocation failed. 
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization 
                  has been completed, but the factor U is exactly 
                  singular, and division by zero will occur if it is used 
                  to solve a system of equations. 
    =====================================================================    */ 
 
     
     /*Workspace*/
     double *AWORK;
     magma_int_t AWORK_LD, AWORK_n;
    
     double **dlpanelT;
     magma_int_t dlpanelT_m, dlpanelT_n;

     int nbcores; /*Number of cores available for the whole factorization*/ 
    // int panel_num_threads; /*Number of threads for the panel*/ 
     double dcpu; /*percentage of the matrix to allocate on the CPUs*/ 
     int nb;
     amc_args_t *args;

     int dd;
#if (dbglevel >=1) 
    double t1;
#endif
    #if (dbglevel==10)
    char sbuff[80];
    #endif

    int N;

//    magma_event_t *A_event; /*Controling bucket*/

    /* Check arguments */ 
    *info = 0; 
    if (m < 0) 
        *info = -1; 
    else if (n < 0) 
        *info = -2; 
    else if (dlA_LD < max(1,m)) 
        *info = -4; 
 
    if (*info != 0) { 
        magma_xerbla( __func__, -(*info) ); 
        return *info; 
    } 
 
    /* Quick return if possible */ 
    if (m == 0 || n == 0) 
        return *info; 
      
    /* Get parameters */  
    
    
    
    args = magma_amc_args_get_default();

    if(args->nb==0)
     nb    = magma_get_dgetrf_nb(m) ;//magma dgetrf block size
    else
     nb = args->nb;

     nbcores = args->P;  
    
     dcpu = args->dcpu;
 
     /*check and fix parameters */
     if(dcpu>1.0) dcpu = 1.0;

     /* Compute the number of blocs columns*/
     N  = (int) ceil( (double) n / nb);

     /*Compute the dimension of the workspace matrix for the cpu*/
     AWORK_LD = m;
     AWORK_n = NSplit(N, dcpu)*nb; //(int) ceil(n*dcpu);

     //if(AWORK_n<nb) 
     //     AWORK_n = nb;//make sure workspace has at least one block column

     //AWORK_n += 1*nb; /* +1 avoid current panel to be overwritten*/

     /*Make LD and n multiple of 32*/
     if(AWORK_LD%32!=0) AWORK_LD = ((AWORK_LD + 31)/32)*32;
     if(AWORK_n%32!=0) AWORK_n = ((AWORK_n + 31)/32)*32; 
     

     /*Allocate the CPU part of the matrix to factorize*/
#if (dbglevel >=1)
    t1 = magma_wtime();
#endif
    
    if (MAGMA_SUCCESS != magma_dmalloc_pinned(&AWORK, AWORK_LD*AWORK_n)) { 
    //if (MAGMA_SUCCESS != magma_dmalloc_cpu(&AWORK, AWORK_LD*AWORK_n)) {
            *info = MAGMA_ERR_HOST_ALLOC; 
            return *info; 
    } 
    
    /* Workspace for the panels on the GPU*/
    dlpanelT_m = AWORK_n; /*assume that the cpu and gpu use the same buffer size*/
    dlpanelT_n = m;
     dlpanelT = (double **)    malloc(num_gpus*sizeof(double*));
      for(dd=0;dd<num_gpus;dd++){
         magma_setdevice(dd);

         if (MAGMA_SUCCESS != magma_dmalloc(&dlpanelT[dd], dlpanelT_m*dlpanelT_n)) { 

                
                *info = MAGMA_ERR_DEVICE_ALLOC; 
                return *info; 
        }
      }

#if (dbglevel >=1)
    printf("[DBG] Time memory malloc (pinned):%f\n",magma_wtime()-t1); 
    t1 = magma_wtime();
#endif

    /*First touch the workspace by each thread*/
    //magma_amc_dmemset(AWORK, 0.0, AWORK_LD*AWORK_n, nb, nbcores);

#if (dbglevel==10)
    //ca_dbg_printMat(AWORK_LD, AWORK_n, AWORK, AWORK_LD,"A after first touch");
#endif

    /* Call the workspace interface */
    *info = magma_dgetrf_mgpu_work_amc_v3(num_gpus, m, n, dlA, dlA_LD, ipiv, info, AWORK, AWORK_LD, AWORK_n);

#if (dbglevel >=1)
    printf("[DBG] Time Factorization:%f\n",magma_wtime()-t1); 
    t1 = magma_wtime();
#endif

    magma_free_pinned(AWORK);

    for(dd=0;dd<num_gpus;dd++){
        magma_setdevice(dd);
        magma_free(dlpanelT[dd]);
    }

    //free(dlAP_set); 
    //free(dlAP_get);
    free(dlpanelT);

#if (dbglevel >=1)
   printf("[DBG] Time memory free memory:%f\n",magma_wtime()-t1); 
   t1 = magma_wtime();
#endif

#if (dbglevel==10)     
//    ca_dbg_printMat_transpose_gpu(m, n, dA, dA_LD,"dA = LU"); 
#endif 



    return *info; 
}   /* End of MAGMA_DGETRF_REC_ASYNC_GPU */





/*
*
*  DGETRF_REC_ASYNC_WORK_GPU
*
*  This function performs the factorization when the workspace is already allocated.
*/


extern "C" magma_int_t 
magma_dgetrf_mgpu_work_amc_v3(magma_int_t num_gpus,
magma_int_t m, magma_int_t n,  
double **dlA, magma_int_t dlA_LD, 
magma_int_t *ipiv, magma_int_t *info,
/*workspace on the cpu side*/
double *AWORK, magma_int_t AWORK_LD, magma_int_t AWORK_n
) 
{ 
/*  -- MAGMA (version 1.1) -- 
       Univ. of Tennessee, Knoxville 
       Univ. of California, Berkeley 
       Univ. of Colorado, Denver 
       November 2011 
 
    Purpose 
    ======= 
 
    DGETRF_REC_ASYNC computes an LU factorization of a general M-by-N matrix A 
    using partial pivoting with row interchanges. The technique used for the panel factorization
    is the parallel recursif LU (see lawn 259).
 
    The factorization has the form 
       A = P * L * U 
    where P is a permutation matrix, L is lower triangular with unit 
    diagonal elements (lower trapezoidal if m > n), and U is upper 
    triangular (upper trapezoidal if m < n). 
 
    This is the right-looking Level 3 BLAS version of the algorithm. 
 
    Arguments 
    ========= 
    NUM_GPUS
            (input) INTEGER
            The number of GPUS to be used for the factorization.
 
    M       (input) INTEGER 
            The number of rows of the matrix A.  M >= 0. 
 
    N       (input) INTEGER 
            The number of columns of the matrix A.  N >= 0. 
 
    A       (input/output) DOUBLE_PRECISION array on the GPU, dimension (LDDA,N). 
            On entry, the M-by-N matrix to be factored. 
            On exit, the factors L and U from the factorization 
            A = P*L*U; the unit diagonal elements of L are not stored. 
 
    LDDA     (input) INTEGER 
            The leading dimension of the array A.  LDDA >= max(1,M). 
 
    IPIV    (output) INTEGER array, dimension (min(M,N)) 
            The pivot indices; for 1 <= i <= min(M,N), row i of the 
            matrix was interchanged with row IPIV(i). 
 
    INFO    (output) INTEGER 
            = 0:  successful exit 
            < 0:  if INFO = -i, the i-th argument had an illegal value 
                  or another error occured, such as memory allocation failed. 
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization 
                  has been completed, but the factor U is exactly 
                  singular, and division by zero will occur if it is used 
                  to solve a system of equations.

    =====================================================================    */ 
 
 
 
    double c_one     = MAGMA_D_ONE; 
    double c_neg_one = MAGMA_D_NEG_ONE; 
 
    int ONE = 1; 
 
    magma_int_t iinfo, nb; 
    magma_int_t mindim; 
    magma_int_t nrows, ncols; 
    //double *work; 
 
 
     magma_int_t dm_max, dn_max; 
     magma_int_t I, J, K, M, N, U_K, L; 
     
     //magma_int_t A_m, A_n, A_N; 
     //magma_int_t Am_max, An_max; 
     //magma_int_t A_nb; 
     
 
     //magma_int_t A_K; 
     double **dlAT; 
     magma_int_t dlAT_LD; 
      
      
     double *dlAP_get[MagmaMaxGPUs]; //*dlAP_set[MagmaMaxGPUs]
      
     
     double *dlAP_set[MagmaMaxGPUs];
     magma_int_t dlAP_LD;

     double *dlpanel[MagmaMaxGPUs];
     magma_int_t dlpanel_LD;

     int *n_local, *nr_local;

     //magma_int_t nrows, ncols; 
     magma_int_t gpu_nrows, gpu_ncols; 
  
     int nbcores; /*Number of cores available for the whole factorization*/ 
     int panel_num_threads; /*Number of threads for the panel*/ 
     double dcpu; /*percentage of the matrix to allocate on the CPUs*/ 
  
    int B_rows;

    double t1;
    
    /*Workspace*/
    // magma_int_t AWORK_NMAX;
    // magma_int_t AWORK_m, AWORK_n, AWORK_N;

     /* Recommanded dimension in the workspace*/ 
     int A_m, A_n, A_N, A_NMAX, A_LD;
     int A_NP1;
     double *A;

     amc_args_t *args;
    /*magma_event_t *A_event;*/ /*Control bucket*/
     magma_queue_t mstream[MagmaMaxGPUs][3]; /*0: H2D, 1: compute, 2:D2H*/
     int dd;

//     double *tmpdA;

    /* Check arguments */ 
    *info = 0; 
    if (m < 0) 
        *info = -1; 
    else if (n < 0) 
        *info = -2; 
    else if (dlA_LD < max(1,m)) 
        *info = -4; 
    else if (AWORK_LD < max(1,m)) 
        *info = -5;

    if (*info != 0) { 
        magma_xerbla( __func__, -(*info) ); 
        return *info; 
    } 
 
    /* Quick return if possible */ 
    if (m == 0 || n == 0) 
        return *info; 
 
 


      
     /*Get parameters*/ 
    args = magma_amc_args_get_default();
     nb= args->nb;

     nbcores = args->P;  
     panel_num_threads = args->Pr; 
     dcpu = args->dcpu;

     /* Check and fix parameters */
     if(nb==0)
        nb     = magma_get_dgetrf_nb(m) ;/*magma dgetrf block size*/ 
    else
        nb = args->nb;

     if(nb>n) nb = n; 
     if(panel_num_threads>nbcores) panel_num_threads = nbcores;

     /*check the buffer size*/

     if(AWORK_n<nb){
         printf("Not enough buffer. Should be greater than the block size: %d\n", nb);
         exit(1);
     }

     /* Compute the number of blocks columns to factorize*/
     N  = (int) ceil( (double) min(m, n) / nb);

     /* Compute the maximum number of panels we can store in the workspace*/
     A_NMAX = (int) (AWORK_n/ nb);

     /*Compute the recommanded number of panels for the cpu part*/
     A_N = NSplit(N, dcpu);

     /* Compute the recommanded number of columns for the cpu part*/
     A_n = A_N*nb;//(int) ceil(n*dcpu);

     //if(A_n<nb) 
     //     A_n = nb;//make sure workspace has at least one block column

     /*Make sure we work with multiple of 32*/
     /*
     if(A_n%32!=0) {
         A_n = ((A_n + 31)/32)*32;
     }
     */

     /* Compute the recommanded number of panels for the cpu part*/
    // A_N = (int) (A_n/ nb);
     
     /* Check if there are enough workspace. In case the user gave a workspace lower than the optimal*/
     /* NOTE: using small workspace may reduce performance*/

     if(A_N>A_NMAX){    

#if (dbglevel >=1)
        printf("[DBG_WARNING] Resizing buffer to feet user preferences. Recommanded:%d, Max given:%d\n",A_N, A_NMAX); 
#endif
        A_N = A_NMAX;

        /*Make A_n a multiple of nb*/
        A_n = A_N*nb;
    }
      
     A = AWORK;
     A_m = m;
     A_LD = AWORK_LD;


#if (dbglevel >=1)
    /* Initialize the tracing*/
    ca_dbg_trace_init(nbcores,num_gpus); //nbcores + 1 GPU
#endif

#if (dbglevel >=1)
    t1 = magma_wtime();
#endif

    /* create the streams */
    //mstream = (magma_queue_t *)    malloc(num_gpus*sizeof(magma_queue_t));

    for(dd=0;dd<num_gpus;dd++){
       magma_setdevice(dd); //required
       magma_queue_create(&mstream[dd][0]);
       magma_queue_create(&mstream[dd][1]);
       magma_queue_create(&mstream[dd][2]);

       /*Set the stream for internal computations*/
       //magmablasSetKernelStream(0); /*Use 0 instead of mstream[dd][1], MagmasetkernelStream is not thread safe*/ /*TODO: mae it safe*/

       //task_dev_set_compute_stream(dd, mstream[dd][1]);
       magma_task_dev_set_compute_stream(dd, 0); //set to mstream 1 later
    }

    


     /* Matrix dimension */
     dm_max = m;
     dn_max = n;

    /*Make sure m and n are multiple of 32*/
     
     if(dm_max%32!=0) dm_max = ((dm_max + 31)/32)*32;
     if(dn_max%32!=0) dn_max = ((dn_max + 31)/32)*32;
     
     

     /* local dimensions of the matrix for each GPU*/
     n_local = (int *)    malloc(num_gpus*sizeof(int)); /*This do no change during the execution*/
     nr_local = (int *)    malloc(num_gpus*sizeof(int)); /*Change after each update of the trailing submatrix*/

     for(dd=0;dd<num_gpus;dd++){
        n_local[dd] = numcols2p(dd, n, nb, num_gpus); //loc2p(dd, N, num_gpus)*nb;    
        nr_local[dd] = n_local[dd];
     }


     /*Allocate a workspace for the panels transposition*/ 

     dlAP_LD = dm_max; 
     //if(dAP_LD%32!=0) dAP_LD = ((dAP_LD + 31)/32)*32;/*Make dAP_LD multiple of 32*/
    /// dlAP_set = (double **)    malloc(num_gpus*sizeof(double*));
     //dlAP_get = (double **)    malloc(num_gpus*sizeof(double*));

     for(dd=0;dd<num_gpus;dd++){

         magma_setdevice(dd);
        
         if (MAGMA_SUCCESS != magma_dmalloc( &dlAP_set[dd], dlAP_LD*nb)) { 
                *info = MAGMA_ERR_DEVICE_ALLOC; 
                return *info; 
        } 
        
        /*
        if (MAGMA_SUCCESS != magma_dmalloc(&tmpdA, dlAP_LD*nb)) { 
                *info = MAGMA_ERR_DEVICE_ALLOC; 
                return *info; 
        }
        */
        if ( magma_is_devptr(dlAP_set[dd] ) == 0 ) {
            fprintf( stderr, "ERROR: dlAP_set[dd] is host pointer.\n" );
            //exit(1);
        }

        
        //cudaMemcpy(dlAP_set[dd],&tmpdA,sizeof(double*), cudaMemcpyDeviceToHost);

        #if (dbglevel==10) 
        printf("0.4\n");
            
            //ca_dbg_printMat_gpu(2, 2, dlAP_set[dd], dlAP_LD, "dlAP_set[dd] for testing");
            //cudaMemcpy(&tmpdA, &dlAP_set[dd], sizeof(double*), cudaMemcpyHostToDevice);
            //ca_dbg_printMat_gpu(2, 2, tmpdA, dlAP_LD, "dlAP_set[dd] for testing");
            //printf("0.5: int to continue"); scanf("%d", &I);
        #endif

         if (MAGMA_SUCCESS != magma_dmalloc(&dlAP_get[dd], dlAP_LD*nb)) { 
                //magma_free(dlAP_set); //TODO: free all previous buffers
                *info = MAGMA_ERR_DEVICE_ALLOC; 
                return *info; 
        }
     }

    /* Workspace for the panels */

    // dlpanel = (double **)    malloc(num_gpus*sizeof(double*));
     
     for(dd=0;dd<num_gpus;dd++){
         magma_setdevice(dd);

         if (MAGMA_SUCCESS != magma_dmalloc(&dlpanel[dd], nb*dm_max)) { 

                
                *info = MAGMA_ERR_DEVICE_ALLOC; 
                return *info; 
        }
      }

      dlpanel_LD = nb;
      

    /*local matrix storage*/
    dlAT = (double **)    malloc(num_gpus*sizeof(double*));

    
    dlAT_LD = n_local[0];

    if(dlAT_LD%32!=0) dlAT_LD = ((dlAT_LD + 31)/32)*32;

    for(dd=0;dd<num_gpus;dd++){
         magma_setdevice(dd);

        if (MAGMA_SUCCESS != magma_dmalloc(&dlAT[dd], dlAT_LD*dm_max )) { 
                for(J=0;J<dd;J++){
                    magma_setdevice(J);
                    magma_free( dlAP_set[J]); 
                    magma_free( dlAP_get[J]);
                    magma_free(dlpanel[J]);
                    magma_free(dlAT[J]);
                }
                //free(dlAP_set); 
                //free(dlAP_get);
                //free(dlpanel);
                free(dlAT);
            *info = MAGMA_ERR_DEVICE_ALLOC; 
            return *info; 
        }



    }


#if (dbglevel >=1)
    printf("[DBG] Time workspace memory alloc (dAP): %f\n",magma_wtime()-t1);
    t1 = magma_wtime();
#endif


    /*1. Transfer the first column blocks of the matrix from the GPU to the CPUs.*/ 
    
    //magma_dgetmatrix(A_m, A_n, dA, dA_LD, A, A_LD); 
    magma_dgetmatrix_1D_col_bcyclic(A_m, A_n, dlA, dlA_LD, A, A_LD, num_gpus, nb);

#if (dbglevel >=1)
    printf("[DBG] Time First getmatrix: %f\n",magma_wtime()-t1);
    t1 = magma_wtime();
#endif

#if (dbglevel==10) 
    printf("1.0\n");
    ca_dbg_printMat(A_m, A_n, A, A_LD,"A after first getMatrix"); 

    /*
    for(dd=0;dd<num_gpus;dd++){
        //Fill the matrix with zero for easy visualization of the matrix in debug mode
        for(I=0;I<dlAT_LD*dm_max;I++)  dlAT[dd][I] = 0.0;
    }
    */
//    ca_dbg_printMat_mgpu(num_gpus,  m, n_local, dlAT, dlAT_LD,"matrix dAlT^T empty");
//    ca_dbg_printMat_transpose_mgpu(num_gpus,  n_local, m, dlAT, dlAT_LD,"matrix dAT empty");
printf("2.0\n");
#endif

    /*Update the remaining number of columns on the GPUs.*/
    for(dd=0;dd<num_gpus;dd++){
        nr_local[dd] = nr_local[dd] - numcols2p(dd, A_n, nb, num_gpus); //;n_local[dd] - loc2p(dd, A_N, num_gpus)*nb;        
    }

#if (dbglevel==10) 
    ca_dbg_printMat_mgpu(num_gpus, m, n_local, dlA, dlA_LD,"matrix dA to factorize");

    printf("3.0\n");    
#endif


for(dd=0;dd<num_gpus;dd++){
    magma_setdevice(dd);
    //magmablasSetKernelStream(mstream[dd][1]);    
    magmablas_dtranspose2(dlAT[dd], dlAT_LD, dlA[dd], dlA_LD, m, n_local[dd]);
}

///


for(dd=0;dd<num_gpus;dd++){
    magma_setdevice(dd);
    magma_task_dev_set_compute_stream(dd, mstream[dd][1]);
}



#if (dbglevel >=1)
    printf("[DBG] Time First transposition: %f\n",magma_wtime()-t1);
    t1 = magma_wtime();
#endif

#if (dbglevel==10) 
    //ca_dbg_printMat_transpose_mgpu(num_gpus, n_local, m, dlAT, dlAT_LD,"matrix dAT to factorize");
/*
    dd = GID(A_N);
    magma_setdevice(dd);

    ca_dbg_printMat_transpose_gpu(nb, m, dlAT(0, A_N), dlAT_LD,"matrix dAT(0, A_N)");
    
    magma_setdevice(0);
    ca_dbg_printMat_transpose_gpu(m, nb, dlA(0, A_N), dlA_LD,"matrix dA(0, A_N)");
    */

    printf("4.0\n");
    printf("int to continue"); scanf("%d", &I);
#endif

/*
#if (dbglevel==10) 
    ca_dbg_printMat_transpose_mgpu(num_gpus, m, n_local, dlAT, dlAT_LD,"matrix dAT to factorize");
#endif
*/


     /* Compute the maximun number of steps*/
     mindim = min(m, n); 
     M      = (int) ceil( (double) m / nb); 
     N      = (int) ceil( (double) mindim / nb); /*N = n/nb*/


     /* 3. Let the asynchronous algorithm begin*/ 
     
#if (dbglevel >=1)
     printf("Starting recursif code ... m:%d, n:%d, nb:%d, nbcores:%d, N:%d, A_N:%d\n", m, n, nb, nbcores, N, A_N); //Summary
#endif



     /*Initialize the scheduler*/ 
     magma_schedule_init(nbcores, num_gpus); 



     K = 0; 
     /*initialize parallel recursif panel environment*/
     CORE_zgetrf_reclap_init();

     magma_schedule_set_task_priority(INT_MAX-1);
     
     /*Schedule the first panel factorization*/ 
     magma_insert_core_dgetrf_rec(A_m, nb, A(0,K), A_LD, ipiv(0), &iinfo, panel_num_threads, colptr(K));  
     //magma_insert_core_dgetrf(A_m, nb, A(0,K), A_LD, ipiv(0), &iinfo, colptr(K)); 
 
     /*Transfer the factorized panel in the buffer of GPU (dlpanel)*/

     for(dd=0;dd<num_gpus;dd++){
         
        
        ///magma_insert_dev_dsetmatrix_transpose(dd, A_m, nb, A(0,K), A_LD, dlpanel(dd,K), dlpanel_LD, dlAP_set[dd], dlAP_LD, colptr(K), dlpanel[dd]);
        magma_insert_dev_dsetmatrix_async_transpose(dd, A_m, nb, A(0,K), A_LD, dlpanel(dd,K), dlpanel_LD, mstream[dd][0], dlAP_set[dd], dlAP_LD, colptr(K), dlpanel(dd,K)); //dlpanel[dd]

     }
#if (dbglevel==10) 
    magma_schedule_barrier();

    for(dd=0;dd<num_gpus;dd++){
        magma_setdevice(dd);
        ca_dbg_printMat_transpose_gpu(nb, m, dlpanel(dd,K), dlpanel_LD,"dlpanel[dd] after setmatrix_async"); //dlpanel[dd]
    }
    printf("4.5: int to continue"); scanf("%d", &I);
#endif
     /*Transfer also the factorized panel on its right position in the final matrix (transposition included)*/ 
     /*TODO: this may use cudaMemcpyDeviceToDevice and initiate the transfer from dlpanel*/
     dd = GID(K);
     //magma_insert_dev_dsetmatrix_transpose(dd, A_m, nb, A(0,K), A_LD, dlAT(0,K), dlAT_LD, dlAP_set[dd], dlAP_LD, colptr(K), dlAT(0,K)); 
     magma_insert_dev_dsetmatrix_async_transpose(dd, A_m, nb, A(0,K), A_LD, dlAT(0,K), dlAT_LD, mstream[dd][0], dlAP_set[dd], dlAP_LD, colptr(K), dlAT(0,K)); 
 
#if (dbglevel==10) 
    magma_schedule_barrier(); 
    ca_dbg_printMat(m, nb, A(0,0), A_LD,"A(0,0)");
    
    for(dd=0;dd<num_gpus;dd++){
        magma_setdevice(dd);
        ca_dbg_printMat_transpose_gpu(nb, m, dlpanel[dd], dlpanel_LD,"dlpanel[dd] after setmatrix to dlAT");
    }

    ca_dbg_printMat_transpose_mgpu(num_gpus, n_local, m, dlAT, dlAT_LD,"dlA"); 
 printf("5.0: int to continue"); scanf("%d", &I);
#endif 

     for(K=0;K<=N-1;K++){ 
     
         /*compute the new value of the cpu number of blocks*/
         A_N = NSplit(N-K, dcpu);

          /*insert the coarse update of the trailing submatrix corresponding to panel K to the GPU, that is submatrix A[K+1:M, K+1+d-1:N]*/ 
          
         //if(K==0) /*TODO: move outside loop*/
          //{

         /*NOTE: Here we work on the matrix transpose*/

         /*Set the priority max for the GPU computations*/
            magma_schedule_set_task_priority(INT_MAX);
            //// magma_schedule_set_task_priority(INT_MAX - N*K);

         gpu_nrows = m - (K+1)*nb;///

         for(J=K+A_N;J<min(K+A_N+num_gpus,N);J++){

            /*Determine the device which own the first column of the group of columns to update*/
             dd = GID(J);

            /*Determine the number of columns to apply the update. */
             nr_local[dd] = numcols2p(dd, n - (K+1+A_N-1)*nb, nb, num_gpus);

              gpu_ncols = nr_local[dd]; //n - (K+1+A_N-1)*nb; 
            
              if(gpu_ncols >0) 
              { 
 
                  /*schedule a swap of the trailing submatrix in the gpus using ipiv[K]*/ 
                  /*dependency dAT((K+1)-1, (K+A_N)-1) = dAT(K, K+A_N-1) with previous dgemm*/              
              
                  magma_insert_dev_dlaswp(dd, gpu_ncols, dlAT(K, J), dlAT_LD, ONE, nb, ipiv(K), ONE, dlAT(K, J-1)); /*non blocking*/                  
                  //printf("debug barrier\n");
                  //magma_schedule_barrier();
                  //&(dlpanel[dd][dlpanel_LD*nb*K])
                  magma_insert_dev_dtrsm(dd, MagmaRight,  MagmaUpper, MagmaNoTrans, MagmaUnit, gpu_ncols, nb, c_one, dlpanel(dd,K), dlpanel_LD, dlAT(K,J), dlAT_LD);/*non blocking*/ 
 
                  /* aij^T = aij^T - (lik.ukj)^T = aij^T - ukj^T.lik^T*/ //&(dlpanel[dd][dlpanel_LD*nb*(K+1)])
                  magma_insert_dev_dgemm(dd, MagmaNoTrans,MagmaNoTrans, gpu_ncols, gpu_nrows, nb, c_neg_one, dlAT(K,J), dlAT_LD, dlpanel(dd,K+1), dlpanel_LD, c_one, dlAT(K+1,J), dlAT_LD);/*non blocking*/    
              

                  /*Transfer asynchronously one column (column K+A_N) from the GPU to the CPU to balance work*/                
                 //// if(K+A_N<N) 
                 //// { 
                    ////ncols = min(nb, gpu_ncols); 
 
                    //////magma_schedule_set_task_priority(INT_MAX);

                    ////magma_insert_dgetmatrix_transpose(gpu_nrows, ncols, dAT(K+1,K+A_N), dAT_LD, A(K+1,K+A_N), A_LD, dAP, dAP_LD, colptr(K+A_N)); //blocking
                 //// }
              
              } 
         }

          //}
          /*iterate over the rest of the columns to update the trailing submatrix on the cpu*/ 
          for(J=K+1;J<=min(K+A_N-1, N-1);J++){ 
 
               ncols = min(nb, n - J*nb); 
 
               /*Set the priority max for column having the next panel (look ahead of deep 1),
               and process the rest of the update in a right looking way*/
               if(J==K+1)
                   magma_schedule_set_task_priority(INT_MAX -2 );
                  //// magma_schedule_set_task_priority(INT_MAX - N*K -1);
               else
                   magma_schedule_set_task_priority(INT_MAX -3 - J );//- N*K
                  //// magma_schedule_set_task_priority(INT_MAX - N*K -3 -J);
               //magma_schedule_set_task_priority(INT_MAX - J);

               /*dependency colptr(J): make sure column J is sent from GPU, and all previous update was done*/
               magma_insert_core_dlaswp(ncols, A(K,J), A_LD, ONE, nb, ipiv(K), ONE, colptr(J)); 
 
               magma_insert_core_dtrsm('L', 'L', 'N', 'U', nb, ncols, c_one, A(K,K), A_LD, A(K,J), A_LD, colptr(J)); 
 
             /*Compute the number of blocs rows to group together before the update. To avoid scheduling overhead.*/
              B_rows = (int) ceil((double) (M-K-1)/panel_num_threads);
              B_rows = max(B_rows,4); /*maximun of 4*/ 
              //B_rows = max(B_rows,1);
              //printf("B_rows:%d\n",B_rows);
               for(I=K+1; I<=M-1; I+=B_rows){ 
     
                    nrows = min(B_rows*nb, m-I*nb); 
                    
                    /*dep colptr(K):make sure the panel is not overwritten or swapped since dgemm use A[I,K]*/
                    /*dep colptr(J): Gather all dgemm on one column and create dependencies with previous dgemm and the next panel*/
                    magma_insert_core_dgemm('N','N', nrows, ncols, nb, c_neg_one, A(I,K), A_LD, A(K,J), A_LD, c_one, A(I,J), A_LD, colptr(K), colptr(J)); 
               } 
 
                

               if(J==K+1) 
               { 
                    /*Look ahead and insert the next panel*/ 
                    nrows = m - (K+1)*nb; 
                    ncols = min(nb, n - (K+1)*nb); 
 
                    /*Schedule the next panel factorization with maximum priority*/ 
                    magma_schedule_set_task_priority(INT_MAX -1);
                    ///magma_schedule_set_task_priority(0); //TEST: testing prio_0
                   //// magma_schedule_set_task_priority(INT_MAX - N*K - 2);

                   magma_insert_core_dgetrf_rec(nrows, ncols, A(K+1,K+1), A_LD, ipiv(K+1), &iinfo, panel_num_threads, colptr(K+1)); 
                   // magma_insert_core_dgetrf(nrows, ncols, A(K+1,K+1), A_LD, ipiv(K+1), &iinfo, colptr(K+1)); 
 
                    /*Transfer the factorized panel in the buffer of GPU (dlpanel)*/

                     for(dd=0;dd<num_gpus;dd++){
                         //&(dlpanel[dd][dlpanel_LD*nb*(K+1)])
                        ///magma_insert_dev_dsetmatrix_transpose(dd, nrows, ncols, A(K+1, K+1), A_LD, dlpanel(dd, K+1), dlpanel_LD, dlAP_set[dd], dlAP_LD, colptr(K+1), dlpanel[dd]);
                        magma_insert_dev_dsetmatrix_async_transpose(dd, nrows, ncols, A(K+1, K+1), A_LD, dlpanel(dd, K+1), dlpanel_LD, mstream[dd][0], dlAP_set[dd], dlAP_LD, colptr(K+1), dlpanel(dd,K+1));//, dlpanel[dd]
                     }

                    /*Determine the upper part of the matrix done by the CPU on that column and send it to the GPU with the panel*/ 
                    U_K = max(0, K+1 - A_N +1); 
                    nrows = m - U_K*nb; 
 
                    ///magma_schedule_set_task_priority(INT_MAX);
                    /*Transfer the upper part of the matrix for that column and the factorized panel to the GPU*/ 
                    ///magma_insert_dsetmatrix_transpose(nrows, ncols, A(U_K, K+1), A_LD, dAT(U_K, K+1), dAT_LD, dAP, dAP_LD, A(K+1,K+1), dAT(K+1,K+1)); 
                    //magma_insert_dev_dsetmatrix_transpose(nrows, ncols, A(U_K, K+1), A_LD, dAT(U_K, K+1), dAT_LD, dAP_set, dAP_LD, colptr(K+1), dAT(K+1,K+1));
 
                    
                     /*Transfer also the factorized panel on its right position in the final matrix (transposition included)*/ 
                     /*TODO: this may use cudaMemcpyDeviceToDevice and initiate the transfer from dlpanel*/
                     dd = GID(K+1);
                     ///magma_insert_dev_dsetmatrix_transpose(dd, nrows, ncols, A(U_K, K+1), A_LD, dlAT(U_K,K+1), dlAT_LD, dlAP_set[dd], dlAP_LD, colptr(K+1), dlAT(K+1,K+1));
                     magma_insert_dev_dsetmatrix_async_transpose(dd, nrows, ncols, A(U_K, K+1), A_LD, dlAT(U_K,K+1), dlAT_LD, mstream[dd][0], dlAP_set[dd], dlAP_LD, colptr(K+1), dlAT(0,K+1));///

               } 
 
          } 
 
            /*compute the next number of blocks colums */
          A_NP1 = NSplit(N-(K+1), dcpu) - NSplit(N-K, dcpu) + 1;

           /*Transfer asynchronously (A_NP1 - A_N) block column (column K+A_N) from the GPU to the CPU to balance work*/  
            /*Make sure this is inserted after all dgemm because it schedules to replace a current panel for the case A_N< N*/
          for(L=K+A_N;L<K+A_N+A_NP1;L++)
          {
               if(L<N) { 

                 /*Determine the device which own column K+A_N*/
                 dd = GID(L);

                 gpu_ncols = nr_local[dd];

                 ncols = min(nb, gpu_ncols); 
 
                 magma_schedule_set_task_priority(INT_MAX);

                 ///magma_insert_dev_dgetmatrix_transpose(dd, gpu_nrows, ncols, dlAT(K+1,K+A_N), dlAT_LD, A(K+1,K+A_N), A_LD, dlAP_get[dd], dlAP_LD, colptr(K+A_N)); //blocking
             
                 /*make sure the computations are done on stream 1 and send a block column on stream 2*/
                 magma_insert_dev_queue_sync(dd, mstream[dd][1], dlAT(K+1,L)); 
                 magma_insert_dev_dgetmatrix_async_transpose(dd, gpu_nrows, ncols, dlAT(K+1,L), dlAT_LD, A(K+1,L), A_LD, mstream[dd][2], dlAP_get[dd], dlAP_LD, colptr(L));
                 /*Update the remaining number of columns*/
                //// nr_local[dd]-=nb;

                  /*if A_N==1, there is no look-ahead, so insert the panel here*/
                   if((A_N==1) && (L==K+A_N)){
                      /*Look ahead and insert the next panel*/ 
                      nrows = m - (K+1)*nb; 
                      ncols = min(nb, n - (K+1)*nb); 
                      /*Schedule the next panel factorization with maximum priority*/ 
                      magma_schedule_set_task_priority(INT_MAX -1);
                            ///magma_schedule_set_task_priority(0); //TEST: testing prio_0
                           //// magma_schedule_set_task_priority(INT_MAX - N*K - 2);

                      magma_insert_core_dgetrf_rec(nrows, ncols, A(K+1,K+1), A_LD, ipiv(K+1), &iinfo, panel_num_threads, colptr(K+1)); 
                      //magma_insert_core_dgetrf(nrows, ncols, A(K+1,K+1), A_LD, ipiv(K+1), &iinfo, colptr(K+1)); 
 
                      /*Transfer the factorized panel in the buffer of GPU (dlpanel)*/

                      for(dd=0;dd<num_gpus;dd++){
                          //&(dlpanel[dd][dlpanel_LD*nb*(K+1)])
                        ///magma_insert_dev_dsetmatrix_transpose(dd, nrows, ncols, A(K+1, K+1), A_LD, dlpanel(dd, K+1), dlpanel_LD, dlAP_set[dd], dlAP_LD, colptr(K+1), dlpanel[dd]);
                        magma_insert_dev_dsetmatrix_async_transpose(dd, nrows, ncols, A(K+1, K+1), A_LD, dlpanel(dd, K+1), dlpanel_LD, mstream[dd][0], dlAP_set[dd], dlAP_LD, colptr(K+1), dlpanel(dd,K+1));//dlpanel[dd]
                      }

                            /*Determine the upper part of the matrix done by the CPU on that column and send it to the GPU with the panel*/ 
                      U_K = max(0, K+1 - A_N +1); 
                      nrows = m - U_K*nb; 
 
                            ///magma_schedule_set_task_priority(INT_MAX);
                            /*Transfer the upper part of the matrix for that column and the factorized panel to the GPU*/ 
                            ///magma_insert_dsetmatrix_transpose(nrows, ncols, A(U_K, K+1), A_LD, dAT(U_K, K+1), dAT_LD, dAP, dAP_LD, A(K+1,K+1), dAT(K+1,K+1)); 
                            //magma_insert_dev_dsetmatrix_transpose(nrows, ncols, A(U_K, K+1), A_LD, dAT(U_K, K+1), dAT_LD, dAP_set, dAP_LD, colptr(K+1), dAT(K+1,K+1));

                      /*Transfer also the factorized panel on its right position in the final matrix (transposition included)*/ 
                      /*TODO: this may use cudaMemcpyDeviceToDevice and initiate the transfer from dlpanel*/
                      dd = GID(K+1);
                      ///magma_insert_dev_dsetmatrix_transpose(dd, nrows, ncols, A(U_K, K+1), A_LD, dlAT(U_K,K+1), dlAT_LD, dlAP_set[dd], dlAP_LD, colptr(K+1), dlAT(K+1,K+1));
                      magma_insert_dev_dsetmatrix_async_transpose(dd, nrows, ncols, A(U_K, K+1), A_LD, dlAT(U_K,K+1), dlAT_LD, mstream[dd][0], dlAP_set[dd], dlAP_LD, colptr(K+1), dlAT(0,K+1));///dlAT(K+1,K+1)
                   }
               }

         }
#if (dbglevel==10) 
  
  magma_schedule_barrier(); 
  ca_dbg_printMat(m, A_n, A, A_LD,"A"); 
  ca_dbg_printMat_transpose_mgpu(num_gpus, n_local, m, dlAT, dlAT_LD,"dAT (Step K)"); 
  
  nrows = m - K*nb; 
  ncols = min(nb, n - K*nb);

  dd = GID(K);
  magma_setdevice(dd);
  ca_dbg_printMat_transpose_gpu(ncols, nrows, dlAT(K,K), dlAT_LD,"dAT(K,K)");

  if(K<=5){
  printf("Step K:%d done. Int to continue: ",K); scanf("%d", &I);
  }

#endif 
           

     } //Step K done
 /*Wait for all thread termination*/
 magma_schedule_barrier(); 


 /*make sure everything arrived*/ ///needed?
 for(dd=0;dd<num_gpus;dd++){
        magma_setdevice(dd);
       magma_queue_sync(mstream[dd][0]);
       magma_queue_sync(mstream[dd][1]);
       magma_queue_sync(mstream[dd][2]);
}

     /*TODO: don't need quark here*/
     /*Perform a sequence of left swap on the matrix corresponding to the different panel*/ 
     for(K=1;K<=N-1;K++){ 
 
#if (dbglevel >=1)
    ca_trace_start();
#endif

    nrows = min(nb,m - K*nb);

    ncols = min(K*nb,n);

    for(dd=0;dd<=min(num_gpus-1, K-1);dd++){
        
        gpu_ncols = numcols2p(dd, ncols, nb, num_gpus); 

        J = dd;
        if(gpu_ncols>0){
            magma_setdevice(dd);
            //pthread_mutex_lock(&mutex_compute_stream);
            magmablasSetKernelStream(mstream[dd][1]);
            magmablas_dlaswp(gpu_ncols, dlAT(K, J), dlAT_LD, ONE, nrows, ipiv(K), ONE);
            //pthread_mutex_lock(&mutex_compute_stream);
        }

    }       
        

        

#if (dbglevel >=1)
ca_trace_end_1gpu('W');
#endif
     } 
     
#if (dbglevel==10) 
    ca_dbg_printMat_transpose_mgpu(num_gpus, n_local, m, dlAT, dlAT_LD,"dAT after lswap"); 
#endif

/*Shutdown the scheduler*/
     magma_schedule_delete();

/*update permutation vector indexes*/ 
     for(K=1;K<=N-1;K++){ 
 
        nrows = min(nb, n-K*nb); 
        for(J=0;J<=nrows-1;J++){ 
            ipiv[K*nb+J] += K*nb; 
        } 
     } 

#if dbglevel>=1
    printf("[DBG] Time Factorization:%f\n",magma_wtime()-t1); 
    t1 = magma_wtime();
#endif

    /* 4. Transpose back the matrix in/out of place*/
    for(dd=0;dd<num_gpus;dd++){

        //n_local[dd] = numcols2p(dd, n, nb, num_gpus); //loc2p(dd, N, num_gpus)*nb;

        magma_setdevice(dd);
        magmablasSetKernelStream(mstream[dd][1]);
        magmablas_dtranspose2(dlA[dd], dlA_LD, dlAT[dd], dlAT_LD, n_local[dd], m);
    }


    for(dd=0;dd<num_gpus;dd++){ //needed
        magma_setdevice(dd);
        magmablasSetKernelStream(NULL);
    }

#if dbglevel>=1
    printf("[DBG] Time Final in/out of place transpose:%f\n",magma_wtime()-t1); 
    t1 = magma_wtime();

#endif



#if (dbglevel==10)     
    ca_dbg_printMat_mgpu(num_gpus, m, n_local, dlA, dlA_LD,"dA = LU"); 
#endif 

    for(dd=0;dd<num_gpus;dd++){
        magma_setdevice(dd);
       magma_queue_destroy(mstream[dd][0]);
       magma_queue_destroy(mstream[dd][1]);
       magma_queue_destroy(mstream[dd][2]);
    }

    //free(mstream);

    // printf("Step 4: time:%f\n",magma_wtime()-t1); 
// t1 = magma_wtime();
    free(n_local);
    free(nr_local);
//    free(k_local);
    for(dd=0;dd<num_gpus;dd++){
        magma_setdevice(dd);
        magma_free( dlAP_set[dd]); 
        magma_free( dlAP_get[dd]);
        magma_free(dlpanel[dd]);
        magma_free(dlAT[dd]);
    }

    //free(dlAP_set); 
    //free(dlAP_get);
    //free(dlpanel);
    free(dlAT);

#if dbglevel>=1
    printf("[DBG] Time memory free (dAP):%f\n",magma_wtime()-t1); 
    t1 = magma_wtime();
#endif

#if dbglevel>=1
    /*Finalize the tracing*/
    ca_dbg_trace_finalize();
    printf("[DBG] Time llog:%f\n",magma_wtime()-t1); 
#endif

    return *info; 
}   /* End of MAGMA_DGETRF_REC_ASYNC_WORK_GPU */

