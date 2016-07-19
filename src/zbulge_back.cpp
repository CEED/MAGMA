/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Stan Tomov
       @author Raffaele Solca
       
       @precisions normal z -> s d c

 */
#include "magma_internal.h"
#include "magma_bulge.h"
#include "magma_zbulge.h"

#define COMPLEX

static void *magma_zapplyQ_parallel_section(void *arg);

static void magma_ztile_bulge_applyQ(magma_int_t core_id, magma_side_t side, magma_int_t n_loc, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
                                     magmaDoubleComplex *E, magma_int_t lde, magmaDoubleComplex *V, magma_int_t ldv,
                                     magmaDoubleComplex *TAU, magmaDoubleComplex *T, magma_int_t ldt);

//////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct magma_zapplyQ_data_s {
    magma_int_t threads_num;
    magma_int_t n;
    magma_int_t ne;
    magma_int_t n_gpu;
    magma_int_t nb;
    magma_int_t Vblksiz;
    magmaDoubleComplex* E;
    magma_int_t lde;
    magmaDoubleComplex* V;
    magma_int_t ldv;
    magmaDoubleComplex* TAU;
    magmaDoubleComplex* T;
    magma_int_t ldt;
    magmaDoubleComplex* dE;
    magma_int_t ldde;
    pthread_barrier_t barrier;
} magma_zapplyQ_data;

void magma_zapplyQ_data_init(
    magma_zapplyQ_data *zapplyQ_data, magma_int_t threads_num, magma_int_t n, magma_int_t ne, magma_int_t n_gpu,
    magma_int_t nb, magma_int_t Vblksiz, magmaDoubleComplex *E, magma_int_t lde,
    magmaDoubleComplex *V, magma_int_t ldv, magmaDoubleComplex *TAU,
    magmaDoubleComplex *T, magma_int_t ldt, magmaDoubleComplex *dE, magma_int_t ldde)
{
    zapplyQ_data->threads_num = threads_num;
    zapplyQ_data->n = n;
    zapplyQ_data->ne = ne;
    zapplyQ_data->n_gpu = n_gpu;
    zapplyQ_data->nb = nb;
    zapplyQ_data->Vblksiz = Vblksiz;
    zapplyQ_data->E = E;
    zapplyQ_data->lde = lde;
    zapplyQ_data->V = V;
    zapplyQ_data->ldv = ldv;
    zapplyQ_data->TAU = TAU;
    zapplyQ_data->T = T;
    zapplyQ_data->ldt = ldt;
    zapplyQ_data->dE = dE;
    zapplyQ_data->ldde = ldde;

    magma_int_t count = zapplyQ_data->threads_num;

    if (zapplyQ_data->threads_num > 1)
        --count;

    pthread_barrier_init(&(zapplyQ_data->barrier), NULL, (unsigned)count);
}

void magma_zapplyQ_data_destroy(
    magma_zapplyQ_data *zapplyQ_data)
{
    pthread_barrier_destroy(&(zapplyQ_data->barrier));
}

typedef struct magma_zapplyQ_id_data {
    magma_int_t id;
    magma_zapplyQ_data* data;
} magma_zapplyQ_id_data;

void magma_zapplyQ_id_data_init(
    magma_zapplyQ_id_data *zapplyQ_id_data,
    magma_int_t id, magma_zapplyQ_data* data)
{
    zapplyQ_id_data->id = id;
    zapplyQ_id_data->data = data;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" magma_int_t
magma_zbulge_back(
    magma_uplo_t uplo,
    magma_int_t n, magma_int_t nb,
    magma_int_t ne, magma_int_t Vblksiz,
    magmaDoubleComplex *Z, magma_int_t ldz,
    magmaDoubleComplex_ptr dZ, magma_int_t lddz,
    magmaDoubleComplex *V, magma_int_t ldv,
    magmaDoubleComplex *TAU,
    magmaDoubleComplex *T, magma_int_t ldt,
    magma_int_t* info)
{
    magma_int_t threads = magma_get_parallel_numthreads();
    magma_int_t mklth   = magma_get_lapack_numthreads();
    magma_set_lapack_numthreads(1);

    real_Double_t timeaplQ2=0.0;
    double f= 1.;
    magma_int_t n_gpu = ne;
    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

//#ifdef REAL
    //double gpu_cpu_perf = 50;  // gpu over cpu performance  //100% ev // SandyB. - Kepler (K20c)
    //double gpu_cpu_perf = 16;  // gpu over cpu performance  //100% ev // SandyB. - Fermi (M2090)
//#else
//    double gpu_cpu_perf = 27.5;  // gpu over cpu performance  //100% ev // Westmere - Fermi (M2090)
    //double gpu_cpu_perf = 37;  // gpu over cpu performance  //100% ev // SandyB. - Kepler (K20c)
//    double gpu_cpu_perf = 130;  // gpu over cpu performance  //100% ev // Bulldozer - Kepler (K20X)
//#endif

    magma_int_t gpu_cpu_perf = magma_get_zbulge_gcperf();
    if (threads > 1) {
        f = 1. / (1. + (double)(threads-1)/ ((double)gpu_cpu_perf)    );
        n_gpu = (magma_int_t)(f*ne);
    }

    /****************************************************
     *  apply V2 from left to the eigenvectors Z. dZ = (I-V2*T2*V2')*Z
     * **************************************************/
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
n_gpu=ne;
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    timeaplQ2 = magma_wtime();
    /*============================
     *  use GPU+CPU's
     *==========================*/

    if (n_gpu < ne) {
        // define the size of Q to be done on CPU's and the size on GPU's
        // note that GPU use Q(1:N_GPU) and CPU use Q(N_GPU+1:N)
        #ifdef ENABLE_DEBUG
        printf("---> calling GPU + CPU(if N_CPU > 0) to apply V2 to Z with NE %ld     N_GPU %ld   N_CPU %ld\n",ne, n_gpu, ne-n_gpu);
        #endif
        magma_zapplyQ_data data_applyQ;
        magma_zapplyQ_data_init(&data_applyQ, threads, n, ne, n_gpu, nb, Vblksiz, Z, ldz, V, ldv, TAU, T, ldt, dZ, lddz);

        magma_zapplyQ_id_data* arg;
        magma_malloc_cpu((void**) &arg, threads*sizeof(magma_zapplyQ_id_data));

        pthread_t* thread_id;
        magma_malloc_cpu((void**) &thread_id, threads*sizeof(pthread_t));

        pthread_attr_t thread_attr;

        // ===============================
        // relaunch thread to apply Q
        // ===============================
        // Set one thread per core
        pthread_attr_init(&thread_attr);
        pthread_attr_setscope(&thread_attr, PTHREAD_SCOPE_SYSTEM);
        pthread_setconcurrency( (unsigned)threads );

        // Launch threads
        for (magma_int_t thread = 1; thread < threads; thread++) {
            magma_zapplyQ_id_data_init(&(arg[thread]), thread, &data_applyQ);
            pthread_create(&thread_id[thread], &thread_attr, magma_zapplyQ_parallel_section, &arg[thread]);
        }
        magma_zapplyQ_id_data_init(&(arg[0]), 0, &data_applyQ);
        magma_zapplyQ_parallel_section(&arg[0]);

        // Wait for completion
        for (magma_int_t thread = 1; thread < threads; thread++) {
            void *exitcodep;
            pthread_join(thread_id[thread], &exitcodep);
        }

        magma_free_cpu(thread_id);
        magma_free_cpu(arg);
        magma_zapplyQ_data_destroy(&data_applyQ);


        magma_zsetmatrix( n, ne-n_gpu, Z + n_gpu*ldz, ldz, dZ + n_gpu*ldz, lddz, queue );

        /*============================
         *  use only GPU
         *==========================*/
    } else {
        magma_zsetmatrix( n, ne, Z, ldz, dZ, lddz, queue );
        magma_zbulge_applyQ_v2(MagmaLeft, ne, n, nb, Vblksiz, dZ, lddz, V, ldv, T, ldt, info);
    }

    timeaplQ2 = magma_wtime()-timeaplQ2;

    magma_queue_destroy( queue );
    magma_set_lapack_numthreads(mklth);
    return MAGMA_SUCCESS;
}

//##################################################################################################
static void *magma_zapplyQ_parallel_section(void *arg)
{
    magma_int_t my_core_id   = ((magma_zapplyQ_id_data*)arg) -> id;
    magma_zapplyQ_data* data = ((magma_zapplyQ_id_data*)arg) -> data;

    magma_int_t allcores_num   = data -> threads_num;
    magma_int_t n              = data -> n;
    magma_int_t ne             = data -> ne;
    magma_int_t n_gpu          = data -> n_gpu;
    magma_int_t nb             = data -> nb;
    magma_int_t Vblksiz        = data -> Vblksiz;
    magmaDoubleComplex *E         = data -> E;
    magma_int_t lde            = data -> lde;
    magmaDoubleComplex *V         = data -> V;
    magma_int_t ldv            = data -> ldv;
    magmaDoubleComplex *TAU       = data -> TAU;
    magmaDoubleComplex *T         = data -> T;
    magma_int_t ldt            = data -> ldt;
    magmaDoubleComplex *dE        = data -> dE;
    magma_int_t ldde           = data -> ldde;
    pthread_barrier_t* barrier = &(data -> barrier);

    magma_int_t info;

    #ifdef ENABLE_TIMER
    real_Double_t timeQcpu=0.0, timeQgpu=0.0;
    #endif

    magma_int_t n_cpu = ne - n_gpu;

    // with MKL and when using omp_set_num_threads instead of mkl_set_num_threads
    // it need that all threads setting it to 1.
    magma_set_lapack_numthreads(1);

#ifndef MAGMA_NOAFFINITY
    //#define PRINTAFFINITY
#ifdef PRINTAFFINITY
    affinity_set print_set;
    print_set.print_affinity(my_core_id, "starting affinity");
#endif
    cpu_set_t old_set, new_set;

    //store current affinity
    CPU_ZERO(&old_set);
    sched_getaffinity( 0, sizeof(old_set), &old_set);
    //set new affinity
    // bind threads
    CPU_ZERO(&new_set);
    CPU_SET(my_core_id, &new_set);
    sched_setaffinity( 0, sizeof(new_set), &new_set);
#ifdef PRINTAFFINITY
    print_set.print_affinity(my_core_id, "set affinity");
#endif
#endif

    if (my_core_id == 0) {
        //=============================================
        //   on GPU on thread 0:
        //    - apply V2*Z(:,1:N_GPU)
        //=============================================
        #ifdef ENABLE_TIMER
        timeQgpu = magma_wtime();
        #endif
        magma_queue_t queue;
        magma_device_t cdev;
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queue );

        magma_zsetmatrix( n, n_gpu, E, lde, dE, ldde, queue );
        magma_zbulge_applyQ_v2(MagmaLeft, n_gpu, n, nb, Vblksiz, dE, ldde, V, ldv, T, ldt, &info);

        magma_queue_destroy( queue );
        
        #ifdef ENABLE_TIMER
        timeQgpu = magma_wtime()-timeQgpu;
        printf("  Finish Q2_GPU GGG timing= %f\n", timeQgpu);
        #endif
    } else {
        //=============================================
        //   on CPU on threads 1:allcores_num-1:
        //    - apply V2*Z(:,N_GPU+1:NE)
        //=============================================
        #ifdef ENABLE_TIMER
        if (my_core_id == 1)
            timeQcpu = magma_wtime();
        #endif

        magma_int_t n_loc = magma_ceildiv(n_cpu, allcores_num-1);
        magmaDoubleComplex* E_loc = E + (n_gpu+ n_loc * (my_core_id-1))*lde;
        n_loc = min(n_loc,n_cpu - n_loc * (my_core_id-1));

        magma_ztile_bulge_applyQ(my_core_id, MagmaLeft, n_loc, n, nb, Vblksiz, E_loc, lde, V, ldv, TAU, T, ldt);
        pthread_barrier_wait(barrier);

        #ifdef ENABLE_TIMER
        if (my_core_id == 1) {
            timeQcpu = magma_wtime()-timeQcpu;
            printf("  Finish Q2_CPU CCC timing= %f\n", timeQcpu);
        }
        #endif
    } // END if my_core_id

#ifndef MAGMA_NOAFFINITY
    //restore old affinity
    sched_setaffinity(0, sizeof(old_set), &old_set);
#ifdef PRINTAFFINITY
    print_set.print_affinity(my_core_id, "restored_affinity");
#endif
#endif

    return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#define E(m,n)   &(E[(m) + lde*(n)])
#define V(m)     &(V[(m)])
#define TAU(m)   &(TAU[(m)])
#define T(m)     &(T[(m)])
static void magma_ztile_bulge_applyQ(
    magma_int_t core_id, magma_side_t side, magma_int_t n_loc, magma_int_t n, magma_int_t nb, magma_int_t Vblksiz,
    magmaDoubleComplex *E, magma_int_t lde,
    magmaDoubleComplex *V, magma_int_t ldv,
    magmaDoubleComplex *TAU, magmaDoubleComplex *T, magma_int_t ldt)
    //, magma_int_t* info)
{
    //%===========================
    //%   local variables
    //%===========================
    magma_int_t firstcolj;
    magma_int_t bg, rownbm;
    magma_int_t st,ed,fst,vlen,vnb,colj;
    magma_int_t vpos,tpos;
    magma_int_t cur_blksiz,avai_blksiz, ncolinvolvd;
    magma_int_t nbgr, colst, coled;

    if (n <= 0)
        return;
    if (n_loc <= 0)
        return;

    //info = 0;
    magma_int_t INFO=0;

    magma_int_t nbGblk  = magma_ceildiv(n-1, Vblksiz);

    /*
     * version v1: for each chunck it apply all the V's then move to
     *                    the other chunck. the locality here inside each
     *                    chunck meaning that thread t apply V_k then move
     *                    to V_k+1 which overlap with V_k meaning that the
     *                    E_k+1 overlap with E_k. so here is the
     *                    locality however thread t had to read V_k+1 and
     *                    T_k+1 at each apply. note that all thread if they
     *                    run at same speed they might reading the same V_k
     *                    and T_k at the same time.
     * */

    magma_int_t nb_loc = 128; //$$$$$$$$

    magma_int_t     lwork = 2*nb_loc*max(Vblksiz,64);
    magmaDoubleComplex *work, *work2;

    magma_zmalloc_cpu(&work, lwork);
    magma_zmalloc_cpu(&work2, lwork);

    magma_int_t nbchunk =  magma_ceildiv(n_loc, nb_loc);

    /* SIDE LEFT  meaning apply E = Q*E = (q_1*q_2*.....*q_n) * E ==> so traverse Vs in reverse order (forward) from q_n to q_1
     *            each q_i consist of applying V to a block of row E(row_i,:) and applies are overlapped meaning
     *            that q_i+1 overlap a portion of the E(row_i, :).
     *            IN parallel E is splitten in vertical block over the threads  */
    /* SIDE RIGHT meaning apply E = E*Q = E * (q_1*q_2*.....*q_n) ==> so tarverse Vs in normal  order (forward) from q_1 to q_n
     *            each q_i consist of applying V to a block of col E(:, col_i,:) and the applies are overlapped meaning
     *            that q_i+1 overlap a portion of the E(:, col_i).
     *            IN parallel E is splitten in horizontal block over the threads  */
    #ifdef ENABLE_DEBUG
    if ((core_id == 0) || (core_id == 1))
        printf("  APPLY Q2_cpu zbulge_back   N %ld  N_loc %ld  nbchunk %ld  NB %ld  Vblksiz %ld  SIDE %c\n", n, n_loc, nbchunk, nb, Vblksiz, side);
    #endif
    for (magma_int_t i = 0; i < nbchunk; i++) {
        magma_int_t ib_loc = min(nb_loc, (n_loc - i*nb_loc));

        if (side == MagmaLeft) {
            for (bg = nbGblk; bg > 0; bg--) {
                firstcolj = (bg-1)*Vblksiz + 1;
                rownbm    = magma_ceildiv((n-(firstcolj+1)),nb);
                if (bg == nbGblk) rownbm = magma_ceildiv((n-(firstcolj)),nb);  // last blk has size=1 used for complex to handle A(N,N-1)
                for (magma_int_t j = rownbm; j > 0; j--) {
                    vlen = 0;
                    vnb  = 0;
                    colj = (bg-1)*Vblksiz; // for k=0; I compute the fst and then can remove it from the loop
                    fst  = (rownbm -j)*nb+colj +1;
                    for (magma_int_t k=0; k < Vblksiz; k++) {
                        colj = (bg-1)*Vblksiz + k;
                        st   = (rownbm -j)*nb+colj +1;
                        ed   = min(st+nb-1,n-1);
                        if (st > ed)
                            break;
                        if ((st == ed) && (colj != n-2))
                            break;
                        vlen = ed-fst+1;
                        vnb = k+1;
                    }
                    colst     = (bg-1)*Vblksiz;
                    magma_bulge_findVTpos(n, nb, Vblksiz, colst, fst, ldv, ldt, &vpos, &tpos);

                    if ((vlen > 0) && (vnb > 0)) {
                        lapackf77_zlarfb( "L", "N", "F", "C", &vlen, &ib_loc, &vnb, V(vpos), &ldv, T(tpos), &ldt, E(fst,i*nb_loc), &lde, work, &ib_loc);
                    }
                    if (INFO != 0)
                        printf("ERROR ZUNMQR INFO %ld\n", long(INFO) );
                }
            }
        } else if (side == MagmaRight) {
            rownbm    = magma_ceildiv((n-1),nb);
            for (magma_int_t k = 1; k <= rownbm; k++) {
                ncolinvolvd = min(n-1, k*nb);
                avai_blksiz = min(Vblksiz,ncolinvolvd);
                nbgr = magma_ceildiv(ncolinvolvd,avai_blksiz);
                for (magma_int_t j = 1; j <= nbgr; j++) {
                    vlen = 0;
                    vnb  = 0;
                    cur_blksiz = min(ncolinvolvd-(j-1)*avai_blksiz, avai_blksiz);
                    colst = (j-1)*avai_blksiz;
                    coled = colst + cur_blksiz -1;
                    fst   = (rownbm -k)*nb+colst +1;
                    for (colj=colst; colj <= coled; colj++) {
                        st = (rownbm -k)*nb+colj +1;
                        ed = min(st+nb-1,n-1);
                        if (st > ed)
                            break;
                        if ((st == ed) && (colj != n-2))
                            break;
                        vlen = ed-fst+1;
                        vnb = vnb+1;
                    }
                    magma_bulge_findVTpos(n, nb, Vblksiz, colst, fst, ldv, ldt, &vpos, &tpos);
                    if ((vlen > 0) && (vnb > 0)) {
                        lapackf77_zlarfb( "R", "N", "F", "C", &ib_loc, &vlen, &vnb, V(vpos), &ldv, T(tpos), &ldt, E(i*nb_loc,fst), &lde, work, &ib_loc);
                    }
                }
            }
        } else {
            printf("ERROR SIDE %d\n", side);
        }
    } // END loop over the chunks

    magma_free_cpu(work);
    magma_free_cpu(work2);
}
#undef E
#undef V
#undef TAU
#undef T
////////////////////////////////////////////////////////////////////////////////////////////////////
