/*
 *   -- MAGMA (version 1.1) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      November 2011
 *
 * @precisions normal z -> s d c
 */

#ifndef _MAGMA_ZBULGEINC_H_
#define _MAGMA_ZBULGEINC_H_

#define PRECISION_z
#ifdef __cplusplus
extern "C" {
#endif


/***************************************************************************//**
 *  Configuration
 **/
#include <sys/time.h>

 // maximum contexts
#define MAX_THREADS_BLG         256

 real_Double_t get_time_azz(void);
 void findVTpos(int N, int NB, int Vblksiz, int sweep, int st, int *Vpos, int *TAUpos, int *Tpos, int *myblkid);
 void findVTsiz(int N, int NB, int Vblksiz, int *blkcnt, int *LDV);
  magma_int_t plasma_ceildiv(magma_int_t a, magma_int_t b);


/*
extern volatile magma_int_t barrier_in[MAX_THREADS_BLG];
extern volatile magma_int_t barrier_out[MAX_THREADS_BLG];
extern volatile magma_int_t *ss_prog;
*/

 /***************************************************************************//**
 *  Static scheduler
 **/
/*
#define ssched_init(nbtiles) \
{ \
        volatile int   prog_ol[2*nbtiles+10];\
                 int   iamdone[MAX_THREADS_BLG]; \
                 int   thread_num[MAX_THREADS_BLG];\
        pthread_t      thread_id[MAX_THREADS_BLG];\
        pthread_attr_t thread_attr;\
}
*/
////////////////////////////////////////////////////////////////////////////////////////////////////



 struct gbstrct_blg {
    int cores_num;
    cuDoubleComplex *T;
    cuDoubleComplex *A;
    cuDoubleComplex *V;
    cuDoubleComplex *TAU;
    cuDoubleComplex *E;
    int NB;
    int NBTILES;
    int N;
    int LDA;
    int LDE;
    int BAND;
    int grsiz;
    int Vblksiz;
    int WANTZ;
    char SIDE;
    real_Double_t *timeblg;
    real_Double_t *timeaplQ;
    volatile int *ss_prog;
} ;
struct gbstrct_blg core_in_all;





////////////////////////////////////////////////////////////////////////////////////////////////////

 real_Double_t get_time_azz(void)
{
    struct timeval  time_val;
    struct timezone time_zone;

    gettimeofday(&time_val, &time_zone);

    return (real_Double_t)(time_val.tv_sec) + (real_Double_t)(time_val.tv_usec) / 1000000.0;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
#define MAX_EVENTSBLG 163840
//#define MAX_EVENTSBLG 1048576

int    event_numblg        [MAX_THREADS_BLG]               __attribute__ ((aligned (128)));
real_Double_t event_start_timeblg [MAX_THREADS_BLG]               __attribute__ ((aligned (128)));
real_Double_t event_end_timeblg   [MAX_THREADS_BLG]               __attribute__ ((aligned (128)));
real_Double_t event_logblg        [MAX_THREADS_BLG][MAX_EVENTSBLG]   __attribute__ ((aligned (128)));
int log_eventsblg = 0;

#define core_event_startblg(my_core_id)\
    event_start_timeblg[my_core_id] = get_time_azz();\

#define core_event_endblg(my_core_id)\
    event_end_timeblg[my_core_id] = get_time_azz();\

#define core_log_eventblg(event, my_core_id)\
    event_logblg[my_core_id][event_numblg[my_core_id]+0] = my_core_id;\
    event_logblg[my_core_id][event_numblg[my_core_id]+1] = event_start_timeblg[my_core_id];\
    event_logblg[my_core_id][event_numblg[my_core_id]+2] = event_end_timeblg[my_core_id];\
    event_logblg[my_core_id][event_numblg[my_core_id]+3] = (event);\
    event_numblg[my_core_id] += (log_eventsblg << 2);\
    event_numblg[my_core_id] &= (MAX_EVENTSBLG-1);

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
}
#endif

#undef PRECISION_z
#endif




