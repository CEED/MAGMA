/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
*/

#include "common_magma.h"

#if defined( _WIN32 ) || defined( _WIN64 )

#  include <time.h>
#  include <sys/timeb.h>
#  if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#    define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#  else
#    define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#  endif

#else

#  include <sys/time.h>

#endif

#if defined(ADD_)
#    define magma_gettime_f        magma_gettime_f_
#    define magma_gettimervalue_f  magma_gettimervalue_f_
#elif defined(NOCHANGE)
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- Get current time
*/ 
#if defined( _WIN32 ) || defined( _WIN64 )
struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};

extern "C"
int gettimeofday(struct timeval* tv, struct timezone* tz)
{
    FILETIME         ft;
    unsigned __int64 tmpres = 0;
    static int       tzflag;

    if (NULL != tv)
        {
            GetSystemTimeAsFileTime(&ft);
            tmpres |=  ft.dwHighDateTime;
            tmpres <<= 32;
            tmpres |=  ft.dwLowDateTime;

            /*converting file time to unix epoch*/
            tmpres /= 10;  /*convert into microseconds*/
            tmpres -= DELTA_EPOCH_IN_MICROSECS;

            tv->tv_sec  = (long)(tmpres / 1000000UL);
            tv->tv_usec = (long)(tmpres % 1000000UL);
        }
    if (NULL != tz)
        {
            if (!tzflag)
                {
                    _tzset();
                    tzflag++;
                }
            tz->tz_minuteswest = _timezone / 60;
            tz->tz_dsttime     = _daylight;
        }
    return 0;
}
#endif

extern "C"
magma_timestr_t get_current_time(void)
{
  static struct timeval  time_val;

  magma_timestr_t time;

  cudaDeviceSynchronize();
  gettimeofday(&time_val, NULL);

  time.sec  = time_val.tv_sec;
  time.usec = time_val.tv_usec;
  return (time);
}

extern "C"
void magma_gettime_f(unsigned int *time) {
  magma_timestr_t tmp = get_current_time();
  time[0] = tmp.sec;
  time[1] = tmp.usec;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- End elapsed time
*/ 
extern "C"
double GetTimerValue(magma_timestr_t time_1, magma_timestr_t time_2)
{
  int sec, usec;

  sec  = time_2.sec  - time_1.sec;
  usec = time_2.usec - time_1.usec;

  return (1000.*(double)(sec) + (double)(usec) * 0.001);
}

extern "C"
void magma_gettimervalue_f(unsigned int *start, unsigned int *end, double *result) {
  magma_timestr_t time1, time2;
  time1.sec  = start[0];
  time1.usec = start[1];
  time2.sec  = end[0];
  time2.usec = end[1];
  *result = GetTimerValue(time1, time2);
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Print the available GPU devices
*/
extern "C"
void printout_devices( )
{
  int ndevices;
  cuDeviceGetCount( &ndevices );
  for( int idevice = 0; idevice < ndevices; idevice++ )
    {
      char name[200];
#if CUDA_VERSION > 3010 
      size_t totalMem;
#else
      unsigned int totalMem;
#endif

      int clock;
      int major, minor;
      CUdevice dev;

      cuDeviceGet( &dev, idevice );
      cuDeviceGetName( name, sizeof(name), dev );
      cuDeviceComputeCapability( &major, &minor, dev );
      cuDeviceTotalMem( &totalMem, dev );
      cuDeviceGetAttribute( &clock,
                            CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev );
      printf( "device %d: %s, %.1f MHz clock, %.1f MB memory, capability %d.%d\n",
              idevice, name, clock/1000.f, totalMem/1024.f/1024.f, major, minor );
    }
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function: ipiv(i) indicates that row i has been swapped with 
      ipiv(i) from top to bottom. This function rearranges ipiv into newipiv
      where row i has to be moved to newipiv(i). The new pivoting allows for
      parallel processing vs the original one assumes a specific ordering and
      has to be done sequentially.
*/
extern "C"
void swp2pswp(char trans, int n, int *ipiv, int *newipiv){
  int i, newind, ind;
  char            trans_[2] = {trans, 0};
  long int    notran = lapackf77_lsame(trans_, "N");

  for(i=0; i<n; i++)
    newipiv[i] = -1;
  
  if (notran){
    for(i=0; i<n; i++){
      newind = ipiv[i] - 1;
      if (newipiv[newind] == -1) {
        if (newipiv[i]==-1){
          newipiv[i] = newind;
          if (newind>i)
            newipiv[newind]= i;
        }
        else
          {
            ind = newipiv[i];
            newipiv[i] = newind;
            if (newind>i)
              newipiv[newind]= ind;
          }
      }
      else {
        if (newipiv[i]==-1){
          if (newind>i){
            ind = newipiv[newind];
            newipiv[newind] = i;
            newipiv[i] = ind;
          }
          else
            newipiv[i] = newipiv[newind];
        }
        else{
          ind = newipiv[i];
          newipiv[i] = newipiv[newind];
          if (newind > i)
            newipiv[newind] = ind;
        }
      }
    }
  } else {
    for(i=n-1; i>=0; i--){
      newind = ipiv[i] - 1;
      if (newipiv[newind] == -1) {
        if (newipiv[i]==-1){
          newipiv[i] = newind;
          if (newind>i)
            newipiv[newind]= i;
        }
        else
          {
            ind = newipiv[i];
            newipiv[i] = newind;
            if (newind>i)
              newipiv[newind]= ind;
          }
      }
      else {
        if (newipiv[i]==-1){
          if (newind>i){
            ind = newipiv[newind];
            newipiv[newind] = i;
            newipiv[i] = ind;
          }
          else
            newipiv[i] = newipiv[newind];
        }
        else{
          ind = newipiv[i];
          newipiv[i] = newipiv[newind];
          if (newind > i)
            newipiv[newind] = ind;
        }
      }
    }
  }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function: used for debugging. Given a pointer to floating
      point number on the GPU memory, the function returns the value
      at that location.
*/
extern "C"
float getv(float *da){
  float res[1];
  cublasGetVector(1, sizeof(float), da, 1, res, 1);
  return res[0];
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Auxiliary function sp_cat
*/
extern "C"
int sp_cat(char *lp, char *rpp[], magma_int_t *rnp, magma_int_t*np, magma_int_t ll)
{
  magma_int_t i, n, nc;
  char *f__rp;

  n = (int)*np;
  for(i = 0 ; i < n ; ++i)
    {
      nc = ll;
      if(rnp[i] < nc)
        nc = rnp[i];
      ll -= nc;
      f__rp = rpp[i];
      while(--nc >= 0)
        *lp++ = *f__rp++;
    }
  while(--ll >= 0)
    *lp++ = ' ';

  return 0;
}
