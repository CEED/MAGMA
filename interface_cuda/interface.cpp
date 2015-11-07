/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
 
       @author Mark Gates
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#if __cplusplus >= 201103  // C++11 standard
#include <atomic>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(MAGMA_WITH_MKL)
#include <mkl_service.h>
#endif

#if defined(MAGMA_WITH_ACML)
#include <acml.h>
#endif

#include <cuda_runtime.h>

// defining MAGMA_LAPACK_H is a hack to NOT include magma_lapack.h
// via common_magma.h here, since it conflicts with acml.h and we don't
// need lapack here, but we want acml.h for the acmlversion() function.
#define MAGMA_LAPACK_H
#include "common_magma.h"
#include "error.h"

#ifdef HAVE_CUBLAS

#define HAVE_PTHREAD_KEY


// ------------------------------------------------------------
// stubs if C++11 atomics are not available.
// these are NOT atomic!
#if ! (__cplusplus >= 201103)  // not C++11 standard
bool atomic_flag_test_and_set( bool* flag )
{
    bool x = flag;
    // note race condition here, if someone else modifies flag!
    *flag = true;
    return x;
}

void atomic_flag_clear( bool* flag )
{
    *flag = false;
}
#endif  // not C++11


// ------------------------------------------------------------
// constants

// bit flags
enum {
    own_none     = 0x0000,
    own_stream   = 0x0001,
    own_cublas   = 0x0002,
    own_cusparse = 0x0004,
    own_opencl   = 0x0008
};


// ------------------------------------------------------------
// globals
#if __cplusplus >= 201103  // C++11 standard
    std::atomic_flag g_init = ATOMIC_FLAG_INIT;
#else
    bool g_init = false;
#endif

magma_queue_t g_null_queues[ MagmaMaxAccelerators ] = { NULL };

#ifdef HAVE_PTHREAD_KEY
    pthread_key_t g_magma_queue_key;
#else
    magma_queue_t g_magma_queue = NULL;
#endif


// --------------------
// subset of the CUDA device properties, set by magma_init()
struct magma_device
{
    size_t memory;
    magma_int_t cuda_arch;
};

int g_magma_devices_cnt = 0;
struct magma_device* g_magma_devices = NULL;


// ========================================
// initialization
// --------------------
/**
    Caches information about available CUDA devices.
    When renumbering devices after calling magma_init,
    call magma_finalize, then cudaSetValidDevices, then magma_init again.
    Ideally magma_init is paired with magma_finalize, but this implementation
    ensures there isn't a memory leak if magma_init is called multiple times
    without calling magma_finalize.
    
    @see magma_finalize
    
    @ingroup magma_init
*/
extern "C"
magma_int_t magma_init()
{
    magma_int_t info = MAGMA_ERR_REINITIALIZED;
    if ( ! atomic_flag_test_and_set( &g_init )) {
        info = 0;
        
        // query devices
        cudaError_t err;
        err = cudaGetDeviceCount( &g_magma_devices_cnt );
        check_error( err );
        magma_malloc_cpu( (void**) &g_magma_devices, g_magma_devices_cnt * sizeof(struct magma_device) );
        if ( g_magma_devices == NULL ) {
            info = MAGMA_ERR_HOST_ALLOC;
            goto cleanup;
        }
        for( int i = 0; i < g_magma_devices_cnt; ++i ) {
            cudaDeviceProp prop;
            err = cudaGetDeviceProperties( &prop, i );
            check_error( err );
            g_magma_devices[i].memory = prop.totalGlobalMem;
            g_magma_devices[i].cuda_arch  = prop.major*100 + prop.minor*10;
        }
        
        // build NULL queues (for backwards compatability with MAGMA 1.x) on each device
        magma_int_t cdev;
        magma_getdevice( &cdev );
        
        for( int dev=0; dev < g_magma_devices_cnt; ++dev ) {
            if ( g_null_queues[dev] == NULL ) {
                // TODO
                //info = 
                magma_queue_create_from_cuda( dev, NULL, NULL, NULL, &g_null_queues[dev] );
                //if ( info != 0 ) {
                //    goto cleanup;
                //}
            }
        }
        
        // set default queue on current device (for backwards compatability with MAGMA 1.x)
        #ifdef HAVE_PTHREAD_KEY
            info = pthread_key_create( &g_magma_queue_key, NULL );
            if ( info != 0 ) {
                info = MAGMA_ERR_NOT_INITIALIZED;
            }
        #endif
        
        magma_setdevice( cdev );
        magmablasSetKernelStream( g_null_queues[cdev] );
    }
    
cleanup:
    return info;
}

// --------------------
/**
    Frees information about CUDA devices.
    @ingroup magma_init
*/
extern "C"
magma_int_t magma_finalize()
{
    magma_int_t info = MAGMA_ERR_NOT_INITIALIZED;
    if ( atomic_flag_test_and_set( &g_init )) {
        info = 0;
        
        magma_free_cpu( g_magma_devices );
        g_magma_devices = NULL;
        
        for( int dev=0; dev < g_magma_devices_cnt; ++dev ) {
            magma_queue_destroy( g_null_queues[dev] );
            g_null_queues[dev] = NULL;
        }
        #ifdef HAVE_PTHREAD_KEY
            pthread_key_delete( g_magma_queue_key );
        #endif
    }
    atomic_flag_clear( &g_init );
    return info;
}

// --------------------
/**
    Print the available GPU devices. Used in testing.
    @ingroup magma_init
*/
extern "C"
void magma_print_environment()
{
    magma_int_t major, minor, micro;
    magma_version( &major, &minor, &micro );
    printf( "%% MAGMA %d.%d.%d %s compiled for CUDA capability >= %.1f, %d-bit magma_int_t, %d-bit pointer.\n",
            (int) major, (int) minor, (int) micro, MAGMA_VERSION_STAGE, MIN_CUDA_ARCH/100.,
            (int) (8*sizeof(magma_int_t)), (int) (8*sizeof(void*)) );
    
    int cuda_runtime, cuda_driver;
    cudaError_t err;
    err = cudaDriverGetVersion( &cuda_driver );
    check_error( err );
    err = cudaRuntimeGetVersion( &cuda_runtime );
    check_error( err );
    printf( "%% CUDA runtime %d, driver %d. ", cuda_runtime, cuda_driver );
    
#if defined(_OPENMP)
    int omp_threads = 0;
    #pragma omp parallel
    {
        omp_threads = omp_get_num_threads();
    }
    printf( "OpenMP threads %d. ", omp_threads );
#else
    printf( "MAGMA not compiled with OpenMP. " );
#endif
    
#if defined(MAGMA_WITH_MKL)
    MKLVersion mkl_version;
    mkl_get_version( &mkl_version );
    printf( "MKL %d.%d.%d, MKL threads %d. ",
            mkl_version.MajorVersion,
            mkl_version.MinorVersion,
            mkl_version.UpdateVersion,
            mkl_get_max_threads() );
#endif
    
#if defined(MAGMA_WITH_ACML)
    // ACML 4 doesn't have acml_build parameter
    int acml_major, acml_minor, acml_patch, acml_build;
    acmlversion( &acml_major, &acml_minor, &acml_patch, &acml_build );
    printf( "ACML %d.%d.%d.%d ", acml_major, acml_minor, acml_patch, acml_build );
#endif
    
    printf( "\n" );
    
    int ndevices = 0;
    err = cudaGetDeviceCount( &ndevices );
    check_error( err );
    for( int dev = 0; dev < ndevices; dev++ ) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties( &prop, dev );
        check_error( err );
        printf( "%% device %d: %s, %.1f MHz clock, %.1f MB memory, capability %d.%d\n",
                dev,
                prop.name,
                prop.clockRate / 1000.,
                prop.totalGlobalMem / (1024.*1024.),
                prop.major,
                prop.minor );
        
        int arch = prop.major*100 + prop.minor*10;
        if ( arch < MIN_CUDA_ARCH ) {
            printf("\n"
                   "==============================================================================\n"
                   "WARNING: MAGMA was compiled only for CUDA capability %.1f and higher;\n"
                   "device %d has only capability %.1f; some routines will not run correctly!\n"
                   "==============================================================================\n\n",
                   MIN_CUDA_ARCH/100., dev, arch/100. );
        }
    }
    
    time_t t = time( NULL );
    printf( "%% %s", ctime( &t ));
}


// ========================================
// device support
// ---------------------------------------------
// Returns CUDA architecture capability for the current device.
// This requires magma_init to be called first (to cache the information).
// Version is integer xyz, where x is major, y is minor, and z is micro,
// the same as __CUDA_ARCH__. Thus for architecture 1.3 it returns 130.
extern "C"
magma_int_t magma_getdevice_arch()
{
    int dev;
    cudaError_t err;
    err = cudaGetDevice( &dev );
    check_error( err );
    if ( g_magma_devices == NULL || dev < 0 || dev >= g_magma_devices_cnt ) {
        fprintf( stderr, "Error in %s: MAGMA not initialized (call magma_init() first) or bad device\n", __func__ );
        return 0;
    }
    return g_magma_devices[dev].cuda_arch;
}


// --------------------
extern "C"
void magma_getdevices(
    magma_int_t* devices,
    magma_int_t  size,
    magma_int_t* numPtr )
{
    cudaError_t err;
    int cnt;
    err = cudaGetDeviceCount( &cnt );
    check_error( err );
    
    cnt = min( cnt, size );
    for( int i = 0; i < cnt; ++i ) {
        devices[i] = i;
    }
    *numPtr = cnt;
}

// --------------------
extern "C"
void magma_getdevice( magma_int_t* device )
{
    cudaError_t err;
    err = cudaGetDevice( device );
    check_error( err );
}

// --------------------
extern "C"
void magma_setdevice( magma_int_t device )
{
    cudaError_t err;
    err = cudaSetDevice( device );
    check_error( err );
}

// --------------------
/// This functionality does not exist in OpenCL, so it is deprecated for CUDA, too.
/// @deprecated
extern "C"
void magma_device_sync()
{
    cudaError_t err;
    err = cudaDeviceSynchronize();
    check_error( err );
}


// ========================================
// queue support
// At the moment, MAGMA queue == CUDA stream.
// In the future, MAGMA queue may be CUBLAS handle.
extern "C"
magma_int_t
magma_queue_get_device( magma_queue_t queue )
{
    return queue->device();
}

extern "C"
cudaStream_t
magma_queue_get_cuda_stream( magma_queue_t queue )
{
    return queue->cuda_stream();
}

extern "C"
cublasHandle_t
magma_queue_get_cublas_handle( magma_queue_t queue )
{
    return queue->cublas_handle();
}

extern "C"
cusparseHandle_t
magma_queue_get_cusparse_handle( magma_queue_t queue )
{
    return queue->cusparse_handle();
}

// --------------------
extern "C"
magma_int_t magmablasSetKernelStream( magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( queue == NULL ) {
        magma_int_t dev;
        magma_getdevice( &dev );
        assert( 0 <= dev && dev < g_magma_devices_cnt );
        assert( g_null_queues != NULL );
        assert( g_null_queues[dev] != NULL );
        queue = g_null_queues[dev];
    }
    #ifdef HAVE_PTHREAD_KEY
    info = pthread_setspecific( g_magma_queue_key, queue );
    #else
    g_magma_queue = queue;
    #endif
    return info;
}


// --------------------
extern "C"
magma_int_t magmablasGetKernelStream( magma_queue_t *queue_ptr )
{
    #ifdef HAVE_PTHREAD_KEY
    *queue_ptr = (magma_queue_t) pthread_getspecific( g_magma_queue_key );
    #else
    *queue_ptr = g_magma_queue;
    #endif
    return 0;
}


// --------------------
extern "C"
magma_queue_t magmablasGetQueue()
{
    magma_queue_t queue;
    #ifdef HAVE_PTHREAD_KEY
        queue = (magma_queue_t) pthread_getspecific( g_magma_queue_key );
    #else
        queue = g_magma_queue;
    #endif
    assert( queue != NULL );
    return queue;
}


// --------------------
extern "C"
void magma_queue_create_internal(
    magma_queue_t* queue_ptr,
    const char* func, const char* file, int line )
{
    int device;
    cudaError_t err;
    err = cudaGetDevice( &device );
    check_xerror( err, func, file, line );
    
    magma_queue_create_v2_internal( device, queue_ptr, func, file, line );
}

// --------------------
extern "C"
void magma_queue_create_v2_internal(
    magma_int_t device, magma_queue_t* queue_ptr,
    const char* func, const char* file, int line )
{
    magma_queue_t queue;
    magma_malloc_cpu( (void**)&queue, sizeof(*queue) );
    assert( queue != NULL );
    *queue_ptr = queue;
    
    queue->own__      = own_none;
    queue->device__   = device;
    queue->stream__   = NULL;
    queue->cublas__   = NULL;
    queue->cusparse__ = NULL;
    
    cudaError_t err;
    err = cudaSetDevice( device );
    check_xerror( err, func, file, line );
    
    err = cudaStreamCreate( &queue->stream__ );
    check_xerror( err, func, file, line );
    queue->own__ |= own_stream;
    
    cublasStatus_t stat;
    stat = cublasCreate( &queue->cublas__ );
    check_xerror( stat, func, file, line );
    queue->own__ |= own_cublas;
    stat = cublasSetStream( queue->cublas__, queue->stream__ );
    check_xerror( stat, func, file, line );
    
    cusparseStatus_t stat2;
    stat2 = cusparseCreate( &queue->cusparse__ );
    check_xerror( stat2, func, file, line );
    queue->own__ |= own_cusparse;
    stat2 = cusparseSetStream( queue->cusparse__, queue->stream__ );
    check_xerror( stat2, func, file, line );
    
}

// --------------------
extern "C"
void magma_queue_create_from_cuda_internal(
    magma_int_t      device,
    cudaStream_t     stream,
    cublasHandle_t   cublas_handle,
    cusparseHandle_t cusparse_handle,
    magma_queue_t*   queue_ptr,
    const char* func, const char* file, int line )
{
    magma_queue_t queue = (magma_queue_t) malloc( sizeof(*queue) );
    assert( queue != NULL );
    *queue_ptr = queue;
    
    queue->own__      = own_none;
    queue->device__   = device;
    queue->stream__   = NULL;
    queue->cublas__   = NULL;
    queue->cusparse__ = NULL;
    
    cudaError_t err;
    err = cudaSetDevice( device );
    check_xerror( err, func, file, line );
    
    // stream can be NULL
    queue->stream__ = stream;
    
    // allocate cublas handle if given as NULL
    cublasStatus_t stat;
    if ( cublas_handle == NULL ) {
        stat  = cublasCreate( &cublas_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_cublas;
    }
    queue->cublas__ = cublas_handle;
    stat  = cublasSetStream( queue->cublas__, queue->stream__ );
    check_xerror( stat, func, file, line );
    
    // allocate cusparse handle if given as NULL
    cusparseStatus_t stat2;
    if ( cusparse_handle == NULL ) {
        stat2 = cusparseCreate( &cusparse_handle );
        check_xerror( stat, func, file, line );
        queue->own__ |= own_cusparse;
    }
    queue->cusparse__ = cusparse_handle;
    stat2 = cusparseSetStream( queue->cusparse__, queue->stream__ );
    check_xerror( stat2, func, file, line );
    
}

// --------------------
extern "C"
void magma_queue_destroy_internal(
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    if ( queue != NULL ) {
        if ( queue->cublas__ != NULL && (queue->own__ & own_cublas)) {
            cublasStatus_t stat = cublasDestroy( queue->cublas__ );
            check_xerror( stat, func, file, line );
        }
        if ( queue->cusparse__ != NULL && (queue->own__ & own_cusparse)) {
            cusparseStatus_t stat = cusparseDestroy( queue->cusparse__ );
            check_xerror( stat, func, file, line );
        }
        if ( queue->stream__ != NULL && (queue->own__ & own_stream)) {
            cudaError_t err = cudaStreamDestroy( queue->stream__ );
            check_xerror( err, func, file, line );
        }
        queue->own__      = own_none;
        queue->device__   = -1;
        queue->stream__   = NULL;
        queue->cublas__   = NULL;
        queue->cusparse__ = NULL;
        magma_free_cpu( queue );
    }
}

// --------------------
extern "C"
void magma_queue_sync_internal(
    magma_queue_t queue,
    const char* func, const char* file, int line )
{
    cudaError_t    err;
    err  = cudaStreamSynchronize( queue->cuda_stream() );
    check_xerror( err, func, file, line );
}


// --------------------
extern "C" size_t
magma_queue_mem_size( magma_queue_t queue )
{
    size_t freeMem, totalMem;
    cudaError_t err = cudaMemGetInfo( &freeMem, &totalMem );
    check_error( err );
    return freeMem;
}


// ========================================
// event support
// --------------------
extern "C"
void magma_event_create( magma_event_t* event )
{
    cudaError_t err;
    err = cudaEventCreate( event );
    check_error( err );
}

// --------------------
extern "C"
void magma_event_destroy( magma_event_t event )
{
    if ( event != NULL ) {
        cudaError_t err;
        err = cudaEventDestroy( event );
        check_error( err );
    }
}

// --------------------
extern "C"
void magma_event_record( magma_event_t event, magma_queue_t queue )
{
    cudaError_t err;
    err = cudaEventRecord( event, queue->cuda_stream() );
    check_error( err );
}

// --------------------
// blocks CPU until event occurs
extern "C"
void magma_event_sync( magma_event_t event )
{
    cudaError_t err;
    err = cudaEventSynchronize( event );
    check_error( err );
}

// --------------------
// blocks queue (but not CPU) until event occurs
extern "C"
void magma_queue_wait_event( magma_queue_t queue, magma_event_t event )
{
    cudaError_t err;
    err = cudaStreamWaitEvent( queue->cuda_stream(), event, 0 );
    check_error( err );
}

#endif // HAVE_CUBLAS
