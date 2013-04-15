#include <stdio.h>

#include "error.h"


// ----------------------------------------
// C++ function is overloaded for different error types,
// which depends on error types being enums to be differentiable.
void magma_xerror( cudaError_t err, const char* func, const char* file, int line )
{
    if ( err != cudaSuccess ) {
        fprintf( stderr, "CUDA runtime error: %s (%d) in %s at %s:%d\n",
                 cudaGetErrorString( err ), err, func, file, line );
    }
}


// --------------------
void magma_xerror( CUresult err, const char* func, const char* file, int line )
{
    if ( err != CUDA_SUCCESS ) {
        fprintf( stderr, "CUDA driver error: %s (%d) in %s at %s:%d\n",
                 cuGetErrorString( err ), err, func, file, line );
    }
}


// --------------------
void magma_xerror( cublasStatus_t err, const char* func, const char* file, int line )
{
    if ( err != CUBLAS_STATUS_SUCCESS ) {
        fprintf( stderr, "CUBLAS error: %s (%d) in %s at %s:%d\n",
                 cublasGetErrorString( err ), err, func, file, line );
    }
}


// --------------------
void magma_xerror( magma_err_t err, const char* func, const char* file, int line )
{
    if ( err != MAGMA_SUCCESS ) {
        fprintf( stderr, "MAGMA error: %s (%d) in %s at %s:%d\n",
                 magma_geterrorstring( err ), err, func, file, line );
    }
}


// ----------------------------------------
// cuda provides cudaGetErrorString, but not cuGetErrorString.
extern "C"
const char* cuGetErrorString( CUresult error )
{
    switch( error ) {
        case CUDA_SUCCESS:
            return "success";
        
        case CUDA_ERROR_INVALID_VALUE:
            return "invalid value";
        
        case CUDA_ERROR_OUT_OF_MEMORY:
            return "out of memory";
        
        case CUDA_ERROR_NOT_INITIALIZED:
            return "not initialized";
        
        case CUDA_ERROR_DEINITIALIZED:
            return "deinitialized";
        
        case CUDA_ERROR_PROFILER_DISABLED:
            return "profiler disabled";
        
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
            return "profiler not initialized";
        
        case CUDA_ERROR_PROFILER_ALREADY_STARTED:
            return "profiler already started";
        
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
            return "profiler already stopped";
        
        case CUDA_ERROR_NO_DEVICE:
            return "no device";
        
        case CUDA_ERROR_INVALID_DEVICE:
            return "invalid device";
        
        case CUDA_ERROR_INVALID_IMAGE:
            return "invalid image";
        
        case CUDA_ERROR_INVALID_CONTEXT:
            return "invalid context";
        
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
            return "context already current";
        
        case CUDA_ERROR_MAP_FAILED:
            return "map failed";
        
        case CUDA_ERROR_UNMAP_FAILED:
            return "unmap failed";
        
        case CUDA_ERROR_ARRAY_IS_MAPPED:
            return "array is mapped";
        
        case CUDA_ERROR_ALREADY_MAPPED:
            return "already mapped";
        
        case CUDA_ERROR_NO_BINARY_FOR_GPU:
            return "no binary for GPU";
        
        case CUDA_ERROR_ALREADY_ACQUIRED:
            return "already acquired";
        
        case CUDA_ERROR_NOT_MAPPED:
            return "not mapped";
        
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
            return "not mapped as array";
        
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
            return "not mapped as pointer";
        
        case CUDA_ERROR_ECC_UNCORRECTABLE:
            return "ECC uncorrectable";
        
        case CUDA_ERROR_UNSUPPORTED_LIMIT:
            return "unsupported limit";
        
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
            return "context already in use";
        
        case CUDA_ERROR_INVALID_SOURCE:
            return "invalid source";
        
        case CUDA_ERROR_FILE_NOT_FOUND:
            return "file not found";
        
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
            return "shared object symbol not found";
        
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
            return "shared object init failed";
        
        case CUDA_ERROR_OPERATING_SYSTEM:
            return "operating system";
        
        case CUDA_ERROR_INVALID_HANDLE:
            return "invalid handle";
        
        case CUDA_ERROR_NOT_FOUND:
            return "not found";
        
        case CUDA_ERROR_NOT_READY:
            return "not ready";
        
        case CUDA_ERROR_LAUNCH_FAILED:
            return "launch failed";
        
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
            return "launch out of resources";
        
        case CUDA_ERROR_LAUNCH_TIMEOUT:
            return "launch timeout";
        
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
            return "launch incompatible texturing";
        
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
            return "peer access already enabled";
        
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
            return "peer access not enabled";
        
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
            return "primary context active";
        
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:
            return "context is destroyed";
        
        case CUDA_ERROR_UNKNOWN:
            return "unknown";
        
        default:
            return "unknown error code";
    }
}


// ----------------------------------------
// cuda provides cudaGetErrorString, but not cublasGetErrorString.
extern "C"
const char* cublasGetErrorString( cublasStatus_t error )
{
    switch( error ) {
        case CUBLAS_STATUS_SUCCESS:
            return "success";
        
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "not initialized";
        
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "out of memory";
        
        case CUBLAS_STATUS_INVALID_VALUE:
            return "invalid value";
        
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "architecture mismatch";
        
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "memory mapping error";
        
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "execution failed";
        
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "internal error";
        
        default:
            return "unknown error code";
    }
}


// ----------------------------------------
extern "C"
const char* magma_geterrorstring( magma_err_t error )
{
    // LAPACK-compliant errors
    if ( error > 0 ) {
        return "see function documentation";
    }
    else if ( error < 0 && error > MAGMA_ERR ) {
        return "bad argument value";
    }
    // MAGMA-specific errors
    switch( error ) {
        case MAGMA_SUCCESS:
            return "success";
        
        case MAGMA_ERR:
            return "unknown error";
        
        case MAGMA_ERR_NOT_INITIALIZED:
            return "not initialized";
        
        case MAGMA_ERR_REINITIALIZED:
            return "reinitialized";
        
        case MAGMA_ERR_NOT_SUPPORTED:
            return "not supported";
        
        case MAGMA_ERR_ILLEGAL_VALUE:
            return "illegal value";
        
        case MAGMA_ERR_NOT_FOUND:
            return "not found";
        
        case MAGMA_ERR_ALLOCATION:
            return "allocation";
        
        case MAGMA_ERR_INTERNAL_LIMIT:
            return "internal limit";
        
        case MAGMA_ERR_UNALLOCATED:
            return "unallocated";
        
        case MAGMA_ERR_FILESYSTEM:
            return "filesystem";
        
        case MAGMA_ERR_UNEXPECTED:
            return "unexpected";
        
        case MAGMA_ERR_SEQUENCE_FLUSHED:
            return "sequence flushed";
        
        case MAGMA_ERR_HOST_ALLOC:
            return "host malloc";
        
        case MAGMA_ERR_DEVICE_ALLOC:
            return "device malloc";
        
        case MAGMA_ERR_CUDASTREAM:
            return "CUDA stream";
        
        case MAGMA_ERR_INVALID_PTR:
            return "invalid pointer";
        
        default:
            return "unknown error code";
    }
}
