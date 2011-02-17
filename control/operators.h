/**
 *
 *  @file operators.h
 *
 *  MAGMA (version 1.0) --
 *  Univ. of Tennessee, Knoxville
 *  Univ. of California, Berkeley
 *  Univ. of Colorado, Denver
 *  November 2010
 *
 **/
#ifndef _MAGMA_OPERATORS_H_
#define _MAGMA_OPERATORS_H_

/*************************************************************
 *              cuDoubleComplex
 */

__host__ __device__ static __inline__ cuDoubleComplex 
operator-(cuDoubleComplex &a)
{
    return make_cuDoubleComplex(-a.x, -a.y);
}

__host__ __device__ static __inline__ cuDoubleComplex 
operator+(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__host__ __device__ static __inline__ void
operator+=(cuDoubleComplex &a, cuDoubleComplex b)
{
    a.x += b.x; a.y += b.y;
}

__host__ __device__ static __inline__ cuDoubleComplex 
operator-(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}

__host__ __device__ static __inline__ void
operator-=(cuDoubleComplex &a, cuDoubleComplex b)
{
    a.x -= b.x; a.y -= b.y;
}

__host__ __device__ static __inline__ cuDoubleComplex 
operator*(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

__host__ __device__ static __inline__ cuDoubleComplex 
operator*(cuDoubleComplex a, double s)
{
    return make_cuDoubleComplex(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ cuDoubleComplex 
operator*(double s, cuDoubleComplex a)
{
    return make_cuDoubleComplex(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ void 
operator*=(cuDoubleComplex &a, double s)
{
    a.x *= s; a.y *= s;
}

/*************************************************************
 *              cuFloatComplex
 */

__host__ __device__ static __inline__ cuFloatComplex 
operator-(cuFloatComplex &a)
{
    return make_cuFloatComplex(-a.x, -a.y);
}

__host__ __device__ static __inline__ cuFloatComplex 
operator+(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}

__host__ __device__ static __inline__ void
operator+=(cuFloatComplex &a, cuFloatComplex b)
{
    a.x += b.x; a.y += b.y;
}

__host__ __device__ static __inline__ cuFloatComplex 
operator-(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}

__host__ __device__ static __inline__ void
operator-=(cuFloatComplex &a, cuFloatComplex b)
{
    a.x -= b.x; a.y -= b.y;
}

__host__ __device__ static __inline__ cuFloatComplex 
operator*(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

__host__ __device__ static __inline__ cuFloatComplex 
operator*(cuFloatComplex a, double s)
{
    return make_cuFloatComplex(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ cuFloatComplex 
operator*(double s, cuFloatComplex a)
{
    return make_cuFloatComplex(a.x * s, a.y * s);
}

__host__ __device__ static __inline__ void 
operator*=(cuFloatComplex &a, double s)
{
    a.x *= s; a.y *= s;
}

#endif



