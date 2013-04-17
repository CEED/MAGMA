/**
 *
 *  @file operators.h
 *
 *  MAGMA (version 1.1) --
 *  Univ. of Tennessee, Knoxville
 *  Univ. of California, Berkeley
 *  Univ. of Colorado, Denver
 *  November 2011
 *
 **/
#ifndef MAGMA_OPERATORS_H
#define MAGMA_OPERATORS_H

// __host__ and __device__ are defined in CUDA headers.
#include "magma.h"

/*************************************************************
 *              magmaDoubleComplex
 */

__host__ __device__ static inline magmaDoubleComplex 
operator - (const magmaDoubleComplex &a)
{
    return make_cuDoubleComplex(-a.x, -a.y);
}

__host__ __device__ static inline magmaDoubleComplex 
operator + (const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__host__ __device__ static inline magmaDoubleComplex&
operator += (magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    a.x += b.x; a.y += b.y;
    return a;
}

__host__ __device__ static inline magmaDoubleComplex 
operator - (const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}

__host__ __device__ static inline magmaDoubleComplex&
operator -= (magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    a.x -= b.x; a.y -= b.y;
    return a;
}

__host__ __device__ static inline magmaDoubleComplex 
operator * (const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

__host__ __device__ static inline magmaDoubleComplex 
operator * (const magmaDoubleComplex a, const double s)
{
    return make_cuDoubleComplex(a.x * s, a.y * s);
}

__host__ __device__ static inline magmaDoubleComplex 
operator * (const double s, const magmaDoubleComplex a)
{
    return make_cuDoubleComplex(a.x * s, a.y * s);
}

__host__ __device__ static inline magmaDoubleComplex& 
operator *= (magmaDoubleComplex &a, const magmaDoubleComplex b)
{
  double tmp = a.y * b.x + a.x * b.y;
  a.x = a.x * b.x - a.y * b.y;
  a.y = tmp;
    return a;
}

__host__ __device__ static inline magmaDoubleComplex& 
operator *= (magmaDoubleComplex &a, const double s)
{
    a.x *= s; a.y *= s;
    return a;
}

/*************************************************************
 *              magmaFloatComplex
 */

__host__ __device__ static inline magmaFloatComplex 
operator - (const magmaFloatComplex &a)
{
    return make_cuFloatComplex(-a.x, -a.y);
}

__host__ __device__ static inline magmaFloatComplex 
operator + (const magmaFloatComplex a, const magmaFloatComplex b)
{
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}

__host__ __device__ static inline magmaFloatComplex&
operator += (magmaFloatComplex &a, const magmaFloatComplex b)
{
    a.x += b.x; a.y += b.y;
    return a;
}

__host__ __device__ static inline magmaFloatComplex 
operator - (const magmaFloatComplex a, const magmaFloatComplex b)
{
    return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}

__host__ __device__ static inline magmaFloatComplex&
operator -= (magmaFloatComplex &a, const magmaFloatComplex b)
{
    a.x -= b.x; a.y -= b.y;
    return a;
}

__host__ __device__ static inline magmaFloatComplex 
operator * (const magmaFloatComplex a, const magmaFloatComplex b)
{
    return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}

__host__ __device__ static inline magmaFloatComplex 
operator * (const magmaFloatComplex a, const float s)
{
    return make_cuFloatComplex(a.x * s, a.y * s);
}

__host__ __device__ static inline magmaFloatComplex 
operator * (const float s, const magmaFloatComplex a)
{
    return make_cuFloatComplex(a.x * s, a.y * s);
}

__host__ __device__ static inline magmaFloatComplex& 
operator *= (magmaFloatComplex &a, const magmaFloatComplex b)
{
  float tmp = a.y * b.x + a.x * b.y;
  a.x = a.x * b.x - a.y * b.y;
  a.y = tmp;
    return a;
}

__host__ __device__ static inline magmaFloatComplex& 
operator *= (magmaFloatComplex &a, const float s)
{
    a.x *= s; a.y *= s;
    return a;
}

#endif /* MAGMA_OPERATORS_H */
