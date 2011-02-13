#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include "magma.h"

#ifndef __OPERATORS__H
#define __OPERATORS__H


// negate
inline __host__ __device__ cuDoubleComplex operator-(cuDoubleComplex &a)
{
	return make_cuDoubleComplex(-a.x, -a.y);
}
// addition
inline __host__ __device__ cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b)
{
	return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(cuDoubleComplex &a, cuDoubleComplex b)
{
	a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ cuDoubleComplex operator-(cuDoubleComplex a, cuDoubleComplex b)
{
	return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(cuDoubleComplex &a, cuDoubleComplex b)
{
	a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b)
{
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}
inline __host__ __device__ cuDoubleComplex operator*(cuDoubleComplex a, double s)
{
	return make_cuDoubleComplex(a.x * s, a.y * s);
}
inline __host__ __device__ cuDoubleComplex operator*(double s, cuDoubleComplex a)
{
	return make_cuDoubleComplex(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(cuDoubleComplex &a, double s)
{
	a.x *= s; a.y *= s;
}



// negate
inline __host__ __device__ cuFloatComplex operator-(cuFloatComplex &a)
{
	return make_cuFloatComplex(-a.x, -a.y);
}
// addition
inline __host__ __device__ cuFloatComplex operator+(cuFloatComplex a, cuFloatComplex b)
{
	return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}
inline __host__ __device__ void operator+=(cuFloatComplex &a, cuFloatComplex b)
{
	a.x += b.x; a.y += b.y;
}

// subtract
inline __host__ __device__ cuFloatComplex operator-(cuFloatComplex a, cuFloatComplex b)
{
	return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(cuFloatComplex &a, cuFloatComplex b)
{
	a.x -= b.x; a.y -= b.y;
}

// multiply
inline __host__ __device__ cuFloatComplex operator*(cuFloatComplex a, cuFloatComplex b)
{
    return make_cuFloatComplex(a.x * b.x - a.y * b.y, a.y * b.x + a.x * b.y);
}
inline __host__ __device__ cuFloatComplex operator*(cuFloatComplex a, float s)
{
	return make_cuFloatComplex(a.x * s, a.y * s);
}
inline __host__ __device__ cuFloatComplex operator*(float s, cuFloatComplex a)
{
	return make_cuFloatComplex(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(cuFloatComplex &a,float s)
{
	a.x *= s; a.y *= s;
}


#endif



