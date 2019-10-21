// this file is merely here to suppress the syntax error detection of visual
// studio 2017 in CUDA source files.

#ifndef __CUDACC__
#pragma once
#define __launch_bounds__(a,b)
#define __constant__
#define __device__
#define __float_as_int(x) 0
#define __float_as_uint(x) 0
#define __int_as_float(x) 0
#define __uint_as_float(x) 0
#define __half22float2(x) make_float2(0)
#define __float22half2_rn(x) 0
#define __halves2half2(a,b) 0
#define __ushort_as_half(x) 0
inline float __expf( float x ) { return 0; }
float RandomFloat( uint& x ) { return 0; }
#define __syncthreads()
#define __all_sync(x,y) 0
#define atomicAdd(x,y) 0
int2 threadIdx, blockIdx, blockDim;
template<class T> void surf2Dwrite(T val, surface<void, cudaSurfaceType2D> surf, 
int x, int y, int s, enum cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap) {}
typedef unsigned int half2;
#define INCLUDEHEADER #include "kernel.h"
#define __sincosf(a,b,c)
#else
#define INCLUDEHEADER
#endif
#define RT_PROGRAM
#define rtDeclareVariable(a,b,c,d) float3 b
template <typename T,int S=1> struct rtBuffer { int x[S]; T *ary; };
#define rtGetExceptionCode(a) 0;
#define rtPrintf printf
#define rtTrace printf
#define rtTerminateRay int x=rand
#define intersect_triangle(a,b,c,d,e,f,g,h) 0
#define rtPotentialIntersection(a) 0
#define rtReportIntersection(a)

