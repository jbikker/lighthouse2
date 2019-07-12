// This file contains a minimal set of Optix functions. From here we will
// dispatch program flow to our own functions that implement the path tracer.

#define IRRATIONAL1		10368890	// 1.6180339887f, in 8:24 fixed point
#define IRRATIONAL2		12281775	// 1.7320508076f, in 8:24 fixed point
#define IRRATIONAL3		6949350		// 1.4142135624f, in 8:24 fixed point

#include "../kernels/noerrors.h"
#include <optix.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include "cuda_fp16.h"

#ifdef __CUDACC__
#if __CUDA_ARCH__ >= 700
#include <cooperative_groups.h>
using namespace cooperative_groups;
namespace cg = cooperative_groups;
#endif
typedef __half half;
typedef unsigned char uchar;
#endif

using namespace optix;

// global include file
#include "../../rendersystem/common_settings.h"
#include "../core_settings.h"

// set automatically
rtDeclareVariable( uint, launch_index, rtLaunchIndex, );
rtDeclareVariable( uint, launch_dim, rtLaunchDim, );

// set from host code
rtDeclareVariable( rtObject, bvhRoot, , );
rtDeclareVariable( uint, sampleBase, , );
rtDeclareVariable( uint, phase, , );

// runtime variables
rtDeclareVariable( float4, payload, rtPayload, );
rtDeclareVariable( uint, visible, rtPayload, );
rtDeclareVariable( float, t_hit, rtIntersectionDistance, );
rtDeclareVariable( Ray, ray, rtCurrentRay, );
rtDeclareVariable( float4, hit_data, attribute hit_data, );

// triangle API data
rtDeclareVariable( float2, barycentrics, attribute barycentrics, );
rtDeclareVariable( unsigned int, instanceid, attribute instanceid, );

// path tracing buffers
rtDeclareVariable( int, instanceIndex, , );
rtBuffer<float4> connectData;
rtBuffer<float4> accumulator;
#ifdef USE_OPTIX_PERSISTENT_THREADS
rtBuffer<Counters> counters;
#endif

// statistics
rtBuffer<uint> performanceCounters;

// results
rtBuffer<float4> hitData;
rtBuffer<float4> pathStates;

// camera parameters
rtDeclareVariable( float4, posLensSize, , );
rtDeclareVariable( float3, right, , );
rtDeclareVariable( float3, up, , );
rtDeclareVariable( float3, p1, , );
rtDeclareVariable( float, geometryEpsilon, , );
rtDeclareVariable( int3, scrsize, , );

// tools
__device__ __inline__ uint WangHash( uint s ) { s = (s ^ 61) ^ (s >> 16), s *= 9, s = s ^ (s >> 4), s *= 0x27d4eb2d, s = s ^ (s >> 15); return s; }
__device__ __inline__ uint RandomInt( uint& s ) { s ^= s << 13, s ^= s >> 17, s ^= s << 5; return s; }
__device__ __inline__ float RandomFloat( uint& s ) { return RandomInt( s ) * 2.3283064365387e-10f; }

static __inline __device__ float3 RandomPointOnLens( const float r0, float r1 )
{
	const float blade = (int)(r0 * 9);
	float r2 = (r0 - blade * (1.0f / 9.0f)) * 9.0f;
	float x1, y1, x2, y2;
	__sincosf( blade * PI / 4.5f, &x1, &y1 );
	__sincosf( (blade + 1.0f) * PI / 4.5f, &x2, &y2 );
	if ((r1 + r2) > 1) r1 = 1.0f - r1, r2 = 1.0f - r2;
	const float xr = x1 * r1 + x2 * r2;
	const float yr = y1 * r1 + y2 * r2;
	const float4 posLens = posLensSize;
	return make_float3( posLens ) + posLens.w * (right * xr + up * yr);
}

static __inline __device__ void generateEyeRay( float3& O, float3& D, const uint pixelIdx, const uint sampleIndex, uint& seed )
{
	// random point on pixel and lens
	int sx = pixelIdx % scrsize.x;
	int sy = pixelIdx / scrsize.x;
	float r0, r1;
	float r2, r3;
	r0 = RandomFloat( seed );
	r1 = RandomFloat( seed );
	r2 = RandomFloat( seed );
	r3 = RandomFloat( seed );
	O = RandomPointOnLens( r2, r3 );
	const float u = ((float)sx + r0) * (1.0f / scrsize.x);
	const float v = ((float)sy + r1) * (1.0f / scrsize.y);
	const float3 pointOnPixel = p1 + u * right + v * up;
	D = normalize( pointOnPixel - O );
}

static __forceinline__ __device__ int atomicAggInc( uint* ptr )
{
#if 1 // __CUDA_ARCH__ < 700
	return atomicAdd( ptr, 1 );
#else
	// not allowed in OptiX code?
	uint lane_mask_lt, active = __activemask();
	asm( "mov.u32 %0, %%lanemask_lt;" : "=r"(lane_mask_lt) );
	int leader = __ffs( active ) - 1, change = __popc( active ), warp_res;
	uint rank = __popc( active & lane_mask_lt );
	if (rank == 0) warp_res = atomicAdd( ptr, change );
	warp_res = __shfl_sync( active, warp_res, leader );
	return warp_res + rank;
#endif
}

#if __CUDA_ARCH__ >= 700
#define THREADMASK	__activemask() // volta, turing
#else
#define THREADMASK	0xffffffff // pascal, kepler, fermi
#endif

__device__ void setupPrimaryRay( const uint jobIdx, const uint stride )
{
	const uint tileIdx = jobIdx >> 8;
	const uint xtiles = scrsize.x / 16;
	const uint tilex = tileIdx % xtiles, tiley = tileIdx / xtiles;
	const uint x_in_tile = (jobIdx & 15);
	const uint y_in_tile = (jobIdx & 255) >> 4;
	const uint pathIdx = tilex * 16 + x_in_tile + (tiley * 16 + y_in_tile) * scrsize.x;
	const uint pixelIdx = pathIdx % (scrsize.x * scrsize.y);
	const uint sampleIdx = pathIdx / (scrsize.x * scrsize.y);
	uint seed = WangHash( pathIdx * 16789 + sampleBase * 1791 );
	// generate eye ray
	float3 O, D;
	generateEyeRay( O, D, pixelIdx, sampleBase + sampleIdx /* 0..SPP-1 */, seed );
	// populate path state array
	pathStates[jobIdx] = make_float4( O, __uint_as_float( (pathIdx << 8) + 1 /* S_SPECULAR in CUDA code */ ) );
	pathStates[jobIdx + stride] = make_float4( D, 0 );
	// trace eye ray
	float4 result = make_float4( 0, 0, __int_as_float( -1 ), 0 );
	rtTrace( bvhRoot, make_Ray( O, D, 0u, geometryEpsilon, RT_DEFAULT_MAX ), result );
	hitData[jobIdx] = result;
}

__device__ void setupSecondaryRay( const uint rayIdx, const uint stride )
{
	const float4 O4 = pathStates[rayIdx];
	const float4 D4 = pathStates[rayIdx + stride];
	float4 result = make_float4( 0, 0, __int_as_float( -1 ), 0 );
	uint pixelIdx = __float_as_uint( O4.w ) >> 8;
	uint px = pixelIdx % 1280;
	uint py = pixelIdx / 1280;
	rtTrace( bvhRoot, make_Ray( make_float3( O4 ), make_float3( D4 ), 0u, geometryEpsilon, RT_DEFAULT_MAX ), result );
	hitData[rayIdx] = result;
}

__device__ void generateShadowRay( const uint rayIdx, const uint stride )
{
	const float4 O4 = connectData[rayIdx]; // O4
	const float4 D4 = connectData[rayIdx + stride * MAXPATHLENGTH]; // D4
	// launch shadow ray
	uint isVisible = 1;
	rtTrace( bvhRoot, make_Ray( make_float3( O4 ), make_float3( D4 ), 1u, 0, D4.w ), isVisible );
	if (isVisible)
	{
		const float4 E4 = connectData[rayIdx + stride * 2 * MAXPATHLENGTH]; // E4
		const int pixelIdx = __float_as_int( E4.w );
		accumulator[pixelIdx] += make_float4( E4.x, E4.y, E4.z, 1 );
	}
}

RT_PROGRAM void generate()
{
	const uint stride = scrsize.x * scrsize.y * scrsize.z;
	if (phase == 0)
	{
		// primary rays
#ifdef USE_OPTIX_PERSISTENT_THREADS
		while (1)
		{
			const uint jobIdx = atomicAggInc( &counters[0].generated );
			if (jobIdx >= stride) break;
			setupPrimaryRay( jobIdx, stride );
		}
#else
		setupPrimaryRay( launch_index, stride );
#endif
	}
	else if (phase == 1)
	{
		// secondary rays
#ifdef USE_OPTIX_PERSISTENT_THREADS
		const uint pathCount = counters[0].activePaths;
		while (1)
		{
			const uint rayIdx = atomicAggInc( &counters[0].extended );
			if (rayIdx >= pathCount) break;
			setupSecondaryRay( rayIdx, stride );
		}
#else
		setupSecondaryRay( launch_index, stride );
#endif
	}
	else
	{
		// shadow rays
#ifdef USE_OPTIX_PERSISTENT_THREADS
		const uint connectionCount = counters[0].shadowRays;
		while (1)
		{
			const uint rayIdx = atomicAggInc( &counters[0].connected );
			if (rayIdx >= connectionCount) break;
			generateShadowRay( rayIdx, stride );
		}
#else
		generateShadowRay( launch_index, stride );
#endif
	}
}

RT_PROGRAM void generateSecondary()
{
	// secondary rays
	const uint stride = scrsize.x * scrsize.y * scrsize.z;
	const float4 O4 = pathStates[launch_index];
	const float4 D4 = pathStates[launch_index + stride];
	float4 result = make_float4( 0, 0, __int_as_float( -1 ), 0 );
	rtTrace( bvhRoot, make_Ray( make_float3( O4 ), make_float3( D4 ), 0u, geometryEpsilon, RT_DEFAULT_MAX ), result );
	hitData[launch_index] = result;
}

RT_PROGRAM void generateShadows()
{
	const uint stride = scrsize.x * scrsize.y * scrsize.z;
	const float4 O4 = connectData[launch_index]; // O4
	const float4 D4 = connectData[launch_index + stride * MAXPATHLENGTH]; // D4
	// launch shadow ray
	uint isVisible = 1;
	rtTrace( bvhRoot, make_Ray( make_float3( O4 ), make_float3( D4 ), 1u, geometryEpsilon, D4.w ), isVisible );
	if (isVisible)
	{
		const float4 E4 = connectData[launch_index + stride * 2 * MAXPATHLENGTH]; // E4
		const int pixelIdx = __float_as_int( E4.w );
		accumulator[pixelIdx] += make_float4( E4.x, E4.y, E4.z, 1 );
	}
}

RT_PROGRAM void closesthit()
{
	// record hit information
	payload = hit_data;
}

RT_PROGRAM void exception()
{
	const unsigned int code = rtGetExceptionCode();
	rtPrintf( "Caught exception 0x%X at launch index (%d)\n", code, launch_index );
}

RT_PROGRAM void any_hit_shadow()
{
	visible = 0;
	rtTerminateRay();
}

RT_PROGRAM void triangle_attributes()
{
	const float2 bary = rtGetTriangleBarycentrics();
	const uint primIdx = rtGetPrimitiveIndex();
	hit_data = make_float4( bary.x, bary.y, __int_as_float( (instanceIndex << 24) + primIdx ), t_hit );
}

// EOF