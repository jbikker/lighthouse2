/* .optix.cu - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file contains a minimal set of Optix functions. From here we will
   dispatch program flow to our own functions that implement the path tracer.
*/

#include "../kernels/noerrors.h"
#include "helper_math.h"

// function defintion helper
#define LH2_DEVFUNC	static __forceinline__ __device__

// global include files
#include "../../RenderSystem/common_settings.h"
#include "../../RenderSystem/common_functions.h"
#include "../../RenderSystem/common_types.h"
#define OPTIX_CU // skip CUDAMaterial definition in core_settings.h; not needed here 
#include "../core_settings.h"

// global path tracing parameters
extern "C" { __constant__ Params params; }

// tools
__device__ __inline__ uint WangHash( uint s ) { s = (s ^ 61) ^ (s >> 16), s *= 9, s = s ^ (s >> 4), s *= 0x27d4eb2d, s = s ^ (s >> 15); return s; }
__device__ __inline__ uint RandomInt( uint& s ) { s ^= s << 13, s ^= s >> 17, s ^= s << 5; return s; }
__device__ __inline__ float RandomFloat( uint& s ) { return RandomInt( s ) * 2.3283064365387e-10f; }

static __inline __device__ float blueNoiseSampler( int x, int y, int sampleIndex, int sampleDimension )
{
	// Adapated from E. Heitz. Arguments:
	// sampleIndex: 0..255
	// sampleDimension: 0..255
	x &= 127, y &= 127, sampleIndex &= 255, sampleDimension &= 255;
	// xor index based on optimized ranking
	int rankedSampleIndex = (sampleIndex ^ params.blueNoise[sampleDimension + (x + y * 128) * 8 + 65536 * 3]) & 255;
	// fetch value in sequence
	int value = params.blueNoise[sampleDimension + rankedSampleIndex * 256];
	// if the dimension is optimized, xor sequence value based on optimized scrambling
	value ^= params.blueNoise[(sampleDimension & 7) + (x + y * 128) * 8 + 65536];
	// convert to float and return
	return (0.5f + value) * (1.0f / 256.0f);
}

LH2_DEVFUNC float4 blueNoiseSampler4( int x, int y, int sampleIndex, int sampleDimension )
{
	// Optimized retrieval of 4 blue noise samples.
	const uint4 bn4 = *((uint4*)(params.blueNoise + sampleDimension + (x + y * 128) * 8 + 65536 * 3));
	const int rsi1 = (sampleIndex ^ bn4.x) & 255, rsi2 = (sampleIndex ^ bn4.y) & 255;
	const int rsi3 = (sampleIndex ^ bn4.z) & 255, rsi4 = (sampleIndex ^ bn4.w) & 255;
	const int v1 = params.blueNoise[sampleDimension + 0 + rsi1 * 256];
	const int v2 = params.blueNoise[sampleDimension + 1 + rsi2 * 256];
	const int v3 = params.blueNoise[sampleDimension + 2 + rsi3 * 256];
	const int v4 = params.blueNoise[sampleDimension + 3 + rsi4 * 256];
	const uint4 bx4 = *((uint4*)(params.blueNoise + (sampleDimension & 7) + (x + y * 128) * 8 + 65536));
	return make_float4( (0.5f + (v1 ^ bx4.x)) * (1.0f / 256.0f), (0.5f + (v2 ^ bx4.y)) * (1.0f / 256.0f),
		(0.5f + (v3 ^ bx4.z)) * (1.0f / 256.0f), (0.5f + (v4 ^ bx4.w)) * (1.0f / 256.0f) );
}

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
	float4 posLens = params.posLensSize;
	return make_float3( posLens ) + posLens.w * (params.right * xr + params.up * yr);
}

static __inline __device__ void generateEyeRay( float3& O, float3& D, const uint pixelIdx, const uint sampleIdx, uint& seed )
{
	// random point on pixel and lens
	int sx = pixelIdx % params.scrsize.x;
	int sy = pixelIdx / params.scrsize.x;
	int shift = params.shift;
	float4 r4;
	if (sampleIdx < 64)
	{
		r4 = blueNoiseSampler4( (sx + (shift & 127)) & 127, (sy + (shift >> 24)) & 127, sampleIdx, 0 );
	}
	else
	{
		r4.x = RandomFloat( seed ), r4.y = RandomFloat( seed );
		r4.z = RandomFloat( seed ), r4.w = RandomFloat( seed );
	}
	O = RandomPointOnLens( r4.x, r4.z );
	float3 posOnPixel = RayTarget( sx, sy, r4.y, r4.w, make_int2( params.scrsize ), params.distortion, params.p1, params.right, params.up );
	D = normalize( posOnPixel - O );
}

#if __CUDA_ARCH__ >= 700
#define THREADMASK	__activemask() // volta, turing
#else
#define THREADMASK	0xffffffff // pascal, kepler, fermi
#endif

__device__ void setupPrimaryRay( const uint pathIdx, const uint stride )
{
	const uint pixelIdx = pathIdx % (params.scrsize.x * params.scrsize.y);
	const uint sampleIdx = pathIdx / (params.scrsize.x * params.scrsize.y) + params.pass;
	uint seed = WangHash( pathIdx * 16789 + params.pass * 1791 );
	// generate eye ray
	float3 O, D;
	generateEyeRay( O, D, pixelIdx, sampleIdx, seed );
	// populate path state array
	params.pathStates[pathIdx] = make_float4( O, __uint_as_float( (pathIdx << 6) + 1 /* S_SPECULAR in CUDA code */ ) );
	params.pathStates[pathIdx + stride] = make_float4( D, 0 );
	// trace eye ray
	uint u0, u1 = 0, u2 = 0xffffffff, u3 = __float_as_uint( 1e34f );
	optixTrace( params.bvhRoot, O, D, params.geometryEpsilon, 1e34f, 0.0f /* ray time */, OptixVisibilityMask( 1 ),
		OPTIX_RAY_FLAG_NONE, 0, 2, 0, u0, u1, u2, u3 );
	if (pixelIdx < stride /* OptiX bug workaround? */) if (u2 != 0xffffffff) /* bandwidth reduction */
		params.hitData[pathIdx] = make_float4( __uint_as_float( u0 ), __uint_as_float( u1 ), __uint_as_float( u2 ), __uint_as_float( u3 ) );
}

__device__ void setupSecondaryRay( const uint rayIdx, const uint stride )
{
	const float4 O4 = params.pathStates[rayIdx];
	const float4 D4 = params.pathStates[rayIdx + stride];
	uint u0, u1 = 0, u2 = 0xffffffff, u3 = __float_as_uint( 1e34f );
	optixTrace( params.bvhRoot, make_float3( O4 ), make_float3( D4 ), params.geometryEpsilon, 1e34f, 0.0f /* ray time */, OptixVisibilityMask( 1 ),
		OPTIX_RAY_FLAG_NONE, 0, 2, 0, u0, u1, u2, u3 );
	if (rayIdx < stride /* OptiX bug workaround? */) if (u2 != 0xffffffff) /* bandwidth reduction */
		params.hitData[rayIdx] = make_float4( __uint_as_float( u0 ), __uint_as_float( u1 ), __uint_as_float( u2 ), __uint_as_float( u3 ) );
}

__device__ void generateShadowRay( const uint rayIdx, const uint stride )
{
	const float4 O4 = params.connectData[rayIdx]; // O4
	const float4 D4 = params.connectData[rayIdx + stride * 2]; // D4
	// launch shadow ray
	uint u0 = 1;
	optixTrace( params.bvhRoot, make_float3( O4 ), make_float3( D4 ), params.geometryEpsilon, D4.w, 0.0f /* ray time */, OptixVisibilityMask( 1 ),
		OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1, 2, 1, u0 );
	if (u0) return;
	const float4 E4 = params.connectData[rayIdx + stride * 2 * 2]; // E4
	const int pixelIdx = __float_as_int( E4.w );
	if (pixelIdx < stride /* OptiX bug workaround? */) params.accumulator[pixelIdx] += make_float4( E4.x, E4.y, E4.z, 1 );
}

extern "C" __global__ void __raygen__rg()
{
	const uint stride = params.scrsize.x * params.scrsize.y * params.scrsize.z;
	const uint3 idx = optixGetLaunchIndex();
	const uint rayIdx = idx.x + idx.y * params.scrsize.x;
	switch (params.phase)
	{
	case Params::SPAWN_PRIMARY: /* primary rays */ setupPrimaryRay( rayIdx, stride ); break;
	case Params::SPAWN_SECONDARY: /* secondary rays */ setupSecondaryRay( rayIdx, stride ); break;
	case Params::SPAWN_SHADOW: /* shadow rays */ generateShadowRay( rayIdx, stride ); break;
	}
}

extern "C" __global__ void __miss__occlusion()
{
	optixSetPayload_0( 0u ); // instead of any hit. suggested by WillUsher.io.
}

extern "C" __global__ void __closesthit__radiance()
{
	const uint prim_idx = optixGetPrimitiveIndex();
	const uint inst_idx = optixGetInstanceIndex();
	const float2 bary = optixGetTriangleBarycentrics();
	const float tmin = optixGetRayTmax();
	optixSetPayload_0( (uint)(65535.0f * bary.x) + ((uint)(65535.0f * bary.y) << 16) );
	optixSetPayload_1( inst_idx );
	optixSetPayload_2( prim_idx );
	optixSetPayload_3( __float_as_uint( tmin ) );
}

// EOF