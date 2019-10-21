/* pathtracer.cu - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file implements the shading stage of the wavefront algorithm.
   It takes a buffer of hit results and populates a new buffer with
   extension rays. Shadow rays are added with 'potential contributions'
   as fire-and-forget rays, to be traced later. Streams are compacted
   using simple atomics. The kernel is a 'persistent kernel': a fixed
   number of threads fights for food by atomically decreasing a counter.

   The implemented path tracer is the reference path tracer. In this
   version NEE is optional. Random numbers are simply uniform.
*/

#include "noerrors.h"

// path state flags
#define S_SPECULAR		1	// previous path vertex was specular

// readability defines; data layout is optimized for 128-bit accesses
#define INSTANCEIDX __float_as_int( hitData.y )
#define PRIMIDX __float_as_int( hitData.z )
#define HIT_U ((__float_as_uint( hitData.x ) & 65535) * (1.0f / 65535.0f))
#define HIT_V ((__float_as_uint( hitData.x ) >> 16) * (1.0f / 65535.0f))
#define HIT_T hitData.w
#define RAY_O make_float3( O4 )
#define FLAGS data
#define PATHIDX (data >> 8)

//  +-----------------------------------------------------------------------------+
//  |  shadeKernel                                                                |
//  |  Implements the shade phase of the wavefront path tracer.             LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 128 /* max block size */, 1 /* min blocks per sm */ )
void shadeKernel( float4* accumulator, const uint stride,
	const Ray4* extensionRays, const float4* pathStateData, const Intersection* hits,
	Ray4* extensionRaysOut, float4* pathStateDataOut, Ray4* connections, float4* potentials,
	const uint R0, const int pass,
	const int probePixelIdx, const int pathLength, const int w, const int h, const float spreadAngle,
	const float3 p1, const float3 p2, const float3 p3, const float3 pos, const int pathCount )
{
	// respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

	// gather data by reading sets of four floats to maximize bandwidth
	const float4 O4 = extensionRays[jobIndex].O4;		// ray origin xyz, w can be ignored
	const float4 D4 = extensionRays[jobIndex].D4;		// ray direction xyz
	const float4 T4 = pathStateData[jobIndex * 2 + 0];	// path thoughput rgb 

	// extract path state from gathered data
	uint data = __float_as_uint( T4.w );
	const float3 D = make_float3( D4 );
	float3 throughput = make_float3( T4 );
	const Intersection hd = hits[jobIndex];
	const float4 hitData = make_float4( __uint_as_float( (uint)(65535.0f * hd.u) + ((uint)(65535.0f * hd.v) << 16) ), __int_as_float( hd.triid == -1 ? 0 : hd.instid ), __int_as_float( hd.triid ), hd.t );
	const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;
	const uint pathIdx = PATHIDX;
	const uint pixelIdx = pathIdx % (w * h);
	const uint sampleIdx = pathIdx / (w * h) + pass;

	// initialize depth in accumulator for DOF shader
	if (pathLength == 1) accumulator[pixelIdx].w += PRIMIDX == NOHIT ? 10000 : HIT_T;

	// use skydome if we didn't hit any geometry
	if (PRIMIDX == NOHIT)
	{
		float3 contribution = throughput * make_float3( SampleSkydome( D, pathLength ) );
		FIXNAN_FLOAT3( contribution );
		accumulator[pixelIdx] += make_float4( contribution, 0 );
		return;
	}

	// object picking
	if (pixelIdx == probePixelIdx && pathLength == 1 && sampleIdx == 0)
		counters->probedInstid = INSTANCEIDX,	// record instace id at the selected pixel
		counters->probedTriid = PRIMIDX,		// record primitive id at the selected pixel
		counters->probedDist = HIT_T;			// record primary ray hit distance

	// get shadingData and normals
	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = RAY_O + HIT_T * D;
	const float coneWidth = spreadAngle * HIT_T;
	GetShadingData( D, HIT_U, HIT_V, coneWidth, instanceTriangles[PRIMIDX], INSTANCEIDX, shadingData, N, iN, fN, T );

	// stop on light
	if (shadingData.IsEmissive() /* r, g or b exceeds 1 */)
	{
		const float DdotNL = -dot( D, N );
		if (DdotNL > 0) /* lights are not double sided */
		{
			float3 contribution = throughput * shadingData.color;
			FIXNAN_FLOAT3( contribution );
			accumulator[pixelIdx] += make_float4( contribution, 0 );
		}
		return;
	}

	// initialize seed based on pixel index
	uint seed = WangHash( pathIdx * 17 + R0 /* well-seeded xor32 is all you need */ );

	// next event estimation: connect eye path to light
	const float r0 = RandomFloat( seed ), r1 = RandomFloat( seed );
	float pickProb, lightPdf = 0;
	float3 lightColor, L = RandomPointOnLight( r0, r1, I, fN, pickProb, lightPdf, lightColor ) - I;
	const float dist = length( L );
	L *= 1.0f / dist;
	const float NdotL = dot( L, fN );
	if (NdotL > 0 && dot( fN, L ) > 0 && lightPdf > 0)
	{
		float dummy;
		const float3 sampledBSDF = EvaluateBSDF( shadingData, fN, T, D * -1.0f, L, dummy );
		// calculate potential contribution
		float3 contribution = throughput * sampledBSDF * lightColor * (NdotL / (pickProb * lightPdf));
		FIXNAN_FLOAT3( contribution );
		// add fire-and-forget shadow ray to the connections buffer
		const uint shadowRayIdx = atomicAdd( &counters->shadowRays, 1 ); // compaction
		connections[shadowRayIdx].O4 = make_float4( SafeOrigin( I, L, N, geometryEpsilon ), 0 );
		connections[shadowRayIdx].D4 = make_float4( L, dist - 2 * geometryEpsilon );
		potentials[shadowRayIdx] = make_float4( contribution, __int_as_float( pixelIdx ) );
	}

	// evaluate bsdf to obtain direction for next path segment
	float3 R;
	float newBsdfPdf;
	const float r3 = RandomFloat( seed ), r4 = RandomFloat( seed ), r5 = RandomFloat( seed );
	const float3 bsdf = SampleBSDF( shadingData, fN, N, T, D * -1.0f, r3, r4, R, newBsdfPdf );
	if (newBsdfPdf < EPSILON || isnan( newBsdfPdf )) return;

	// russian roulette
	const float p = pathLength == MAXPATHLENGTH ? 0 : SurvivalProbability( bsdf );
	if (p < r5) return;
	throughput *= (max( 0.0f, dot( fN, R ) ) / (p * newBsdfPdf)) * bsdf;

	// write extension ray
	const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 ); // compact
	FIXNAN_FLOAT3( throughput );
	extensionRaysOut[extensionRayIdx].O4 = make_float4( SafeOrigin( I, R, N, geometryEpsilon ), 0 );
	extensionRaysOut[extensionRayIdx].D4 = make_float4( R, 1e34f );
	pathStateDataOut[extensionRayIdx * 2 + 0] = make_float4( throughput, __uint_as_float( data ) );
}

//  +-----------------------------------------------------------------------------+
//  |  shadeKernel                                                                |
//  |  Host-side access point for the shadeKernel code.                     LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void shade( const int pathCount, float4* accumulator, const uint stride,
	const Ray4* extensionRays, const float4* pathStateData, const Intersection* hits,
	Ray4* extensionRaysOut, float4* pathStateDataOut,
	Ray4* connections, float4* potentials,
	const uint R0, const int pass,
	const int probePixelIdx, const int pathLength, const int scrwidth, const int scrheight, const float spreadAngle,
	const float3 p1, const float3 p2, const float3 p3, const float3 pos )
{
	const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 128 ) / 128, 1 ), blockDim( 128, 1 );
	shadeKernel << < gridDim.x, 128 >> > (accumulator, stride,
		extensionRays, pathStateData, hits,
		extensionRaysOut, pathStateDataOut, connections, potentials,
		R0, pass,
		probePixelIdx, pathLength, scrwidth, scrheight, spreadAngle, p1, p2, p3, pos, pathCount);
}

// EOF