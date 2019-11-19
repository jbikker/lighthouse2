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

   The implemented path tracer is deliberately simple.
   This file is as similar as possible to the one in OptixPrime_B.
*/

#include "noerrors.h"

// path state flags
#define S_SPECULAR		1	// previous path vertex was specular
#define S_BOUNCED		2	// path encountered a diffuse vertex
#define S_VIASPECULAR	4	// path has seen at least one specular vertex

// readability defines; data layout is optimized for 128-bit accesses
#define PRIMIDX __float_as_int( hitData.z )
#define INSTANCEIDX __float_as_int( hitData.y )
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
#if __CUDA_ARCH__ > 700 // Volta deliberately excluded
__global__  __launch_bounds__( 128 /* max block size */, 4 /* min blocks per sm TURING */ )
#else
__global__  __launch_bounds__( 128 /* max block size */, 8 /* min blocks per sm, PASCAL, VOLTA */ )
#endif
void shadeKernel( float4* accumulator, const uint stride,
	float4* pathStates, const float4* hits, float4* connections,
	const uint R0, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int w, const int h, const float spreadAngle,
	const float3 p1, const float3 p2, const float3 p3, const float3 pos, const uint pathCount )
{
	// respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

	// gather data by reading sets of four floats for optimal throughput
	const float4 O4 = pathStates[jobIndex];				// ray origin xyz, w can be ignored
	const float4 D4 = pathStates[jobIndex + stride];	// ray direction xyz
	float4 T4 = pathLength == 1 ? make_float4( 1 ) /* faster */ : pathStates[jobIndex + stride * 2]; // path thoughput rgb 
	const float4 hitData = hits[jobIndex];
	const float bsdfPdf = T4.w;

	// derived data
	uint data = __float_as_uint( O4.w ); // prob.density of the last sampled dir, postponed because of MIS
	const float3 D = make_float3( D4 );
	float3 throughput = make_float3( T4 );
	const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;
	const uint pathIdx = PATHIDX;
	const uint pixelIdx = pathIdx % (w * h);
	const uint sampleIdx = pathIdx / (w * h) + pass;

	// initialize depth in accumulator for DOF shader
	if (pathLength == 1) accumulator[pixelIdx].w += PRIMIDX == NOHIT ? 10000 : HIT_T;

	// use skydome if we didn't hit any geometry
	if (PRIMIDX == NOHIT)
	{
		float3 contribution = throughput * make_float3( SampleSkydome( D, pathLength ) ) * (1.0f / bsdfPdf);
		CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
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

	// we need to detect alpha in the shading code.
	if (shadingData.flags & 1)
	{
		if (pathLength < MAXPATHLENGTH)
		{
			const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 );
			pathStates[extensionRayIdx] = make_float4( I + D * geometryEpsilon, O4.w );
			pathStates[extensionRayIdx + stride] = D4;
			if (!(isfinite( T4.x + T4.y + T4.z ))) T4 = make_float4( 0, 0, 0, T4.w );
			pathStates[extensionRayIdx + stride * 2] = T4;
		}
		return;
	}

	// path regularization
	// if (FLAGS & S_BOUNCED) shadingData.roughness2 = max( 0.7f, shadingData.roughness2 );

	// stop on light
	if (shadingData.IsEmissive() /* r, g or b exceeds 1 */)
	{
		const float DdotNL = -dot( D, N );
		float3 contribution = make_float3( 0 ); // initialization required.
		if (DdotNL > 0 /* lights are not double sided */)
		{
			if (pathLength == 1 || (FLAGS & S_SPECULAR) > 0)
			{
				// accept light contribution if previous vertex was specular
				contribution = shadingData.color;
			}
			else
			{
				// last vertex was not specular: apply MIS
				const float3 lastN = UnpackNormal( __float_as_uint( D4.w ) );
				const CoreTri& tri = (const CoreTri&)instanceTriangles[PRIMIDX];
				const float lightPdf = CalculateLightPDF( D, HIT_T, tri.area, N );
				const float pickProb = LightPickProb( tri.ltriIdx, RAY_O, lastN, I /* the N at the previous vertex */ );
				if ((bsdfPdf + lightPdf * pickProb) > 0) contribution = throughput * shadingData.color * (1.0f / (bsdfPdf + lightPdf * pickProb));
				contribution = throughput * shadingData.color * (1.0f / (bsdfPdf + lightPdf));
			}
			CLAMPINTENSITY;
			FIXNAN_FLOAT3( contribution );
			accumulator[pixelIdx] += make_float4( contribution, 0 );
		}
		return;
	}

	// detect specular surfaces
	if (ROUGHNESS == 0.001f || TRANSMISSION > 0.999f) FLAGS |= S_SPECULAR; /* detect pure speculars; skip NEE for these */ else FLAGS &= ~S_SPECULAR;

	// initialize seed based on pixel index
	uint seed = WangHash( pathIdx + R0  /* well-seeded xor32 is all you need */ );

	// normal alignment for backfacing polygons
	const float flip = (dot( D, N ) > 0) ? -1 : 1;
	N *= flip;		// fix geometric normal
	iN *= flip;		// fix interpolated normal (consistent normal interpolation)
	fN *= flip;		// fix final normal (includes normal map)
	if (flip > 0)
	{
		shadingData.InvertETA(); // leaving medium; eta ==> 1 / eta
		shadingData.transmittance = make_float3( 0 );
	}

	// apply postponed bsdf pdf
	throughput *= 1.0f / bsdfPdf;

	// next event estimation: connect eye path to light
	if (!(FLAGS & S_SPECULAR)) // skip for specular vertices
	{
		float r0, r1, pickProb, lightPdf = 0;
		if (sampleIdx < 256)
		{
			const uint x = (pixelIdx % w) & 127, y = (pixelIdx / w) & 127;
			r0 = blueNoiseSampler( blueNoise, x, y, sampleIdx, 4 + 4 * pathLength );
			r1 = blueNoiseSampler( blueNoise, x, y, sampleIdx, 5 + 4 * pathLength );
		}
		else
		{
			r0 = RandomFloat( seed );
			r1 = RandomFloat( seed );
		}
		float3 lightColor, L = RandomPointOnLight( r0, r1, I, fN, pickProb, lightPdf, lightColor ) - I;
		const float dist = length( L );
		L *= 1.0f / dist;
		const float NdotL = dot( L, fN );
		if (NdotL > 0 && dot( fN, L ) > 0 && lightPdf > 0)
		{
			float bsdfPdf;
			const float3 sampledBSDF = EvaluateBSDF( shadingData, fN, T, D * -1.0f, L, bsdfPdf ) * ROUGHNESS;
			if (bsdfPdf > 0)
			{
				// calculate potential contribution
				float3 contribution = throughput * sampledBSDF * lightColor * (NdotL / (pickProb * lightPdf + bsdfPdf));
				FIXNAN_FLOAT3( contribution );
				CLAMPINTENSITY;
				// add fire-and-forget shadow ray to the connections buffer
				const uint shadowRayIdx = atomicAdd( &counters->shadowRays, 1 ); // compaction
				connections[shadowRayIdx] = make_float4( SafeOrigin( I, L, N, geometryEpsilon ), 0 ); // O4
				connections[shadowRayIdx + stride * 2] = make_float4( L, dist - 2 * geometryEpsilon ); // D4
				connections[shadowRayIdx + stride * 2 * 2] = make_float4( contribution, __int_as_float( pixelIdx ) ); // E4
			}
		}
	}

	// cap at one diffuse bounce (because of this we also don't need Russian roulette)
	if (FLAGS & S_BOUNCED) return;

	// depth cap
	if (pathLength == MAXPATHLENGTH /* don't fill arrays with rays we won't trace */) return;

	// evaluate bsdf to obtain direction for next path segment
	float3 R;
	float newBsdfPdf, r3, r4;
	if (sampleIdx < 256)
	{
		const uint x = (pixelIdx % w) & 127, y = (pixelIdx / w) & 127;
		r3 = blueNoiseSampler( blueNoise, x, y, sampleIdx, 6 + 4 * pathLength );
		r4 = blueNoiseSampler( blueNoise, x, y, sampleIdx, 7 + 4 * pathLength );
	}
	else
	{
		r3 = RandomFloat( seed );
		r4 = RandomFloat( seed );
	}
	bool specular = false;
	const float3 bsdf = SampleBSDF( shadingData, fN, N, T, D * -1.0f, HIT_T, r3, r4, R, newBsdfPdf, specular );
	if (newBsdfPdf < EPSILON || isnan( newBsdfPdf )) return;
	if (specular) FLAGS |= S_SPECULAR;

	// write extension ray
	const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 ); // compact
	const uint packedNormal = PackNormal( fN );
	if (!(FLAGS & S_SPECULAR)) FLAGS |= S_BOUNCED; else FLAGS |= S_VIASPECULAR;
	pathStates[extensionRayIdx] = make_float4( SafeOrigin( I, R, N, geometryEpsilon ), __uint_as_float( FLAGS ) );
	pathStates[extensionRayIdx + stride] = make_float4( R, __uint_as_float( packedNormal ) );
	FIXNAN_FLOAT3( throughput );
	pathStates[extensionRayIdx + stride * 2] = make_float4( throughput * bsdf * abs( dot( fN, R ) ), newBsdfPdf );
}

//  +-----------------------------------------------------------------------------+
//  |  shadeKernel                                                                |
//  |  Host-side access point for the shadeKernel code.                     LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void shade( const int pathCount, float4* accumulator, const uint stride,
	float4* pathStates, const float4* hits, float4* connections,
	const uint R0, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int scrwidth, const int scrheight, const float spreadAngle,
	const float3 p1, const float3 p2, const float3 p3, const float3 pos )
{
	const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 128 ) / 128, 1 ), blockDim( 128, 1 );
	shadeKernel << <gridDim.x, 128 >> > (accumulator, stride, pathStates, hits, connections, R0, blueNoise,
		pass, probePixelIdx, pathLength, scrwidth, scrheight, spreadAngle, p1, p2, p3, pos, pathCount);
}

// EOF