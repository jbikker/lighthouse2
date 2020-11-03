/* pathtracer.h - Copyright 2019/2020 Utrecht University

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
   This file is as similar as possible to the one in Optix7.
*/

#include "noerrors.h"

// path state flags
#define S_SPECULAR		1	// previous path vertex was specular
#define S_BOUNCED		2	// path encountered a diffuse vertex
#define S_VIASPECULAR	4	// path has seen at least one specular vertex
#define S_BOUNCEDTWICE	8	// this core will stop after two diffuse bounces
#define ENOUGH_BOUNCES	S_BOUNCED // or S_BOUNCEDTWICE

// readability defines; data layout is optimized for 128-bit accesses
#define PRIMIDX __float_as_int( hitData.z )
#define INSTANCEIDX __float_as_int( hitData.y )
#define HIT_U ((__float_as_uint( hitData.x ) & 65535) * (1.0f / 65535.0f))
#define HIT_V ((__float_as_uint( hitData.x ) >> 16) * (1.0f / 65535.0f))
#define HIT_T hitData.w
#define RAY_O make_float3( O4 )
#define FLAGS data
#define PATHIDX (data >> 6)

//  +-----------------------------------------------------------------------------+
//  |  shadeKernel                                                                |
//  |  Implements the shade phase of the wavefront path tracer.             LH2'19|
//  +-----------------------------------------------------------------------------+
#if __CUDA_ARCH__ > 700 // Volta deliberately excluded
__global__  __launch_bounds__( 128 /* max block size */, 2 /* min blocks per sm TURING */ )
#else
__global__  __launch_bounds__( 256 /* max block size */, 2 /* min blocks per sm, PASCAL, VOLTA */ )
#endif
void shadeKernel( float4* accumulator, float4* features, const uint stride,
	const Ray4* extensionRays, const float4* pathStateData, const Intersection* hits,
	Ray4* extensionRaysOut, float4* pathStateDataOut, Ray4* connections, float4* potentials,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int probePixelIdx, const int pathLength, const int w, const int h, const float spreadAngle,
	const int pathCount )
{
	// respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= pathCount) return;

	// gather data by reading sets of four floats for optimal throughput
	const float3 D = make_float3( extensionRays[jobIndex].D4 );
	const float4 T4 = pathStateData[jobIndex * 2 + 0];	// path thoughput rgb
	const float4 Q4 = pathStateData[jobIndex * 2 + 1];	// x, y: pd of the previous bounce, normal at the previous vertex
	const Intersection hd = hits[jobIndex];				// TODO: when using instances, Optix Prime needs 5x4 bytes here...
	const float4 hitData = make_float4( __uint_as_float( (uint)(65535.0f * hd.u) + ((uint)(65535.0f * hd.v) << 16) ), __int_as_float( hd.triid == -1 ? 0 : hd.instid ), __int_as_float( hd.triid ), hd.t );
	uint data = __float_as_uint( T4.w );

	// derived data
	const float bsdfPdf = Q4.x;							// prob.density of the last sampled dir, postponed because of MIS
	float3 throughput = make_float3( T4 );
	const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;
	const uint pathIdx = PATHIDX;
	const uint sampleIdx = pathIdx / (w * h) + pass;
	bool firstHitToBeStored = (FLAGS & S_BOUNCED) == 0;

	// update sample count for adaptive sampling
	if (pathLength == 1) accumulator[pathIdx].w += 1;

	// use skydome if we didn't hit any geometry
	if (PRIMIDX == NOHIT)
	{
		float3 tD = -worldToSky.TransformVector( D );
		float3 skyPixel = FLAGS & S_BOUNCED ? SampleSmallSkydome( tD ) : SampleSkydome( tD );
		float3 contribution = throughput * skyPixel * (1.0f / bsdfPdf);
		CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
		FIXNAN_FLOAT3( contribution );
		accumulator[pathIdx] += make_float4( contribution, 0 );
		if (firstHitToBeStored)
		{
			// we only initialized the feature data, but the path ends here: store something reasonable for the variance estimator.
			const uint packedNormal = PackNormal2( D * -1.0f );
			features[pathIdx] = make_float4( contribution, __uint_as_float( packedNormal ) );
		}
		return;
	}

	// final data gathering
	const float3 O = make_float3( extensionRays[jobIndex].O4 );

	// object picking
	if (pathIdx == probePixelIdx && pathLength == 1)
	{
		counters->probedInstid = INSTANCEIDX;	// record instace id at the selected pixel
		counters->probedTriid = PRIMIDX;		// record primitive id at the selected pixel
		counters->probedDist = HIT_T;			// record primary ray hit distance
	}

	// get shadingData and normals
	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = O + HIT_T * D;
	const float coneWidth = spreadAngle * HIT_T;
	GetShadingData( D, HIT_U, HIT_V, coneWidth, instanceTriangles[PRIMIDX], INSTANCEIDX, shadingData, N, iN, fN, T );

	// store primary hit albedo and normal
	if (firstHitToBeStored)
	{
		// we only initialized the feature data, but the path ends here: store something reasonable for the variance estimator.
		features[pathIdx] = make_float4( shadingData.color, __uint_as_float( PackNormal2( iN  ) ) );
	}

	// we need to detect alpha in the shading code.
	if (shadingData.flags & 1)
	{
		if (pathLength < MAXPATHLENGTH)
		{
			const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 );
			extensionRaysOut[extensionRayIdx].O4 = make_float4( I, EPSILON );
			extensionRaysOut[extensionRayIdx].D4 = make_float4( D, 1e34f );
			FIXNAN_FLOAT3( throughput );
			pathStateDataOut[extensionRayIdx * 2 + 0] = make_float4( throughput, __uint_as_float( data ) );
			pathStateDataOut[extensionRayIdx * 2 + 1] = make_float4( bsdfPdf, 0, 0, 0 );
		}
		return;
	}

	// stop on light
	if (shadingData.IsEmissive() /* r, g or b exceeds 1 */)
	{
		const float DdotNL = -dot( D, N );
		float3 contribution = make_float3( 0 ); // initialization required.
		if (DdotNL > 0 /* lights are not double sided */)
		{
			if (pathLength == 1 || (FLAGS & S_SPECULAR) > 0 || connections == 0)
			{
				// accept light contribution if previous vertex was specular
				contribution = shadingData.color;
			}
			else
			{
				// last vertex was not specular: apply MIS
				const float3 lastN = UnpackNormal( __float_as_uint( Q4.y ) );
				const CoreTri& tri = (const CoreTri&)instanceTriangles[PRIMIDX];
				const float lightPdf = CalculateLightPDF( D, HIT_T, tri.area, N );
				const float pickProb = LightPickProb( tri.ltriIdx, O, lastN, I /* the N at the previous vertex */ );
				if ((bsdfPdf + lightPdf * pickProb) > 0) contribution = throughput * shadingData.color * (1.0f / (bsdfPdf + lightPdf * pickProb));
			}
			CLAMPINTENSITY;
			FIXNAN_FLOAT3( contribution );
			accumulator[pathIdx] += make_float4( contribution, 0 );
		}
		return;
	}

	// path regularization
	if (FLAGS & S_BOUNCED) shadingData.parameters.x |= 255u << 24; // set roughness to 1 after a bounce

	// detect specular surfaces
	if (ROUGHNESS <= 0.001f || TRANSMISSION > 0.5f) FLAGS |= S_SPECULAR; /* detect pure speculars; skip NEE for these */ else FLAGS &= ~S_SPECULAR;

	// normal alignment for backfacing polygons
	const float faceDir = (dot( D, N ) > 0) ? -1 : 1;
	if (faceDir == 1) shadingData.transmittance = make_float3( 0 );

	// apply postponed bsdf pdf
	throughput *= 1.0f / bsdfPdf;

	// prepare random numbers
	uint seed = WangHash( pathIdx * 17 + R0 /* well-seeded xor32 is all you need */ );
	float4 r4;
	if (sampleIdx < 64)
	{
		const uint x = ((pathIdx % w) + (shift & 127)) & 127;
		const uint y = ((pathIdx / w) + (shift >> 24)) & 127;
		r4 = blueNoiseSampler4( blueNoise, x, y, sampleIdx, 4 * pathLength - 4 );
	}
	else
	{
		r4.x = RandomFloat( seed ), r4.y = RandomFloat( seed );
		r4.z = RandomFloat( seed ), r4.w = RandomFloat( seed );
	}

	// next event estimation: connect eye path to light
	if ((FLAGS & S_SPECULAR) == 0 && connections != 0) // skip for specular vertices
	{
		float pickProb, lightPdf = 0;
		float3 lightColor, L = RandomPointOnLight( r4.x, r4.y, I, fN * faceDir, pickProb, lightPdf, lightColor ) - I;
		const float dist = length( L );
		L *= 1.0f / dist;
		const float NdotL = dot( L, fN * faceDir );
		if (NdotL > 0 && lightPdf > 0)
		{
			float bsdfPdf;
		#ifdef BSDF_HAS_PURE_SPECULARS // see note in lambert.h
			const float3 sampledBSDF = EvaluateBSDF( shadingData, fN /* * faceDir */, T, D * -1.0f, L, bsdfPdf ) * ROUGHNESS;
		#else
			const float3 sampledBSDF = EvaluateBSDF( shadingData, fN /* * faceDir */, T, D * -1.0f, L, bsdfPdf );
		#endif
			if (bsdfPdf > 0)
			{
				// calculate potential contribution
				float3 contribution = throughput * sampledBSDF * lightColor * (NdotL / (pickProb * lightPdf + bsdfPdf));
				FIXNAN_FLOAT3( contribution );
				CLAMPINTENSITY;
				// add fire-and-forget shadow ray to the connections buffer
				const uint shadowRayIdx = atomicAdd( &counters->shadowRays, 1 ); // compaction
				connections[shadowRayIdx].O4 = make_float4( SafeOrigin( I, L, N, geometryEpsilon ), 0 );
				connections[shadowRayIdx].D4 = make_float4( L, dist - 2 * geometryEpsilon );
				potentials[shadowRayIdx] = make_float4( contribution, __int_as_float( pathIdx ) );
			}
		}
	}

	// cap at two diffuse bounces, or a maxium path length
	if (FLAGS & ENOUGH_BOUNCES || pathLength == MAXPATHLENGTH) return;

	// evaluate bsdf to obtain direction for next path segment
	float3 R;
	float newBsdfPdf;
	bool specular = false;
	const float3 bsdf = SampleBSDF( shadingData, fN, N, T, D * -1.0f, HIT_T, r4.z, r4.w, RandomFloat( seed ), R, newBsdfPdf, specular );
	if (newBsdfPdf < EPSILON || isnan( newBsdfPdf )) return;
	if (specular) FLAGS |= S_SPECULAR;

	// russian roulette
	const float p = ((FLAGS & S_SPECULAR) || ((FLAGS & S_BOUNCED) == 0)) ? 1 : SurvivalProbability( bsdf );
	if (p < RandomFloat( seed )) return; else throughput *= 1 / p;

	// write extension ray, with compaction. Note: nvcc will aggregate automatically, 
	// https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics 
	const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 ); // compact
	const uint packedNormal = PackNormal( fN * faceDir );
	if (!(FLAGS & S_SPECULAR)) FLAGS |= FLAGS & S_BOUNCED ? S_BOUNCEDTWICE : S_BOUNCED; else FLAGS |= S_VIASPECULAR;
	extensionRaysOut[extensionRayIdx].O4 = make_float4( SafeOrigin( I, R, N, geometryEpsilon ), 0 );
	extensionRaysOut[extensionRayIdx].D4 = make_float4( R, 1e34f );
	FIXNAN_FLOAT3( throughput );
	pathStateDataOut[extensionRayIdx * 2 + 0] = make_float4( throughput * bsdf * abs( dot( fN * faceDir, R ) ), __uint_as_float( data ) );
	pathStateDataOut[extensionRayIdx * 2 + 1] = make_float4( newBsdfPdf, __uint_as_float( packedNormal ), 0, 0 );
}

//  +-----------------------------------------------------------------------------+
//  |  shadeKernel                                                                |
//  |  Host-side access point for the shadeKernel code.                     LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void shade( const int pathCount, float4* accumulator, float4* features,
	const Ray4* extensionRays, const float4* pathStateData, const Intersection* hits,
	Ray4* extensionRaysOut, float4* pathStateDataOut,
	Ray4* connections, float4* potentials,
	const uint R0, const uint shift, const uint* blueNoise, const int pass,
	const int2 probePos, const int pathLength, const int scrwidth, const int scrheight,
	const ViewPyramid& view )
{
	const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 128 ) / 128, 1 );
	shadeKernel << < gridDim.x, 128 >> > (accumulator, features, scrwidth * scrheight,
		extensionRays, pathStateData, hits, extensionRaysOut, pathStateDataOut, connections, potentials, R0, shift, blueNoise, pass,
		probePos.x + scrwidth * probePos.y, pathLength, scrwidth, scrheight, view.spreadAngle,
		pathCount);
}

// EOF