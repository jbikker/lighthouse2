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
#define PATHIDX (data >> 8)

//  +-----------------------------------------------------------------------------+
//  |  RandomPointOnLightPNEE                                                     |
//  |  Selects a random point on a random light, with a probability steered by    |
//  |  guidance data.                                                       LH2'20|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float3 RandomPointOnLightPNEE( float r0, float r1, const float3& I, const float3& II, const float3& N, float& pickProb, float& lightPdf, float3& lightColor )
{
	bool debugFlag = (lightPdf == -999);
	// determine PNEE grid cell
	const float3 G = (II - PNEEmin) * PNEErext;
	const int gx = clamp( (int)G.x, 0, GRIDDIMX - 1 );
	const int gy = clamp( (int)G.y, 0, GRIDDIMY - 1 );
	const int gz = clamp( (int)G.z, 0, GRIDDIMZ - 1 );
	const int cellIdx = gx + (gy << BITS_TO_REPRESENT( GRIDDIMX - 1 )) + (gz << BITS_TO_REPRESENT( GRIDDIMX * GRIDDIMY - 1 ));
	// sample cell
	int lightIdx = 0;
	const uint4 g1 = guidance ? guidance[cellIdx * 4 + 0] : make_uint4( 0 );
	const uint4 g2 = guidance ? guidance[cellIdx * 4 + 1] : make_uint4( 0 );
	const uint4 g3 = guidance ? guidance[cellIdx * 4 + 2] : make_uint4( 0 );
	const uint4 g4 = guidance ? guidance[cellIdx * 4 + 3] : make_uint4( 0 );
	if (g1.x == 0)
	{
		// nothing here; sample lights uniformly
		lightIdx = (int)(r1 * (AREALIGHTCOUNT + POINTLIGHTCOUNT));
		pickProb = 1.0f / (AREALIGHTCOUNT + POINTLIGHTCOUNT);
	}
	else
	{
		if (debugFlag)
		{
			printf( "cellIdx: %i", cellIdx );
			printf( " [%1i:%4i,%1i:%4i,%1i:%4i,%1i:%4i,%1i:%4i,%1i:%4i,%1i:%4i,%1i:%4i] ==> ",
				g1.x & 0xfffff, g1.x >> 20, g1.y & 0xfffff, g1.y >> 20, g1.z & 0xfffff, g1.z >> 20, g1.w & 0xfffff, g1.w >> 20,
				g2.x & 0xfffff, g2.x >> 20, g2.y & 0xfffff, g2.y >> 20, g2.z & 0xfffff, g2.z >> 20, g2.w & 0xfffff, g2.w >> 20 );
		}
		pickProb = 1.0f; // safety net, so it doesn't go uninitialized.
		// calculate summed 
		if (r1 < CDFFLOOR)
		{
			// sample a random light
			r1 /= CDFFLOOR; // normalize r1 so we can reuse it
			lightIdx = (int)(r1 * (AREALIGHTCOUNT + POINTLIGHTCOUNT));
			pickProb = CDFFLOOR / (AREALIGHTCOUNT + POINTLIGHTCOUNT);
			// account for probability of sampling this light using the other method
			if ((g1.x & 0xfffff) == lightIdx) pickProb += (g1.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g1.y & 0xfffff) == lightIdx) pickProb += (g1.y >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g1.z & 0xfffff) == lightIdx) pickProb += (g1.z >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g1.w & 0xfffff) == lightIdx) pickProb += (g1.w >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g2.x & 0xfffff) == lightIdx) pickProb += (g2.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g2.y & 0xfffff) == lightIdx) pickProb += (g2.y >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g2.z & 0xfffff) == lightIdx) pickProb += (g2.z >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g2.w & 0xfffff) == lightIdx) pickProb += (g2.w >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g3.x & 0xfffff) == lightIdx) pickProb += (g3.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g3.y & 0xfffff) == lightIdx) pickProb += (g3.y >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g3.z & 0xfffff) == lightIdx) pickProb += (g3.z >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g3.w & 0xfffff) == lightIdx) pickProb += (g3.w >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g4.x & 0xfffff) == lightIdx) pickProb += (g4.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g4.y & 0xfffff) == lightIdx) pickProb += (g4.y >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g4.z & 0xfffff) == lightIdx) pickProb += (g4.z >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if ((g4.w & 0xfffff) == lightIdx) pickProb += (g4.w >> 20) * ((1 - CDFFLOOR) / 4096.0f);
			if (debugFlag) printf( "UNIFORM, prob: %5.3f\n", pickProb );
		}
		else
		{
			// sample one of the important lights
			r1 = (r1 - CDFFLOOR) / (1 - CDFFLOOR); // normalize r1 so we can reuse it
			int s = (int)(r1 * 4096.0f);
			int total = g1.x >> 20, ti;
			if (total >= s) lightIdx = g1.x & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (total / 4096.0f); else
			{
				ti = g1.y >> 20, total += ti; if (total >= s) lightIdx = g1.y & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
				{
					ti = g1.z >> 20, total += ti; if (total >= s) lightIdx = g1.z & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
					{
						ti = g1.w >> 20, total += ti; if (total >= s) lightIdx = g1.w & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
						{
							ti = g2.x >> 20, total += ti; if (total >= s) lightIdx = g2.x & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
							{
								ti = g2.y >> 20, total += ti; if (total >= s) lightIdx = g2.y & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
								{
									ti = g2.z >> 20, total += ti; if (total >= s) lightIdx = g2.z & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
									{
										ti = g2.w >> 20, total += ti; if (total >= s) lightIdx = g2.w & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
										{
											ti = g3.x >> 20, total += ti; if (total >= s) lightIdx = g3.x & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
											{
												ti = g3.y >> 20, total += ti; if (total >= s) lightIdx = g3.y & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
												{
													ti = g3.z >> 20, total += ti; if (total >= s) lightIdx = g3.z & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
													{
														ti = g3.w >> 20, total += ti; if (total >= s) lightIdx = g3.w & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
														{
															ti = g4.x >> 20, total += ti; if (total >= s) lightIdx = g4.x & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
															{
																ti = g4.y >> 20, total += ti; if (total >= s) lightIdx = g4.y & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
																{
																	ti = g4.z >> 20, total += ti; if (total >= s) lightIdx = g4.z & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f); else
																	{
																		ti = g4.w >> 20, lightIdx = g4.w & 0xfffff, pickProb = (1.0f - CDFFLOOR) * (ti / 4096.0f);
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
			pickProb += CDFFLOOR / (AREALIGHTCOUNT + POINTLIGHTCOUNT);
			if (debugFlag) printf( "S: %4i; prob: %5.3f\n", s, pickProb );
		}
	}
	// sample selected light
	if (lightIdx < AREALIGHTCOUNT)
	{
		float3 bary = RandomBarycentrics( r0 );
		const CoreLightTri4& light = (const CoreLightTri4&)areaLights[lightIdx];
		const float4 V0 = light.data3;				// vertex0
		const float4 V1 = light.data4;				// vertex1
		const float4 V2 = light.data5;				// vertex2
		lightColor = make_float3( light.data2 );	// radiance
		const float4 LN = light.data1;				// N
		const float3 P = make_float3( bary.x * V0 + bary.y * V1 + bary.z * V2 );
		float3 L = I - P; // reversed: from light to intersection point
		const float sqDist = dot( L, L );
		L = normalize( L );
		const float LNdotL = L.x * LN.x + L.y * LN.y + L.z * LN.z;
		const float reciSolidAngle = sqDist / (LN.w * LNdotL); // LN.w contains area
		lightPdf = (LNdotL > 0 && dot( L, N ) < 0) ? reciSolidAngle : 0;
		return P;
	}
	else
	{
		const CorePointLight4& light = (const CorePointLight4&)pointLights[lightIdx - AREALIGHTCOUNT];
		const float3 P = make_float3( light.data0 );	// position
		lightColor = make_float3( light.data1 );	// radiance
		const float3 L = P - I;
		const float sqDist = dot( L, L );
		lightPdf = dot( L, N ) > 0 ? sqDist : 0;
		return P;
	}
}

LH2_DEVFUNC float LightPickProbPNEE( int idx, const float3& O, const float3& N, const float3& I, uint& seed )
{
	// jitter ray origin
	float3 jitter = make_float3( 1.0f / PNEErext.x, 1.0f / PNEErext.y, 1.0f / PNEErext.z );
	float3 II = make_float3(
		I.x + jitter.x * (RandomFloat( seed ) - 0.5f),
		I.y + jitter.y * (RandomFloat( seed ) - 0.5f),
		I.z + jitter.z * (RandomFloat( seed ) - 0.5f) );
	// determine PNEE grid cell for ray origin
	const float3 G = (II - PNEEmin) * PNEErext;
	const int gx = clamp( (int)G.x, 0, GRIDDIMX - 1 );
	const int gy = clamp( (int)G.y, 0, GRIDDIMY - 1 );
	const int gz = clamp( (int)G.z, 0, GRIDDIMZ - 1 );
	const int cellIdx = gx + (gy << BITS_TO_REPRESENT( GRIDDIMX - 1 )) + (gz << BITS_TO_REPRESENT( GRIDDIMX * GRIDDIMY - 1 ));
	// what are the odds of selecting the specified light
	float pickProb = CDFFLOOR / (AREALIGHTCOUNT + POINTLIGHTCOUNT);
	if (guidance)
	{
		const uint4 g1 = guidance[cellIdx * 4 + 0];
		const uint4 g2 = guidance[cellIdx * 4 + 1];
		const uint4 g3 = guidance[cellIdx * 4 + 2];
		const uint4 g4 = guidance[cellIdx * 4 + 3];
		if ((g1.x & 0xfffff) == idx) pickProb += (g1.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g1.y & 0xfffff) == idx) pickProb += (g1.y >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g1.z & 0xfffff) == idx) pickProb += (g1.z >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g1.w & 0xfffff) == idx) pickProb += (g1.w >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g2.x & 0xfffff) == idx) pickProb += (g2.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g2.y & 0xfffff) == idx) pickProb += (g2.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g2.z & 0xfffff) == idx) pickProb += (g2.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g2.w & 0xfffff) == idx) pickProb += (g2.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g3.x & 0xfffff) == idx) pickProb += (g3.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g3.y & 0xfffff) == idx) pickProb += (g3.y >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g3.z & 0xfffff) == idx) pickProb += (g3.z >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g3.w & 0xfffff) == idx) pickProb += (g3.w >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g4.x & 0xfffff) == idx) pickProb += (g4.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g4.y & 0xfffff) == idx) pickProb += (g4.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g4.z & 0xfffff) == idx) pickProb += (g4.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
		if ((g4.w & 0xfffff) == idx) pickProb += (g4.x >> 20) * ((1 - CDFFLOOR) / 4096.0f);
	}
	return pickProb;
}

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
		float3 contribution = throughput * make_float3( SampleSkydome( -worldToSky.TransformVector( D ), pathLength ) ) * (1.0f / bsdfPdf);
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

	// initialize seed based on pixel index
	uint seed = WangHash( pathIdx * 17 + R0 /* well-seeded xor32 is all you need */ );

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
				const float pickProb = LightPickProbPNEE( tri.ltriIdx, RAY_O, lastN /* the N at the previous vertex */, I, seed );
				if ((bsdfPdf + lightPdf * pickProb) > 0) contribution = throughput * shadingData.color * (1.0f / (bsdfPdf + lightPdf * pickProb));
			}
			CLAMPINTENSITY;
			FIXNAN_FLOAT3( contribution );
			accumulator[pixelIdx] += make_float4( contribution, 0 );
		}
		return;
	}

	// detect specular surfaces
	if (ROUGHNESS <= 0.001f || TRANSMISSION > 0.999f) FLAGS |= S_SPECULAR; /* detect pure speculars; skip NEE for these */ else FLAGS &= ~S_SPECULAR;

	// normal alignment for backfacing polygons
	const float faceDir = (dot( D, N ) > 0) ? -1 : 1;
	if (faceDir == 1) shadingData.transmittance = make_float3( 0 );

	// apply postponed bsdf pdf
	throughput *= 1.0f / bsdfPdf;

	// next event estimation: connect eye path to light
	if (!(FLAGS & S_SPECULAR)) // skip for specular vertices
	{
		float r0, r1, pickProb, lightPdf = 0;
		if (sampleIdx < 2)
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
		lightPdf = 0; // (pixelIdx == probePixelIdx && pathLength == 1) ? -999 : 0;
		float3 lightColor, L;
		if (pixelIdx % SCRWIDTH < SCRWIDTH / 2)
		{
			float3 jitter = make_float3( 1.0f / PNEErext.x, 1.0f / PNEErext.y, 1.0f / PNEErext.z );
			float3 II = make_float3(
				I.x + jitter.x * (RandomFloat( seed ) * 0.5f - 0.25f),
				I.y + jitter.y * (RandomFloat( seed ) * 0.5f - 0.25f),
				I.z + jitter.z * (RandomFloat( seed ) * 0.5f - 0.25f) );
			L = RandomPointOnLightPNEE( r0, r1, I, II, fN * faceDir, pickProb, lightPdf, lightColor ) - I;
		}
		else
		{
			L = RandomPointOnLight( r0, r1, I, fN * faceDir, pickProb, lightPdf, lightColor ) - I;
		}
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
				connections[shadowRayIdx] = make_float4( SafeOrigin( I, L, N, geometryEpsilon ), 0 ); // O4
				connections[shadowRayIdx + stride * 2] = make_float4( L, dist - 2 * geometryEpsilon ); // D4
				connections[shadowRayIdx + stride * 2 * 2] = make_float4( contribution, __int_as_float( pixelIdx ) ); // E4
			}
		}
	}

	// cap at two diffuse bounces, or a maxium path length
	if (FLAGS & ENOUGH_BOUNCES || pathLength == MAXPATHLENGTH) return;

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

	// russian roulette (TODO: greatly increases variance.)
	const float p = ((FLAGS & S_SPECULAR) || ((FLAGS & S_BOUNCED) == 0)) ? 1 : SurvivalProbability( bsdf );
	if (p < RandomFloat( seed )) return; else throughput *= 1 / p;

	// write extension ray
	const uint extensionRayIdx = atomicAdd( &counters->extensionRays, 1 ); // compact
	const uint packedNormal = PackNormal( fN * faceDir );
	if (!(FLAGS & S_SPECULAR)) FLAGS |= FLAGS & S_BOUNCED ? S_BOUNCEDTWICE : S_BOUNCED; else FLAGS |= S_VIASPECULAR;
	pathStates[extensionRayIdx] = make_float4( SafeOrigin( I, R, N, geometryEpsilon ), __uint_as_float( FLAGS ) );
	pathStates[extensionRayIdx + stride] = make_float4( R, __uint_as_float( packedNormal ) );
	FIXNAN_FLOAT3( throughput );
	pathStates[extensionRayIdx + stride * 2] = make_float4( throughput * bsdf * abs( dot( fN, R ) ), newBsdfPdf );
}

//  +-----------------------------------------------------------------------------+
//  |  shade                                                                      |
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

//  +-----------------------------------------------------------------------------+
//  |  ProcessPhotonHits                                                          |
//  |  Adjust photon intensities using attenuation and cos theta.           LH2'19|
//  +-----------------------------------------------------------------------------+
__global__ void ProcessPhotonHitsKernel( float4* photonData, const uint photonCount )
{
	// respect boundaries
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= photonCount) return;

	// gather hit information
	const float t = photonData[jobIndex * 3 + 0].w;
	if (t >= 1e30f) return; // photon left the scene
	const uint instIdx = __float_as_uint( photonData[jobIndex * 3 + 2].y );
	const uint primIdx = __float_as_uint( photonData[jobIndex * 3 + 2].z );

	// obtain triangle normal
	const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[instIdx].triangles;
	const CoreTri4& tri = instanceTriangles[primIdx];
	const float3 edge1 = make_float3( tri.vertex[1] - tri.vertex[0] );
	const float3 edge2 = make_float3( tri.vertex[2] - tri.vertex[0] );
	float3 N = normalize( cross( edge1, edge2 ) );
	N = normalize( N.x * make_float3( instanceDescriptors[instIdx].invTransform.A ) +
		N.y * make_float3( instanceDescriptors[instIdx].invTransform.B ) +
		N.z * make_float3( instanceDescriptors[instIdx].invTransform.C ) );

	// scale photon power by 1/dist^2
	photonData[jobIndex * 3 + 2].x *= 1.0f / (t * t);

	// scale photon power by N dot L
	const float3 L = make_float3( photonData[jobIndex * 3 + 1] );
	photonData[jobIndex * 3 + 2].x *= abs( dot( N, L ) );
}

__host__ void ProcessPhotonHits( float4* photonData, const uint photonCount )
{
	const dim3 gridDim( NEXTMULTIPLEOF( photonCount, 128 ) / 128, 1 ), blockDim( 128, 1 );
	ProcessPhotonHitsKernel << <gridDim.x, 128 >> > (photonData, photonCount);
}

// EOF