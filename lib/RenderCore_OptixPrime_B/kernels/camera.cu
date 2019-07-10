/* camera.cu - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "noerrors.h"

//  +-----------------------------------------------------------------------------+
//  |  RandomPointOnLens                                                          |
//  |  Generate a random point on the lens.                                 LH2'19|
//  +-----------------------------------------------------------------------------+
__inline __device__ float3 RandomPointOnLens( const float r0, float r1, const float3 pos, const float aperture, const float3 right, const float3 up )
{
	const float blade = (int)(r0 * 9);
	float r2 = (r0 - blade * (1.0f / 9.0f)) * 9.0f;
	float x1, y1, x2, y2;
	__sincosf( blade * PI / 4.5f, &x1, &y1 );
	__sincosf( (blade + 1.0f) * PI / 4.5f, &x2, &y2 );
	if ((r1 + r2) > 1) r1 = 1.0f - r1, r2 = 1.0f - r2;
	const float xr = x1 * r1 + x2 * r2;
	const float yr = y1 * r1 + y2 * r2;
	return pos + aperture * (right * xr + up * yr);
}

//  +-----------------------------------------------------------------------------+
//  |  generateEyeRaysKernel                                                      |
//  |  Generate primary rays, to be traced by Optix Prime.                  LH2'19|
//  +-----------------------------------------------------------------------------+
__device__ void generateEyeRaysKernel( const uint rayIdx, Ray4* rayBuffer, float4* pathStateData,
	const uint R0, const int sampleBase,
	const float3 pos, const float3 right, const float3 up, const float aperture,
	const float3 p1, const int4 screenParams )
{
	// get pixel coordinate
	const int scrhsize = screenParams.x & 0xffff;
	const int scrvsize = screenParams.x >> 16;
	const uint tileIdx = rayIdx >> 8;
	const uint xtiles = scrhsize / 16;
	const uint tilex = tileIdx % xtiles, tiley = tileIdx / xtiles;
	const uint x_in_tile = (rayIdx & 15);
	const uint y_in_tile = (rayIdx & 255) >> 4;
	uint x = tilex * 16 + x_in_tile, y = tiley * 16 + y_in_tile, sampleIndex = sampleBase + y / scrvsize;
	y %= scrvsize;
	// get random numbers
	float3 posOnPixel, posOnLens;
	// depth of field camera for no filter
	uint seed = WangHash( rayIdx + R0 );
	float r0 = RandomFloat( seed ), r1 = RandomFloat( seed );
	float r2 = RandomFloat( seed ), r3 = RandomFloat( seed );
	posOnPixel = p1 + ((float)x + r0) * (right / (float)scrhsize) + ((float)y + r1) * (up / (float)scrvsize);
	posOnLens = RandomPointOnLens( r2, r3, pos, aperture, right, up );
	const float3 rayDir = normalize( posOnPixel - posOnLens );
	// initialize path state
	rayBuffer[rayIdx].O4 = make_float4( posOnLens, geometryEpsilon );
	rayBuffer[rayIdx].D4 = make_float4( rayDir, 1e34f );
	pathStateData[rayIdx * 2 + 0] = make_float4( 1, 1, 1, __uint_as_float( ((x + (y + (sampleIndex - sampleBase) * scrvsize) * scrhsize) << 8) + 1 /* S_SPECULAR */ ) );
	pathStateData[rayIdx * 2 + 1] = make_float4( 1, 0, 0, 0 );
}

//  +-----------------------------------------------------------------------------+
//  |  generateEyeRaysPersistent                                                  |
//  |  Persistent kernel for generating primary rays.                       LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  void __launch_bounds__( 256 /* max block size */, 1 /* min blocks per sm */ )
generateEyeRaysPersistent( int pathCount, Ray4* rayBuffer, float4* pathStateData,
	const uint R0, const int pass,
	const float3 pos, const float3 right, const float3 up, const float aperture, const float3 p1,
	const int4 screenParams )
{
	__shared__ volatile int baseIdx[32];
	int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
	__syncthreads();
	while (1)
	{
		if (lane == 0) baseIdx[warp] = atomicAdd( &counters->generated, 32 );
		int jobIndex = baseIdx[warp] + lane;
		if (__all_sync( THREADMASK, jobIndex >= pathCount )) break; // need to do the path with all threads in the warp active
		if (jobIndex < pathCount) generateEyeRaysKernel( jobIndex,
			rayBuffer, pathStateData,
			R0, pass,
			pos, right, up, aperture, p1,
			screenParams );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  generateEyeRays                                                            |
//  |  Entry point for the persistent generateEyeRays kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void generateEyeRays( int smcount, Ray4* rayBuffer, float4* pathStateData,
	const uint R0, const int pass,
	const float aperture, const float3 camPos, const float3 right, const float3 up, const float3 p1,
	const int4 screenParams )
{
	const int scrwidth = screenParams.x & 0xffff;
	const int scrheight = screenParams.x >> 16;
	const int scrspp = screenParams.y & 255;
	const int pathCount = scrwidth * scrheight * scrspp;
	InitCountersForExtend_Kernel << <1, 32 >> > (pathCount);
	generateEyeRaysPersistent << < smcount, 256 >> > (pathCount, rayBuffer, pathStateData, R0, pass, camPos, right, up, aperture, p1, screenParams);
}

// EOF