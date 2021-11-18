/* camera.h - Copyright 2019/2021 Utrecht University

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
//  |  generateAdaptiveSamplesKernel                                              |
//  |  Adaptively generate primary rays, to be traced by Optix Prime.       LH2'20|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256 /* max block size */, 1 /* min blocks per sm */ )
void generateAdaptiveSamplesKernel( const float* deviation, float4* pathState,
	const uint R0, const uint shift, const uint pass, const uint* blueNoise,
	const float3 pos, const float3 right, const float3 up, const float aperture,
	const float3 p1, const float distortion, 
	const uint scrwidth, const uint scrheight, const uint spp, const int threadCount )
{
	int threadIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (threadIndex >= threadCount) return;
	// get pixel coordinate
	const uint sx = threadIndex % scrwidth;
	uint sy = threadIndex / scrwidth;
	const uint S = sy / scrheight + 1;
	sy %= scrheight;
	// see if this pixel needs sample S
	const float samplesNeeded = deviation[sx + sy * scrwidth];
	if (S > samplesNeeded) return;
	// generate primary ray
	float4 r4;
	const uint sampleIdx = S + spp + pass;
	if (sampleIdx < 64)
	{
		r4 = blueNoiseSampler4( blueNoise, sx & 127, sy & 127, sampleIdx, 0 );
	}
	else
	{
		uint seed = WangHash( threadIndex + R0 );
		r4.x = RandomFloat( seed ), r4.y = RandomFloat( seed );
		r4.z = RandomFloat( seed ), r4.w = RandomFloat( seed );
	}
	const float3 posOnLens = RandomPointOnLens( r4.x, r4.z, pos, aperture, right, up );
	const float3 posOnPixel = RayTarget( sx, sy, r4.y, r4.w, make_int2( scrwidth, scrheight ), distortion, p1, right, up );
	const float3 rayDir = normalize( posOnPixel - posOnLens );
	// initialize path state
	const uint maxRayIdx = scrwidth * scrheight * spp;
	const uint primaryRayIdx = atomicAdd( &counters->activePaths, 1 ); // compact
	if (primaryRayIdx >= maxRayIdx) { /* bad; we have no room */ return; }
	const uint stride = scrwidth * scrheight * spp;
	pathState[primaryRayIdx] = make_float4( posOnLens, __uint_as_float( ((sx + sy * scrwidth + (S % spp) * scrwidth * scrheight) << 6) + 1 ) );
	pathState[primaryRayIdx + stride] = make_float4( rayDir, 0 );
}

//  +-----------------------------------------------------------------------------+
//  |  generateAdaptiveSamples                                                    |
//  |  Entry point for the persistent generateEyeRaysAdaptive kernel.       LH2'20|
//  +-----------------------------------------------------------------------------+
__host__ void generateAdaptiveSamples( const float* deviation, float4* pathStateData, const uint R0, const uint shift, 
	const uint pass, const uint* blueNoise, const ViewPyramid& view, const uint w, const uint h, const uint spp )
{
	// We will spawn 24 * pixelCount threads, which together create up to 24 additional samples per pixel.
	// Per pixel, each invocation S will decide if the pixel needs S or more additional samples.
	// This way, rays generated for S and S + 1 are maximally apart in thread space.
	const int threadCount = w * h * 24;
	const dim3 gridDim( NEXTMULTIPLEOF( threadCount, 256 ) / 256, 1 ), blockDim( 256, 1 );
	generateAdaptiveSamplesKernel << < gridDim.x, 256 >> > (deviation, pathStateData, R0, shift, pass, blueNoise,
		view.pos, view.p2 - view.p1, view.p3 - view.p1, view.aperture, view.p1, view.distortion, w, h, spp, threadCount);
}

// EOF