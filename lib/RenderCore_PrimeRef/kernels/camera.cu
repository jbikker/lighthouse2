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
__global__  __launch_bounds__( 256 /* max block size */, 1 /* min blocks per sm */ )
void generateEyeRaysKernel( Ray4* rayBuffer, float4* pathStateData,
	const uint R0, const int pass,
	const float3 pos, const float3 right, const float3 up, const float aperture,
	const float3 p1, const int4 screenParams, const int jobCount )
{
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= jobCount) return;
	// get pixel coordinate
	const int scrhsize = screenParams.x & 0xffff;
	const int scrvsize = screenParams.x >> 16;
	const uint x = jobIndex % scrhsize;
	uint y = jobIndex / scrhsize;
	const uint sampleIndex = pass + y / scrvsize;
	y %= scrvsize;
	// get random numbers
	float3 posOnPixel, posOnLens;
	// depth of field camera for no filter
	uint seed = WangHash( jobIndex + R0 );
	const float r0 = RandomFloat( seed ), r1 = RandomFloat( seed );
	const float r2 = RandomFloat( seed ), r3 = RandomFloat( seed );
	posOnPixel = p1 + ((float)x + r0) * (right / (float)scrhsize) + ((float)y + r1) * (up / (float)scrvsize);
	posOnLens = RandomPointOnLens( r2, r3, pos, aperture, right, up );
	const float3 rayDir = normalize( posOnPixel - posOnLens );
	// initialize path state
	rayBuffer[jobIndex].O4 = make_float4( posOnLens, geometryEpsilon );
	rayBuffer[jobIndex].D4 = make_float4( rayDir, 1e34f );
	pathStateData[jobIndex * 2 + 0] = make_float4( 1, 1, 1, __uint_as_float( ((x + (y + (sampleIndex - pass) * scrvsize) * scrhsize) << 8) + 1 /* S_SPECULAR */ ) );
	pathStateData[jobIndex * 2 + 1] = make_float4( 1, 0, 0, 0 );
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
	const dim3 gridDim( NEXTMULTIPLEOF( pathCount, 256 ) / 256, 1 ), blockDim( 256, 1 );
	generateEyeRaysKernel << < gridDim.x, 256 >> > (rayBuffer, pathStateData, R0, pass, camPos, right, up, aperture, p1, screenParams, pathCount);
}

// EOF