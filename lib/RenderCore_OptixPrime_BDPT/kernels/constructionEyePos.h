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
//  |  constructionLightPosKernel                                                      |
//  |  Generate the first vertex of the light path including pos and direction.                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256, 1 )
void constructionEyePosKernel( uint* constructEyeBuffer, BiPathState* pathStateData,
	Ray4* visibilityRays, Ray4* randomWalkRays,
	const uint R0, const float aperture, const float imgPlaneSize, const float3 cam_pos,
	const float3 right, const float3 up, const float3 forward, const float3 p1,
	const int4 screenParams, const uint* blueNoise )
{
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= counters->constructionEyePos) return;

	int jobIndex = constructEyeBuffer[gid];

	float data = __uint_as_float( ((uint)jobIndex << 8) + 1 /* S_SPECULAR */ );

	uint path_s_t_type_pass = __float_as_uint( pathStateData[jobIndex].eye_normal.w );

	uint pass, type, t, s;
	getPathInfo( path_s_t_type_pass, pass, s, t, type );

	const int scrhsize = screenParams.x & 0xffff;
	const int scrvsize = screenParams.x >> 16;
	const uint x = jobIndex % scrhsize;
	uint y = jobIndex / scrhsize;
	y %= scrvsize;

	uint sampleIdx = pass * MAX_LIGHTPATH + t - 1;

	float3 posOnPixel, posOnLens;

	// depth of field camera for no filter
	float r0, r1, r2, r3;

	if (BLUENOISER_ON && sampleIdx < 256)
	{
		r0 = blueNoiseSampler( blueNoise, x, y, sampleIdx, 2 );
		r1 = blueNoiseSampler( blueNoise, x, y, sampleIdx, 3 );
		r2 = blueNoiseSampler( blueNoise, x, y, sampleIdx, 4 );
		r3 = blueNoiseSampler( blueNoise, x, y, sampleIdx, 5 );
	}
	else
	{
		uint seed = WangHash( jobIndex * 17 + R0 );
		r0 = RandomFloat( seed ), r1 = RandomFloat( seed );
		r2 = RandomFloat( seed ), r3 = RandomFloat( seed );
	}

	posOnPixel = p1 + ((float)x + r0) * (right / (float)scrhsize) + ((float)y + r1) * (up / (float)scrvsize);
	posOnLens = RandomPointOnLens( r2, r3, cam_pos, aperture, right, up );
	const float3 rayDir = normalize( posOnPixel - posOnLens );

	const uint randomWalkRayIdx = atomicAdd( &counters->randomWalkRays, 1 );
	randomWalkRays[randomWalkRayIdx].O4 = make_float4( posOnLens, EPSILON );
	randomWalkRays[randomWalkRayIdx].D4 = make_float4( rayDir, 1e34f );

	float4 value = make_float4( make_float3( 1.0f ), 0.0f );
	float3 normal = normalize( forward );
	float cosTheta = fabs( dot( normal, rayDir ) );

	float eye_pdf_solid = 1.0f / (imgPlaneSize * cosTheta * cosTheta * cosTheta);

	pathStateData[jobIndex].data4 = value;
	pathStateData[jobIndex].data5 = value;
	pathStateData[jobIndex].data6 = make_float4( posOnLens, eye_pdf_solid );
	pathStateData[jobIndex].data7 = make_float4( rayDir, __int_as_float( randomWalkRayIdx ) );
	pathStateData[jobIndex].eye_normal = make_float4( normal, 0.0f );
	pathStateData[jobIndex].light_normal.w = data;

	// when type == EXTEND_LIGHTPATH, reset the length of eye path
	s = 0;

	path_s_t_type_pass = (s << 27) + (t << 22) + (type << 19) + pass;
	pathStateData[jobIndex].eye_normal.w = __uint_as_float( path_s_t_type_pass );
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void constructionEyePos( int smcount, uint* constructEyeBuffer, BiPathState* pathStateBuffer,
	Ray4* visibilityRays, Ray4* randomWalkRays,
	const uint R0, const float aperture, const float imgPlaneSize, const float3 camPos,
	const float3 right, const float3 up, const float3 forward, const float3 p1,
	const int4 screenParams, const uint* blueNoise )
{
	const dim3 gridDim( NEXTMULTIPLEOF( smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
	constructionEyePosKernel << < gridDim.x, 256 >> > (constructEyeBuffer,
		pathStateBuffer, visibilityRays, randomWalkRays,
		R0, aperture, imgPlaneSize, camPos, right, up, forward, p1, screenParams, blueNoise);
}

// EOF