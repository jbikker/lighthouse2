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
//  |  getPathInfo                                                          |
//  |  get the path info from path_s_t_type_pass.                     LH2'19|
//  +-----------------------------------------------------------------------------+
/*
	pass: the sample number of current pixel, [0, 2^19-1]
	s   : the length of eye path, [0, 31]
	t   : the length of light path, [0, 31]
	type: the type of extend path, 0: create new path; 1: extend eye path; 2: extend light path
*/
LH2_DEVFUNC void getPathInfo( const uint& path_s_t_type_pass, uint& pass, uint& s, uint& t, uint& type )
{
	pass = (path_s_t_type_pass & 524287);
	type = (path_s_t_type_pass >> 19);
	t = (type >> 3);
	s = (t >> 5);
	type = (type & 7);
	t = (t & 31);
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPosKernel                                                      |
//  |  Generate the first vertex of the light path including pos and direction.                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256, 1 )
void constructionLightPosKernel( int smcount,
	BiPathState* pathStateData, const uint R0, const uint* blueNoise, const int4 screenParams,
	Ray4* randomWalkRays, float4* accumulatorOnePass,
	const int probePixelIdx, uint* constructEyeBuffer )
{
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= smcount) return;

	float data = __uint_as_float( ((uint)jobIndex << 8) + 1 /* S_SPECULAR */ );
	float data_L = __uint_as_float( 1 /* S_SPECULAR */ );

	const int scrhsize = screenParams.x & 0xffff;
	const int scrvsize = screenParams.x >> 16;
	const uint x = jobIndex % scrhsize;
	uint y = jobIndex / scrhsize;
	y %= scrvsize;

	uint path_s_t_type_pass = __float_as_uint( pathStateData[jobIndex].eye_normal.w );

	uint s = 0;
	uint t = 1;
	uint type = NEW_PATH;
	uint sampleIdx = path_s_t_type_pass & 524287;//2^19-1

	float r0, r1, r2, r3;

	uint seed = WangHash( jobIndex + R0 );
	uint idxS = sampleIdx * MAX_LIGHTPATH;
	/* blue Noise is bad here*/
	if (BLUENOISER_ON && idxS < 256) // sampleIdx < 256
	{
		r0 = blueNoiseSampler( blueNoise, x, y, idxS, 0 );
		r1 = blueNoiseSampler( blueNoise, x, y, idxS, 1 );
		// r2 = blueNoiseSampler( blueNoise, x, y, sampleIdx, 2 );
		// r3 = blueNoiseSampler( blueNoise, x, y, sampleIdx, 3 );
		r2 = RandomFloat( seed );
		r3 = RandomFloat( seed );
	}
	else
	{
		r0 = RandomFloat( seed );
		r1 = RandomFloat( seed );
		r2 = RandomFloat( seed );
		r3 = RandomFloat( seed );
	}

	float3 normal, throughput, lightDir;
	float lightPdf, pdfPos, pdfDir;

	// get the pos and direction of the light source with pos pdf and dir pdf
	float3 pos = Sample_Le( r0, r1, r2, r3, normal, lightDir, throughput, lightPdf, pdfPos, pdfDir );

	// PBR book equation [16.15]
	float3 beta = throughput * fabs( dot( normal, lightDir ) ) / (lightPdf * pdfPos * pdfDir);

	// area pdf
	float light_p = lightPdf * pdfPos;
	// dL Recursive MIS Equation [14]
	float dL = NKK / light_p;
	float light_pdf_solid = pdfDir;

	int randomWalkRayIdx = -1;
	if (t < MAX_LIGHTPATH)
	{
		randomWalkRayIdx = atomicAdd( &counters->randomWalkRays, 1 );
		randomWalkRays[randomWalkRayIdx].O4 = make_float4( SafeOrigin( pos, lightDir, normal, geometryEpsilon ), 0 );
		randomWalkRays[randomWalkRayIdx].D4 = make_float4( lightDir, 1e34f );
	}


	pathStateData[jobIndex].data0 = make_float4( throughput, dL );
	pathStateData[jobIndex].data1 = make_float4( beta, light_p );
	pathStateData[jobIndex].data2 = make_float4( pos, light_pdf_solid );
	pathStateData[jobIndex].data3 = make_float4( lightDir, __int_as_float( randomWalkRayIdx ) );

	pathStateData[jobIndex].light_normal = make_float4( normal, data );
	pathStateData[jobIndex].pre_light_dir.w = data_L;

	sampleIdx++;
	path_s_t_type_pass = (s << 27) + (t << 22) + (type << 19) + sampleIdx;

	pathStateData[jobIndex].eye_normal.w = __uint_as_float( path_s_t_type_pass );

	const uint eyeIdx = atomicAdd( &counters->constructionEyePos, 1 );
	constructEyeBuffer[eyeIdx] = jobIndex;
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void constructionLightPos( int smcount,
	BiPathState* pathStateData, const uint R0, const uint* blueNoise, const int4 screenParams,
	Ray4* randomWalkRays, float4* accumulatorOnePass,
	const int probePixelIdx, uint* constructEyeBuffer )
{
	const dim3 gridDim( NEXTMULTIPLEOF( smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
	constructionLightPosKernel << < gridDim.x, 256 >> > (smcount,
		pathStateData, R0, blueNoise, screenParams, randomWalkRays,
		accumulatorOnePass, probePixelIdx, constructEyeBuffer);
}

// EOF