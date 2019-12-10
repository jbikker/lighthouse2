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

#define INSTANCEIDX (prim >> 20)
#define HIT_U hitData.x
#define HIT_V hitData.y
#define HIT_T hitData.w

//  +-----------------------------------------------------------------------------+
//  |  generateEyeRaysKernel                                                      |
//  |  Generate primary rays, to be traced by Optix Prime.                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256, 1 )
void connectionPathKernel( int smcount, BiPathState* pathStateData,
	const Intersection* randomWalkHitBuffer,
	float4* accumulatorOnePass,
	const int4 screenParams,
	uint* constructEyeBuffer, uint* eyePathBuffer )
{
	int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= smcount) return;

	uint data = __float_as_uint( pathStateData[jobIndex].light_normal.w );
	int contribIdx = (data >> 8);


	uint path_s_t_type_pass = __float_as_uint( pathStateData[jobIndex].eye_normal.w );

	uint pass, type, t, s;
	getPathInfo( path_s_t_type_pass, pass, s, t, type );

	if (type == DEAD)
	{
		return;
	}

	const float3 empty_color = make_float3( 0.0f );
	float misWeight = 0.0f;

	int eye_hit = -1;
	int eye_hit_idx = __float_as_int( pathStateData[jobIndex].data7.w );
	float eye_pdf = pathStateData[jobIndex].data6.w;

	if (eye_pdf < EPSILON || isnan( eye_pdf ))
	{
		eye_hit = -1;
		pathStateData[jobIndex].data7.w = __int_as_float( -1 );
	}
	else if (eye_hit_idx > -1)
	{
		const Intersection hd = randomWalkHitBuffer[eye_hit_idx];

		eye_hit = hd.triid;

		const float4 hitData = make_float4( hd.u, hd.v, __int_as_float( hd.triid + (hd.triid == -1 ? 0 : (hd.instid << 20)) ), hd.t );
		pathStateData[jobIndex].eye_intersection = hitData;

		pathStateData[jobIndex].data7.w = __int_as_float( -1 );
	}

	int light_hit = -1;
	int light_hit_idx = __float_as_int( pathStateData[jobIndex].data3.w );
	float light_pdf_test = pathStateData[jobIndex].data2.w;
	if (light_pdf_test < EPSILON || isnan( light_pdf_test ))
	{
		light_hit = -1;
		pathStateData[jobIndex].data3.w = __int_as_float( -1 );
	}
	else if (light_hit_idx > -1)
	{
		const Intersection hd = randomWalkHitBuffer[light_hit_idx];
		light_hit = hd.triid;
		const float4 hitData = make_float4( hd.u, hd.v, __int_as_float( hd.triid + (hd.triid == -1 ? 0 : (hd.instid << 20)) ), hd.t );

		pathStateData[jobIndex].light_intersection = hitData;
		pathStateData[jobIndex].data3.w = __int_as_float( -1 );
	}
	else
	{
		const float4 hitData = pathStateData[jobIndex].light_intersection;

		const int prim = __float_as_int( hitData.z );
		const int primIdx = prim == -1 ? prim : (prim & 0xfffff);

		light_hit = primIdx;
	}

	if (eye_hit == -1 && type < EXTEND_LIGHTPATH)
	{
		if (!(eye_pdf < EPSILON || isnan( eye_pdf )))
		{
			float3 hit_dir = make_float3( pathStateData[jobIndex].data7 );
			float3 background = make_float3( SampleSkydome( hit_dir, s + 1 ) );

			// hit miss : beta 
			float3 throughput = make_float3( pathStateData[jobIndex].data5 );

			float3 contribution = throughput * background;

			CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
			FIXNAN_FLOAT3( contribution );

			float dE = pathStateData[jobIndex].data4.w;
			misWeight = 1.0f / (dE * (1.0f / (SCENE_AREA)) + NKK);

			if (type == NEW_PATH)
			{
				misWeight = 1.0f;
			}


			accumulatorOnePass[contribIdx] += make_float4( (contribution * misWeight), 0.0f );
		}
	}

	if (eye_hit != -1 && s + t < MAX_EYEPATH)
	{
		type = EXTEND_EYEPATH;
		const uint eyePIdx = atomicAdd( &counters->extendEyePath, 1 );
		eyePathBuffer[eyePIdx] = jobIndex;
	}
	else if (light_hit != -1 && t < MAX_LIGHTPATH)
	{
		type = EXTEND_LIGHTPATH;

		const uint eyeIdx = atomicAdd( &counters->constructionEyePos, 1 );
		constructEyeBuffer[eyeIdx] = jobIndex;

		const uint lightPIdx = atomicAdd( &counters->extendLightPath, 1 );
	}
	else
	{
		type = DEAD;
	}

	path_s_t_type_pass = (s << 27) + (t << 22) + (type << 19) + pass;
	pathStateData[jobIndex].eye_normal.w = __uint_as_float( path_s_t_type_pass );
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void connectionPath( int smcount,
	BiPathState* pathStateData, const Intersection* randomWalkHitBuffer,
	float4* accumulatorOnePass,
	const int4 screenParams,
	uint* constructEyeBuffer, uint* eyePathBuffer )
{
	const dim3 gridDim( NEXTMULTIPLEOF( smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
	connectionPathKernel << < gridDim.x, 256 >> > (smcount,
		pathStateData, randomWalkHitBuffer, accumulatorOnePass,
		screenParams,
		constructEyeBuffer, eyePathBuffer);
}

// EOF