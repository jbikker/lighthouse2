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
#define RAY_O pos

#define FLAGS_L data_L

LH2_DEVFUNC void Sample_Wi( const float aperture, const float imgPlaneSize, const float3 eye_pos,
	const float3 forward, const float3 light_pos, const float focalDistance,
	const float3 p1, const float3 right, const float3 up,
	float3& throughput, float& pdf, float& u, float& v )
{
	throughput = make_float3( 0.0f );
	pdf = 0.0f;

	float3 dir = light_pos - eye_pos;
	float dist = length( dir );

	dir /= dist;

	float cosTheta = dot( normalize( forward ), dir );

	// check direction
	if (cosTheta <= 0)
	{
		return;
	}

	float x_length = length( right );
	float y_length = length( up );

	float distance = focalDistance / cosTheta;

	float3 raster_pos = eye_pos + distance * dir;
	float3 pos2p1 = raster_pos - p1;

	float3 unit_up = up / y_length;
	float3 unit_right = right / x_length;

	float x_offset = dot( unit_right, pos2p1 );
	float y_offset = dot( unit_up, pos2p1 );

	// check view fov
	if (x_offset<0 || x_offset > x_length
		|| y_offset<0 || y_offset > y_length)
	{
		return;
	}

	u = x_offset / x_length;
	v = y_offset / y_length;

	float cos2Theta = cosTheta * cosTheta;
	float lensArea = aperture != 0 ? aperture * aperture * PI : 1;
	lensArea = 1.0f; // because We / pdf
	float We = 1.0f / (imgPlaneSize * lensArea * cos2Theta * cos2Theta);

	throughput = make_float3( We );
	pdf = dist * dist / (cosTheta * lensArea);
}

//  +-----------------------------------------------------------------------------+
//  |  extendPathKernel                                                      |
//  |  extend eye path or light path.                                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256, 1 )
void extendLightPathKernel( int smcount, BiPathState* pathStateData,
	Ray4* visibilityRays, Ray4* randomWalkRays, const uint R0, const uint* blueNoise,
	const float3 cam_pos, const float spreadAngle, const int4 screenParams,
	uint* lightPathBuffer, float4* contribution_buffer,
	const float aperture, const float imgPlaneSize,
	const float3 forward, const float focalDistance, const float3 p1,
	const float3 right, const float3 up )
{
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= counters->extendLightPath) return;

	int jobIndex = lightPathBuffer[gid];

	uint path_s_t_type_pass = __float_as_uint( pathStateData[jobIndex].eye_normal.w );
	uint data = __float_as_uint( pathStateData[jobIndex].light_normal.w );
	uint data_L = __float_as_uint( pathStateData[jobIndex].pre_light_dir.w );

	uint pass, type, t, s;
	getPathInfo( path_s_t_type_pass, pass, s, t, type );

	const int scrhsize = screenParams.x & 0xffff;
	const int scrvsize = screenParams.x >> 16;
	const uint x = jobIndex % scrhsize;
	uint y = jobIndex / scrhsize;
	const uint sampleIndex = pass;
	y %= scrvsize;

	float3 pos, dir;
	float4 hitData;

	float d, pdf_area, pdf_solidangle;
	float3 throughput, beta;


	throughput = make_float3( pathStateData[jobIndex].data0 );
	beta = make_float3( pathStateData[jobIndex].data1 );
	pos = make_float3( pathStateData[jobIndex].data2 );
	dir = make_float3( pathStateData[jobIndex].data3 );

	d = pathStateData[jobIndex].data0.w;
	pdf_area = pathStateData[jobIndex].data1.w;
	pdf_solidangle = pathStateData[jobIndex].data2.w;

	hitData = pathStateData[jobIndex].light_intersection;

	const int prim = __float_as_int( hitData.z );
	const int primIdx = prim == -1 ? prim : (prim & 0xfffff);

	const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;

	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = RAY_O + HIT_T * dir;
	const float coneWidth = spreadAngle * HIT_T;
	GetShadingData( dir, HIT_U, HIT_V, coneWidth, instanceTriangles[primIdx],
		INSTANCEIDX, shadingData, N, iN, fN, T );

	if (ROUGHNESS < 0.01f) FLAGS_L |= S_SPECULAR; else FLAGS_L &= ~S_SPECULAR;

	throughput = beta;
	pdf_area = pdf_solidangle * fabs( dot( -dir, fN ) ) / (HIT_T * HIT_T);

	float3 R;
	float r4, r5;
	if (false && sampleIndex < 256)
	{
		r4 = blueNoiseSampler( blueNoise, x, y, sampleIndex, 0 );
		r5 = blueNoiseSampler( blueNoise, x, y, sampleIndex, 1 );
	}
	else
	{
		uint seed = WangHash( jobIndex * 17 + R0 );
		r4 = RandomFloat( seed );
		r5 = RandomFloat( seed );
	}

	bool specular = false;
	const float3 bsdf = SampleBSDF( shadingData, fN, N, T, dir * -1.0f, HIT_T, r4, r5, R, pdf_solidangle, specular, type );
	if (specular) FLAGS_L |= S_SPECULAR;

	if (!(pdf_solidangle < EPSILON || isnan( pdf_solidangle )))
	{
		beta *= bsdf * fabs( dot( fN, R ) ) / pdf_solidangle;
	}


	// correct shading normal when it is importance
	float shading_normal_num = fabs( dot( dir, fN ) ) * fabs( dot( R, N ) );
	float shading_normal_denom = fabs( dot( dir, N ) ) * fabs( dot( R, fN ) );
	/**/
	float fCorrectNormal = (shading_normal_num / shading_normal_denom);

	if ((shading_normal_denom < EPSILON || isnan( shading_normal_denom )))
	{
		fCorrectNormal = 0.0f;
	}

	beta *= fCorrectNormal;

	t++;

	float3 eye_pos = cam_pos;
	float3 eye2lightU = normalize( eye_pos - I );

	float bsdfPdf;
	const float3 sampledBSDF = EvaluateBSDF( shadingData, fN, T, eye2lightU, dir * -1.0f, bsdfPdf );

	float3 normal = make_float3( pathStateData[jobIndex].light_normal );
	float eye_p = bsdfPdf * fabs( dot( normal, dir ) ) / (HIT_T * HIT_T);
	float dL = (1.0f + eye_p * d) / pdf_area;

	if (ROUGHNESS < 0.01f)
	{
		dL = eye_p * d / pdf_area;
	}

	int randomWalkRayIdx = -1;
	float pdf_ = pdf_solidangle;
#ifdef FLAGS_ON
	if ((FLAGS_L & S_BOUNCED))
	{
		pdf_ = 0.0f; // terminate the eye path extension
	}
	else if (t < MAX_LIGHTPATH) // reduce this query
	#endif
	{
		randomWalkRayIdx = atomicAdd( &counters->randomWalkRays, 1 );
		randomWalkRays[randomWalkRayIdx].O4 = make_float4( SafeOrigin( I, R, N, geometryEpsilon ), 0 );
		randomWalkRays[randomWalkRayIdx].D4 = make_float4( R, 1e34f );
	}

	if (!(FLAGS_L & S_SPECULAR)) FLAGS_L |= S_BOUNCED; else FLAGS_L |= S_VIASPECULAR;

	pathStateData[jobIndex].data0 = make_float4( throughput, dL );
	pathStateData[jobIndex].data1 = make_float4( beta, pdf_area );
	pathStateData[jobIndex].data2 = make_float4( I, pdf_ );
	pathStateData[jobIndex].data3 = make_float4( R, __int_as_float( randomWalkRayIdx ) );

	pathStateData[jobIndex].light_normal = make_float4( fN, __uint_as_float( data ) );
	pathStateData[jobIndex].pre_light_dir = make_float4( dir, 0.0f );
	pathStateData[jobIndex].currentLight_hitData = hitData;

	path_s_t_type_pass = (s << 27) + (t << 22) + (type << 19) + pass;
	pathStateData[jobIndex].eye_normal.w = __uint_as_float( path_s_t_type_pass );

	/**/
	if (shadingData.IsEmissive())
	{
		pathStateData[jobIndex].data2.w = 0;
		pathStateData[jobIndex].data6.w = 0;
		return;
	}

	float3 light_pos = I;
	float3 eye2light = eye_pos - light_pos;
	const float dist = length( eye2light );
	eye2light = eye2light / dist;

	float3 light2eye = eye2light;
	float length_l2e = dist;

	float3 throughput_eye;
	float pdf_eye;
	float u, v;
	Sample_Wi( aperture, imgPlaneSize, eye_pos, forward, light_pos,
		focalDistance, p1, right, up, throughput_eye, pdf_eye, u, v );

	if (pdf_eye > EPSILON)
	{
		float bsdfPdf;
		const float3 sampledBSDF = EvaluateBSDF( shadingData, fN, T, dir * -1.0f, light2eye, bsdfPdf );

		float3 light_throught = throughput;
		float cosTheta = fabs( dot( fN, light2eye ) );

		float eye_cosTheta = fabs( dot( normalize( forward ), light2eye * -1.0f ) );
		float eye_pdf_solid = 1.0f / (imgPlaneSize * eye_cosTheta * eye_cosTheta * eye_cosTheta);
		float p_forward = eye_pdf_solid * cosTheta / (length_l2e * length_l2e);

		float misWeight = 1.0f / (1 + dL * p_forward);

		if (pdf_solidangle < EPSILON || isnan( pdf_solidangle ))
		{
			return;
		}

		uint x = (scrhsize * u + 0.5);
		uint y = (scrvsize * v + 0.5);
		uint idx = y * scrhsize + x;

		float3 contribution = light_throught * sampledBSDF * (throughput_eye / pdf_eye) * cosTheta;

		contribution = contribution * misWeight;
		CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
		FIXNAN_FLOAT3( contribution );

		const uint contib_idx = atomicAdd( &counters->contribution_count, 1 );
		contribution_buffer[contib_idx] = make_float4( contribution, __uint_as_float( idx ) );

		visibilityRays[contib_idx].O4 = make_float4( SafeOrigin( light_pos, eye2light, fN, geometryEpsilon ), 0 );
		visibilityRays[contib_idx].D4 = make_float4( eye2light, dist - 2 * geometryEpsilon );
	}
}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void extendLightPath( int smcount, BiPathState* pathStateBuffer,
	Ray4* visibilityRays, Ray4* randomWalkRays, const uint R0, const uint* blueNoise,
	const float3 camPos, const float spreadAngle, const int4 screenParams,
	uint* lightPathBuffer, float4* contribution_buffer,
	const float aperture, const float imgPlaneSize,
	const float3 forward, const float focalDistance, const float3 p1,
	const float3 right, const float3 up )
{
	const dim3 gridDim( NEXTMULTIPLEOF( smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
	extendLightPathKernel << < gridDim.x, 256 >> > (smcount, pathStateBuffer,
		visibilityRays, randomWalkRays,
		R0, blueNoise, camPos, spreadAngle, screenParams,
		lightPathBuffer, contribution_buffer,
		aperture, imgPlaneSize, forward, focalDistance, p1, right, up);
}

// EOF