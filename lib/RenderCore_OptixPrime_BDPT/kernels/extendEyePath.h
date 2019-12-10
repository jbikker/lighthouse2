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

// path state flags
#define S_SPECULAR		1	// previous path vertex was specular
#define S_BOUNCED		2	// path encountered a diffuse vertex
#define S_VIASPECULAR	4	// path has seen at least one specular vertex

#define FLAGS data

//  +-----------------------------------------------------------------------------+
//  |  extendPathKernel                                                      |
//  |  extend eye path or light path.                                  LH2'19|
//  +-----------------------------------------------------------------------------+
__global__  __launch_bounds__( 256, 1 )
void extendEyePathKernel( int smcount, BiPathState* pathStateData,
	Ray4* visibilityRays, Ray4* randomWalkRays, const uint R0, const uint* blueNoise,
	const float spreadAngle, const int4 screenParams, const int probePixelIdx,
	uint* eyePathBuffer, float4* contribution_buffer,
	float4* accumulatorOnePass )
{
	int gid = threadIdx.x + blockIdx.x * blockDim.x;
	if (gid >= counters->extendEyePath) return;

	int jobIndex = eyePathBuffer[gid];

	uint path_s_t_type_pass = __float_as_uint( pathStateData[jobIndex].eye_normal.w );

	uint data = __float_as_uint( pathStateData[jobIndex].light_normal.w );

	int contribIdx = (data >> 8);

	uint pass, type, t, s;
	getPathInfo( path_s_t_type_pass, pass, s, t, type );

	const int scrhsize = screenParams.x & 0xffff;
	const int scrvsize = screenParams.x >> 16;
	const uint x = jobIndex % scrhsize;
	uint y = jobIndex / scrhsize;

	const uint sampleIndex = pass * MAX_LIGHTPATH + t - 1;
	y %= scrvsize;

	float3 pos, dir;
	float4 hitData;

	float d, pdf_area, pdf_solidangle;
	float3 throughput, beta;

	throughput = make_float3( pathStateData[jobIndex].data4 );
	beta = make_float3( pathStateData[jobIndex].data5 );
	pos = make_float3( pathStateData[jobIndex].data6 );
	dir = make_float3( pathStateData[jobIndex].data7 );

	d = pathStateData[jobIndex].data4.w;
	pdf_area = pathStateData[jobIndex].data5.w;
	pdf_solidangle = pathStateData[jobIndex].data6.w;

	hitData = pathStateData[jobIndex].eye_intersection;

	const int prim = __float_as_int( hitData.z );
	const int primIdx = prim == -1 ? prim : (prim & 0xfffff);

	const CoreTri4* instanceTriangles = (const CoreTri4*)instanceDescriptors[INSTANCEIDX].triangles;

	ShadingData shadingData;
	float3 N, iN, fN, T;
	const float3 I = RAY_O + HIT_T * dir;
	const float coneWidth = spreadAngle * HIT_T;
	GetShadingData( dir, HIT_U, HIT_V, coneWidth, instanceTriangles[primIdx], INSTANCEIDX, shadingData, N, iN, fN, T );

	if (ROUGHNESS < 0.01f) FLAGS |= S_SPECULAR; else FLAGS &= ~S_SPECULAR;

	throughput = beta;
	pdf_area = pdf_solidangle * fabs( dot( -dir, fN ) ) / (HIT_T * HIT_T);

	float3 R;
	float r4, r5;

	if (BLUENOISER_ON && sampleIndex < 256)
	{
		r4 = blueNoiseSampler( blueNoise, x, y, sampleIndex, 6 );
		r5 = blueNoiseSampler( blueNoise, x, y, sampleIndex, 7 );
	}
	else
	{
		uint seed = WangHash( contribIdx * 17 + R0 );
		r4 = RandomFloat( seed );
		r5 = RandomFloat( seed );
	}

	bool specular = false;
	const float3 bsdf = SampleBSDF( shadingData, fN, N, T, dir * -1.0f, HIT_T, r4, r5, R, pdf_solidangle, specular, type );
	if (specular) FLAGS |= S_SPECULAR;

	if (!(pdf_solidangle < EPSILON || isnan( pdf_solidangle )))
	{
		beta *= bsdf * fabs( dot( fN, R ) ) / pdf_solidangle;
	}

	s++;

	// the ray is from eye to the pixel directly
	if (jobIndex == probePixelIdx && s == 1)
		counters->probedInstid = INSTANCEIDX,	// record instace id at the selected pixel
		counters->probedTriid = primIdx,		// record primitive id at the selected pixel
		counters->probedDist = HIT_T;			// record primary ray hit distance

	// initialize depth in accumulator for DOF shader
	if (s == 1) accumulatorOnePass[jobIndex].w += (prim == NOHIT ? 10000 : HIT_T);

	float dE = 1.0f / pdf_area; // N0k

	float3 light_pos = make_float3( pathStateData[jobIndex].data2 );

	if (s > 1)
	{
		float3 light2eye = normalize( light_pos - I );

		float bsdfPdf;
		const float3 sampledBSDF = EvaluateBSDF( shadingData, fN, T, light2eye, dir * -1.0f, bsdfPdf );

		float3 normal = make_float3( pathStateData[jobIndex].eye_normal );
		float light_p = bsdfPdf * fabs( dot( normal, dir ) ) / (HIT_T * HIT_T);

		dE = (1.0f + light_p * d) / pdf_area;

		// Equation [18]
		if (ROUGHNESS < 0.01f)
		{
			dE = light_p * d / pdf_area;
		}
	}

	int randomWalkRayIdx = -1;
	float pdf_ = pdf_solidangle;

#ifdef FLAGS_ON
	if ((FLAGS & S_BOUNCED))
	{
		pdf_ = 0.0f; // terminate the eye path extension        
	}
	else if (s < MAX_EYEPATH)
	#endif
	{
		randomWalkRayIdx = atomicAdd( &counters->randomWalkRays, 1 );
		randomWalkRays[randomWalkRayIdx].O4 = make_float4( SafeOrigin( I, R, N, geometryEpsilon ), 0 );
		randomWalkRays[randomWalkRayIdx].D4 = make_float4( R, 1e34f );
	}

	if (!(FLAGS & S_SPECULAR)) FLAGS |= S_BOUNCED; else FLAGS |= S_VIASPECULAR;

	path_s_t_type_pass = (s << 27) + (t << 22) + (type << 19) + pass;

	pathStateData[jobIndex].data4 = make_float4( throughput, dE );
	pathStateData[jobIndex].data5 = make_float4( beta, pdf_area );
	pathStateData[jobIndex].data6 = make_float4( I, pdf_ );
	pathStateData[jobIndex].data7 = make_float4( R, __int_as_float( randomWalkRayIdx ) );
	pathStateData[jobIndex].eye_normal = make_float4( fN,
		__uint_as_float( path_s_t_type_pass ) );
	pathStateData[jobIndex].light_normal.w = __uint_as_float( data );

	float3 eye_pos = I;

	float3 eye2light = light_pos - eye_pos;
	float3 eye_normal = fN;
	const float dist = length( eye2light );
	eye2light = eye2light / dist;

	if (shadingData.IsEmissive())
	{
		float3 pre_pos = pos;

		if (dot( N, dir ) < 0) // single side light
		{
			float3 contribution = throughput * shadingData.color;

			const CoreTri& tri = (const CoreTri&)instanceTriangles[primIdx];
			const float pickProb = LightPickProb( tri.ltriIdx, pre_pos, dir, eye_pos );
			const float pdfPos = 1.0f / tri.area;
			const float p_rev = pickProb * pdfPos; // surface area

			// Equation [16] when s == k
			float misWeight = 1.0f / (dE * p_rev + NKK);

			contribution = contribution * misWeight;
			CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
			FIXNAN_FLOAT3( contribution );

			accumulatorOnePass[contribIdx] += make_float4( (contribution), 0.0f );
		}

		pathStateData[jobIndex].data6.w = 0;
	}
	else if (t == 1)
	{
		if ((s + t) > MAXPATHLENGTH)
			return;

		float3 light2eye = eye2light;
		float length_l2e = dist;

		float bsdfPdf;
		const float3 sampledBSDF = EvaluateBSDF( shadingData, fN, T, dir * -1.0f, light2eye, bsdfPdf );

		// specular connection
		/*
		if (ROUGHNESS < 0.01f)
		{
			return;
		}
		*/

		if (bsdfPdf < EPSILON || isnan( bsdfPdf ))
		{
			return;
		}

		float3 light_throughput = make_float3( pathStateData[jobIndex].data0 );
		float light_pdf = pathStateData[jobIndex].data1.w;

		float3 light_normal = make_float3( pathStateData[jobIndex].light_normal );
		float light_cosTheta = fabs( dot( light2eye * -1.0f, light_normal ) );

		// area to solid angle: r^2 / (Area * cos)
		light_pdf *= length_l2e * length_l2e / light_cosTheta;

		float cosTheta = fabs( dot( fN, light2eye ) );

		float eye_cosTheta = fabs( dot( light2eye, eye_normal ) );

		float p_forward = bsdfPdf * light_cosTheta / (length_l2e * length_l2e);
		float p_rev = light_cosTheta * INVPI * eye_cosTheta / (length_l2e * length_l2e);

		float dL = pathStateData[jobIndex].data0.w;

		float misWeight = 1.0 / (dE * p_rev + 1 + dL * p_forward);

		float3 contribution = throughput * sampledBSDF * light_throughput * (1.0f / light_pdf)  * cosTheta;

		contribution = contribution * misWeight;
		CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
		FIXNAN_FLOAT3( contribution );

		const uint contib_idx = atomicAdd( &counters->contribution_count, 1 );
		contribution_buffer[contib_idx] = make_float4( contribution, __uint_as_float( contribIdx ) );

		visibilityRays[contib_idx].O4 = make_float4( SafeOrigin( eye_pos, eye2light, eye_normal, geometryEpsilon ), 0 );
		visibilityRays[contib_idx].D4 = make_float4( eye2light, dist - 2 * geometryEpsilon );
	}
	else
	{

		if ((s + t) > MAXPATHLENGTH)
			return;

		float3 light2eye = eye2light;
		float length_l2e = dist;

		if (length_l2e < SCENE_RADIUS * RadiusFactor)
		{
			return;
		}

		float eye_bsdfPdf;
		const float3 sampledBSDF_s = EvaluateBSDF( shadingData, fN, T, dir * -1.0f, light2eye, eye_bsdfPdf );

		hitData = pathStateData[jobIndex].currentLight_hitData;

		float3 dir_light = make_float3( pathStateData[jobIndex].pre_light_dir );

		const int prim_light = __float_as_int( hitData.z );
		const int primIdx_light = prim_light == -1 ? prim_light : (prim_light & 0xfffff);
		int idx = (prim_light >> 20);

		const CoreTri4* instanceTriangles_eye = (const CoreTri4*)instanceDescriptors[idx].triangles;

		ShadingData shadingData_light;
		float3 N_light, iN_light, fN_light, T_light;
		const float coneWidth_light = spreadAngle * HIT_T;

		GetShadingData( dir_light, HIT_U, HIT_V, coneWidth_light, instanceTriangles_eye[primIdx_light],
			idx, shadingData_light, N_light, iN_light, fN_light, T_light );

		float r_Light = (max( 0.001f, CHAR2FLT( shadingData_light.parameters.x, 24 ) ));

		// specular connection
		/*
		if (ROUGHNESS < 0.01f || r_Light < 0.01f)
		{
			return;
		}
		*/

		float light_bsdfPdf;
		float3 sampledBSDF_t = EvaluateBSDF( shadingData_light, fN_light, T_light,
			dir_light * -1.0f, light2eye * -1.0f, light_bsdfPdf );

		float shading_normal_num = fabs( dot( dir_light, fN_light ) ) * fabs( dot( light2eye, N_light ) );
		float shading_normal_denom = fabs( dot( dir_light, N_light ) ) * fabs( dot( light2eye, fN_light ) );
		float fCorrectNormal = (shading_normal_num / shading_normal_denom);

		/**/
		if ((shading_normal_denom < EPSILON || isnan( shading_normal_denom )))
		{
			fCorrectNormal = 0.0f;
		}
		sampledBSDF_t *= fCorrectNormal;

		float3 throughput_light = make_float3( pathStateData[jobIndex].data0 );

		// fabs keep safety
		float cosTheta_eye = (dot( fN, light2eye ));
		float cosTheta_light = fabs( dot( fN_light, light2eye* -1.0f ) );
		float G = cosTheta_eye * cosTheta_light / (length_l2e * length_l2e);

		if (cosTheta_eye < EPSILON || cosTheta_light < EPSILON)
		{
			return;
		}

		if (eye_bsdfPdf < EPSILON || isnan( eye_bsdfPdf )
			|| light_bsdfPdf < EPSILON || isnan( light_bsdfPdf ))
		{
			return;
		}

		float3 contribution = throughput * sampledBSDF_s * sampledBSDF_t * throughput_light * G;

		float p_forward = eye_bsdfPdf * cosTheta_light / (length_l2e * length_l2e);
		float p_rev = light_bsdfPdf * cosTheta_eye / (length_l2e * length_l2e);

		float dL = pathStateData[jobIndex].data0.w;

		float misWeight = 1.0 / (dE * p_rev + 1 + dL * p_forward);

		contribution = contribution * misWeight;
		CLAMPINTENSITY; // limit magnitude of thoughput vector to combat fireflies
		FIXNAN_FLOAT3( contribution );

		const uint contib_idx = atomicAdd( &counters->contribution_count, 1 );
		contribution_buffer[contib_idx] = make_float4( contribution, __uint_as_float( contribIdx ) );

		visibilityRays[contib_idx].O4 = make_float4( SafeOrigin( eye_pos, eye2light, eye_normal, geometryEpsilon ), 0 );
		visibilityRays[contib_idx].D4 = make_float4( eye2light, dist - 2 * geometryEpsilon );
	}

}

//  +-----------------------------------------------------------------------------+
//  |  constructionLightPos                                                            |
//  |  Entry point for the persistent constructionLightPos kernel.               LH2'19|
//  +-----------------------------------------------------------------------------+
__host__ void extendEyePath( int smcount, BiPathState* pathStateBuffer,
	Ray4* visibilityRays, Ray4* randomWalkRays, const uint R0, const uint* blueNoise,
	const float spreadAngle, const int4 screenParams, const int probePixelIdx,
	uint* eyePathBuffer, float4* contribution_buffer,
	float4* accumulatorOnePass )
{
	const dim3 gridDim( NEXTMULTIPLEOF( smcount, 256 ) / 256, 1 ), blockDim( 256, 1 );
	extendEyePathKernel << < gridDim.x, 256 >> > (smcount, pathStateBuffer,
		visibilityRays, randomWalkRays,
		R0, blueNoise, spreadAngle, screenParams, probePixelIdx, eyePathBuffer,
		contribution_buffer, accumulatorOnePass);
}

// EOF