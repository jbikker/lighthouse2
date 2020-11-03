/* lights_shared.h - Copyright 2019/2020 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   THIS IS A SHARED FILE: used in
   - RenderCore_OptixPrime_b
   - RenderCore_Optix7
   - RenderCore_Optix7Filter
   - RenderCore_PrimeRef
*/

#include "noerrors.h"

#define ISLIGHTS
#define MAXISLIGHTS	64

#define TRILIGHTCOUNT			(lightCounts.x & 0xffff)
#define POINTLIGHTCOUNT			lightCounts.y
#define SPOTLIGHTCOUNT			lightCounts.z
#define DIRECTIONALLIGHTCOUNT	lightCounts.w

//  +-----------------------------------------------------------------------------+
//  |  PotentialTriLightContribution                                              |
//  |  Calculates the potential contribution of an area light.              LH2'20|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float PotentialTriLightContribution( const int idx, const float3& O, const float3& N, const float3& I, const float3& bary )
{
	const CoreLightTri4& light = (const CoreLightTri4&)triLights[idx];
	const float4 centre4 = light.data0; // holds area light energy in w
	const float4 LN = light.data1;
	float3 L = I;
	if (bary.x >= 0)
	{
		const float4 V0 = light.data3; // vertex0
		const float4 V1 = light.data4; // vertex1
		const float4 V2 = light.data5; // vertex2
		L = make_float3( bary.x * V0 + bary.y * V1 + bary.z * V2 );
	}
	L -= O;
	const float att = 1.0f / dot( L, L );
	L = normalize( L );
	const float LNdotL = max( 0.0f, -dot( make_float3( LN ), L ) );
	const float NdotL = max( 0.0f, dot( N, L ) );
	return AREALIGHT_ENERGY * LNdotL * NdotL * att;
}

//  +-----------------------------------------------------------------------------+
//  |  PotentialPointLightContribution                                            |
//  |  Calculates the potential contribution of a point light.              LH2'19|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float PotentialPointLightContribution( const int idx, const float3& I, const float3& N )
{
	const CorePointLight4& light = (const CorePointLight4&)pointLights[idx];
	const float4 position4 = light.data0;
	const float3 L = make_float3( position4 ) - I;
	const float NdotL = max( 0.0f, dot( N, normalize( L ) ) );
	const float att = 1.0f / dot( L, L );
	return POINTLIGHT_ENERGY * NdotL * att;
}

//  +-----------------------------------------------------------------------------+
//  |  PotentialSpotLightContribution                                             |
//  |  Calculates the potential contribution of a spot light.               LH2'19|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float PotentialSpotLightContribution( const int idx, const float3& I, const float3& N )
{
	const CoreSpotLight4& light = (const CoreSpotLight4&)spotLights[idx];
	const float4 position4 = light.data0;
	const float4 radiance4 = light.data1;
	const float4 direction4 = light.data2;
	float3 L = make_float3( position4 ) - I;
	const float att = 1.0f / dot( L, L );
	L = normalize( L );
	const float d = (max( 0.0f, -dot( L, make_float3( direction4 ) ) ) - SPOTLIGHT_OUTER) / (SPOTLIGHT_INNER - SPOTLIGHT_OUTER);
	const float NdotL = max( 0.0f, dot( N, L ) );
	const float LNdotL = max( 0.0f, min( 1.0f, d ) );
	return (radiance4.x + radiance4.y + radiance4.z) * LNdotL * NdotL * att;
	// TODO: other lights have radiance4.x+y+z precalculated as 'float energy'. For spots, this
	// does not help, as we need position4.w and direction4.w for the inner and outer angle anyway,
	// so we are touching 4 float4's. If we reduce the inner and outer angles to 16-bit values
	// however, the precalculated energy helps once more, and one float4 read disappears.
}

//  +-----------------------------------------------------------------------------+
//  |  PotentialDirectionalLightContribution                                      |
//  |  Calculates the potential contribution of a directional light.        LH2'19|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float PotentialDirectionalLightContribution( const int idx, const float3& I, const float3& N )
{
	const CoreDirectionalLight4& light = (const CoreDirectionalLight4&)directionalLights[idx];
	const float4 direction4 = light.data0;
	const float LNdotL = max( 0.0f, -(direction4.x * N.x + direction4.y * N.y + direction4.z * N.z) );
	return DIRLIGHT_ENERGY * LNdotL;
}

//  +-----------------------------------------------------------------------------+
//  |  CalculateLightPDF                                                          |
//  |  Calculates the solid angle of a light source.                        LH2'19|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float CalculateLightPDF( const float3& D, const float t, const float lightArea, const float3 lightNormal )
{
	return (t * t) / (abs( dot( D, lightNormal ) ) * lightArea);
}

//  +-----------------------------------------------------------------------------+
//  |  LightPickProb                                                              |
//  |  Calculates the probability with which the specified light woukd be picked  |
//  |  from the specified world space location and normal. Used in MIS.     LH2'19|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float LightPickProb( int idx, const float3& O, const float3& N, const float3& I )
{
#ifdef ISLIGHTS
	// for implicit connections; calculates the chance that the light would have been explicitly selected
	float potential[MAXISLIGHTS];
	float sum = 0;
	int lights = 0;
	for (int i = 0; i < TRILIGHTCOUNT; i++) { float c = PotentialTriLightContribution( i, O, N, I, make_float3( -1 ) ); potential[lights++] = c; sum += c; }
	for (int i = 0; i < POINTLIGHTCOUNT; i++) { float c = PotentialPointLightContribution( i, O, N ); potential[lights++] = c; sum += c; }
	for (int i = 0; i < SPOTLIGHTCOUNT; i++) { float c = PotentialSpotLightContribution( i, O, N ); potential[lights++] = c; sum += c; }
	for (int i = 0; i < DIRECTIONALLIGHTCOUNT; i++) { float c = PotentialDirectionalLightContribution( i, O, N ); potential[lights++] = c; sum += c; }
	if (sum <= 0) return 0; // no potential lights found
	return potential[idx] / sum;
#else
	return 1.0f / (TRILIGHTCOUNT + POINTLIGHTCOUNT + SPOTLIGHTCOUNT + DIRECTIONALLIGHTCOUNT);
#endif
}

//  +-----------------------------------------------------------------------------+
//  |  RandomPointOnLight                                                         |
//  |  Selects a random point on a random light. Returns a position, a normal on  |
//  |  the light source, the probability that this particular light would have    |
//  |  been picked and the importance of the explicit connection.           LH2'19|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float3 RandomPointOnLight( float r0, float r1, const float3& I, const float3& N, float& pickProb, float& lightPdf, float3& lightColor )
{
	const float lightCount = TRILIGHTCOUNT + POINTLIGHTCOUNT + SPOTLIGHTCOUNT + DIRECTIONALLIGHTCOUNT;
	// predetermine the barycentrics for any area light we sample
	float3 bary = RandomBarycentrics( r0 );
#ifdef ISLIGHTS
	// importance sampling of lights, pickProb is per-light probability
	float potential[MAXISLIGHTS];
	float sum = 0, total = 0;
	int lights = 0, lightIdx = 0;
	for (int i = 0; i < TRILIGHTCOUNT; i++) { float c = PotentialTriLightContribution( i, I, N, I, bary ); potential[lights++] = c; sum += c; }
	for (int i = 0; i < POINTLIGHTCOUNT; i++) { float c = PotentialPointLightContribution( i, I, N ); potential[lights++] = c; sum += c; }
	for (int i = 0; i < SPOTLIGHTCOUNT; i++) { float c = PotentialSpotLightContribution( i, I, N ); potential[lights++] = c; sum += c; }
	for (int i = 0; i < DIRECTIONALLIGHTCOUNT; i++) { float c = PotentialDirectionalLightContribution( i, I, N ); potential[lights++] = c; sum += c; }
	if (sum <= 0) // no potential lights found
	{
		lightPdf = 0;
		return make_float3( 1 /* light direction; don't return 0 or nan, this will be slow */ );
	}
	r1 *= sum;
	for (int i = 0; i < lights; i++)
	{
		total += potential[i];
		if (total >= r1) { lightIdx = i; break; }
	}
	pickProb = potential[lightIdx] / sum;
#else
	// uniform random sampling of lights, pickProb is simply 1.0 / lightCount
	pickProb = 1.0f / lightCount;
	int lightIdx = (int)(r1 * lightCount);
#endif
	lightIdx = clamp( lightIdx, 0, (int)lightCount - 1 );
	if (lightIdx < TRILIGHTCOUNT)
	{
		// pick an area light
		const CoreLightTri4& light = (const CoreLightTri4&)triLights[lightIdx];
		const float4 V0 = light.data3;			// vertex0
		const float4 V1 = light.data4;			// vertex1
		const float4 V2 = light.data5;			// vertex2
		lightColor = make_float3( light.data2 );	// radiance
		const float4 LN = light.data1;			// N
		const float3 P = make_float3( bary.x * V0 + bary.y * V1 + bary.z * V2 );
		float3 L = I - P; // reversed: from light to intersection point
		const float sqDist = dot( L, L );
		L = normalize( L );
		const float LNdotL = L.x * LN.x + L.y * LN.y + L.z * LN.z;
		const float reciSolidAngle = sqDist / (LN.w * LNdotL); // LN.w contains area
		lightPdf = (LNdotL > 0 && dot( L, N ) < 0) ? reciSolidAngle : 0;
		return P;
	}
	else if (lightIdx < (TRILIGHTCOUNT + POINTLIGHTCOUNT))
	{
		// pick a pointlight
		const CorePointLight4& light = (const CorePointLight4&)pointLights[lightIdx - TRILIGHTCOUNT];
		const float3 P = make_float3( light.data0 );	// position
		const float3 L = P - I;
		const float sqDist = dot( L, L );
		lightColor = make_float3( light.data1 ) / sqDist;		// radiance
		lightPdf = dot( L, N ) > 0 ? 1 : 0;
		return P;
	}
	else if (lightIdx < (TRILIGHTCOUNT + POINTLIGHTCOUNT + SPOTLIGHTCOUNT))
	{
		// pick a spotlight
		const CoreSpotLight4& light = (const CoreSpotLight4&)spotLights[lightIdx - (TRILIGHTCOUNT + POINTLIGHTCOUNT)];
		const float4 V0 = light.data0;			// position + cos_inner
		const float4 V1 = light.data1;			// radiance + cos_outer
		const float4 D = light.data2;			// direction
		const float3 P = make_float3( V0 );
		float3 L = I - P;
		const float sqDist = dot( L, L );
		L = normalize( L );
		float d = (max( 0.0f, L.x * D.x + L.y * D.y + L.z * D.z ) - V1.w) / (V0.w - V1.w);
		const float LNdotL = min( 1.0f, d );
		lightPdf = (LNdotL > 0 && dot( L, N ) < 0) ? (sqDist / LNdotL) : 0;
		lightColor = make_float3( V1 );
		return P;
	}
	else
	{
		// pick a directional light
		const CoreDirectionalLight4& light = (const CoreDirectionalLight4&)directionalLights[lightIdx - (TRILIGHTCOUNT + POINTLIGHTCOUNT + SPOTLIGHTCOUNT)];
		const float3 L = make_float3( light.data0 );	// direction
		lightColor = make_float3( light.data1 );		// radiance
		const float NdotL = dot( L, N );
		lightPdf = NdotL < 0 ? 1 : 0;
		return I - 1000.0f * L;
	}
}

//  +-----------------------------------------------------------------------------+
//  |  Sample_Le                                                                  |
//  |  Part of the BDPT core.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float3 Sample_Le( const float& r0, float r1, const float& r2, const float& r3,
	float3& normal, float3& lightDir, float3& lightColor,
	float& lightPdf, float& pdfPos, float& pdfDir )
{
	const float lightCount = TRILIGHTCOUNT + POINTLIGHTCOUNT + SPOTLIGHTCOUNT + DIRECTIONALLIGHTCOUNT;
	// predetermine the barycentrics for any area light we sample
	float3 bary = RandomBarycentrics( r0 );
#ifdef ISLIGHTS
	// importance sampling of lights, pickProb is per-light probability
	float potential[MAXISLIGHTS];
	float sum = 0, total = 0;
	int lights = 0, lightIdx = 0;
	for (int i = 0; i < TRILIGHTCOUNT; i++)
	{
		const CoreLightTri4& light = (const CoreLightTri4&)triLights[i];
		const float4 centre4 = light.data0; // holds area light energy in w
		float c = AREALIGHT_ENERGY;
		potential[lights++] = c;
		sum += c;
	}
	for (int i = 0; i < POINTLIGHTCOUNT; i++)
	{
		const CorePointLight4& light = (const CorePointLight4&)pointLights[i];
		const float4 position4 = light.data0;
		float c = POINTLIGHT_ENERGY;
		potential[lights++] = c;
		sum += c;
	}
	for (int i = 0; i < SPOTLIGHTCOUNT; i++)
	{
		const CoreSpotLight4& light = (const CoreSpotLight4&)spotLights[i];
		const float4 radiance4 = light.data1;
		float c = radiance4.x + radiance4.y + radiance4.z;
		potential[lights++] = c;
		sum += c;
	}
	for (int i = 0; i < DIRECTIONALLIGHTCOUNT; i++)
	{
		const CoreDirectionalLight4& light = (const CoreDirectionalLight4&)directionalLights[i];
		const float4 direction4 = light.data0;
		float c = DIRLIGHT_ENERGY;
		potential[lights++] = c;
		sum += c;
	}
	if (sum <= 0) // no potential lights found
	{
		lightPdf = 0;
		return make_float3( 1 /* light direction; don't return 0 or nan, this will be slow */ );
	}
	r1 *= sum;
	for (int i = 0; i < lights; i++)
	{
		total += potential[i];
		if (total >= r1) { lightIdx = i; break; }
	}
	lightPdf = potential[lightIdx] / sum;
#else
	// uniform random sampling of lights, pickProb is simply 1.0 / lightCount
	lightPdf = 1.0f / lightCount;
	int lightIdx = (int)(r1 * lightCount);
#endif
	lightIdx = clamp( lightIdx, 0, (int)lightCount - 1 );
	float3 pos;
	if (lightIdx < TRILIGHTCOUNT)
	{
		const CoreLightTri4& light = (const CoreLightTri4&)triLights[lightIdx];
		const float4 V0 = light.data3;			// vertex0
		const float4 V1 = light.data4;			// vertex1
		const float4 V2 = light.data5;			// vertex2
		lightColor = make_float3( light.data2 );	// radiance
		const float4 LN = light.data1;			// N+area
		float area = LN.w;
		pdfPos = 1.0f / area;
		pos = make_float3( bary.x * V0 + bary.y * V1 + bary.z * V2 );
		normal = make_float3( LN );
		float3 dir_loc = DiffuseReflectionCosWeighted( r2, r3 );
		pdfDir = dir_loc.z * INVPI;
		float3 u, v;
		if (fabsf( normal.x ) > 0.99f)
		{
			u = normalize( cross( normal, make_float3( 0.0f, 1.0f, 0.0f ) ) );
		}
		else
		{
			u = normalize( cross( normal, make_float3( 1.0f, 0.0f, 0.0f ) ) );
		}
		v = cross( u, normal );
		lightDir = normalize( dir_loc.x * u + dir_loc.y * v + dir_loc.z * normal );
		return pos;
	}
	else if (lightIdx < (TRILIGHTCOUNT + POINTLIGHTCOUNT))
	{
		// pick a pointlight
		const CorePointLight4& light = (const CorePointLight4&)pointLights[lightIdx - TRILIGHTCOUNT];
		const float3 pos = make_float3( light.data0 );			// position
		lightColor = make_float3( light.data1 );	// radiance
		normal = lightDir = UniformSampleSphere( r2, r3 );
		pdfPos = 1.0f;
		pdfDir = INVPI * 0.25f; // UniformSpherePdf
		return pos;
	}
	else if (lightIdx < (TRILIGHTCOUNT + POINTLIGHTCOUNT + SPOTLIGHTCOUNT))
	{
		// pick a spotlight
		const CoreSpotLight4& light = (const CoreSpotLight4&)spotLights[lightIdx - (TRILIGHTCOUNT + POINTLIGHTCOUNT)];
		const float4 P = light.data0;			// position + cos_inner
		const float4 E = light.data1;			// radiance + cos_outer
		const float4 D = light.data2;			// direction
		const float3 pos = make_float3( P );
		lightColor = make_float3( E );
		float3 dir_loc = UniformSampleCone( r2, r3, E.w );
		const float3 light_direction = make_float3( D );
		float3 u, v;
		if (fabsf( light_direction.x ) > 0.99f)
		{
			u = normalize( cross( light_direction, make_float3( 0.0f, 1.0f, 0.0f ) ) );
		}
		else
		{
			u = normalize( cross( light_direction, make_float3( 1.0f, 0.0f, 0.0f ) ) );
		}
		v = cross( u, light_direction );
		normal = lightDir = normalize( dir_loc.x * u + dir_loc.y * v + dir_loc.z * light_direction );
		pdfPos = 1.0f;
		pdfDir = 1.0f / (TWOPI * (1.0f - E.w)); // UniformConePdf
		return pos;
	}
	else
	{
		// pick a directional light
		const CoreDirectionalLight4& light = (const CoreDirectionalLight4&)directionalLights[lightIdx - (TRILIGHTCOUNT + POINTLIGHTCOUNT + SPOTLIGHTCOUNT)];
		const float3 L = make_float3( light.data0 );	// direction
		lightColor = make_float3( light.data1 );		// radiance
	#ifdef DIRECTIONAL_LIGHT
		const float3 pos = SCENE_CENTER - SCENE_RADIUS * L;
		normal = lightDir = L;
		pdfPos = 1.0f / SCENE_AREA;
		pdfDir = 1.0f;
	#endif
		return pos;
	}
	return make_float3( 0 );
}

// EOF