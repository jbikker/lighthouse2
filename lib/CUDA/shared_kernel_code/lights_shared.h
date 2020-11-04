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
// #define LIGHTTREE
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
//  |  CalculateChildNodeWeights                                                  |
//  |  Helper to compute child node weights.                                LH2'20|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float CalculateChildNodeWeights( const int node, const float3& I, const float3& N, uint& seed, const bool debug = false )
{
	const int left = lightTree[node].left;
	const int right = lightTree[node].right;
	const float3 b1j = make_float3( lightTree[left].bmin ), b1k = make_float3( lightTree[right].bmin );
	const float3 b2j = make_float3( lightTree[left].bmax ), b2k = make_float3( lightTree[right].bmax );
	const float3 diag_j = b2j - b1j;
	const float3 diag_k = b2k - b1k;
	const float3 LN = make_float3( lightTree[node].N );
	// calculate (squared) minimum and maximum distance from I to aabb
	// method: GPU-Accelerated Minimum Distance and Clearance Queries, Krishnamurthy et al., 2011
	const float3 Bj = 0.5f * diag_j;
	const float3 Bk = 0.5f * diag_k;
	const float3 Cj = (b1j + b2j) * 0.5f;
	const float3 Ck = (b1k + b2k) * 0.5f;
	const float3 Dj = Cj - I;
	const float3 Dk = Ck - I;
	const float3 min_j = make_float3( max( Dj.x - Bj.x, 0.0f ), max( Dj.y - Bj.y, 0.0f ), max( Dj.z - Bj.z, 0.0f ) );
	const float3 min_k = make_float3( max( Dk.x - Bk.x, 0.0f ), max( Dk.y - Bk.y, 0.0f ), max( Dk.z - Bk.z, 0.0f ) );
	const float dist2j = dot( min_j, min_j );
	const float dist2k = dot( min_k, min_k );
	const float3 max_j = Dj + Bj;
	const float3 max_k = Dk + Bk;
	const float dist2j_max = dot( max_j, max_j );
	const float dist2k_max = dot( max_k, max_k );
	// get the left and right cluster intensities
	const float Ij = lightTree[left].intensity;
	const float Ik = lightTree[right].intensity;
	// get a reasonable value for F using the normals at I and the light
	const float3 Rj = b1j + (b2j - b1j) * make_float3( RandomFloat( seed ), RandomFloat( seed ), RandomFloat( seed ) );
	const float3 Rk = b1k + (b2k - b1k) * make_float3( RandomFloat( seed ), RandomFloat( seed ), RandomFloat( seed ) );
	const float3 Lj = normalize( Rj - I );
	const float3 Lk = normalize( Rk - I );
	float Fj = max( 0.001f, dot( N, Lj ) );
	float Fk = max( 0.001f, dot( N, Lk ) );
	if (dot( LN, LN ) > 0.001f)
		Fj *= max( 0.001f, dot( LN, Lj * -1.0f ) ),
		Fk *= max( 0.001f, dot( LN, Lk * -1.0f ) );
	// calculate final probabilities according to the realtime stochastic lightcuts paper
	const bool insideBoth = dist2j == 0 && dist2k == 0;
	const float wmin_j = (Fj * Ij) / (insideBoth ? 1 : max( 0.0001f, dist2j) );
	const float wmin_k = (Fk * Ik) / (insideBoth ? 1 : max( 0.0001f, dist2k) );
	const float wmax_j = (Fj * Ij) / max( 0.0001f, dist2j_max );
	const float wmax_k = (Fj * Ij) / max( 0.0001f, dist2k_max );
	const float pmin_j = wmin_j / (wmin_j + wmin_k);
	const float pmax_j = wmax_j / (wmax_j + wmax_k);
	return 0.5f * (pmin_j + pmax_j);
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
//  |  LightPickProbLTree                                                         |
//  |  Calculates the probability with which the specified light woukd be picked  |
//  |  from the specified world space location and normal using the stochastic    |
//  |  lightcuts approach.                                                  LH2'20|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float LightPickProbLTree( int idx, const float3& O, const float3& N, const float3& I, uint& seed )
{
#ifndef LIGHTTREE
	return LightPickProb( idx, O, N, I );
#else
	LightCluster* tree = lightTree;
	int node = idx + 1; // leaf for light i is at index i + 1, see UpdateLightTree in rendercore.cpp.
	float pickProb = 1;
	while (1)
	{
		if (node == 0) break; // we are the root node
		// determine probability of selecting the current node over its sibling
		int parent = __float_as_int( tree[node].N.w /* we abused N.w to store the parent node index */ );
		const float p = CalculateChildNodeWeights( parent, I, N, seed );
		if (tree[parent].left == node) pickProb *= p /* we are the left child */; else pickProb *= 1 - p;
		node = parent;
	}
	return pickProb;
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
//  |  RandomPointOnLightLTree                                                    |
//  |  Selects a random point on a random light, using the stochastic lightcuts   |
//  |  approach, via a binary light tree. Default method for the Optix7 core.     |
//  |  Returns a position, a normal on the light source, the pick probability,    |
//  |  and the importance of the explicit connection.                       LH2'20|
//  +-----------------------------------------------------------------------------+
LH2_DEVFUNC float3 RandomPointOnLightLTree( float r0, float r1, uint& seed, const float3& I, const float3& N, float& pickProb, float& lightPdf, float3& lightColor, const bool debug = false )
{
#ifndef LIGHTTREE
	return RandomPointOnLight( r0, r1, I, N, pickProb, lightPdf, lightColor );
#else
	LightCluster* tree = lightTree;
	int node = 0; // index of root node
	int lightIdx = 0;
	pickProb = 1;
	while (1)
	{
		if (tree[node].left == -1)
		{
			// reached a leaf, use this light
			lightIdx = tree[node].light;
			break;
		}
		// interior node; randomly pick a child
		const float p_j = CalculateChildNodeWeights( node, I, N, seed, debug );
		// decide
		if (r1 < p_j)
			node = tree[node].left, r1 *= 1.0f / p_j, pickProb *= p_j;
		else
			node = tree[node].right, r1 = (r1 - p_j) / (1 - p_j), pickProb *= 1 - p_j;
	}
	if (lightIdx & (1 << 30))
	{
		// pick a pointlight
		const CorePointLight4& light = (const CorePointLight4&)pointLights[lightIdx - (1 << 30)];
		const float3 P = make_float3( light.data0 );	// position
		const float3 L = P - I;
		const float sqDist = dot( L, L );
		lightColor = make_float3( light.data1 ) / sqDist;		// radiance
		lightPdf = dot( L, N ) > 0 ? 1 : 0;
		return P;
	}
	else if (lightIdx & (1 << 29))
	{
		// spotlight
		const CoreSpotLight4& light = (const CoreSpotLight4&)spotLights[lightIdx - (1 << 29)];
		const float4 V0 = light.data0;				// position + cos_inner
		const float4 V1 = light.data1;				// radiance + cos_outer
		const float4 D = light.data2;				// direction
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
		// light triangle
		float3 bary = RandomBarycentrics( r0 );
		const CoreLightTri4& light = (const CoreLightTri4&)triLights[lightIdx];
		const float4 V0 = light.data3;				// vertex0
		const float4 V1 = light.data4;				// vertex1
		const float4 V2 = light.data5;				// vertex2
		lightColor = make_float3( light.data2 );	// radiance
		const float4 LN = light.data1;				// N
		const float3 P = make_float3( bary.x * V0 + bary.y * V1 + bary.z * V2 );
		float3 L = I - P;							// reversed: from light to intersection point
		const float sqDist = dot( L, L );
		L = normalize( L );
		const float LNdotL = L.x * LN.x + L.y * LN.y + L.z * LN.z;
		const float reciSolidAngle = sqDist / (LN.w * LNdotL); // LN.w contains area
		lightPdf = (LNdotL > 0 && dot( L, N ) < 0) ? reciSolidAngle : 0;
		return P;
	}
#endif
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