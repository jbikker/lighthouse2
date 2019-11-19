/* material.glsl - Copyright 2019 Utrecht University

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

#include "structures.glsl"
#include "tools.glsl"
#include "sampling.glsl"

layout( set = 0, binding = cMATERIALS ) buffer materialBuffer { Material data[]; } materials;

void GetShadingData(
	const vec3 D, // IN: incoming ray direction
	const float u, const float v, // barycentrics
	const float coneWidth, // ray cone width, for texture LOD
	const CoreTri tri, // triangle data
	const int instIdx, // instance index
	inout ShadingData retVal, // material properties for current intersection
	inout vec3 N, inout vec3 iN, inout vec3 fN, // geometric normal, interpolated normal, final normal
	inout vec3 T,  // tangent vector
	const mat3 invTransform // inverse instance transformation matrix
)
{
	const float w = 1.0f - u - v;
	const vec4 tdata1 = tri.v4;
	const vec4 tdata2 = tri.vN0;
	const vec4 tdata3 = tri.vN1;
	const vec4 tdata4 = tri.vN2;
	const vec4 tdata5 = tri.T4;

	const Material mat = materials.data[TRI_MATERIAL]; // material data

	const uvec4 baseData = mat.baseData4;

	const vec2 base_rg = unpackHalf2x16( baseData.x );
	const vec2 base_b_medium_r = unpackHalf2x16( baseData.y );
	const vec2 medium_gb = unpackHalf2x16( baseData.z );
	const uint flags = MAT_FLAGS;

	retVal.color = vec3( base_rg.x, base_rg.y, base_b_medium_r.x ), retVal.flags = 0; // int flags;
	retVal.transmittance = vec3( base_b_medium_r.y, medium_gb.x, medium_gb.y ), retVal.matID = 0;  // int matID;
	retVal.parameters = mat.parameters;

	N = TRI_N, iN = N, fN;
	T = TRI_T;

	if (MAT_HASSMOOTHNORMALS)
		iN = normalize( w * TRI_N0 + u * TRI_N1 + v * TRI_N2 );

	// Transform normals from local space to world space
	N = invTransform * N, iN = invTransform * iN;

	// "Consistent Normal Interpolation", Reshetov et al., 2010
	const vec4 alpha4 = tri.alpha4;
	const float alpha = w * alpha4.x + u * alpha4.y + v * alpha4.z;
	const bool backSide = dot( D, N ) > 0;
	iN = ConsistentNormal( D * -1.0f, backSide ? (iN * -1.0f) : iN, alpha );
	if (backSide) iN *= -1.0f;
	fN = iN;

	// Texturing
	float tu, tv;
	if (MAT_HASDIFFUSEMAP || MAT_HAS2NDDIFFUSEMAP || MAT_HAS3RDDIFFUSEMAP || MAT_HASSPECULARITYMAP ||
		MAT_HASNORMALMAP || MAT_HAS2NDNORMALMAP || MAT_HAS3RDNORMALMAP || MAT_HASROUGHNESSMAP)
	{
		const vec4 tdata0 = tri.u4;
		tu = w * TRI_U0 + u * TRI_U1 + v * TRI_U2;
		tv = w * TRI_V0 + u * TRI_V1 + v * TRI_V2;
	}

	if (MAT_HASDIFFUSEMAP)
	{
		// Determine LOD
		const float lambda = TRI_LOD + log2( coneWidth * (1.0 / abs( dot( D, N ) )) );
		uvec4 data = mat.t0data4;

		vec2 uvscale = unpackHalf2x16( data.y );
		vec2 uvoffs = unpackHalf2x16( data.z );

		// Fetch texels
		const vec4 texel = FetchTexelTrilinear( lambda, uvscale * (uvoffs + vec2( tu, tv )), int( data.w ), int( data.x & 0xFFFF ), int( data.x >> 16 ) );
		if (MAT_HASALPHA && texel.w < 0.5f)
		{
			retVal.flags |= 1;
			return;
		}
		retVal.color = retVal.color * texel.xyz;
		if (MAT_HAS2NDDIFFUSEMAP) // must have base texture; second and third layers are additive
		{
			data = mat.t1data4;
			uvscale = unpackHalf2x16( data.y );
			uvoffs = unpackHalf2x16( data.z );
			retVal.color += FetchTexel( uvscale * (uvoffs + vec2( tu, tv )), int( data.w ), int( data.x & 0xFFFF ), int( data.x >> 16 ), ARGB32 ).xyz;
		}
		if (MAT_HAS3RDDIFFUSEMAP)
		{
			data = mat.t2data4;
			uvscale = unpackHalf2x16( data.y );
			uvoffs = unpackHalf2x16( data.z );
			retVal.color += FetchTexel( uvscale * (uvoffs + vec2( tu, tv )), int( data.w ), int( data.x & 0xFFFF ), int( data.x >> 16 ), ARGB32 ).xyz;
		}
	}
	// Normal mapping
	if (MAT_HASNORMALMAP)
	{
		vec4 tdata6 = tri.B4;
		const vec3 B = TRI_B;
		uvec4 data = mat.n0data4;
		const uint part3 = baseData.z;
		const float n0scale = -0.0001f + 0.0001f * exp( 0.1 * abs( float( (part3 >> 8) & 255 ) - 128.0f ) )* sign( float( (part3 >> 8) & 255 ) - 128.0f );
		vec2 uvscale = unpackHalf2x16( data.y );
		vec2 uvoffs = unpackHalf2x16( data.z );
		vec3 shadingNormal = (FetchTexel( uvscale * (uvoffs + vec2( tu, tv )), int( data.w ), int( data.x & 0xFFFF ), int( data.x >> 16 ), NRM32 ).xyz - vec3( 0.5f )) * 2.0f;
		shadingNormal.x = shadingNormal.x * n0scale;
		shadingNormal.y = shadingNormal.y * n0scale;

		if (MAT_HAS2NDNORMALMAP)
		{
			data = mat.n1data4;
			const float n1scale = -0.0001f + 0.0001f * exp( 0.1 * abs( float( (part3 >> 8) & 255 ) - 128.0f ) )* sign( float( (part3 >> 8) & 255 ) - 128.0f );
			vec2 uvscale = unpackHalf2x16( data.y );
			vec2 uvoffs = unpackHalf2x16( data.z );
			vec3 normalLayer1 = (FetchTexel( uvscale * (uvoffs + vec2( tu, tv )), int( data.w ), int( data.x & 0xFFFF ), int( data.x >> 16 ), NRM32 ).xyz - vec3( 0.5f )) * 2.0f;
			normalLayer1.x = normalLayer1.x * n1scale;
			normalLayer1.y = normalLayer1.y * n1scale;
			shadingNormal += normalLayer1;
		}
		if (MAT_HAS3RDNORMALMAP)
		{
			data = mat.n2data4;
			const float n2scale = -0.0001f + 0.0001f * exp( 0.1 * abs( float( (part3 >> 8) & 255 ) - 128.0f ) )* sign( float( (part3 >> 8) & 255 ) - 128.0f );
			vec2 uvscale = unpackHalf2x16( data.y );
			vec2 uvoffs = unpackHalf2x16( data.z );
			vec3 normalLayer2 = (FetchTexel( uvscale * (uvoffs + vec2( tu, tv )), int( data.w ), int( data.x & 0xFFFF ), int( data.x >> 16 ), NRM32 ).xyz - vec3( 0.5f )) * 2.0f;
			normalLayer2.x = normalLayer2.x * n2scale;
			normalLayer2.y = normalLayer2.y * n2scale;
			shadingNormal += normalLayer2;
		}
		shadingNormal = normalize( shadingNormal );
		fN = normalize( shadingNormal * T + shadingNormal.y * B + shadingNormal.z * iN );
	}

	if (MAT_HASROUGHNESSMAP)
	{
		const uvec4 data = mat.rdata4;
		const vec2 uvscale = unpackHalf2x16( data.y );
		const vec2 uvoffs = unpackHalf2x16( data.z );
		const uint blend = (retVal.parameters.x & 0xffffff) +
			(uint(FetchTexel( uvscale * (uvoffs + vec2( tu, tv )), int( data.w ), int( data.x & 0xffff ), int( data.x >> 16 ), ARGB32 ).x * 255.0f) << 24);
		retVal.parameters.x = blend;
	}
}