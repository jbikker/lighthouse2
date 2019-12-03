/* material_shared.cu - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   THIS IS A SHARED FILE:
   used in RenderCore_OptixPrime and RenderCore_OptixRTX.
*/

LH2_DEVFUNC float3 linear_rgb_to_ciexyz( const float3 rgb )
{
	return make_float3(
		max( 0.0f, 0.412453f * rgb.x + 0.357580f * rgb.y + 0.180423f * rgb.z ),
		max( 0.0f, 0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z ),
		max( 0.0f, 0.019334f * rgb.x + 0.119193f * rgb.y + 0.950227f * rgb.z ) );
}

LH2_DEVFUNC float3 ciexyz_to_linear_rgb( const float3 xyz )
{
	return make_float3(
		max( 0.0f, 3.240479f * xyz.x - 1.537150f * xyz.y - 0.498535f * xyz.z ),
		max( 0.0f, -0.969256f * xyz.x + 1.875992f * xyz.y + 0.041556f * xyz.z ),
		max( 0.0f, 0.055648f * xyz.x - 0.204043f * xyz.y + 1.057311f * xyz.z ) );
}

LH2_DEVFUNC void GetShadingData(
	const float3 D,							// IN:	incoming ray direction, used for consistent normals
	const float u, const float v,			//		barycentric coordinates of intersection point
	const float coneWidth,					//		ray cone width, for texture LOD
	const CoreTri4& tri,					//		triangle data
	const int instIdx,						//		instance index, for normal transform
	ShadingData& retVal,					// OUT:	material properties of the intersection point
	float3& N, float3& iN, float3& fN,		//		geometric normal, interpolated normal, final normal (normal mapped)
	float3& T,								//		tangent vector
	const float waveLength = -1.0f			// IN:	wavelength (optional)
)
{
	// Note: GetShadingData is called from the 'shade' code, which is in turn
	// only called for intersections. We thus can assume that we have a valid
	// triangle reference.
	const float4 tdata1 = tri.v4;
	const float4 tdata2 = tri.vN0;
	const float4 tdata3 = tri.vN1;
	const float4 tdata4 = tri.vN2;
	const float4 tdata5 = tri.T4;
	// fetch initial set of data from material
	const CoreMaterial4& mat = (const CoreMaterial4&)materials[TRI_MATERIAL];
	const uint4 baseData = mat.baseData4;
	// process common data (unconditional)
	const uint part0 = baseData.x; // diffuse_r, diffuse_g
	const uint part1 = baseData.y; // diffuse_b, medium_r
	const uint part2 = baseData.z; // medium_g, medium_b
	const uint flags = baseData.w;
	const float2 base_rg = __half22float2( __halves2half2( __ushort_as_half( part0 & 0xffff ), __ushort_as_half( part0 >> 16 ) ) );
	const float2 base_b_medium_r = __half22float2( __halves2half2( __ushort_as_half( part1 & 0xffff ), __ushort_as_half( part1 >> 16 ) ) );
	const float2 medium_gb = __half22float2( __halves2half2( __ushort_as_half( part2 & 0xffff ), __ushort_as_half( part2 >> 16 ) ) );
	ShadingData4& retVal4 = (ShadingData4&)retVal;
	retVal4.data0 = make_float4( base_rg.x, base_rg.y, base_b_medium_r.x, __uint_as_float( 0 ) );
	retVal4.data1 = make_float4( base_b_medium_r.y, medium_gb.x, medium_gb.y, __uint_as_float( 0 /* matid? */ ) );
	retVal4.data2 = mat.parameters;
	const float3 tint_xyz = linear_rgb_to_ciexyz( make_float3( base_rg.x, base_rg.y, base_b_medium_r.x ) );
	retVal4.tint4 = make_float4( tint_xyz.y > 0 ? ciexyz_to_linear_rgb( tint_xyz * (1.0f / tint_xyz.y) ) : make_float3( 1 ), tint_xyz.y );
	// initialize normals
	N = iN = fN = TRI_N;
	T = TRI_T;
	const float w = 1 - (u + v);
	// calculate interpolated normal
#ifdef OPTIXPRIMEBUILD
	if (MAT_HASSMOOTHNORMALS) iN = normalize( u * TRI_N0 + v * TRI_N1 + w * TRI_N2 );
#else
	if (MAT_HASSMOOTHNORMALS) iN = normalize( w * TRI_N0 + u * TRI_N1 + v * TRI_N2 );
#endif
	// transform the normals for the current instance
	const float3 A = make_float3( instanceDescriptors[instIdx].invTransform.A );
	const float3 B = make_float3( instanceDescriptors[instIdx].invTransform.B );
	const float3 C = make_float3( instanceDescriptors[instIdx].invTransform.C );
	N = N.x * A + N.y * B + N.z * C, iN = iN.x * A + iN.y * B + iN.z * C;
	// "Consistent Normal Interpolation", Reshetov et al., 2010
	const float4 vertexAlpha = tri.alpha4;
	const bool backSide = dot( D, N ) > 0;
#ifdef CONSISTENTNORMALS
#ifdef OPTIXPRIMEBUILD
	const float alpha = u * vertexAlpha.x + v * vertexAlpha.y + w * vertexAlpha.z;
#else
	const float alpha = w * vertexAlpha.x + u * vertexAlpha.y + v * vertexAlpha.z;
#endif
	iN = (backSide ? -1.0f : 1.0f) * ConsistentNormal( D * -1.0f, backSide ? (iN * -1.0f) : iN, alpha );
#endif
	fN = iN;
	// texturing
	float tu, tv;
	if (MAT_HASDIFFUSEMAP || MAT_HAS2NDDIFFUSEMAP || MAT_HAS3RDDIFFUSEMAP || MAT_HASSPECULARITYMAP ||
		MAT_HASNORMALMAP || MAT_HAS2NDNORMALMAP || MAT_HAS3RDNORMALMAP || MAT_HASROUGHNESSMAP)
	{
		const float4 tdata0 = tri.u4;
		const float w = 1 - (u + v);
	#ifdef OPTIXPRIMEBUILD
		tu = u * TRI_U0 + v * TRI_U1 + w * TRI_U2;
		tv = u * TRI_V0 + v * TRI_V1 + w * TRI_V2;
	#else
		tu = w * TRI_U0 + u * TRI_U1 + v * TRI_U2;
		tv = w * TRI_V0 + u * TRI_V1 + v * TRI_V2;
	#endif
	}
	if (MAT_HASDIFFUSEMAP)
	{
		// determine LOD
		const float lambda = TRI_LOD + log2f( coneWidth * (1.0f / fabs( dot( D, N ) )) ); // eq. 26
		const uint4 data = mat.t0data4;
		// fetch texels
		const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( data.y & 0xffff ), __ushort_as_half( data.y >> 16 ) ) );
		const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( data.z & 0xffff ), __ushort_as_half( data.z >> 16 ) ) );
		const float4 texel = FetchTexelTrilinear( lambda, uvscale * (uvoffs + make_float2( tu, tv )), data.w, data.x & 0xffff, data.x >> 16 );
		if (MAT_HASALPHA && texel.w < 0.5f)
		{
			retVal.flags |= 1;
			return;
		}
		retVal.color = retVal.color * make_float3( texel );
		if (MAT_HAS2NDDIFFUSEMAP) // must have base texture; second and third layers are additive
		{
			const uint4 data = mat.t1data4;
			const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( data.y & 0xffff ), __ushort_as_half( data.y >> 16 ) ) );
			const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( data.z & 0xffff ), __ushort_as_half( data.z >> 16 ) ) );
			retVal.color += make_float3( FetchTexel( uvscale * (uvoffs + make_float2( tu, tv )), data.w, data.x & 0xffff, data.x >> 16 ) ) - make_float3( 0.5f );
		}
		if (MAT_HAS3RDDIFFUSEMAP)
		{
			const uint4 data = mat.t2data4;
			const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( data.y & 0xffff ), __ushort_as_half( data.y >> 16 ) ) );
			const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( data.z & 0xffff ), __ushort_as_half( data.z >> 16 ) ) );
			retVal.color += make_float3( FetchTexel( uvscale * (uvoffs + make_float2( tu, tv )), data.w, data.x & 0xffff, data.x >> 16 ) ) - make_float3( 0.5f );
		}
	}
	// normal mapping
	if (MAT_HASNORMALMAP)
	{
		// fetch bitangent for applying normal map vector to geometric normal
		float4 tdata6 = tri.B4;
		float3 B = TRI_B;
		const uint4 data = mat.n0data4;
		const uint part3 = baseData.z;
		const float n0scale = copysignf( -0.0001f + 0.0001f * __expf( 0.1f * fabsf( (float)((part3 >> 8) & 255) - 128.0f ) ), (float)((part3 >> 8) & 255) - 128.0f );
		const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( data.y & 0xffff ), __ushort_as_half( data.y >> 16 ) ) );
		const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( data.z & 0xffff ), __ushort_as_half( data.z >> 16 ) ) );
		float3 shadingNormal = (make_float3( FetchTexel( uvscale * (uvoffs + make_float2( tu, tv )), data.w, data.x & 0xffff, data.x >> 16, NRM32 ) ) - make_float3( 0.5f )) * 2.0f;
		shadingNormal.x *= n0scale, shadingNormal.y *= n0scale;
		if (MAT_HAS2NDNORMALMAP)
		{
			const uint4 data = mat.n1data4;
			const float n1scale = copysignf( -0.0001f + 0.0001f * __expf( 0.1f * ((float)((part3 >> 16) & 255) - 128.0f) ), (float)((part3 >> 16) & 255) - 128.0f );
			const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( data.y & 0xffff ), __ushort_as_half( data.y >> 16 ) ) );
			const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( data.z & 0xffff ), __ushort_as_half( data.z >> 16 ) ) );
			float3 normalLayer1 = (make_float3( FetchTexel( uvscale * (uvoffs + make_float2( tu, tv )), data.w, data.x & 0xffff, data.x >> 16, NRM32 ) ) - make_float3( 0.5f )) * 2.0f;
			normalLayer1.x *= n1scale, normalLayer1.y *= n1scale;
			shadingNormal += normalLayer1;
		}
		if (MAT_HAS3RDNORMALMAP)
		{
			const uint4 data = mat.n2data4;
			const float n2scale = copysignf( -0.0001f + 0.0001f * __expf( 0.1f * ((float)(part3 >> 24) - 128.0f) ), (float)(part3 >> 24) - 128.0f );
			const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( data.y & 0xffff ), __ushort_as_half( data.y >> 16 ) ) );
			const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( data.z & 0xffff ), __ushort_as_half( data.z >> 16 ) ) );
			float3 normalLayer2 = (make_float3( FetchTexel( uvscale * (uvoffs + make_float2( tu, tv )), data.w, data.x & 0xffff, data.x >> 16, NRM32 ) ) - make_float3( 0.5f )) * 2.0f;
			normalLayer2.x *= n2scale, normalLayer2.y *= n2scale;
			shadingNormal += normalLayer2;
		}
		shadingNormal = normalize( shadingNormal );
		fN = normalize( shadingNormal.x * T + shadingNormal.y * B + shadingNormal.z * iN );
	}
	// roughness map
	if (MAT_HASROUGHNESSMAP)
	{
		const uint4 data = mat.rdata4;
		const float2 uvscale = __half22float2( __halves2half2( __ushort_as_half( data.y & 0xffff ), __ushort_as_half( data.y >> 16 ) ) );
		const float2 uvoffs = __half22float2( __halves2half2( __ushort_as_half( data.z & 0xffff ), __ushort_as_half( data.z >> 16 ) ) );
		const uint blend = (retVal.parameters.x & 0xffffff) +
			(((uint)(FetchTexel( uvscale * (uvoffs + make_float2( tu, tv )), data.w, data.x & 0xffff, data.x >> 16 ).x * 255.0f)) << 24);
		retVal.parameters.x = blend;
	}
#ifdef FILTERINGCORE
	// prevent r, g and b from becoming zero, for albedo separation
	retVal.color.x = max( 0.05f, retVal.color.x );
	retVal.color.y = max( 0.05f, retVal.color.y );
	retVal.color.z = max( 0.05f, retVal.color.z );
#endif
}

// EOF