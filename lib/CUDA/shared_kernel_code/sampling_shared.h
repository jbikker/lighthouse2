/* sampling_shared.cu - Copyright 2019 Utrecht University

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

LH2_DEVFUNC float mitchellNetravali( const float v )
{
	const float B = 1.0f / 3.0f, C = 1.0f / 3.0f, x = fabs( v ), x2 = x * x, x3 = x2 * x;
	if (x < 1) return (1.0f / 6.0f) * ((12 - 9 * B - 6 * C) * x3 + (-18 + 12 * B + 6 * C) * x2 + (6 - 2 * B));
	else if (x < 2) return 1.0f / 6.0f * ((-B - 6 * C) * x3 + (6 * B + 30 * C) * x2 + (-12 * B - 48 * C) * x + (8 * B + 24 * C)); else return 0.0f;
}

LH2_DEVFUNC float4 __uchar4_to_float4( const uchar4 v4 )
{
	const float r = 1.0f / 256.0f;
	return make_float4( v4.x * r, v4.y * r, v4.z * r, v4.w * r );
}

LH2_DEVFUNC float4 FetchTexel( const float2 texCoord, const int o, const int w, const int h,
	const TexelStorage storage = ARGB32 )
{
	const float2 tc = make_float2( (max( texCoord.x + 1000, 0.0f ) * w) - 0.5f, (max( texCoord.y + 1000, 0.0f ) * h) - 0.5f );
	const int iu = ((int)tc.x) % w;
	const int iv = ((int)tc.y) % h;
#ifdef BILINEAR
	const float fu = tc.x - floor( tc.x );
	const float fv = tc.y - floor( tc.y );
	const float w0 = (1 - fu) * (1 - fv);
	const float w1 = fu * (1 - fv);
	const float w2 = (1 - fu) * fv;
	const float w3 = 1 - (w0 + w1 + w2);
	float4 p0, p1, p2, p3;
	const uint iu1 = (iu + 1) % w, iv1 = (iv + 1) % h;
	if (storage == ARGB32)
		p0 = __uchar4_to_float4( argb32[o + iu + iv * w] ),
		p1 = __uchar4_to_float4( argb32[o + iu1 + iv * w] ),
		p2 = __uchar4_to_float4( argb32[o + iu + iv1 * w] ),
		p3 = __uchar4_to_float4( argb32[o + iu1 + iv1 * w] );
	else if (storage == ARGB128)
		p0 = argb128[o + iu + iv * w],
		p1 = argb128[o + iu1 + iv * w],
		p2 = argb128[o + iu + iv1 * w],
		p3 = argb128[o + iu1 + iv1 * w];
	else /* if (storage == NRM32) */
		p0 = __uchar4_to_float4( nrm32[o + iu + iv * w] ),
		p1 = __uchar4_to_float4( nrm32[o + iu1 + iv * w] ),
		p2 = __uchar4_to_float4( nrm32[o + iu + iv1 * w] ),
		p3 = __uchar4_to_float4( nrm32[o + iu1 + iv1 * w] );
	return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
#else
	if (storage == ARGB32) return __uchar4_to_float4( argb32[o + iu + iv * w] );
	else if (storage == ARGB128) return argb128[o + iu + iv * w];
	/* else if (storage == NRM32) */ return __uchar4_to_float4( nrm32[o + iu + iv * w] );
#endif
}

LH2_DEVFUNC float4 FetchTexelTrilinear( const float lambda, const float2 texCoord, const int offset, const int width, const int height )
{
	const int level0 = min( MIPLEVELCOUNT - 1, (int)lambda );
	const int level1 = min( MIPLEVELCOUNT - 1, level0 + 1 );
	const float f = lambda - floor( lambda );
	// select first MIP level
	int o0 = offset, w0 = width, h0 = height;
	for (int i = 0; i < level0; i++) o0 += w0 * h0, w0 >>= 1, h0 >>= 1;
	// select second MIP level
	int o1 = offset, w1 = width, h1 = height;
	for (int i = 0; i < level1; i++) o1 += w1 * h1, w1 >>= 1, h1 >>= 1; // TODO: start at o0, h0, w0
	// read actual data
	const float4 p0 = FetchTexel( texCoord, o0, w0, h0 );
	const float4 p1 = FetchTexel( texCoord, o1, w1, h1 );
	// final interpolation
	return (1 - f) * p0 + f * p1;
}

LH2_DEVFUNC float3 ReadTexelPoint( const float4* buffer, const float u, const float v, const int w, const int h )
{
	return make_float3( buffer[(int)u + (int)v * w] );
}

LH2_DEVFUNC float4 ReadWorldPos( const float4* buffer, const int2 uv, const int w, const int h )
{
	if (uv.x >= 0 && uv.y >= 0 && uv.x < w && uv.y < h) return buffer[uv.x + uv.y * w];
	return make_float4( 1e20f, 1e20f, 1e20f, __uint_as_float( 0 ) );
}

LH2_DEVFUNC float3 ReadTexelBmitchellNetravali( const float4* buffer, float u, float v, int w, int h )
{
	int x1 = (int)(u - 2.0f), y1 = (int)(v - 2.0f);
	float totalWeight = 0;
	float4 total = make_float4( 0 );
	for (int y = y1; y < y1 + 4; y++) for (int x = x1; x < x1 + 4; x++)
	{
		if (x >= 0 && y > 0 && x < w && y < h)
		{
			float dx = (float)x - u;
			float dy = (float)y - v;
			float weight = mitchellNetravali( dx ) * mitchellNetravali( dy );
			total += buffer[x + y * w] * weight;
			totalWeight += weight;
		}
	}
	return make_float3( total * (1.0f / totalWeight) );
}

LH2_DEVFUNC float4 ReadTexelConsistent( const float4* buffer, const float4* prevWorldPos,
	const float4 localPos, const float allowedDist, const float3 localNormal, float u, float v, int w, int h )
{
	// part of reprojection:
	// read a texel from the specified history buffer with bilinear interpolation,
	// while checking each tap for consistentency (similar world space position and normal).
	const int iu1 = (int)floor( u ), iv1 = (int)floor( v ), iu0 = max( 0, iu1 - 1 ), iv0 = max( 0, iv1 - 1 );
	if (iu1 >= w || iv1 >= h || iu1 < 0 || iv1 < 0) return make_float4( -1 );
	const float2 fuv = make_float2( u - floor( u ), v - floor( v ) );
	const float4 p0 = buffer[iu0 + iv0 * w], pp0 = prevWorldPos[iu0 + iv0 * w];
	const float4 p1 = buffer[iu1 + iv0 * w], pp1 = prevWorldPos[iu1 + iv0 * w];
	const float4 p2 = buffer[iu0 + iv1 * w], pp2 = prevWorldPos[iu0 + iv1 * w];
	const float4 p3 = buffer[iu1 + iv1 * w], pp3 = prevWorldPos[iu1 + iv1 * w];
	const uint localSpecularity = __float_as_uint( localPos.w ) & 3;
	float w0 = (1 - fuv.x) * (1 - fuv.y), w1 = fuv.x * (1 - fuv.y), w2 = (1 - fuv.x) * fuv.y, w3 = 1 - (w0 + w1 + w2);
	{	// scope reduction
		const float posDiff0 = sqrLength( make_float3( pp0 - localPos ) ), posDiff1 = sqrLength( make_float3( pp1 - localPos ) );
		const float posDiff2 = sqrLength( make_float3( pp2 - localPos ) ), posDiff3 = sqrLength( make_float3( pp3 - localPos ) );
		const uint n0 = __float_as_uint( pp0.w ), n1 = __float_as_uint( pp1.w );
		const uint n2 = __float_as_uint( pp2.w ), n3 = __float_as_uint( pp3.w );
		const float dot0 = dot( UnpackNormal2( n0 ), localNormal ), dot1 = dot( UnpackNormal2( n1 ), localNormal );
		const float dot2 = dot( UnpackNormal2( n2 ), localNormal ), dot3 = dot( UnpackNormal2( n3 ), localNormal );
	#if 0
		// suppress but don't kill neighbors
		const float allowedDist2 = allowedDist * allowedDist; // distances are squared too; faster this way
		if (posDiff0 > allowedDist2 || dot0 < 0.99f || (n0 & 3) != localSpecularity) w0 = 0;
		if (posDiff1 > allowedDist2 || dot1 < 0.99f || (n1 & 3) != localSpecularity) w1 = 0;
		if (posDiff2 > allowedDist2 || dot2 < 0.99f || (n2 & 3) != localSpecularity) w2 = 0;
		if (posDiff3 > allowedDist2 || dot3 < 0.99f || (n3 & 3) != localSpecularity) w3 = 0;
	#else
		// relax; let the neighborhood clipping worry about the details
		if (dot0 < 0.95f || (n0 & 3) != localSpecularity) w0 = 0;
		if (dot1 < 0.95f || (n1 & 3) != localSpecularity) w1 = 0;
		if (dot2 < 0.95f || (n2 & 3) != localSpecularity) w2 = 0;
		if (dot3 < 0.95f || (n3 & 3) != localSpecularity) w3 = 0;
	#endif
	}
	const float sum = w0 + w1 + w2 + w3;
	if (sum == 0 /* shouldn't happen */) return make_float4( -1 ); else return (w0 * p0 + w1 * p1 + w2 * p2 + w3 * p3) * (1.0f / sum);
}

LH2_DEVFUNC void ReadTexelConsistent2( const float4* buffer, const float4* prevWorldPos,
	const float4 localPos, const float allowedDist2, const float3 localNormal, float u, float v, int w, int h,
	float3& direct, float3& indirect )
{
	// part of reprojection:
	// read a texel from the specified history buffer with bilinear interpolation,
	// while checking each tap for consistentency (similar world space position and normal).
	// this version interpolates and returns two values (for direct and indirect light).
	direct.x = -1;
	const int iu1 = (int)floor( u ), iv1 = (int)floor( v ), iu0 = max( 0, iu1 - 1 ), iv0 = max( 0, iv1 - 1 );
	if (iu1 >= w || iv1 >= h || iu1 < 0 || iv1 < 0) return;
	const float2 fuv = make_float2( u - floor( u ), v - floor( v ) );
	const float4 p0 = buffer[iu0 + iv0 * w], pp0 = prevWorldPos[iu0 + iv0 * w], pd0 = make_float4( GetDirectFromFloat4( p0 ), 1 ), pi0 = make_float4( GetIndirectFromFloat4( p0 ) );
	const float4 p1 = buffer[iu1 + iv0 * w], pp1 = prevWorldPos[iu1 + iv0 * w], pd1 = make_float4( GetDirectFromFloat4( p1 ), 1 ), pi1 = make_float4( GetIndirectFromFloat4( p1 ) );
	const float4 p2 = buffer[iu0 + iv1 * w], pp2 = prevWorldPos[iu0 + iv1 * w], pd2 = make_float4( GetDirectFromFloat4( p2 ), 1 ), pi2 = make_float4( GetIndirectFromFloat4( p2 ) );
	const float4 p3 = buffer[iu1 + iv1 * w], pp3 = prevWorldPos[iu1 + iv1 * w], pd3 = make_float4( GetDirectFromFloat4( p3 ), 1 ), pi3 = make_float4( GetIndirectFromFloat4( p3 ) );
	const uint localSpecularity = __float_as_uint( localPos.w ) & 3;
	float w0 = (1 - fuv.x) * (1 - fuv.y), w1 = fuv.x * (1 - fuv.y), w2 = (1 - fuv.x) * fuv.y, w3 = 1 - (w0 + w1 + w2);
	{	// scope reduction
		const uint n0 = __float_as_uint( pp0.w ), n1 = __float_as_uint( pp1.w );
		const uint n2 = __float_as_uint( pp2.w ), n3 = __float_as_uint( pp3.w );
		const float dot0 = dot( UnpackNormal2( n0 ), localNormal ), dot1 = dot( UnpackNormal2( n1 ), localNormal );
		const float dot2 = dot( UnpackNormal2( n2 ), localNormal ), dot3 = dot( UnpackNormal2( n3 ), localNormal );
	#if 0
		// suppress but don't kill neighbors
		const float posDiff0 = sqrLength( make_float3( pp0 - localPos ) ), posDiff1 = sqrLength( make_float3( pp1 - localPos ) );
		const float posDiff2 = sqrLength( make_float3( pp2 - localPos ) ), posDiff3 = sqrLength( make_float3( pp3 - localPos ) );
		if (posDiff0 > allowedDist2 || dot0 < 0.95f || (n0 & 3) != localSpecularity) w0 = 0;
		if (posDiff1 > allowedDist2 || dot1 < 0.95f || (n1 & 3) != localSpecularity) w1 = 0;
		if (posDiff2 > allowedDist2 || dot2 < 0.95f || (n2 & 3) != localSpecularity) w2 = 0;
		if (posDiff3 > allowedDist2 || dot3 < 0.95f || (n3 & 3) != localSpecularity) w3 = 0;
	#else
		// relax; let the neighborhood clipping worry about the details
		if (dot0 < 0.975f || (n0 & 3) != localSpecularity) w0 = 0;
		if (dot1 < 0.975f || (n1 & 3) != localSpecularity) w1 = 0;
		if (dot2 < 0.975f || (n2 & 3) != localSpecularity) w2 = 0;
		if (dot3 < 0.975f || (n3 & 3) != localSpecularity) w3 = 0;
	#endif
	}
	const float sum = w0 + w1 + w2 + w3;
	if (sum == 0) return;
	direct = make_float3( w0 * pd0 + w1 * pd1 + w2 * pd2 + w3 * pd3 ) * (1.0f / sum);
	indirect = make_float3( w0 * pi0 + w1 * pi1 + w2 * pi2 + w3 * pi3 ) * (1.0f / sum);
}

#if 0

template <typename T>
__device__ T CoreTexture<T>::Evaluate( float2 uv ) const
{
	switch (type)
	{
	case Constant:
		return constant;
	case Imagemap:
		if (imagemap.trilinear)
			return FetchTexelTrilinear(
				0, uv,
				imagemap.textureOffset,
				imagemap.width,
				imagemap.height );
		else
			return FetchTexel(
				uv,
				imagemap.textureOffset,
				imagemap.width,
				imagemap.height );
	}
	return T{};
}

template <>
__device__ float CoreTexture<float>::Evaluate( float2 uv ) const
{
	switch (type)
	{
	case Constant:
		return constant;
	}
	return 0.f;
}

template <>
__device__ float3 CoreTexture<float3>::Evaluate( float2 uv ) const
{
	switch (type)
	{
	case Constant:
		return constant;
	case Imagemap:
		return make_float3(
			imagemap.trilinear
			? FetchTexelTrilinear(
				0, uv,
				imagemap.textureOffset,
				imagemap.width,
				imagemap.height )
			: FetchTexel(
				uv,
				imagemap.textureOffset,
				imagemap.width,
				imagemap.height ) );
	}
	return make_float3( 0.f );
}

#endif

// EOF