/* tools.glsl - Copyright 2019 Utrecht University

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

#ifndef TOOLS_H
#define TOOLS_H

#define PI 3.14159265358979323846264f
#define INVPI 0.31830988618379067153777f
#define INV2PI 0.15915494309189533576888f
#define TWOPI 6.28318530717958647692528f
#define SQRT_PI_INV 0.56418958355f
#define LARGE_FLOAT 1e34f
#define EPSILON 0.0001f
#define MINROUGHNESS 0.0001f // minimal GGX roughness
#define BLACK vec( 0 )
#define WHITE vec( 1 )
#define MIPLEVELCOUNT 5
#define BILINEAR 1

#define ARGB32 0
#define ARGB128 1
#define NRM32 2

#define NOHIT -1

#define APPLYSAFENORMALS if (dot( N, wi ) <= 0) pdf = 0;

void FIXNAN_VEC3( inout vec3 x )
{
	if (isnan( x.x ) || isnan( x.y ) || isnan( x.z ))
	{
		x = vec3( 0.0 );
	}
}

void FIXNAN_VEC4( inout vec4 x )
{
	if (isnan( x.x ) || isnan( x.y ) || isnan( x.z ) || isnan( x.w ))
	{
		x = vec4( 0.0 );
	}
}

void CLAMPINTENSITY( inout vec3 contribution, const float clampValue )
{
	const float v = max( contribution.x, max( contribution.y, contribution.z ) );
	if (v > clampValue)
	{
		const float m = clampValue / v;
		contribution.xyz = contribution.xyz * m;
		/* don't touch w */
	}
}

vec3 min3( const vec3 a, const vec3 b ) { return vec3( min( a.x, b.x ), min( a.y, b.y ), min( a.z, b.z ) ); }
vec3 min3( const vec3 a, const float b ) { return vec3( min( a.x, b ), min( a.y, b ), min( a.z, b ) ); }
vec3 min3( const float a, const vec3 b ) { return vec3( min( a, b.x ), min( a, b.y ), min( a, b.z ) ); }
vec3 max3( const vec3 a, const vec3 b ) { return vec3( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ) ); }
vec3 max3( const vec3 a, const float b ) { return vec3( max( a.x, b ), max( a.y, b ), max( a.z, b ) ); }
vec3 max3( const float a, const vec3 b ) { return vec3( max( a, b.x ), max( a, b.y ), max( a, b.z ) ); }
vec4 max4( const vec4 a, const vec4 b ) { return vec4( max( a.x, b.x ), max( a.y, b.y ), max( a.z, b.z ), max( a.w, b.w ) ); }

uint PackNormal( const vec3 N )
{
#if 1
	// more efficient
	const float f = 65535.0f / sqrt( 8.0f * N.z + 8.0f );
	return uint(N.x * f + 32767.0f) + (uint(N.y * f + 32767.0f) << 16);
#else
	vec2 enc = normalize( vec2( N ) ) * (sqrt( -N.z * 0.5f + 0.5f ));
	enc = enc * 0.5f + 0.5f;
	return uint(enc.x * 65535.0f) + (uint(enc.y * 65535.0f) << 16);
#endif
}

vec3 UnpackNormal( uint p )
{
	vec4 nn = vec4( float( p & 65535 ) * (2.0f / 65535.0f), float( p >> 16 ) * (2.0f / 65535.0f), 0, 0 );
	nn += vec4( -1, -1, 1, -1 );
	float l = dot( nn.xyz, -nn.xyz );
	nn.z = l, l = sqrt( l ), nn.x *= l, nn.y *= l;
	return vec3( nn ) * 2.0f + vec3( 0, 0, -1 );
}

// alternative method
uint PackNormal2( vec3 N )
{
	// simple, and good enough discrimination of normals for filtering.
	const uint x = clamp( uint((N.x + 1) * 511), 0u, 1023u );
	const uint y = clamp( uint((N.y + 1) * 511), 0u, 1023u );
	const uint z = clamp( uint((N.z + 1) * 511), 0u, 1023u );
	return (x << 2u) + (y << 12u) + (z << 22u);
}

vec3 UnpackNormal2( uint pi )
{
	const uint x = (pi >> 2u) & 1023u;
	const uint y = (pi >> 12u) & 1023u;
	const uint z = pi >> 22u;
	return vec3( x * (1.0f / 511.0f) - 1, y * (1.0f / 511.0f) - 1, z * (1.0f / 511.0f) - 1 );
}

// color conversions
vec3 RGBToYCoCg( const vec3 RGB )
{
	const vec3 rgb = min3( vec3( 4 ), RGB ); // clamp helps AA for strong HDR
	const float Y = dot( rgb, vec3( 1, 2, 1 ) ) * 0.25f;
	const float Co = dot( rgb, vec3( 2, 0, -2 ) ) * 0.25f + (0.5f * 256.0f / 255.0f);
	const float Cg = dot( rgb, vec3( -1, 2, -1 ) ) * 0.25f + (0.5f * 256.0f / 255.0f);
	return vec3( Y, Co, Cg );
}

vec3 YCoCgToRGB( const vec3 YCoCg )
{
	const float Y = YCoCg.x;
	const float Co = YCoCg.y - (0.5f * 256.0f / 255.0f);
	const float Cg = YCoCg.z - (0.5f * 256.0f / 255.0f);
	return vec3( Y + Co - Cg, Y + Cg, Y - Co - Cg );
}

float Luminance( vec3 rgb )
{
	return 0.299f * min( rgb.x, 10.0f ) + 0.587f * min( rgb.y, 10.0f ) + 0.114f * min( rgb.z, 10.0f );
}

uint HDRtoRGB32( const vec3 c )
{
	const uint r = uint(1023.0f * min( 1.0f, c.x ));
	const uint g = uint(2047.0f * min( 1.0f, c.y ));
	const uint b = uint(2047.0f * min( 1.0f, c.z ));
	return (r << 22) + (g << 11) + b;
}
vec3 RGB32toHDR( const uint c )
{
	return vec3(
		float( c >> 22 ) * (1.0f / 1023.0f),
		float( (c >> 11) & 2047 ) * (1.0f / 2047.0f),
		float( c & 2047 ) * (1.0f / 2047.0f) );
}
vec3 RGB32toHDRmin1( const uint c )
{
	return vec3(
		float( max( 1u, c >> 22 ) * (1.0f / 1023.0f) ),
		float( max( 1u, (c >> 11) & 2047 ) * (1.0f / 2047.0f) ),
		float( max( 1u, c & 2047 ) * (1.0f / 2047.0f) ) );
}

//vec4 SampleSkydome( vec3 D, const int pathLength )
//{
//	// formulas by Paul Debevec, http://www.pauldebevec.com/Probes
//	uint u = uint( skywidth * 0.5f * ( 1.0f + atan2( D.x, -D.z ) * INVPI ) );
//	uint v = uint( skyheight * acos( D.y ) * INVPI );
//	uint idx = u + v * skywidth;
//	return idx < skywidth * skyheight ? vec4( skyPixels[idx], 1.0f ) : vec4( 0 );
//}

float SurvivalProbability( const vec3 diffuse )
{
	return min( 1.0f, max( max( diffuse.x, diffuse.y ), diffuse.z ) );
}

float FresnelDielectricExact( const vec3 wo, const vec3 N, float eta )
{
	if (eta <= 1.0f) return 0.0f;
	const float cosThetaI = max( 0.0f, dot( wo, N ) );
	float scale = 1 / eta, cosThetaTSqr = 1 - (1 - cosThetaI * cosThetaI) * (scale * scale);
	if (cosThetaTSqr <= 0.0f) return 1.0f;
	float cosThetaT = sqrt( cosThetaTSqr );
	float Rs = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
	float Rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
	return 0.5f * (Rs * Rs + Rp * Rp);
}

vec3 Tangent2World( const vec3 V, const vec3 N )
{
	// "Building an Orthonormal Basis, Revisited"
#if 0
	const float signf = 1.0f * sign( N.z );
	const float a = -1.0f / (signf + N.z);
	const float b = N.x * N.y * a;
	const vec3 B = vec3( 1.0f + signf * N.x * N.x * a, signf * b, -signf * N.x );
	const vec3 T = vec3( b, signf + N.y * N.y * a, -N.y );
#else
	vec3 B, T;
	if (N.z < 0.)
	{
		const float a = 1.0f / (1.0f - N.z);
		const float b = N.x * N.y * a;
		B = vec3( 1.0f - N.x * N.x * a, -b, N.x );
		T = vec3( b, N.y * N.y * a - 1.0f, -N.y );
	}
	else
	{
		const float a = 1.0f / (1.0f + N.z);
		const float b = -N.x * N.y * a;
		B = vec3( 1.0f - N.x * N.x * a, b, -N.x );
		T = vec3( b, 1.0f - N.y * N.y * a, -N.y );
	}
#endif
	return V.x * T + V.y * B + V.z * N;
}

vec3 World2Tangent( const vec3 V, const vec3 N )
{
	float signf = sign( N.z );
	const float a = -1.0f / (signf + N.z);
	const float b = N.x * N.y * a;
	const vec3 B = vec3( 1.0f + signf * N.x * N.x * a, signf * b, -signf * N.x );
	const vec3 T = vec3( b, signf + N.y * N.y * a, -N.y );
	return vec3( dot( V, T ), dot( V, B ), dot( V, N ) );
}

vec3 DiffuseReflectionUniform( const float r0, const float r1 )
{
	const float term1 = TWOPI * r0, term2 = sqrt( 1 - r1 * r1 );
	float s = sin( term1 );
	float c = cos( term1 );
	return vec3( c * term2, s * term2, r1 );
}

vec3 DiffuseReflectionCosWeighted( const float r0, const float r1 )
{
	const float term1 = TWOPI * r0;
	const float  term2 = sqrt( 1.0 - r1 );
	const float s = sin( term1 );
	const float c = cos( term1 );
	return vec3( c * term2, s * term2, sqrt( r1 ) );
}

// origin offset

vec3 SafeOrigin( const vec3 O, const vec3 R, const vec3 N, const float geoEpsilon )
{
	// offset outgoing ray direction along R and / or N: along N when strongly parallel to the origin surface; mostly along R otherwise
	const float parallel = 1 - abs( dot( N, R ) );
	const float v = parallel * parallel;
#if 0
	// we can go slightly into the surface when iN != N; negate the offset along N in that case
	const float side = dot( N, R ) < 0 ? -1 : 1;
#else
	// negating offset along N only makes sense once we backface cull
	const float side = 1.0f;
#endif
	return O + R * geoEpsilon * (1 - v) + N * side * geoEpsilon * v;
}

vec3 ConsistentNormal( const vec3 D, const vec3 iN, const float alpha )
{
	// part of the implementation of "Consistent Normal Interpolation", Reshetov et al., 2010
	// calculates a safe normal given an incoming direction, phong normal and alpha
#ifndef CONSISTENTNORMALS
	return iN;
#else
#if 1
	// Eq. 1, exact
	const float q = (1 - sin( alpha )) / (1 + sin( alpha ));
#else
	// Eq. 1 approximation, as in Figure 6 (not the wrong one in Table 8)
	const float t = PI - 2 * alpha;
	const float q = (t * t) / (PI * (PI + (2 * PI - 4) * alpha));
#endif
	const float b = dot( D, iN );
	const float g = 1 + q * (b - 1);
	const float rho = sqrt( q * (1 + g) / (1 + b) );
	const vec3 Rc = (g + rho * b) * iN - (rho * D);
	return normalize( D + Rc );
#endif
}

vec4 CombineToVec4( const vec3 A, const vec3 B )
{
	// convert two float4's to a single uint4, where each int stores two components of the input vectors.
	// assumptions:
	// - the input is possitive
	// - the input can be safely clamped to 31.999
	// with this in mind, the data is stored as 5:11 unsigned fixed point, which should be plenty.
	const uint Ar = uint(min( A.x, 31.999f ) * 2048.0f);
	const uint Ag = uint(min( A.y, 31.999f ) * 2048.0f);
	const uint Ab = uint(min( A.z, 31.999f ) * 2048.0f);
	const uint Br = uint(min( B.x, 31.999f ) * 2048.0f);
	const uint Bg = uint(min( B.y, 31.999f ) * 2048.0f);
	const uint Bb = uint(min( B.z, 31.999f ) * 2048.0f);
	return vec4( uintBitsToFloat( (Ar << 16) + Ag ), uintBitsToFloat( Ab ), uintBitsToFloat( (Br << 16) + Bg ), uintBitsToFloat( Bb ) );
}

vec3 GetDirectFromFloat4( const vec4 X )
{
	const uint v0 = floatBitsToUint( X.x ), v1 = floatBitsToUint( X.y );
	return vec3( float( v0 >> 16 ) * (1.0f / 2048.0f), float( v0 & 65535 ) * (1.0f / 2048.0f), float( v1 ) * (1.0f / 2048.0f) );
}

vec3 GetIndirectFromFloat4( const vec4 X )
{
	const uint v2 = floatBitsToUint( X.z ), v3 = floatBitsToUint( X.w );
	return vec3( float( v2 >> 16 ) * (1.0f / 2048.0f), float( v2 & 65535 ) * (1.0f / 2048.0f), float( v3 ) * (1.0f / 2048.0f) );
}

#endif