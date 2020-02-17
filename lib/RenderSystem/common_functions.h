/* common_functions.h - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   In this file: functions that are availble both in CPU and core code.
*/

#pragma once

#ifdef __CUDACC__
#define FUNCTYPE LH2_DEVFUNC
#else
#define FUNCTYPE static
#ifndef __USE_GNU // sincosf is a GNU extension
static inline void sincosf( const float a, float* s, float* c ) { *s = sinf( a ); *c = cosf( a ); }
#endif
#endif

FUNCTYPE float3 RandomBarycentrics( const float r0 )
{
	const uint uf = (uint)(r0 * (1ull << 32));			// convert to 0:32 fixed point
	float2 A = make_float2( 1, 0 ), B = make_float2( 0, 1 ), C = make_float2( 0, 0 ); // barycentrics
	for (int i = 0; i < 16; ++i)						// for each base-4 digit
	{
		const int d = (uf >> (2 * (15 - i))) & 0x3;		// get the digit
		float2 An, Bn, Cn;
		switch (d)
		{
		case 0: An = (B + C) * 0.5f; Bn = (A + C) * 0.5f; Cn = (A + B) * 0.5f; break;
		case 1: An = A; Bn = (A + B) * 0.5f; Cn = (A + C) * 0.5f; break;
		case 2: An = (B + A) * 0.5f; Bn = B; Cn = (B + C) * 0.5f; break;
		case 3: An = (C + A) * 0.5f; Bn = (C + B) * 0.5f; Cn = C; break;
		}
		A = An, B = Bn, C = Cn;
	}
	const float2 r = (A + B + C) * 0.3333333f;
	return make_float3( r.x, r.y, 1 - r.x - r.y );
}

FUNCTYPE void SetupTangentSpace( const float3& N, float3& T, float3& B )
{
	// "Building an Orthonormal Basis, Revisited"
	float sign = copysignf( 1.0f, N.z );
	const float a = -1.0f / (sign + N.z);
	const float b = N.x * N.y * a;
	B = make_float3( 1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x );
	T = make_float3( b, sign + N.y * N.y * a, -N.y );
}

FUNCTYPE float3 Tangent2World( const float3& V, const float3& N )
{
	float3 T, B;
	SetupTangentSpace( N, T, B );
	return V.x * T + V.y * B + V.z * N;
}

FUNCTYPE float3 Tangent2World( const float3& V, const float3& N, const float3& T, const float3& B )
{
	return V.x * T + V.y * B + V.z * N;
}

FUNCTYPE float3 World2Tangent( const float3& V, const float3& N )
{
	float sign = copysignf( 1.0f, N.z );
	const float a = -1.0f / (sign + N.z);
	const float b = N.x * N.y * a;
	const float3 B = make_float3( 1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x );
	const float3 T = make_float3( b, sign + N.y * N.y * a, -N.y );
	return make_float3( dot( V, T ), dot( V, B ), dot( V, N ) );
}

FUNCTYPE float3 World2Tangent( const float3& V, const float3& N, const float3& T, const float3& B )
{
	return make_float3( dot( V, T ), dot( V, B ), dot( V, N ) );
}

FUNCTYPE float3 DiffuseReflectionCosWeighted( const float r0, const float r1 )
{
	const float term1 = TWOPI * r0, term2 = sqrtf( 1 - r1 );
	float s, c;
	sincosf( term1, &s, &c );
	return make_float3( c * term2, s * term2, sqrtf( r1 ) );
}

FUNCTYPE float3 DiffuseReflectionUniform( const float r0, const float r1 )
{
	const float term1 = TWOPI * r0, term2 = sqrtf( 1 - r1 * r1 );
	float s, c;
	sincosf( term1, &s, &c );
	return make_float3( c * term2, s * term2, r1 );
}

FUNCTYPE float3 UniformSampleSphere( const float r0, const float r1 )
{
	const float z = 1.0f - 2.0f * r1; // [-1~1]
	const float term1 = TWOPI * r0, term2 = sqrtf( 1 - z * z );
	float s, c;
	sincosf( term1, &s, &c );
	return make_float3( c * term2, s * term2, z );
}

FUNCTYPE float3 UniformSampleCone( const float r0, const float r1, const float cos_outer )
{
	float cosTheta = 1.0f - r1 + r1 * cos_outer;
	float term2 = sqrtf( 1 - cosTheta * cosTheta );
	const float term1 = TWOPI * r0;
	float s, c;
	sincosf( term1, &s, &c );
	return make_float3( c * term2, s * term2, cosTheta );
}

FUNCTYPE float3 CatmullRom( const float3& p0, const float3& p1, const float3& p2, const float3& p3, const float t )
{
	const float3 a = 2 * p1;
	const float3 b = p2 - p0;
	const float3 c = 2 * p0 - 5 * p1 + 4 * p2 - p3;
	const float3 d = -1 * p0 + 3 * p1 - 3 * p2 + p3;
	return 0.5f * (a + (b * t) + (c * t * t) + (d * t * t * t));
}

// EOF