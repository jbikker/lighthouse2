#pragma once

struct TBN
{
	float3 T, B, N;

	__device__ TBN( const float3& T, const float3& N )
	{
		// Setup TBN (Not using Tangent2World/World2Tangent because we already have T, besides N)
		this->N = N;
		this->B = normalize( cross( T, N ) );
		this->T = cross( B, N );
	}

	__device__ float3 WorldToLocal( const float3& v ) const
	{
		return make_float3( dot( v, T ), dot( v, B ), dot( v, N ) );
	}

	__device__ float3 LocalToWorld( const float3& v ) const
	{
		return T * v.x + B * v.y + N * v.z;
	}
};
