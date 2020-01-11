/**
 * BXDF interfaces and reflection/transmission types.
 *
 * Based on PBRT interface, modified for the GPU.
 */

#pragma once

namespace pbrt
{
#include "pbrt/util.h"
};
using namespace pbrt;

class BxDF : public HasPlacementNewOperator
{
  protected:
	__device__ BxDF( BxDFType type ) : type( type ) {}

  public:
	const BxDFType type;

	__device__ bool MatchesFlags( BxDFType t ) const
	{
		return ( type & t ) == type;
	}

	__device__ bool HasFlags( BxDFType t ) const
	{
		return ( type & t ) == t;
	}

	__device__ virtual float3 f( const float3& wo, const float3& wi ) const = 0;

	__device__ virtual float3 Sample_f( const float3 wo, float3& wi,
										/*  const Point2f& u, */ const float r0, const float r1,
										float& pdf, BxDFType& sampledType ) const
	{
		// Cosine-sample the hemisphere, flipping the direction if necessary
		wi = CosineSampleHemisphere( r0, r1 );
		if ( wo.z < 0 ) wi.z *= -1;
		pdf = Pdf( wo, wi );
		return f( wo, wi );
	}

	__device__ virtual float Pdf( const float3& wo, const float3& wi ) const
	{
		return SameHemisphere( wo, wi ) ? AbsCosTheta( wi ) * INVPI : 0;
	}
};

// Templated type that may ever be helpful
template <typename Derived, BxDFType _type>
class BxDF_T : public BxDF
{
  protected:
	__device__ BxDF_T() : BxDF( _type ) {}

  public:
	static constexpr BxDFType type = _type;
};
