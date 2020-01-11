/**
 * BSDF implementation to sample multiple BxDFs.
 *
 * Based on the `BSDF' structure from PBRT, altered to work on the GPU
 * with a type "invariant" list on the stack / in registers.
 */

#pragma once

#ifndef NDEBUG
#define NDEBUG
#endif

#include "VariantStore.h"
#include "bxdf.h"
#include "tbn.h"

template <typename... BxDFs>
class BSDFStackMaterial : public MaterialIntf
{
  protected:
	VariantStore<BxDF, BxDFs...> bxdfs;

	// ----------------------------------------------------------------

  private:
	__device__ float Pdf( const float3 wo, const float3 wi,
						  const BxDFType flags ) const
	{
		int matches = (int)bxdfs.size();
		float pdf = 0.f;
		for ( const auto& bxdf : bxdfs )
		{
			if ( bxdf.MatchesFlags( flags ) )
				pdf += bxdf.Pdf( wo, wi );
			else
				--matches;
		}

		return matches > 0 ? pdf / matches : 0.f;
	}

	// ----------------------------------------------------------------
	// Overrides:

  public:
	/**
	 * Create BxDF stack
	 */
	__device__ void Setup(
		const float3 D,									   // IN:	incoming ray direction, used for consistent normals
		const float u, const float v,					   //		barycentric coordinates of intersection point
		const float coneWidth,							   //		ray cone width, for texture LOD
		const CoreTri4& tri,							   //		triangle data
		const int instIdx,								   //		instance index, for normal transform
		const int materialInstance,						   //		Material instance id/location
		float3& N, float3& iN, float3& fN,				   //		geometric normal, interpolated normal, final normal (normal mapped)
		float3& T,										   //		tangent vector
		const float waveLength = -1.0f,					   // IN:	wavelength (optional)
		const bool allowMultipleLobes = true,			   // IN:	Integrator samples multiple lobes (optional)
		const TransportMode mode = TransportMode::Radiance // IN:	Mode based on integrator (optional)
		) override
	{
		// Empty BxDF stack by default

		// Extract _common_ normal/frame info from triangle.
		// TODO: This should _not_ be the responsibility of the material. REFACTOR!
		float w;
		SetupFrame(
			// In:
			D, u, v, tri, instIdx, /* TODO: Extract smoothnormal information elsewhere */ true,
			// Out:
			N, iN, fN, T, w );
	}

	__device__ bool IsEmissive() const override
	{
		return false;
	}

	__device__ bool IsAlpha() const override
	{
		return false;
	}

	/**
	 * Used to retrieve color for emissive surfaces.
	 */
	__device__ float3 Color() const override
	{
		// TODO:
		return make_float3( 1, 0, 1 );
	}

	__device__ float3 Evaluate( const float3 iN, const float3 Tinit,
								const float3 woWorld, const float3 wiWorld,
								const BxDFType flags,
								float& pdf ) const override
	{
		const TBN tbn( Tinit, iN );
		const float3 wo = tbn.WorldToLocal( woWorld ), wi = tbn.WorldToLocal( wiWorld );

		const bool reflect = dot( wiWorld, iN ) * dot( woWorld, iN ) > 0;
		const BxDFType reflectFlag = reflect ? BxDFType::BSDF_REFLECTION : BxDFType::BSDF_TRANSMISSION;

		// pdf = Pdf( wo, wi, flags );
		// NOTE: Instead of calling a separate function, we are already iterating
		// over and matching bxdfs, so might as well do the sum here.
		pdf = 0.f;
		int matches = (int)bxdfs.size();

		float3 r = make_float3( 0.f );
		for ( const auto& bxdf : bxdfs )
			if ( bxdf.MatchesFlags( flags ) )
			{
				if ( bxdf.HasFlags( reflectFlag ) )
					r += bxdf.f( wo, wi );

				pdf += bxdf.Pdf( wo, wi );
			}
			else
				--matches;

		if ( matches > 0 )
			pdf /= (float)matches;

		return r;
	}

	__device__ float3 Sample( float3 iN, const float3 /* N */, const float3 Tinit,
							  const float3 woWorld, const float distance,
							  float r3, float r4,
							  const BxDFType type,
							  float3& wiWorld, float& pdf,
							  BxDFType& sampledType ) const override
	{

		pdf = 0.f;
		sampledType = BxDFType( 0 );

		const TBN tbn( Tinit, iN );
		const float3 wo = tbn.WorldToLocal( woWorld );

		int matchingComps = (int)bxdfs.size();
		for ( const auto& bxdf : bxdfs )
			if ( !bxdf.MatchesFlags( type ) )
				--matchingComps;

		if ( !matchingComps )
			return make_float3( 0.f );

		// Select a random BxDF (that matches the flags) to sample:
		const int comp = min( (int)floor( r3 * matchingComps ), matchingComps - 1 );

		// Rescale r3:
		r3 = min( r3 * matchingComps - comp, OneMinusEpsilon );

		const BxDF* bxdf = nullptr;
		int count = comp;
		for ( const auto& bxdf_i : bxdfs )
			if ( bxdf_i.MatchesFlags( type ) && count-- == 0 )
			{
				bxdf = &bxdf_i;
				break;
			}

		assert( bxdf );

		sampledType = bxdf->type;
		float3 wi;
		auto f = bxdf->Sample_f( wo, wi, r3, r4, pdf, sampledType );
		wiWorld = tbn.LocalToWorld( wi );

		if ( pdf == 0 )
		{
			sampledType = BxDFType( 0 );
			return make_float3( 0.f );
		}

		// If the selected bxdf is specular (and thus with
		// a specifically chosen direction, wi)
		// this is the only bxdf that is supposed to be sampled.
		if ( bxdf->HasFlags( BxDFType::BSDF_SPECULAR ) )
			return f;

		if ( matchingComps > 1 )
		{
			// TODO: Interpolated normal or geometric normal?
			const bool reflect = dot( wiWorld, iN ) * dot( woWorld, iN ) > 0;
			const BxDFType reflectFlag = reflect
											 ? BxDFType::BSDF_REFLECTION
											 : BxDFType::BSDF_TRANSMISSION;

			for ( const auto& bxdf_i : bxdfs )
				if ( bxdf != &bxdf_i && bxdf_i.MatchesFlags( type ) )
				{
					// Compute overall PDF with all matching _BxDF_s
					pdf += bxdf_i.Pdf( wo, wi );

					// Compute value of BSDF for sampled direction

					// PBRT Resets f to zero and evaluates all bxdfs again.
					// We however keep the evaluation of `bxdf`, just like
					// PBRT does in the pdf sum calculation.
					if ( bxdf_i.HasFlags( reflectFlag ) )
						f += bxdf_i.f( wo, wi );
				}

			pdf /= matchingComps;
		}

		return f;
	}
};

/**
 * Helper class that abstracts away the _massive_ setup function overhead
 */
template <typename... BxDFs>
class SimpleMaterial : public BSDFStackMaterial<BxDFs...>
{
  protected:
	__device__ virtual void ComputeScatteringFunctions( const CoreMaterial& props,
														const float2 uv,
														const bool allowMultipleLobes,
														const TransportMode mode ) = 0;

  public:
	__device__ void Setup(
		const float3 D,									   // IN:	incoming ray direction, used for consistent normals
		const float u, const float v,					   //		barycentric coordinates of intersection point
		const float coneWidth,							   //		ray cone width, for texture LOD
		const CoreTri4& tri,							   //		triangle data
		const int instIdx,								   //		instance index, for normal transform
		const int materialInstance,						   //		Material instance id/location
		float3& N, float3& iN, float3& fN,				   //		geometric normal, interpolated normal, final normal (normal mapped)
		float3& T,										   //		tangent vector
		const float waveLength = -1.0f,					   // IN:	wavelength (optional)
		const bool allowMultipleLobes = true,			   // IN:	Integrator samples multiple lobes (optional)
		const TransportMode mode = TransportMode::Radiance // IN:	Mode based on integrator (optional)
		)
		/* Don't allow overriding this function any further */ final override
	{
		BSDFStackMaterial<BxDFs...>::Setup( D, u, v, coneWidth, tri, instIdx, materialInstance,
											// Output
											N, iN, fN, T, waveLength, allowMultipleLobes, mode );

		const auto props = pbrtMaterials[materialInstance];

		float tu, tv;

		const float4 tdata0 = tri.u4;
		const float4 tdata1 = tri.v4;
		const float w = 1 - ( u + v );
#ifdef OPTIXPRIMEBUILD
		tu = u * TRI_U0 + v * TRI_U1 + w * TRI_U2;
		tv = u * TRI_V0 + v * TRI_V1 + w * TRI_V2;
#else
		tu = w * TRI_U0 + u * TRI_U1 + v * TRI_U2;
		tv = w * TRI_V0 + u * TRI_V1 + v * TRI_V2;
#endif

		ComputeScatteringFunctions( props, make_float2( tu, tv ), allowMultipleLobes, mode );
	}
};
