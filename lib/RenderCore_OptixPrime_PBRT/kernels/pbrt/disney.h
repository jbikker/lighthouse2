/*
    pbrt source code is Copyright(c) 1998-2017
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#pragma once

// ----------------------------------------------------------------

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
//
// The Schlick Fresnel approximation is:
//
// R = R(0) + (1 - R(0)) (1 - cos theta)^5,
//
// where R(0) is the reflectance at normal indicence.
LH2_DEVFUNC float SchlickWeight( float cosTheta )
{
	float m = clamp( 1.f - cosTheta, 0.f, 1.f );
	return ( m * m ) * ( m * m ) * m;
}

LH2_DEVFUNC float FrSchlick( float R0, float cosTheta )
{
	return pbrt_Lerp( SchlickWeight( cosTheta ), R0, 1.f );
}

LH2_DEVFUNC float3 FrSchlick( float3 R0, float cosTheta )
{
	return pbrt_Lerp( SchlickWeight( cosTheta ), R0, make_float3( 1.f ) );
}

// For a dielectric, R(0) = (eta - 1)^2 / (eta + 1)^2, assuming we're
// coming from air..
LH2_DEVFUNC float SchlickR0FromEta( float eta ) { return sqr( eta - 1 ) / sqr( eta + 1 ); }

LH2_DEVFUNC float pbrt_GTR1( float cosTheta, float alpha )
{
	float alpha2 = alpha * alpha;
	return ( alpha2 - 1.f ) /
		   ( PI * logf( alpha2 ) *
			 ( 1.f + ( alpha2 - 1.f ) * cosTheta * cosTheta ) );
}

// Smith masking/shadowing term.
LH2_DEVFUNC float smithG_GGX( float cosTheta, float alpha )
{
	float alpha2 = alpha * alpha;
	float cosTheta2 = cosTheta * cosTheta;
	return 1.f / ( cosTheta + sqrtf( alpha2 + cosTheta2 - alpha2 * cosTheta2 ) );
}

// ----------------------------------------------------------------

class DisneyDiffuse : public BxDF_T<DisneyDiffuse,
									BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE )>
{
	const float3 R;

  public:
	__device__ DisneyDiffuse( const float3& r ) : R( r )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		const float Fo = SchlickWeight( AbsCosTheta( wo ) ),
					Fi = SchlickWeight( AbsCosTheta( wi ) );

		// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
		// Burley 2015, eq (4).
		return R * INVPI * ( 1.f - Fo * .5f ) * ( 1.f - Fi * .5f );
	}
};

class DisneyFakeSS : public BxDF_T<DisneyFakeSS,
								   BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE )>
{
	const float3 R;
	const float roughness;

  public:
	__device__ DisneyFakeSS( const float3& r, float roughness ) : R( r ), roughness( roughness )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		const float3 wh = normalize( wo + wi );
		if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0.f );

		const float Fo = SchlickWeight( AbsCosTheta( wo ) ),
					Fi = SchlickWeight( AbsCosTheta( wi ) );
		const float cosThetaD = dot( wi, wh );

		// Fss90 used to "flatten" retroreflection based on roughness
		const float Fss90 = cosThetaD * cosThetaD * roughness;
		const float Fss = pbrt_Lerp( Fo, 1.0f, Fss90 ) * pbrt_Lerp( Fi, 1.0f, Fss90 );
		// 1.25 scale is used to (roughly) preserve albedo
		const float ss =
			1.25f * ( Fss * ( 1 / ( AbsCosTheta( wo ) + AbsCosTheta( wi ) ) - .5f ) + .5f );

		return R * INVPI * ss;
	}
};

class DisneyRetro : public BxDF_T<DisneyRetro,
								  BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE )>
{
	const float3 R;
	const float roughness;

  public:
	__device__ DisneyRetro( const float3& r, float roughness ) : R( r ), roughness( roughness )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		const float3 wh = normalize( wo + wi );
		if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0.f );

		const float Fo = SchlickWeight( AbsCosTheta( wo ) ),
					Fi = SchlickWeight( AbsCosTheta( wi ) );
		const float cosThetaD = dot( wi, wh );

		const float Rr = 2 * cosThetaD * cosThetaD * roughness;

		// Burley 2015, eq (4).
		return R * INVPI * Rr * ( Fo + Fi + Fo * Fi * ( Rr - 1 ) );
	}
};

class DisneySheen : public BxDF_T<DisneySheen,
								  BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE )>
{
	const float3 R;

  public:
	__device__ DisneySheen( const float3& r ) : R( r )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		const float3 wh = normalize( wo + wi );
		if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0.f );

		const float cosThetaD = dot( wi, wh );

		return R * SchlickWeight( cosThetaD );
	}
};

class DisneyClearcoat : public BxDF_T<DisneyClearcoat,
									  BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE )>
{
	const float weight, gloss;

  public:
	__device__ DisneyClearcoat( const float weight, const float gloss ) : weight( weight ), gloss( gloss )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		const float3 wh = normalize( wo + wi );
		if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0.f );

		const float cosThetaD = dot( wi, wh );
		// Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
		// GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
		// (which is GTR2).
		const float Dr = pbrt_GTR1( AbsCosTheta( wh ), gloss );
		const float Fr = FrSchlick( .04, dot( wo, wh ) );
		// The geometric term always based on alpha = 0.25.
		const float Gr =
			smithG_GGX( AbsCosTheta( wo ), .25 ) * smithG_GGX( AbsCosTheta( wi ), .25 );

		return make_float3( weight * Gr * Fr * Dr / 4 );
	}

	__device__ float3 Sample_f( const float3 wo, float3& wi,
								/*  const Point2f& u, */ const float r0, const float r1,
								float& pdf, BxDFType& sampledType ) const override
	{
		if ( wo.z == 0 ) return make_float3( 0.f );

		const float alpha2 = gloss * gloss;
		const float cosTheta = sqrtf(
			max( 0.f, ( 1.f - powf( alpha2, 1.f - r0 ) ) / ( 1.f - alpha2 ) ) );
		const float sinTheta = sqrtf( max( 0.f, 1.f - cosTheta * cosTheta ) );
		const float phi = 2 * PI * r1;
		float3 wh = SphericalDirection( sinTheta, cosTheta, phi );
		if ( !SameHemisphere( wo, wh ) ) wh = -wh;

		wi = pbrt_Reflect( wo, wh );
		if ( !SameHemisphere( wo, wi ) ) return make_float3( 0.f );

		pdf = Pdf( wo, wi );
		return f( wo, wi );
	}

	__device__ float Pdf( const float3& wo, const float3& wi ) const override
	{
		if ( !SameHemisphere( wo, wi ) )
			return 0;

		const float3 wh = normalize( wo + wi );
		if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return 0.f;

		// The sampling routine samples wh exactly from the GTR1 distribution.
		// Thus, the final value of the PDF is just the value of the
		// distribution for wh converted to a mesure with respect to the
		// surface normal.
		float Dr = pbrt_GTR1( AbsCosTheta( wh ), gloss );
		return Dr * AbsCosTheta( wh ) / ( 4 * dot( wo, wh ) );
	}
};

class DisneyFresnel : public Fresnel
{
	const float3 R0;
	const float metallic, eta;

  public:
	__device__ DisneyFresnel( const float3 R0, const float metallic, const float eta )
		: R0( R0 ), metallic( metallic ), eta( eta )
	{
	}

	__device__ float3 Evaluate( float cosI ) const override
	{
		return pbrt_Lerp( metallic,
						  make_float3( FrDielectric( cosI, 1.f, eta ) ),
						  FrSchlick( R0, cosI ) );
	}
};

class DisneyMicrofacetDistribution : public TrowbridgeReitzDistribution</* sampleVisibleArea */>
{
  public:
	__device__ DisneyMicrofacetDistribution( float alphax, float alphay )
		: TrowbridgeReitzDistribution( alphax, alphay ) {}

	__device__ float G( const float3& wo, const float3& wi ) const override
	{
		// Disney uses the separable masking-shadowing model.
		return G1( wo ) * G1( wi );
	}
};

using DisneyMicrofacetReflection = MicrofacetReflection<DisneyMicrofacetDistribution, DisneyFresnel>;

// ----------------------------------------------------------------

/**
 * DisneyGltf: Disney material expressed as PBRT BxDF stack.
 * Material input data does not match it entirely, so be cautious.
 */
class DisneyGltf : public BSDFStackMaterial<
					   DisneyDiffuse, DisneyFakeSS, DisneyRetro, DisneySheen,
					   DisneyClearcoat, DisneyMicrofacetReflection,
					   MicrofacetTransmission<DisneyMicrofacetDistribution>,
					   MicrofacetTransmission<TrowbridgeReitzDistribution<>>,
					   LambertianTransmission>
{
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
		) override
	{
		// metallic: Controls how "metal" the object appears. Higher values reduce diffuse scattering and shift the highlight color towards the material's color. Range: [0,1].
		// spectrans: Controls contribution of glossy specular transmission. Range: [0,1].

		ShadingData shadingData;
		const bool thin = false;

		GetShadingData( D, u, v, coneWidth, tri, instIdx,
						// Returns:
						shadingData, N, iN, fN, T, waveLength );

		const float metallicWeight = METALLIC;
		const float strans = TRANSMISSION;
		const float diffuseWeight = ( 1.f - metallicWeight ) * ( 1.f - strans );
		const float rough = ROUGHNESS;
		const float e = ETA * 2.f; // Multiplied by .5 in host_material
		const float3 c = shadingData.color;

		// WRONG! This should be _diffuse_ transmission, which is different
		// from _specular_ transmission
		const float dt = TRANSMISSION / 2;

		const float lum = 0.212671f * c.x + 0.715160f * c.y + 0.072169f * c.z;
		// normalize lum. to isolate hue+sat
		const float3 Ctint = lum > 0 ? ( c / lum ) : make_float3( 1.f );

		if ( diffuseWeight > 0 )
		{
			if ( thin )
			{
				const float flat = -1337.f; // TODO: Evaluate from texture/value
				const float3 thin_color = ( 1 - dt ) * diffuseWeight * c;

				// Blend between DisneyDiffuse and fake subsurface based on
				// flatness.  Additionally, weight using diffTrans.
				bxdfs.emplace_back<DisneyDiffuse>( ( 1 - flat ) * thin_color );
				bxdfs.emplace_back<DisneyFakeSS>( flat * thin_color, rough );
			}
			else
			{
				if ( /* scatterdistance == 0 */ true )
				{
					bxdfs.emplace_back<DisneyDiffuse>( diffuseWeight * c );
				}
				else
				{
					// TODO:

					// Implementation of the empirical BSSRDF described in "Extending the
					// Disney BRDF to a BSDF with integrated subsurface scattering" (Brent
					// Burley) and "Approximate Reflectance Profiles for Efficient Subsurface
					// Scattering (Christensen and Burley).
				}
			}

			// Retro-reflection.
			bxdfs.emplace_back<DisneyRetro>( diffuseWeight * c, rough );

			// Sheen (if enabled)
			const float sheenWeight = SHEEN;
			if ( sheenWeight > 0 )
			{
				const float stint = SHEENTINT;
				const float3 Csheen = pbrt_Lerp( stint, make_float3( 1.f ), Ctint );

				bxdfs.emplace_back<DisneySheen>( diffuseWeight * sheenWeight * Csheen );
			}
		}

		// Create the microfacet distribution for metallic and/or specular
		// transmission.
		const float aspect = sqrtf( 1.f - ANISOTROPIC * .9f );
		const float ax = max( .001f, sqr( rough ) / aspect );
		const float ay = max( .001f, sqr( rough ) * aspect );
		const DisneyMicrofacetDistribution distrib( ax, ay );

		// Specular is Trowbridge-Reitz with a modified Fresnel function.
		const float specTint = SPECTINT;
		const float3 Cspec0 = pbrt_Lerp( metallicWeight, SchlickR0FromEta( e ) * pbrt_Lerp( specTint, make_float3( 1.f ), Ctint ), c );
		const DisneyFresnel fresnel( Cspec0, metallicWeight, e );
		// https://github.com/mmp/pbrt-v3/issues/224
#if 1
		// HACK! The fix for this issue replaces the color component with 1.f,
		// which solves an incorrect reflection color, but makes the scene extremely
		// overexposed. This could be an issue specific to stuffing a GLTF material
		// into a PBRT BxDF expression.
		bxdfs.emplace_back<DisneyMicrofacetReflection>(
			pbrt_Lerp( metallicWeight, c, make_float3( 1.f ) ),
			distrib, fresnel );
#else
		bxdfs.emplace_back<DisneyMicrofacetReflection>( make_float3( 1.f ), distrib, fresnel );
#endif

		const float cc = CLEARCOAT;
		if ( cc > 0 )
			bxdfs.emplace_back<DisneyClearcoat>( cc, pbrt_Lerp( CLEARCOATGLOSS, .1f, .001f ) );

		// BTDF
		if ( strans > 0 )
		{
			// Walter et al's model, with the provided transmissive term scaled
			// by sqrt(color), so that after two refractions, we're back to the
			// provided color.
			const float3 T = strans * Sqrt( c );
			if ( thin )
			{
				// Scale roughness based on IOR (Burley 2015, Figure 15).
				const float rscaled = ( .65f * e - .35f ) * rough;
				const float ax = max( .001f, sqr( rscaled ) / aspect );
				const float ay = max( .001f, sqr( rscaled ) * aspect );
				const TrowbridgeReitzDistribution<> scaledDistrib( ax, ay );
				bxdfs.emplace_back<MicrofacetTransmission<
					std::remove_const_t<decltype( scaledDistrib )>>>( T, scaledDistrib, 1.f, e, mode );
			}
			else
				bxdfs.emplace_back<MicrofacetTransmission<
					std::remove_const_t<decltype( distrib )>>>( T, distrib, 1.f, e, mode );
		}

		if ( thin )
			// Lambertian, weighted by (1 - diffTrans)
			bxdfs.emplace_back<LambertianTransmission>( dt * c );
	}
};

/**
 * DisneyGltf: Disney material expressed as PBRT BxDF stack.
 * Data is a full CoreMaterial, not the slimmed-down CUDAMaterial, just like in PBRT.
 */
class Disney : public SimpleMaterial<
				   DisneyDiffuse, DisneyFakeSS, DisneyRetro, DisneySheen,
				   DisneyClearcoat, DisneyMicrofacetReflection,
				   MicrofacetTransmission<DisneyMicrofacetDistribution>,
				   MicrofacetTransmission<TrowbridgeReitzDistribution<>>,
				   LambertianTransmission>
{
  public:
	__device__ void ComputeScatteringFunctions(
		const CoreMaterial& params,
		const float2 uv,
		const bool allowMultipleLobes,
		const TransportMode mode ) override
	{
		// TODO: Bumpmapping

		const float3 c = SampleCoreTexture( params.color, uv );
		const float metallicWeight = SampleCoreTexture( params.metallic, uv );
		const float e = SampleCoreTexture( params.eta, uv );
		const float strans = SampleCoreTexture( params.specTrans, uv );
		const float diffuseWeight = ( 1.f - metallicWeight ) * ( 1.f - strans );
		const float dt = SampleCoreTexture( params.diffTrans, uv ) / 2.f; // 0: all diffuse is reflected -> 1, transmitted
		const float rough = SampleCoreTexture( params.roughness, uv );
		const float lum = 0.212671f * c.x + 0.715160f * c.y + 0.072169f * c.z;
		// normalize lum. to isolate hue+sat
		const float3 Ctint = lum > 0 ? ( c / lum ) : make_float3( 1.f );

		if ( diffuseWeight > 0 )
		{
			if ( params.thin )
			{
				const float flat = -1337.f; // TODO: Evaluate from texture/value
				const float3 thin_color = ( 1 - dt ) * diffuseWeight * c;

				// Blend between DisneyDiffuse and fake subsurface based on
				// flatness.  Additionally, weight using diffTrans.
				bxdfs.emplace_back<DisneyDiffuse>( ( 1 - flat ) * thin_color );
				bxdfs.emplace_back<DisneyFakeSS>( flat * thin_color, rough );
			}
			else
			{
				const auto sd = SampleCoreTexture( params.scatterDistance, uv );
				if ( IsBlack( sd ) )
				{
					bxdfs.emplace_back<DisneyDiffuse>( diffuseWeight * c );
				}
				else
				{
					// TODO:

					// Implementation of the empirical BSSRDF described in "Extending the
					// Disney BRDF to a BSDF with integrated subsurface scattering" (Brent
					// Burley) and "Approximate Reflectance Profiles for Efficient Subsurface
					// Scattering (Christensen and Burley).
				}
			}

			// Retro-reflection.
			bxdfs.emplace_back<DisneyRetro>( diffuseWeight * c, rough );

			// Sheen (if enabled)
			const float sheenWeight = SampleCoreTexture( params.sheen, uv );
			if ( sheenWeight > 0 )
			{
				const float stint = SampleCoreTexture( params.sheenTint, uv );
				const float3 Csheen = pbrt_Lerp( stint, make_float3( 1.f ), Ctint );

				bxdfs.emplace_back<DisneySheen>( diffuseWeight * sheenWeight * Csheen );
			}
		}

		// Create the microfacet distribution for metallic and/or specular
		// transmission.
		const float aspect = sqrtf( 1.f - SampleCoreTexture( params.anisotropic, uv ) * .9f );
		const float ax = max( .001f, sqr( rough ) / aspect );
		const float ay = max( .001f, sqr( rough ) * aspect );
		const DisneyMicrofacetDistribution distrib( ax, ay );

		// Specular is Trowbridge-Reitz with a modified Fresnel function.
		const float specTint = SampleCoreTexture( params.specularTint, uv );
		const float3 Cspec0 = pbrt_Lerp( metallicWeight, SchlickR0FromEta( e ) * pbrt_Lerp( specTint, make_float3( 1.f ), Ctint ), c );
		const DisneyFresnel fresnel( Cspec0, metallicWeight, e );
		// https://github.com/mmp/pbrt-v3/issues/224
		bxdfs.emplace_back<DisneyMicrofacetReflection>( make_float3( 1.f ), distrib, fresnel );

		// Clearcoat
		const float cc = SampleCoreTexture( params.clearcoat, uv );
		if ( cc > 0 )
			bxdfs.emplace_back<DisneyClearcoat>( cc, pbrt_Lerp( SampleCoreTexture( params.clearcoatGloss, uv ), .1f, .001f ) );

		// BTDF
		if ( strans > 0 )
		{
			// Walter et al's model, with the provided transmissive term scaled
			// by sqrt(color), so that after two refractions, we're back to the
			// provided color.
			const float3 T = strans * Sqrt( c );
			if ( params.thin )
			{
				const float rscaled = ( .65f * e - .35f ) * rough;
				const float ax = max( .001f, sqr( rscaled ) / aspect );
				const float ay = max( .001f, sqr( rscaled ) * aspect );
				const TrowbridgeReitzDistribution<> scaledDistrib( ax, ay );
				bxdfs.emplace_back<MicrofacetTransmission<
					std::remove_const_t<decltype( scaledDistrib )>>>( T, scaledDistrib, 1.f, e, mode );
			}
			else
				bxdfs.emplace_back<MicrofacetTransmission<
					std::remove_const_t<decltype( distrib )>>>( T, distrib, 1.f, e, mode );
		}

		if ( params.thin )
			// Lambertian, weighted by (1 - diffTrans)
			bxdfs.emplace_back<LambertianTransmission>( dt * c );
	}
};
