/*
    pbrt source code is Copyright(c) 1998-2016
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

template <typename _BxDF>
class ScaledBxDF : public BxDF_T<ScaledBxDF<_BxDF>,
								 _BxDF::type>
{
	const _BxDF bxdf;
	const float3 scale;

  public:
	__device__ ScaledBxDF( const BxDF& bxdf, const float3& scale )
		: bxdf( bxdf ), scale( scale ) {}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		return scale * bxdf.f( wo, wi );
	}

	__device__ float3 Sample_f( const float3 wo, float3& wi,
								const float r0, const float r1,
								float& pdf, BxDFType& sampledType ) const override
	{
		return scale * bxdf.Sample_f( wo, wi, r0, r1, pdf, sampledType );
	}

	__device__ float Pdf( const float3& wo, const float3& wi ) const override
	{
		return bxdf.Pdf( wo, wi );
	}
};

template <typename Fresnel_T>
class SpecularReflection : public BxDF_T<SpecularReflection<Fresnel_T>,
										 BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_SPECULAR )>
{
	const float3 R;
	const Fresnel_T fresnel;

  public:
	__device__ SpecularReflection( const float3& R, const Fresnel_T& fresnel ) : R( R ), fresnel( fresnel )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		return make_float3( 0.f );
	}

	__device__ float3 Sample_f( const float3 wo, float3& wi,
								const float r0, const float r1,
								float& pdf, BxDFType& sampledType ) const override
	{
		// Compute perfect specular reflection direction
		wi = make_float3( -wo.x, -wo.y, wo.z );
		pdf = 1.f;
		return fresnel.Evaluate( CosTheta( wi ) ) * R / AbsCosTheta( wi );
	}

	__device__ float Pdf( const float3& wo, const float3& wi ) const override
	{
		return 0.f;
	}
};

class SpecularTransmission : public BxDF_T<SpecularTransmission,
										   BxDFType( BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_SPECULAR )>
{
	const float3 T;
	const float etaA, etaB;
	const TransportMode mode;

  public:
	__device__ SpecularTransmission( const float3& T,
									 const float etaA, const float etaB,
									 const TransportMode mode )
		: T( T ),
		  etaA( etaA ),
		  etaB( etaB ),
		  mode( mode )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		return make_float3( 0.f );
	}

	__device__ float3 Sample_f( const float3 wo, float3& wi,
								const float r0, const float r1,
								float& pdf, BxDFType& sampledType ) const override
	{
		// Figure out which $\eta$ is incident and which is transmitted
		const bool entering = CosTheta( wo ) > 0;
		const float etaI = entering ? etaA : etaB;
		const float etaT = entering ? etaB : etaA;

		// Compute ray direction for specular transmission
		if ( !pbrt_Refract( wo, Faceforward( make_float3( 0, 0, 1 ), wo ), etaI / etaT, wi ) )
			return make_float3( 0.f );
		pdf = 1.f;

		float f = 1.f - FrDielectric( CosTheta( wi ), etaA, etaB );
		// Account for non-symmetry with transmission to different medium
		if ( mode == TransportMode::Radiance ) f *= ( etaI * etaI ) / ( etaT * etaT );
		return ( T * f / AbsCosTheta( wi ) );
	}

	__device__ float Pdf( const float3& wo, const float3& wi ) const override
	{
		return 0.f;
	}
};

class FresnelSpecular : public BxDF_T<FresnelSpecular,
									  BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_SPECULAR )>
{
	const float3 R, T;
	const float etaA, etaB;
	const TransportMode mode;

  public:
	__device__ FresnelSpecular( const float3& R,
								const float3& T,
								const float etaA, const float etaB,
								const TransportMode mode )
		: R( R ),
		  T( T ),
		  etaA( etaA ),
		  etaB( etaB ),
		  mode( mode )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		return make_float3( 0.f );
	}

	__device__ float3 Sample_f( const float3 wo, float3& wi,
								const float r0, const float r1,
								float& pdf, BxDFType& sampledType ) const override
	{
		const float F = FrDielectric( CosTheta( wo ), etaA, etaB );
		if ( r0 < F )
		{
			// Compute specular reflection for _FresnelSpecular_

			// Compute perfect specular reflection direction
			wi = make_float3( -wo.x, -wo.y, wo.z );
			sampledType = BxDFType( BSDF_SPECULAR | BSDF_REFLECTION );
			pdf = F;
			return F * R / AbsCosTheta( wi );
		}
		else
		{
			// Compute specular transmission for _FresnelSpecular_

			// Figure out which $\eta$ is incident and which is transmitted
			bool entering = CosTheta( wo ) > 0;
			const float etaI = entering ? etaA : etaB;
			const float etaT = entering ? etaB : etaA;

			// Compute ray direction for specular transmission
			if ( !pbrt_Refract( wo, Faceforward( make_float3( 0, 0, 1 ), wo ), etaI / etaT, wi ) )
				return make_float3( 0 );
			float3 ft = T * ( 1.f - F );

			// Account for non-symmetry with transmission to different medium
			if ( mode == TransportMode::Radiance )
				ft *= ( etaI * etaI ) / ( etaT * etaT );
			sampledType = BxDFType( BSDF_SPECULAR | BSDF_TRANSMISSION );

			pdf = 1.f - F;
			return ft / AbsCosTheta( wi );
		}
	}

	__device__ float Pdf( const float3& wo, const float3& wi ) const override
	{
		return 0.f;
	}
};

class LambertianReflection : public BxDF_T<LambertianReflection,
										   BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE )>
{
	const float3 R;

  public:
	__device__ LambertianReflection( const float3& R )
		: R( R )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		return R * INVPI;
	}
};

class LambertianTransmission : public BxDF_T<LambertianTransmission, BxDFType( BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_DIFFUSE )>
{
	const float3 T;

  public:
	__device__ LambertianTransmission( const float3& T ) : T( T )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		return T * INVPI;
	}

	__device__ float3 Sample_f( const float3 wo, float3& wi,
								const float r0, const float r1,
								float& pdf, BxDFType& sampledType ) const override
	{
		wi = CosineSampleHemisphere( r0, r1 );
		if ( wo.z > 0 ) wi.z *= -1;
		pdf = Pdf( wo, wi );
		return f( wo, wi );
	}

	__device__ float Pdf( const float3& wo, const float3& wi ) const override
	{
		return !SameHemisphere( wo, wi ) ? AbsCosTheta( wi ) * INVPI : 0;
	}
};

class OrenNayar : public BxDF_T<OrenNayar,
								BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_DIFFUSE )>
{
	const float3 R;
	float A, B;

  public:
	__device__ OrenNayar( const float3& R, float sigma )
		: R( R )
	{
		sigma = Radians( sigma );
		const float sigma2 = sigma * sigma;
		A = 1.f - ( sigma2 / ( 2.f * ( sigma2 + 0.33f ) ) );
		B = 0.45f * sigma2 / ( sigma2 + 0.09f );
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		const float sinThetaI = SinTheta( wi );
		const float sinThetaO = SinTheta( wo );
		// Compute cosine term of Oren-Nayar model
		float maxCos = 0.f;
		if ( sinThetaI > 1e-4 && sinThetaO > 1e-4 )
		{
			const float sinPhiI = SinPhi( wi ), cosPhiI = CosPhi( wi );
			const float sinPhiO = SinPhi( wo ), cosPhiO = CosPhi( wo );
			const float dCos = cosPhiI * cosPhiO + sinPhiI * sinPhiO;
			maxCos = max( 0.f, dCos );
		}

		// Compute sine and tangent terms of Oren-Nayar model
		float sinAlpha, tanBeta;
		if ( AbsCosTheta( wi ) > AbsCosTheta( wo ) )
		{
			sinAlpha = sinThetaO;
			tanBeta = sinThetaI / AbsCosTheta( wi );
		}
		else
		{
			sinAlpha = sinThetaI;
			tanBeta = sinThetaO / AbsCosTheta( wo );
		}
		return R * INVPI * ( A + B * maxCos * sinAlpha * tanBeta );
	}
};

template <typename MicrofacetDistribution_T, typename Fresnel_T>
class MicrofacetReflection : public BxDF_T<MicrofacetReflection<MicrofacetDistribution_T, Fresnel_T>,
										   BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_GLOSSY )>
{
	// Not strictly necessary. Any type exposing the used functions would fit (or otherwise throw a compiler error there)
	// static_assert( std::is_convertible<MicrofacetDistribution_T, MicrofacetDistribution>::value );
	// static_assert( std::is_convertible<Fresnel_T, Fresnel>::value );

	const float3 R;
	const MicrofacetDistribution_T distribution;
	const Fresnel_T fresnel;

	__device__ float distribution_Pdf( const float3& wo, const float3& wh ) const
	{
		return distribution.Pdf( wo, wh ) / ( 4.f * dot( wo, wh ) );
	}

  public:
	__device__ MicrofacetReflection( const float3& R,
									 const MicrofacetDistribution_T& distribution,
									 const Fresnel_T& fresnel )
		: R( R ),
		  distribution( distribution ),
		  fresnel( fresnel )
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		float cosThetaO = AbsCosTheta( wo ), cosThetaI = AbsCosTheta( wi );
		auto wh = wi + wo;
		// Handle degenerate cases for microfacet reflection
		if ( cosThetaI == 0 || cosThetaO == 0 ) return make_float3( 0.f );
		if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0.f );
		wh = normalize( wh );
		// For the Fresnel call, make sure that wh is in the same hemisphere
		// as the surface normal, so that TIR is handled correctly.
		// https://github.com/mmp/pbrt-v3/issues/229
		const auto F = fresnel.Evaluate(
			dot( wi, Faceforward( wh, make_float3( 0, 0, 1 ) ) ) );
		return R * distribution.D( wh ) * distribution.G( wo, wi ) * F /
			   ( 4 * cosThetaI * cosThetaO );
	}

	__device__ float3 Sample_f( const float3 wo, float3& wi,
								const float r0, const float r1,
								float& pdf, BxDFType& sampledType ) const override
	{
		// Sample microfacet orientation $\wh$ and reflected direction $\wi$
		if ( wo.z == 0.f ) return make_float3( 0.f );
		const auto wh = distribution.Sample_wh( wo, r0, r1 );
		if ( dot( wo, wh ) < 0 ) return make_float3( 0.f ); // Should be rare
		wi = pbrt_Reflect( wo, wh );
		if ( !SameHemisphere( wo, wi ) ) return make_float3( 0.f );

		// Compute PDF of _wi_ for microfacet reflection
		pdf = distribution_Pdf( wo, wh );
		return f( wo, wi );
	}

	__device__ float Pdf( const float3& wo, const float3& wi ) const override
	{
		if ( !SameHemisphere( wo, wi ) ) return 0.f;
		const auto wh = normalize( wo + wi );
		return distribution_Pdf( wo, wh );
	}
};

template <typename MicrofacetDistribution_T>
class MicrofacetTransmission : public BxDF_T<MicrofacetTransmission<MicrofacetDistribution_T>,
											 BxDFType( BxDFType::BSDF_TRANSMISSION | BxDFType::BSDF_GLOSSY )>
{
	const float3 T;
	const MicrofacetDistribution_T distribution;
	const float etaA, etaB;
	const TransportMode mode;

  public:
	__device__ MicrofacetTransmission( const float3& T,
									   const MicrofacetDistribution_T& distribution,
									   const float etaA,
									   const float etaB,
									   const TransportMode mode )
		: T( T ),
		  distribution( distribution ),
		  etaA( etaA ),
		  etaB( etaB ),
		  mode( mode ){}
	{
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		if ( SameHemisphere( wo, wi ) ) return make_float3( 0.f ); // transmission only

		const float cosThetaO = CosTheta( wo );
		const float cosThetaI = CosTheta( wi );
		if ( cosThetaI == 0 || cosThetaO == 0 ) return make_float3( 0.f );

		// Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
		const float eta = CosTheta( wo ) > 0.f ? ( etaB / etaA ) : ( etaA / etaB );
		float3 wh = normalize( wo + wi * eta );
		if ( wh.z < 0 ) wh = -wh;

		float F = FrDielectric( dot( wo, wh ), etaA, etaB );

		const float sqrtDenom = dot( wo, wh ) + eta * dot( wi, wh );
		const float factor = ( mode == TransportMode::Radiance ) ? ( 1 / eta ) : 1;

		return ( 1.f - F ) *
			   abs( distribution.D( wh ) * distribution.G( wo, wi ) * eta * eta *
						 AbsDot( wi, wh ) * AbsDot( wo, wh ) * factor * factor /
						 ( cosThetaI * cosThetaO * sqrtDenom * sqrtDenom ) ) *
			   T;
	}

	__device__ float3 Sample_f( const float3 wo, float3& wi,
								const float r0, const float r1,
								float& pdf, BxDFType& sampledType ) const override
	{
		// Sample microfacet orientation $\wh$ and reflected direction $\wi$
		if ( wo.z == 0.f ) return make_float3( 0.f );
		const auto wh = distribution.Sample_wh( wo, r0, r1 );
		if ( dot( wo, wh ) < 0 ) return make_float3( 0.f ); // Should be rare

		const float eta = CosTheta( wo ) > 0 ? ( etaA / etaB ) : ( etaB / etaA );
		if ( !pbrt_Refract( wo, wh, eta, wi ) )
			return make_float3( 0.f );
		pdf = Pdf( wo, wi );
		return f( wo, wi );
	}

	__device__ float Pdf( const float3& wo, const float3& wi ) const override
	{
		if ( SameHemisphere( wo, wi ) ) return 0.f;

		// Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
		const float eta = CosTheta( wo ) > 0 ? ( etaB / etaA ) : ( etaA / etaB );
		const float3 wh = normalize( wo + wi * eta );

		// Compute change of variables _dwh\_dwi_ for microfacet transmission
		const float sqrtDenom = dot( wo, wh ) + eta * dot( wi, wh );
		const float dwh_dwi =
			abs( ( eta * eta * dot( wi, wh ) ) / ( sqrtDenom * sqrtDenom ) );
		return distribution.Pdf( wo, wh ) * dwh_dwi;
	}
};

template <typename MicrofacetDistribution_T>
class FresnelBlend : public BxDF_T<FresnelBlend<MicrofacetDistribution_T>,
								   BxDFType( BxDFType::BSDF_REFLECTION | BxDFType::BSDF_GLOSSY )>
{
	const float3 Rd, Rs;
	const MicrofacetDistribution_T distribution;

  public:
	__device__ FresnelBlend( const float3& Rd, const float3& Rs,
							 const MicrofacetDistribution_T& distribution )
		: Rd( Rd ),
		  Rs( Rs ),
		  distribution( distribution )
	{
	}

	LH2_DEVFUNC float pow5( float v ) { return ( v * v ) * ( v * v ) * v; };

	__device__ float3 SchlickFresnel( float cosTheta ) const
	{
		return Rs + pow5( 1.f - cosTheta ) * ( make_float3( 1.f ) - Rs );
	}

	__device__ float3 f( const float3& wo, const float3& wi ) const override
	{
		const float3 diffuse = ( 28.f / ( 23.f * PI ) ) * Rd * ( make_float3( 1.f ) - Rs ) *
							   ( 1 - pow5( 1 - .5f * AbsCosTheta( wi ) ) ) *
							   ( 1 - pow5( 1 - .5f * AbsCosTheta( wo ) ) );
		float3 wh = wi + wo;
		if ( wh.x == 0 && wh.y == 0 && wh.z == 0 ) return make_float3( 0.f );
		wh = normalize( wh );
		const float3 specular =
			distribution.D( wh ) /
			( 4 * AbsDot( wi, wh ) * max( AbsCosTheta( wi ), AbsCosTheta( wo ) ) ) *
			SchlickFresnel( dot( wi, wh ) );
		return diffuse + specular;
	}

	__device__ float3 Sample_f( const float3 wo, float3& wi,
								const float r0_orig, const float r1,
								float& pdf, BxDFType& sampledType ) const override
	{
		float r0 = r0_orig;
		if ( r0 < .5 )
		{
			r0 = min( 2.f * r0, OneMinusEpsilon );
			// Cosine-sample the hemisphere, flipping the direction if necessary
			wi = CosineSampleHemisphere( r0, r1 );
			if ( wo.z < 0 ) wi.z *= -1;
		}
		else
		{
			r0 = min( 2.f * ( r0 - .5f ), OneMinusEpsilon );
			// Sample microfacet orientation $\wh$ and reflected direction $\wi$
			const float3 wh = distribution.Sample_wh( wo, r0, r1 );
			wi = pbrt_Reflect( wo, wh );
			if ( !SameHemisphere( wo, wi ) ) return make_float3( 0.f );
		}
		pdf = Pdf( wo, wi );
		return f( wo, wi );
	}

	__device__ float Pdf( const float3& wo, const float3& wi ) const override
	{
		if ( !SameHemisphere( wo, wi ) ) return 0.f;
		const float3 wh = normalize( wo + wi );
		const float pdf_wh = distribution.Pdf( wo, wh );
		return .5f * ( AbsCosTheta( wi ) * INVPI + pdf_wh / ( 4.f * dot( wo, wh ) ) );
	}
};
