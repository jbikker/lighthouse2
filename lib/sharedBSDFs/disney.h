/* disney2.h - License information:

   This code has been adapted from AppleSeed: https://appleseedhq.net
   The AppleSeed software is released under the MIT license.
   Copyright (c) 2014-2018 Esteban Tovagliari, The appleseedhq Organization.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   // https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/renderer/modeling/bsdf/disneybrdf.cpp
*/

#ifndef DISNEY2_H
#define DISNEY2_H

#include "compatibility.h"

#define DIFFWEIGHT	weights.x
#define SHEENWEIGHT	weights.y
#define SPECWEIGHT	weights.z
#define COATWEIGHT	weights.w

#define GGXMDF		1001
#define GTR1MDF		1002

LH2_DEVFUNC float schlick_fresnel( const float u )
{
	const float m = saturate( 1.0f - u ), m2 = sqr( m ), m4 = sqr( m2 );
	return m4 * m;
}
LH2_DEVFUNC void mix_spectra( const float3 a, const float3 b, const float t, REFERENCE_OF( float3 ) result ) { result = (1.0f - t) * a + t * b; }
LH2_DEVFUNC void mix_one_with_spectra( const float3 b, const float t, REFERENCE_OF( float3 ) result ) { result = (1.0f - t) + t * b; }
LH2_DEVFUNC void mix_spectra_with_one( const float3 a, const float t, REFERENCE_OF( float3 ) result ) { result = (1.0f - t) * a + t; }
LH2_DEVFUNC float microfacet_alpha_from_roughness( const float roughness ) { return max( 0.001f, roughness * roughness ); }
LH2_DEVFUNC void microfacet_alpha_from_roughness( const float roughness, const float anisotropy, REFERENCE_OF( float ) alpha_x, REFERENCE_OF( float ) alpha_y )
{
	const float square_roughness = roughness * roughness;
	const float aspect = sqrtf( 1.0f + anisotropy * (anisotropy < 0 ? 0.9f : -0.9f) );
	alpha_x = max( 0.001f, square_roughness / aspect );
	alpha_y = max( 0.001f, square_roughness * aspect );
}
LH2_DEVFUNC float clearcoat_roughness( const ShadingData& shadingData ) { return mix( 0.1f, 0.001f, CLEARCOATGLOSS ); }
LH2_DEVFUNC void DisneySpecularFresnel( CONSTREF_OF( ShadingData ) shadingData, const float3 o, const float3 h, REFERENCE_OF( float3 ) value )
{
	mix_one_with_spectra( TINT, SPECTINT, value );
	value *= SPECULAR * 0.08f;
	mix_spectra( value, shadingData.color, METALLIC, value );
	const float cos_oh = fabs( dot( o, h ) );
	mix_spectra_with_one( value, schlick_fresnel( cos_oh ), value );
}
LH2_DEVFUNC void DisneyClearcoatFresnel( CONSTREF_OF( ShadingData ) shadingData, const float3 o, const float3 h, REFERENCE_OF( float3 ) value )
{
	const float cos_oh = fabs( dot( o, h ) );
	value = make_float3( mix( 0.04f, 1.0f, schlick_fresnel( cos_oh ) ) * 0.25f * CLEARCOAT );
}
LH2_DEVFUNC bool force_above_surface( REFERENCE_OF( float3 ) direction, const float3 normal )
{
	const float Eps = 1.0e-4f;
	const float cos_theta = dot( direction, normal );
	const float correction = Eps - cos_theta;
	if (correction <= 0) return false;
	direction = normalize( direction + correction * normal );
	return true;
}
LH2_DEVFUNC float Fr_L( float VDotN, float eio )
{
	if (VDotN < 0.0f) eio = 1.0f / eio, VDotN = fabs( VDotN );
	const float SinThetaT2 = sqr( eio ) * (1.0f - VDotN * VDotN);
	if (SinThetaT2 > 1.0f) return 1.0f; // TIR
	const float LDotN = sqrtf( 1.0f - SinThetaT2 );
	const float r1 = (VDotN - eio * LDotN) / (VDotN + eio * LDotN);
	const float r2 = (LDotN - eio * VDotN) / (LDotN + eio * VDotN);
	return 0.5f * (sqr( r1 ) + sqr( r2 ));
}
LH2_DEVFUNC bool Refract_L( const float3 wi, const float3 n, const float eta, REFERENCE_OF( float3 ) wt )
{
	const float cosThetaI = fabs( dot( n, wi ) );
	const float sin2ThetaI = max( 0.0f, 1.0f - cosThetaI * cosThetaI );
	const float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1) return false; // TIR
	const float cosThetaT = sqrtf( 1.0f - sin2ThetaT );
	wt = eta * (wi * -1.0f) + (eta * cosThetaI - cosThetaT) * float3( n );
	return true;
}

template <uint MDF, bool flip>
LH2_DEVFUNC void sample_mf( CONSTREF_OF( ShadingData ) shadingData, const float r0, const float r1, const float alpha_x, const float alpha_y, 
	const float3 N, const float3 T, const float3 B, const float3 gN, const float3 wow,
	/* OUT: */ REFERENCE_OF( float3 ) wiw, REFERENCE_OF( float ) pdf, REFERENCE_OF( float3 ) value )
{
	float3 wo = World2Tangent( wow, N, T, B ); // local_geometry.m_shading_basis.transform_to_local( outgoing );
	if (wo.z == 0) return;
	if (flip) wo.z = fabs( wo.z );
	// compute the incoming direction by sampling the MDF
	float3 m = MDF == GGXMDF ? GGXMDF_sample( wo, r0, r1, alpha_x, alpha_y ) : GTR1MDF_sample( r0, r1, alpha_x, alpha_y );
	float3 wi = reflect( wo * -1.0f, m );
	// force the outgoing direction to lie above the geometric surface.
	const float3 ng = World2Tangent( gN, N, T, B );
	if (force_above_surface( wi, ng )) m = normalize( wo + wi );
	if (wi.z == 0) return;
	const float cos_oh = dot( wo, m );
	pdf = (MDF == GGXMDF ? GGXMDF_pdf( wo, m, alpha_x, alpha_y ) : GTR1MDF_pdf( wo, m, alpha_x, alpha_y )) / fabs( 4.0f * cos_oh );
	if (pdf < 1.0e-6f) return; // skip samples with very low probability
	const float D = MDF == GGXMDF ? GGXMDF_D( m, alpha_x, alpha_y ) : GTR1MDF_D( m, alpha_x, alpha_y );
	const float G = MDF == GGXMDF ? GGXMDF_G( wi, wo, m, alpha_x, alpha_y ) : GTR1MDF_G( wi, wo, m, alpha_x, alpha_y );
	if (MDF == GGXMDF) DisneySpecularFresnel( shadingData, wo, m, value ); else DisneyClearcoatFresnel( shadingData, wo, m, value );
	value *= D * G / fabs( 4.0f * wo.z * wi.z );
	wiw = Tangent2World( wi, N, T, B );
}

template <uint MDF, bool flip>
LH2_DEVFUNC float evaluate_mf( CONSTREF_OF( ShadingData ) shadingData, const float alpha_x, const float alpha_y, const float3 N, const float3 T, const float3 B, const float3 wow, const float3 wiw, REFERENCE_OF( float3 ) bsdf )
{
	float3 wo = World2Tangent( wow, N, T, B );
	float3 wi = World2Tangent( wiw, N, T, B );
	if (wo.z == 0 || wi.z == 0) return 0;
	// flip the incoming and outgoing vectors to be in the same hemisphere as the shading normal if needed.
	if (flip) wo.z = fabs( wo.z ), wi.z = fabs( wi.z );
	const float3 m = normalize( wi + wo );
	const float cos_oh = dot( wo, m );
	if (cos_oh == 0) return 0;
	const float D = MDF == GGXMDF ? GGXMDF_D( m, alpha_x, alpha_y ) : GTR1MDF_D( m, alpha_x, alpha_y );
	const float G = MDF == GGXMDF ? GGXMDF_G( wi, wo, m, alpha_x, alpha_y ) : GTR1MDF_G( wi, wo, m, alpha_x, alpha_y );
	if (MDF == GGXMDF) DisneySpecularFresnel( shadingData, wo, m, bsdf ); else DisneyClearcoatFresnel( shadingData, wo, m, bsdf );
	bsdf *= D * G / fabs( 4.0f * wo.z * wi.z );
	return (MDF == GGXMDF ? GGXMDF_pdf( wo, m, alpha_x, alpha_y ) : GTR1MDF_pdf( wo, m, alpha_x, alpha_y )) / fabs( 4.0f * cos_oh );
}

LH2_DEVFUNC float evaluate_diffuse( CONSTREF_OF( ShadingData ) shadingData, const float3 iN, const float3 wow, const float3 wiw, REFERENCE_OF( float3 ) value )
{
	// this code is mostly ported from the GLSL implementation in Disney's BRDF explorer.
	const float3 n = iN;
	const float3 h( normalize( wiw + wow ) );
	// using the absolute values of cos_on and cos_in creates discontinuities
	const float cos_on = dot( n, wow );
	const float cos_in = dot( n, wiw );
	const float cos_ih = dot( wiw, h );
	const float fl = schlick_fresnel( cos_in );
	const float fv = schlick_fresnel( cos_on );
	float fd = 0;
	if (SUBSURFACE != 1.0f)
	{
		const float fd90 = 0.5f + 2.0f * sqr( cos_ih ) * ROUGHNESS;
		fd = mix( 1.0f, fd90, fl ) * mix( 1.0f, fd90, fv );
	}
	if (SUBSURFACE > 0)
	{
		// Based on Hanrahan-Krueger BRDF approximation of isotropic BSRDF. The 1.25 scale is used 
		// to (roughly) preserve albedo. Fss90 is used to "flatten" retroreflection based on roughness.
		const float fss90 = sqr( cos_ih ) * ROUGHNESS;
		const float fss = mix( 1.0f, fss90, fl ) * mix( 1.0f, fss90, fv );
		const float ss = 1.25f * (fss * (1.0f / (fabs( cos_on ) + fabs( cos_in )) - 0.5f) + 0.5f);
		fd = mix( fd, ss, SUBSURFACE );
	}
	value = shadingData.color * fd * INVPI * (1.0f - METALLIC);
	return fabs( cos_in ) * INVPI;
}

LH2_DEVFUNC void sample_diffuse( CONSTREF_OF( ShadingData ) shadingData, const float r0, const float r1, 
	const float3 N, const float3 T, const float3 B, const float3 wow,
	/* OUT: */ REFERENCE_OF( float3 ) wiw, REFERENCE_OF( float ) pdf, REFERENCE_OF( float3 ) value )
{
	// compute the incoming direction
	const float3 wi = DiffuseReflectionCosWeighted( r0, r1 );
	wiw = normalize( Tangent2World( wi, N, T, B ) );
	// compute the component value and the probability density of the sampled direction.
	pdf = evaluate_diffuse( shadingData, N, wow, wiw, value );
	if (pdf < 1.0e-6f) return;
}

LH2_DEVFUNC float evaluate_sheen( CONSTREF_OF( ShadingData ) shadingData, const float3 wow, const float3 wiw, REFERENCE_OF( float3 ) value )
{
	// this code is mostly ported from the GLSL implementation in Disney's BRDF explorer.
	const float3 h( normalize( wow + wow ) );
	const float cos_ih = dot( wiw, h );
	const float fh = schlick_fresnel( cos_ih );
	mix_one_with_spectra( TINT, SHEENTINT, value );
	value *= fh * SHEEN * (1.0f - METALLIC);
	return 1.0f / (2 * PI); // return the probability density of the sampled direction
}

LH2_DEVFUNC void sample_sheen( CONSTREF_OF( ShadingData ) shadingData, const float r0, const float r1, 
	const float3 N, const float3 T, const float3 B, const float3 wow,
	/* OUT: */ REFERENCE_OF( float3 ) wiw, REFERENCE_OF( float ) pdf, REFERENCE_OF( float3 ) value )
{
	// compute the incoming direction
	const float3 wi = DiffuseReflectionCosWeighted( r0, r1 );
	wiw = normalize( Tangent2World( wi, N, T, B ) );
	// compute the component value and the probability density of the sampled direction
	pdf = evaluate_sheen( shadingData, wow, wiw, value );
	if (pdf < 1.0e-6f) return;
}

LH2_DEVFUNC float3 SampleBSDF( CONSTREF_OF( ShadingData ) shadingData, float3 iN, const float3 N, const float3 iT, const float3 wow, const float distance,
	const float r0, const float r1, REFERENCE_OF( float3 ) wiw, REFERENCE_OF( float ) pdf, REFERENCE_OF( bool ) specular
#ifdef __CUDACC__
	, bool adjoint = false
#endif
)
{
	// detect backside hits
	float flip = (dot( wow, N ) < 0) ? -1 : 1;
	iN *= flip;
	// use the pure specular dielectric path from the Lambert shader for now
	if (r0 < TRANSMISSION)
	{
		// specular
		specular = true, pdf = 1;
		const float eio = flip < 0 ? (1.0f / ETA) : ETA, F = Fr_L( dot( iN, wow ), eio );
		float3 beer = make_float3( 1 );
		beer.x = expf( -shadingData.transmittance.x * distance * 2.0f );
		beer.y = expf( -shadingData.transmittance.y * distance * 2.0f );
		beer.z = expf( -shadingData.transmittance.z * distance * 2.0f );
		if (r1 < F)
		{
			wiw = reflect( wow * -1.0f, iN );
			if (dot( N * flip, wiw ) <= 0) pdf = 0; // APPLYSAFENORMALS
			return shadingData.color * beer * (1 / abs( dot( iN, wiw ) ));
		}
		else
		{
			if (!Refract_L( wow, iN, eio, wiw )) return make_float3( 0 );
			float ajointCorrection = 1.0f;
			if (adjoint) ajointCorrection = (eio * eio);
			return shadingData.color * beer * ajointCorrection * (1 / abs( dot( iN, wiw ) ));
		}
	}
	// not a dielectric: normalize r0 to r3
	const float r3 = (r0 - TRANSMISSION) / (1 - TRANSMISSION);
	// calculate tangent matrix
	const float3 B = normalize( cross( iN, iT ) );
	const float3 T = normalize( cross( iN, B ) );
	// compute component weights and cdf
	float4 weights = make_float4( 
		lerp( LUMINANCE, 0, METALLIC ), 
		lerp( SHEEN, 0, METALLIC ), 
		lerp( SPECULAR, 1, METALLIC ), 
		CLEARCOAT * 0.25f 
	);
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	const float4 cdf = make_float4( weights.x, weights.x + weights.y, weights.x + weights.y + weights.z, 0 );
	// sample a random component
	float probability, component_pdf;
	float3 contrib, value = make_float3( 0 );
	if (r3 < cdf.x)
	{
		const float r2 = r3 / cdf.x; // reuse r3 after normalization
		sample_diffuse( shadingData, r2, r1, iN, T, B, wow, wiw, component_pdf, value );
		probability = DIFFWEIGHT * component_pdf, DIFFWEIGHT = 0;
	}
	else if (r3 < cdf.y)
	{
		const float r2 = (r3 - cdf.x) / (cdf.y - cdf.x); // reuse r3 after normalization
		sample_sheen( shadingData, r2, r1, iN, T, B, wow, wiw, component_pdf, value );
		probability = SHEENWEIGHT * component_pdf, SHEENWEIGHT = 0;
	}
	else if (r3 < cdf.z)
	{
		const float r2 = (r3 - cdf.y) / (cdf.z - cdf.y); // reuse r3 after normalization
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
		sample_mf<GGXMDF, false>( shadingData, r2, r1, alpha_x, alpha_y, iN, T, B, N * flip, wow, wiw, component_pdf, value );
		probability = SPECWEIGHT * component_pdf, SPECWEIGHT = 0;
	}
	else
	{
		const float r2 = (r3 - cdf.z) / (1 - cdf.z); // reuse r3 after normalization
		const float alpha = clearcoat_roughness( shadingData );
		sample_mf<GTR1MDF, false>( shadingData, r2, r1, alpha, alpha, iN, T, B, N * flip, wow, wiw, component_pdf, value );
		probability = COATWEIGHT * component_pdf, COATWEIGHT = 0;
	}
	if (DIFFWEIGHT > 0) probability += DIFFWEIGHT * evaluate_diffuse( shadingData, iN, wow, wiw, contrib ), value += contrib;
	if (SHEENWEIGHT > 0) probability += SHEENWEIGHT * evaluate_sheen( shadingData, wow, wiw, contrib ), value += contrib;
	if (SPECWEIGHT > 0)
	{
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
		probability += SPECWEIGHT * evaluate_mf<GGXMDF, false>( shadingData, alpha_x, alpha_y, iN, T, B, wow, wiw, contrib );
		value += contrib;
	}
	if (COATWEIGHT > 0)
	{
		const float alpha = clearcoat_roughness( shadingData );
		probability += COATWEIGHT * evaluate_mf<GTR1MDF, false>( shadingData, alpha, alpha, iN, T, B, wow, wiw, contrib );
		value += contrib;
	}
	if (probability > 1.0e-6f) pdf = probability; else pdf = 0;
	return value;
}

LH2_DEVFUNC float3 EvaluateBSDF( CONSTREF_OF( ShadingData ) shadingData, const float3 iN, const float3 iT, const float3 wow, const float3 wiw, REFERENCE_OF( float ) pdf )
{
	if (TRANSMISSION > 0.999f || ROUGHNESS <=0.001f)
	{
		// no transport via explicit connections for specular vertices
		pdf = 0;
		return make_float3( 0 );
	}
	// calculate tangent matrix
	const float3 B = normalize( cross( iN, iT ) );
	const float3 T = normalize( cross( iN, B ) );
	// compute component weights
	float4 weights = make_float4( lerp( LUMINANCE, 0, METALLIC ), lerp( SHEEN, 0, METALLIC ), lerp( SPECULAR, 1, METALLIC ), CLEARCOAT * 0.25f );
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	// compute pdf
	pdf = 0;
	float3 value = make_float3( 0 );
	if (DIFFWEIGHT > 0) pdf += DIFFWEIGHT * evaluate_diffuse( shadingData, iN, wow, wiw, value );
	if (SHEENWEIGHT > 0) pdf += SHEENWEIGHT * evaluate_sheen( shadingData, wow, wiw, value );
	if (SPECWEIGHT > 0)
	{
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
		float3 contrib;
		const float spec_pdf = evaluate_mf<GGXMDF, false>( shadingData, alpha_x, alpha_y, iN, T, B, wow, wiw, contrib );
		if (spec_pdf > 0) pdf += SPECWEIGHT * spec_pdf, value += contrib;
	}
	if (COATWEIGHT > 0)
	{
		const float alpha = clearcoat_roughness( shadingData );
		float3 contrib;
		const float clearcoat_pdf = evaluate_mf<GTR1MDF, false>( shadingData, alpha, alpha, iN, T, B, wow, wiw, contrib );
		if (clearcoat_pdf > 0) pdf += COATWEIGHT * clearcoat_pdf, value += contrib;
	}
	return value;
}

/* template <uint MDF, bool flip>
LH2_DEVFUNC float pdf_mf( const float alpha_x, const float alpha_y, const float3 iN, const float3 wow, const float3 wiw )
{
	float3 wo = World2Tangent( wow, iN ); // local_geometry.m_shading_basis.transform_to_local( outgoing );
	float3 wi = World2Tangent( wiw, iN ); // local_geometry.m_shading_basis.transform_to_local( incoming );
	// flip the incoming and outgoing vectors to be in the same hemisphere as the shading normal if needed.
	if (flip) wo.z = fabs( wo.z ), wi.z = fabs( wi.z );
	const float3 m = normalize( wi + wo );
	const float cos_oh = dot( wo, m );
	if (cos_oh == 0) return 0;
	return (MDF == GGXMDF ? GGXMDF_pdf( wo, m, alpha_x, alpha_y ) : GTR1MDF_pdf( wo, m, alpha_x, alpha_y )) / fabs( 4.0f * cos_oh );
}

LH2_DEVFUNC float evaluate_pdf( CONSTREF_OF( ShadingData ) shadingData, const float3 iN, const float3 wow, const float3 wiw )
{
	// compute component weights
	float4 weights = make_float4( lerp( LUMINANCE, 0, METALLIC ), lerp( SHEEN, 0, METALLIC ), lerp( SPECULAR, 1, METALLIC ), CLEARCOAT * 0.25f );
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	// compute pdf
	float pdf = 0;
	if (DIFFWEIGHT > 0) pdf += DIFFWEIGHT * fabs( dot( wiw, iN ) ) * INVPI;
	if (SHEENWEIGHT > 0) pdf += SHEENWEIGHT * (1.0f / (2 * PI));
	if (SPECWEIGHT > 0)
	{
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
		pdf += SPECWEIGHT * pdf_mf<GGXMDF, false>( alpha_x, alpha_y, iN, wow, wiw );
	}
	if (COATWEIGHT > 0)
	{
		const float alpha = clearcoat_roughness( shadingData );
		pdf += COATWEIGHT * pdf_mf<GTR1MDF, false>( alpha, alpha, iN, wow, wiw );
	}
	// assert( pdf >= 0 ); 
	return pdf;
} */

#endif