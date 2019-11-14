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

struct InputValues
{
	float sheen, metallic, specular, clearcoat, clearcoat_gloss, roughness, anisotropic, subsurface, sheen_tint, specular_tint;
	float3 tint_color, base_color;
	float base_color_luminance;
};

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
LH2_DEVFUNC float clearcoat_roughness( const InputValues disney ) { return mix( 0.1f, 0.001f, disney.clearcoat_gloss ); }
LH2_DEVFUNC void DisneySpecularFresnel( const InputValues disney, const float3 o, const float3 h, const float3 n, REFERENCE_OF( float3 ) value )
{
	mix_one_with_spectra( disney.tint_color, disney.specular_tint, value );
	value *= disney.specular * 0.08f;
	mix_spectra( value, disney.base_color, disney.metallic, value );
	const float cos_oh = fabs( dot( o, h ) );
	mix_spectra_with_one( value, schlick_fresnel( cos_oh ), value );
}
LH2_DEVFUNC void DisneyClearcoatFresnel( const InputValues disney, const float3 o, const float3 h, const float3 n, REFERENCE_OF( float3 ) value )
{
	const float cos_oh = fabs( dot( o, h ) );
	value = make_float3( mix( 0.04f, 1.0f, schlick_fresnel( cos_oh ) ) * 0.25f * disney.clearcoat );
}

template <uint MDF, bool flip>
LH2_DEVFUNC void sample_mf( const InputValues disney, REFERENCE_OF( uint ) seed, const float alpha_x, const float alpha_y, const float3 iN, const float3 outgoing, 
	/* OUT: */ REFERENCE_OF( float3 ) incoming, REFERENCE_OF( float ) pdf, REFERENCE_OF( float3 ) value )
{
	float3 wo = local_geometry.m_shading_basis.transform_to_local( outgoing );
	if (wo.y == 0) return;
	if (flip) wo.y = fabs( wo.y ); // TODO: z = up
	// compute the incoming direction by sampling the MDF
	sampling_context.split_in_place( 2, 1 );
	const float2 s = make_float2( RandomFloat( seed ), RandomFloat( seed ) );
	float3 m = MDF == GGXMDF ? GGXMDF_sample( wo, s, alpha_x, alpha_y ) : GTR1MDF_sample( wo, s, alpha_x, alpha_y );
	float3 wi = reflect( wo, m );
	// force the outgoing direction to lie above the geometric surface.
	const float3 ng = local_geometry.m_shading_basis.transform_to_local( local_geometry.m_geometric_normal );
	if (BSDF::force_above_surface( wi, ng )) m = normalize( wo + wi );
	if (wi.y == 0) return;
	const float cos_oh = dot( wo, m );
	pdf = MDF == GGXMDF ? (GGXMDF_pdf( wo, m, alpha_x, alpha_y ) : GTR1MDF_pdf( wo, m, alpha_x, alpha_y )) / fabs( 4.0f * cos_oh );
	/* assert( pdf >= 0 ); */
	if (pdf < 1.0e-6f) return; // skip samples with very low probability
	const float D = MDF == GGXMDF ? GGXMDF_D( m, alpha_x, alpha_y ) : GTR1MDF_D( m, alpha_x, alpha_y );
	const float G = MDF == GGXMDF ? GGXMDF_G( wi, wo, m, alpha_x, alpha_y ) : GTR1MDF_G( wi, wo, m, alpha_x, alpha_y );
	const float3 n = make_float3( 0, 1, 0 );
	const float cos_on = wo.y;
	const float cos_in = wi.y;
	if (MDF == GGXMDF) DisneySpecularFresnel( disney, wo, m, n, value ); else DisneyClearcoatFresnel( wo, m, n, value );
	value *= D * G / fabs( 4.0f * cos_on * cos_in );
	incoming = local_geometry.m_shading_basis.transform_to_parent( wi );
	// sample.set_to_scattering( ScatteringMode::Glossy, pdf );
	// sample.m_incoming = Dual<float3>( incoming );
	// sample.compute_reflected_differentials( iN, outgoing );
}

template <uint MDF, bool flip>
LH2_DEVFUNC float evaluate_mf( const InputValues disney, const float alpha_x, const float alpha_y, const float3 iN, const float3 outgoing, const float3 incoming, REFERENCE_OF( float3 ) value )
{
	float3 wo = local_geometry.m_shading_basis.transform_to_local( outgoing );
	float3 wi = local_geometry.m_shading_basis.transform_to_local( incoming );
	if (wo.y == 0 || wi.y == 0) return 0;
	// flip the incoming and outgoing vectors to be in the same hemisphere as the shading normal if needed.
	if (flip) wo.y = fabs( wo.y ), wi.y = fabs( wi.y );
	const float3 m = normalize( wi + wo );
	const float cos_oh = dot( wo, m );
	if (cos_oh == 0) return 0;
	const float D = MDF == GGXMDF ? GGXMDF_D( m, alpha_x, alpha_y ) : GTR1MDF_D( m, alpha_x, alpha_y );
	const float G = MDF == GGXMDF ? GGXMDF_G( wi, wo, m, alpha_x, alpha_y ) : GTR1MDF_G( wi, wo, m, alpha_x, alpha_y );
	const float3 n( 0, 1, 0 );
	if (MDF == GGXMDF) DisneySpecularFresnel( disney, wo, m, n, value ); else DisneyClearcoatFresnelFun( wo, m, n, value );
	const float cos_on = wo.y;
	const float cos_in = wi.y;
	value *= D * G / fabs( 4.0f * cos_on * cos_in );
	return (MDF == GGXMDF ? GGXMDF_pdf( wo, m, alpha_x, alpha_y ) : GTR1MDF_pdf( wo, m, alpha_x, alpha_y )) / fabs( 4.0f * cos_oh );
}

template <uint MDF, bool flip>
LH2_DEVFUNC float pdf_mf( const float alpha_x, const float alpha_y, const float3 iN, const float3 outgoing, const float3 incoming )
{
	float3 wo = local_geometry.m_shading_basis.transform_to_local( outgoing );
	float3 wi = local_geometry.m_shading_basis.transform_to_local( incoming );
	// flip the incoming and outgoing vectors to be in the same hemisphere as the shading normal if needed.
	if (flip) wo.y = fabs( wo.y ), wi.y = fabs( wi.y );
	const float3 m = normalize( wi + wo );
	const float cos_oh = dot( wo, m );
	if (cos_oh == 0) return 0;
	return (MDF == GGXMDF ? GGXMDF_pdf( wo, m, alpha_x, alpha_y ) : GTR1MDF_pdf( wo, m, alpha_x, alpha_y )) / fabs( 4.0f * cos_oh );
}

LH2_DEVFUNC float evaluate_diffuse( const InputValues disney, const float3 iN, const float3 outgoing, const float3 incoming, REFERENCE_OF( float3 ) value )
{
	// this code is mostly ported from the GLSL implementation in Disney's BRDF explorer.
	// const float3 n( local_geometry.m_shading_basis.get_normal() );
	const float3 n = iN;
	const float3 h( normalize( incoming + outgoing ) );
	// using the absolute values of cos_on and cos_in creates discontinuities
	const float cos_on = dot( n, outgoing );
	const float cos_in = dot( n, incoming );
	const float cos_ih = dot( incoming, h );
	const float fl = schlick_fresnel( cos_in );
	const float fv = schlick_fresnel( cos_on );
	float fd = 0;
	if (disney.subsurface != 1.0f)
	{
		const float fd90 = 0.5f + 2.0f * sqr( cos_ih ) * disney.roughness;
		fd = mix( 1.0f, fd90, fl ) * mix( 1.0f, fd90, fv );
	}
	if (disney.subsurface > 0)
	{
		// based on Hanrahan-Krueger BRDF approximation of isotropic BSRDF.
		// the 1.25 scale is used to (roughly) preserve albedo.
		// Fss90 is used to "flatten" retroreflection based on roughness.
		const float fss90 = sqr( cos_ih ) * disney.roughness;
		const float fss = mix( 1.0f, fss90, fl ) * mix( 1.0f, fss90, fv );
		const float ss = 1.25f * (fss * (1.0f / (fabs( cos_on ) + fabs( cos_in )) - 0.5f) + 0.5f);
		fd = mix( fd, ss, disney.subsurface );
	}
	value = disney.base_color * fd * INVPI * (1.0f - disney.metallic);
	return fabs( cos_in ) * INVPI;
}

LH2_DEVFUNC void sample_diffuse( const InputValues disney, REFERENCE_OF( uint) seed, const float3 iN, const float3 outgoing, 
	/* OUT: */ REFERENCE_OF( float3 ) incoming, REFERENCE_OF( float ) pdf, REFERENCE_OF( float3 ) value )
{
	// compute the incoming direction
	// const float2 s = make_float2( RandomFloat( seed ), RandomFloat( seed ) );
	// const float3 wi = sample_hemisphere_cosine( s );
	// incoming = local_geometry.m_shading_basis.transform_to_parent( wi );
	const float3 wi = DiffuseReflectionCosWeighted( RandomFloat( seed ), RandomFloat( seed ) );
	incoming = normalize( Tangent2World( wi, iN ) );
	// compute the component value and the probability density of the sampled direction.
	pdf = evaluate_diffuse( disney, iN, outgoing, incoming, value );
	/* assert( pdf > 0 ); */
	if (pdf < 1.0e-6f) return;
	// sample.set_to_scattering( ScatteringMode::Diffuse, pdf );
	// sample.m_aov_components.m_albedo = disney->base_color;
	// sample.compute_reflected_differentials( iN, outgoing );
}

LH2_DEVFUNC float evaluate_sheen( const InputValues disney, const float3 outgoing, const float3 incoming, REFERENCE_OF( float3 ) value )
{
	// this code is mostly ported from the GLSL implementation in Disney's BRDF explorer.
	const float3 h( normalize( incoming + outgoing ) );
	const float cos_ih = dot( incoming, h );
	const float fh = schlick_fresnel( cos_ih );
	mix_one_with_spectra( disney.tint_color, disney.sheen_tint, value );
	value *= fh * disney.sheen * (1.0f - disney.metallic);
	return 1.0f / (2 * PI); // return the probability density of the sampled direction
}

LH2_DEVFUNC void sample_sheen( const InputValues disney, REFERENCE_OF( uint ) seed, const float3 iN, const float3 outgoing, 
	/* OUT: */ REFERENCE_OF( float3 ) incoming, REFERENCE_OF( float ) pdf, REFERENCE_OF( float3 ) value )
{
	// compute the incoming direction
	// const float2 s = make_float2( RandomFloat( seed ), RandomFloat( seed ) );
	// const float3 wi = sample_hemisphere_uniform( s );
	// incoming = local_geometry.m_shading_basis.transform_to_parent( wi );
	const float3 wi = DiffuseReflectionCosWeighted( RandomFloat( seed ), RandomFloat( seed ) );
	incoming = normalize( Tangent2World( wi, iN ) );
	// compute the component value and the probability density of the sampled direction
	pdf = evaluate_sheen( disney, outgoing, incoming, value );
	/* assert( pdf > 0 ); */
	if (pdf < 1.0e-6f) return;
	// sample.m_incoming = Dual3f( incoming );
	// sample.set_to_scattering( ScatteringMode::Glossy, pdf );
	// sample.compute_reflected_differentials( iN, outgoing );
}

/* void prepare_inputs( Arena arena, const ShadingPoint shading_point, void* data )
{
	disney.roughness = max( disney.roughness, shading_point.get_ray().m_min_roughness );
	new (&disney.m_precomputed) InputValues::Precomputed();
	const Color3f tint_xyz = disney.base_color.to_ciexyz( g_std_lighting_conditions );
	disney.m_precomputed.tint_color.set( tint_xyz[1] > 0 ? ciexyz_to_linear_rgb( tint_xyz / tint_xyz[1] ) : Color3f( 1.0f ), g_std_lighting_conditions, Spectrum::Reflectance );
	disney.m_precomputed.base_color_luminance = tint_xyz[1];
} */

LH2_DEVFUNC void sample_disney( const InputValues disney, REFERENCE_OF( uint ) seed, const float3 iN, const float3 outgoing, 
	/* OUT: */ REFERENCE_OF( float3 ) incoming, REFERENCE_OF( float) pdf, REFERENCE_OF( float3 ) value )
{
	// compute component weights and cdf
	float4 weights = make_float4( lerp( disney.base_color_luminance, 0, disney.metallic ), lerp( disney.sheen, 0, disney.metallic ), lerp( disney.specular, 1, disney.metallic ), disney.clearcoat * 0.25f );
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	const float4 cdf = make_float4( weights.x, weights.x + weights.y, weights.x + weights.y + weights.z, 0 );
	// sample a random component
	float probability;
	float component_pdf;
	float3 contrib;
	const float s = RandomFloat( seed );
	if (s < cdf.x)
	{
		sample_diffuse( disney, seed, iN, outgoing, incoming, component_pdf, value );
		probability = DIFFWEIGHT * component_pdf, DIFFWEIGHT = 0;
	}
	else if (s < cdf.y)
	{
		sample_sheen( disney, seed, iN, outgoing, incoming, component_pdf, value );
		probability = SHEENWEIGHT * component_pdf, SHEENWEIGHT = 0;
	}
	else if (s < cdf.z)
	{
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness( disney.roughness, disney.anisotropic, alpha_x, alpha_y );
		sample_mf<GGXMDF, false>( disney, seed, alpha_x, alpha_y, iN, outgoing, incoming, component_pdf, value );
		probability = SPECWEIGHT * component_pdf, SPECWEIGHT = 0;
	}
	else
	{
		const float alpha = clearcoat_roughness( disney );
		sample_mf<GTR1MDF, false>( disney, seed, alpha, alpha, iN, outgoing, incoming, component_pdf, value );
		probability = COATWEIGHT * component_pdf, COATWEIGHT = 0;
	}
	if (DIFFWEIGHT > 0) probability += DIFFWEIGHT * evaluate_diffuse( disney, iN, outgoing, incoming, contrib ), value += contrib;
	if (SHEENWEIGHT > 0) probability += SHEENWEIGHT * evaluate_sheen( disney, outgoing, incoming, contrib ), value += contrib;
	if (SPECWEIGHT > 0)
	{
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness( disney.roughness, disney.anisotropic, alpha_x, alpha_y );
		probability += SPECWEIGHT * evaluate_mf<GGXMDF, false>( disney, alpha_x, alpha_y, iN, outgoing, incoming, contrib );
		value += contrib;
	}
	if (COATWEIGHT > 0)
	{
		const float alpha = clearcoat_roughness( disney );
		probability += COATWEIGHT * evaluate_mf<GTR1MDF, false>( disney, alpha, alpha, iN, outgoing, incoming, contrib );
		value += contrib;
	}
	if (probability > 1.0e-6f)
	{
		pdf = probability;
		// sample.set_to_scattering( sample.get_mode(), probability );
		// sample.m_value.m_beauty = sample.m_value.m_diffuse;
		// sample.m_value.m_beauty += sample.m_value.m_glossy;
		// sample.m_min_roughness = disney->roughness;
	}
	else 
	{
		pdf = 0;
		// sample.set_to_absorption();
	}
}

LH2_DEVFUNC float evaluate_disney( const InputValues disney, const float3 iN, const float3 outgoing, const float3 incoming, REFERENCE_OF( float3 ) value )
{
	// compute component weights
	float4 weights = make_float4( lerp( disney.base_color_luminance, 0, disney.metallic ), lerp( disney.sheen, 0, disney.metallic ), lerp( disney.specular, 1, disney.metallic ), disney.clearcoat * 0.25f );
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	// compute pdf
	float pdf = 0;
	value = make_float3( 0 );
	if (DIFFWEIGHT > 0) pdf += DIFFWEIGHT * evaluate_diffuse( disney, iN, outgoing, incoming, value );
	if (SHEENWEIGHT > 0) pdf += SHEENWEIGHT * evaluate_sheen( disney, outgoing, incoming, value );
	if (SPECWEIGHT > 0)
	{
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness( disney.roughness, disney.anisotropic, alpha_x, alpha_y );
		float3 contrib;
		const float spec_pdf = evaluate_mf<GGXMDF, false>( disney, alpha_x, alpha_y, iN, outgoing, incoming, contrib );
		if (spec_pdf > 0) pdf += SPECWEIGHT * spec_pdf, value += contrib;
	}
	if (COATWEIGHT > 0)
	{
		const float alpha = clearcoat_roughness( disney );
		float3 contrib;
		const float clearcoat_pdf = evaluate_mf<GTR1MDF, false>( disney, alpha, alpha, iN, outgoing, incoming, contrib );
		if (clearcoat_pdf > 0) pdf += COATWEIGHT * clearcoat_pdf, value += contrib;
	}
	/* assert( pdf >= 0 ); */ return pdf;
}

LH2_DEVFUNC float evaluate_pdf( const InputValues disney, const float3 iN, const float3 wow, const float3 wiw )
{
	// compute component weights
	float4 weights = make_float4( lerp( disney.base_color_luminance, 0, disney.metallic ), lerp( disney.sheen, 0, disney.metallic ), lerp( disney.specular, 1, disney.metallic ), disney.clearcoat * 0.25f );
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	// compute pdf
	float pdf = 0;
	if (DIFFWEIGHT > 0) pdf += DIFFWEIGHT * fabs( dot( wiw, iN ) ) * INVPI;
	if (SHEENWEIGHT > 0) pdf += SHEENWEIGHT * (1.0f / (2 * PI));
	if (SPECWEIGHT > 0)
	{
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness( disney.roughness, disney.anisotropic, alpha_x, alpha_y );
		pdf += SPECWEIGHT * pdf_mf<GGXMDF, false>( alpha_x, alpha_y, iN, wow, wiw );
	}
	if (COATWEIGHT > 0)
	{
		const float alpha = clearcoat_roughness( disney );
		pdf += COATWEIGHT * pdf_mf<GTR1MDF, false>( alpha, alpha, iN, wow, wiw );
	}
	/* assert( pdf >= 0 ); */ return pdf;
}

#endif