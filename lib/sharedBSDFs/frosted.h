/* frosted.h - License information:

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

   // https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/renderer/modeling/bsdf/glassbsdf.cpp
*/

#include "noerrors.h"

// HELPERS

LH2_DEVFUNC void fresnel_reflectance_dielectric( float& reflectance, const float& eta, const float cos_theta_i, const float cos_theta_t )
{
	if (cos_theta_i == 0 && cos_theta_t == 0) { reflectance = 1; return; }
	const float k0 = eta * cos_theta_t, k1 = eta * cos_theta_i;
	reflectance = 0.5f * (sqr( (cos_theta_i - k0) / (cos_theta_i + k0) ) + sqr( (cos_theta_t - k1) / (cos_theta_t + k1) ));
}

LH2_DEVFUNC float fresnel_reflectance( const float cos_theta_i, const float eta, float& cos_theta_t )
{
	const float sin_theta_t2 = (1 - sqr( cos_theta_i )) * sqr( eta );
	if (sin_theta_t2 > 1) { cos_theta_t = 0; return 1; }
	cos_theta_t = min( sqrtf( max( 1 - sin_theta_t2, 0.0f ) ), 1.0f );
	float F;
	fresnel_reflectance_dielectric( F, eta, abs( cos_theta_i ), cos_theta_t );
	return F;
}

LH2_DEVFUNC float fresnel_reflectance( const float cos_theta_i, const float eta )
{
	float cos_theta_t;
	return fresnel_reflectance( cos_theta_i, eta, cos_theta_t );
}

LH2_DEVFUNC float3 improve_normalization( const float3 v )
{
	return v * ((3 - dot( v, v )) * 0.5f);
}

LH2_DEVFUNC float3 refracted_direction( const float3& wo, const float3& m, const float cos_wom, const float cos_theta_t, const float rcp_eta )
{
	const float3 wi = cos_wom > 0
		? (rcp_eta * cos_wom - cos_theta_t) * m - rcp_eta * wo
		: (rcp_eta * cos_wom + cos_theta_t) * m - rcp_eta * wo;
	return improve_normalization( wi );
}

LH2_DEVFUNC float choose_reflection_probability( const float reflection_weight, const float refraction_weight, const float F )
{
	const float r_probability = F * reflection_weight;
	const float t_probability = (1 - F) * refraction_weight;
	const float sum_probabilities = r_probability + t_probability;
	return sum_probabilities != 0 ? r_probability / sum_probabilities : 1;
}

LH2_DEVFUNC float3 half_reflection_vector( const float3& wo, const float3& wi )
{
	const float3 h = normalize( wi + wo );
	return h.z < 0 ? (h * -1) : h;
}

LH2_DEVFUNC void evaluate_reflection( const float3& reflection_color, const float3& wo, const float3& wi, const float3& m, const float alpha_x, const float alpha_y, const float F, float3& value )
{
	const float denom = abs( 4 * wo.z * wi.z );
	if (denom == 0) { value = make_float3( 0 ); return; }
	const float D = GGXMDF_D( m, alpha_x, alpha_y );
	const float G = GGXMDF_G( wi, wo, m, alpha_x, alpha_y );
	value = reflection_color * (F * D * G / denom);
}

LH2_DEVFUNC float3 half_refraction_vector( const float3& wo, const float3& wi, const float eta )
{
	const float3 h = normalize( wo + eta * wi );
	return h.z < 0 ? (h * -1) : h;
}

LH2_DEVFUNC void evaluate_refraction( const float eta, const float3& refraction_color, const bool adjoint, const float3& wo,
	const float3& wi, const float3& m, const float alpha_x, const float alpha_y, const float T, float3& value )
{
	if (wo.z == 0 || wi.z == 0) { value = make_float3( 0 ); return; }
	const float cos_ih = dot( m, wi ), cos_oh = dot( m, wo );
	const float dots = (cos_ih * cos_oh) / (wi.z * wo.z);
	const float sqrt_denom = cos_oh + eta * cos_ih;
	if (abs( sqrt_denom ) < 1.0e-6f) { value = make_float3( 0 ); return; }
	const float D = GGXMDF_D( m, alpha_x, alpha_y );
	const float G = GGXMDF_G( wi, wo, m, alpha_x, alpha_y );
	float multiplier = abs( dots ) * T * D * G / sqr( sqrt_denom );
	if (!adjoint) multiplier *= sqr( eta );
	value = refraction_color * multiplier;
}

LH2_DEVFUNC float reflection_jacobian( const float3& wo, const float3& m, const float cos_oh, const float alpha_x, const float alpha_y )
{
	if (cos_oh == 0) return 0;
	return 1 / (4 * abs( cos_oh ));
}

LH2_DEVFUNC float refraction_jacobian( const float3& wo, const float3& wi, const float3& m, const float alpha_x, const float alpha_y, const float eta )
{
	const float cos_ih = dot( m, wi ), cos_oh = dot( m, wo );
	const float sqrt_denom = cos_oh + eta * cos_ih;
	if (abs( sqrt_denom ) < 1.0e-6f) return 0;
	return abs( cos_ih ) * sqr( eta / sqrt_denom );
}

// IMPLEMENTATION

LH2_DEVFUNC float3 SampleBSDF_frosted( const ShadingData shadingData, float3 iN, const float3 N, const float3 iT, const float3 wow, const float distance,
	const float r3, const float r4, const float r5, REFERENCE_OF( float3 ) wiw, REFERENCE_OF( float ) pdf, REFERENCE_OF( bool ) specular
#ifdef __CUDACC__
	, bool adjoint = false
#endif
)
{
	specular = true;
	const float flip = (dot( wow, N ) < 0) ? -1 : 1;
	iN *= flip;
	const float3 B = normalize( cross( iN, iT ) );
	const float3 T = normalize( cross( iN, B ) );
	const float3 wol = World2Tangent( wow, iN, T, B );
	const float eta = flip < 0 ? (1 / ETA) : ETA;
	if (eta == 1) return make_float3( 0 );
	const float3 beer = make_float3(
		expf( -shadingData.transmittance.x * distance * 2.0f ),
		expf( -shadingData.transmittance.y * distance * 2.0f ),
		expf( -shadingData.transmittance.z * distance * 2.0f ) );
	float alpha_x, alpha_y;
	microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
	const float3 m = GGXMDF_sample( wol, r3, r4, alpha_x, alpha_y );
	const float rcp_eta = 1 / eta;
	const float cos_wom = clamp( dot( wol, m ), -1.0f, 1.0f );
	float cos_theta_t, jacobian;
	const float F = fresnel_reflectance( cos_wom, eta, cos_theta_t );
	float3 wil, retVal;
	if (r5 < F) // compute the reflected direction
	{
		wil = reflect( wol * -1.0f, m );
		if (wil.z * wol.z <= 0) return make_float3( 0 );
		evaluate_reflection( shadingData.color, wol, wil, m, alpha_x, alpha_y, F, retVal );
		pdf = F, jacobian = reflection_jacobian( wol, m, cos_wom, alpha_x, alpha_y );
	}
	else // compute refracted direction
	{
		wil = refracted_direction( wol, m, cos_wom, cos_theta_t, eta );
		if (wil.z * wol.z > 0) return make_float3( 0 );
		evaluate_refraction( rcp_eta, shadingData.color, adjoint, wol, wil, m, alpha_x, alpha_y, 1 - F, retVal );
		pdf = 1 - F, jacobian = refraction_jacobian( wol, wil, m, alpha_x, alpha_y, rcp_eta );
	}
	pdf *= jacobian * GGXMDF_pdf( wol, m, alpha_x, alpha_y );
	if (pdf > 1.0e-6f) wiw = Tangent2World( wil, iN, T, B );
	return retVal * beer;
}

LH2_DEVFUNC float3 EvaluateBSDF_frosted( const ShadingData shadingData, const float3 iN, const float3 iT,
	const float3 wow, const float3 wiw, REFERENCE_OF( float ) pdf )
{
	const float3 B = normalize( cross( iN, iT ) );
	const float3 T = normalize( cross( iN, B ) );
	const float3 wol = World2Tangent( wow, iN, T, B );
	const float3 wil = World2Tangent( wiw, iN, T, B );
	const float eta = wol.z > 0 ? ETA : (1.0f / ETA);
	if (eta == 1) { pdf = 0; return make_float3( 0 ); }
	float alpha_x, alpha_y, jacobian;
	microfacet_alpha_from_roughness( ROUGHNESS, ANISOTROPIC, alpha_x, alpha_y );
	float3 retVal, m;
	if (wil.z * wol.z >= 0) // reflection
	{
		m = half_reflection_vector( wol, wil );
		const float cos_wom = dot( wol, m );
		const float F = fresnel_reflectance( cos_wom, 1 / eta );
		evaluate_reflection( shadingData.color, wol, wil, m, alpha_x, alpha_y, F, retVal );
		const float r_probability = choose_reflection_probability( 1, 1, F );
		pdf = r_probability, jacobian = reflection_jacobian( wol, m, cos_wom, alpha_x, alpha_y );
	}
	else // refraction
	{
		m = half_refraction_vector( wol, wil, eta );
		const float cos_wom = dot( wol, m );
		const float F = fresnel_reflectance( cos_wom, 1 / eta );
		evaluate_refraction( eta, shadingData.color, false /* adjoint */, wol, wil, m, alpha_x, alpha_y, 1 - F, retVal );
		const float r_probability = choose_reflection_probability( 1, 1, F );
		pdf = 1 - r_probability, jacobian = refraction_jacobian( wol, wil, m, alpha_x, alpha_y, eta );
	}
	pdf *= jacobian * GGXMDF_pdf( wol, m, alpha_x, alpha_y );
	return retVal;
}