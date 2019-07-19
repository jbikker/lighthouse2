/*
# Copyright Disney Enterprises, Inc.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License
# and the following modification to it: Section 6 Trademarks.
# deleted and replaced with:
#
# 6. Trademarks. This License does not grant permission to use the
# trade names, trademarks, service marks, or product names of the
# Licensor and its affiliates, except as required for reproducing
# the content of the NOTICE file.
#
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Adapted to C++ by Miles Macklin 2016

*/

enum BSDFType
{
	eReflected,
	eTransmitted,
	eSpecular
};

__device__ static inline bool Refract( const float3 &wi, const float3 &n, const float eta, float3& wt )
{
	float cosThetaI = dot( n, wi );
	float sin2ThetaI = max( 0.0f, 1.0f - cosThetaI * cosThetaI );
	float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1) return false; // TIR
	float cosThetaT = sqrtf( 1.0f - sin2ThetaT );
	wt = eta * (wi * -1.0f) + (eta * cosThetaI - cosThetaT) * float3( n );
	return true;
}

__device__ static inline float SchlickFresnel( const float u )
{
	const float m = clamp( 1 - u, 0.0f, 1.0f );
	return float( m * m ) * (m * m) * m;
}

__device__ static inline float GTR1( const float NDotH, const float a )
{
	if (a >= 1) return INVPI;
	const float a2 = a * a;
	const float t = 1 + (a2 - 1) * NDotH * NDotH;
	return (a2 - 1) / (PI * logf( a2 ) * t);
}

__device__ static inline float GTR2( const float NDotH, const float a )
{
	const float a2 = a * a;
	const float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
	return a2 / (PI * t * t);
}

__device__ static inline float SmithGGX( const float NDotv, const float alphaG )
{
	const float a = alphaG * alphaG;
	const float b = NDotv * NDotv;
	return 1 / (NDotv + sqrtf( a + b - a * b ));
}

__device__ static float Fr( const float VDotN, const float eio )
{
	const float SinThetaT2 = sqr( eio ) * (1.0f - VDotN * VDotN);
	if (SinThetaT2 > 1.0f) return 1.0f; // TIR
	const float LDotN = sqrtf( 1.0f - SinThetaT2 );
	// todo: reformulate to remove this division
	const float eta = 1.0f / eio;
	const float r1 = (VDotN - eta * LDotN) / (VDotN + eta * LDotN);
	const float r2 = (LDotN - eta * VDotN) / (LDotN + eta * VDotN);
	return 0.5f * (sqr( r1 ) + sqr( r2 ));
}

__device__ static inline float3 SafeNormalize( const float3& a )
{
	const float ls = dot( a, a );
	if (ls > 0.0f) return a * (1.0f / sqrtf( ls )); else return make_float3( 0 );
}

__device__ static float BSDFPdf( const ShadingData& shadingData, const float3& N, const float3& wo, const float3& wi )
{
	float bsdfPdf = 0.0f, brdfPdf;
	if (dot( wi, N ) <= 0.0f) brdfPdf = INV2PI * SUBSURFACE * 0.5f; else
	{
		const float F = Fr( dot( N, wo ), ETA );
		const float3 half = SafeNormalize( wi + wo );
		const float cosThetaHalf = abs( dot( half, N ) );
		const float pdfHalf = GTR2( cosThetaHalf, ROUGHNESS ) * cosThetaHalf;
		// calculate pdf for each method given outgoing light vector
		const float pdfSpec = 0.25f * pdfHalf / max( 1.e-6f, dot( wi, half ) );
		const float pdfDiff = abs( dot( wi, N ) ) * INVPI * (1.0f - SUBSURFACE);
		bsdfPdf = pdfSpec * F;
		brdfPdf = lerp( pdfDiff, pdfSpec, 0.5f );
	}
	return lerp( brdfPdf, bsdfPdf, TRANSMISSION );
}

// evaluate the BSDF for a given pair of directions
__device__ static float3 BSDFEval( const ShadingData& shadingData,
	const float3& N, const float3& wo, const float3& wi )
{
	const float NDotL = dot( N, wi );
	const float NDotV = dot( N, wo );
	const float3 H = normalize( wi + wo );
	const float NDotH = dot( N, H );
	const float LDotH = dot( wi, H );
	const float3 Cdlin = shadingData.color;
	const float Cdlum = .3f * Cdlin.x + .6f * Cdlin.y + .1f * Cdlin.z; // luminance approx.
	const float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3( 1.0f ); // normalize lum. to isolate hue+sat
	const float3 Cspec0 = lerp( SPECULAR * .08f * lerp( make_float3( 1.0f ), Ctint, SPECTINT ), Cdlin, METALLIC );
	float3 bsdf = make_float3( 0 );
	float3 brdf = make_float3( 0 );
	if (TRANSMISSION > 0.0f)
	{
		// evaluate BSDF
		if (NDotL <= 0)
		{
			// transmission Fresnel
			const float F = Fr( NDotV, ETA );
			bsdf = make_float3( (1.0f - F) / abs( NDotL ) * (1.0f - METALLIC) * TRANSMISSION );
		}
		else
		{
			// specular lobe
			const float a = ROUGHNESS;
			const float Ds = GTR2( NDotH, a );

			// Fresnel term with the microfacet normal
			const float FH = Fr( LDotH, ETA );
			const float3 Fs = lerp( Cspec0, make_float3( 1.0f ), FH );
			const float Gs = SmithGGX( NDotV, a ) * SmithGGX( NDotL, a );
			bsdf = (Gs * Ds) * Fs;
		}
	}
	if (TRANSMISSION < 1.0f)
	{
		// evaluate BRDF
		if (NDotL <= 0)
		{
			if (SUBSURFACE > 0.0f)
			{
				// take sqrt to account for entry/exit of the ray through the medium
				// this ensures transmitted light corresponds to the diffuse model
				const float3 s = make_float3( sqrtf( shadingData.color.x ), sqrtf( shadingData.color.y ), sqrtf( shadingData.color.z ) );
				const float FL = SchlickFresnel( abs( NDotL ) ), FV = SchlickFresnel( NDotV );
				const float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);
				brdf = INVPI * s * SUBSURFACE * Fd * (1.0f - METALLIC);
			}
		}
		else
		{
			// specular
			const float a = ROUGHNESS;
			const float Ds = GTR2( NDotH, a );

			// Fresnel term with the microfacet normal
			const float FH = SchlickFresnel( LDotH );
			const float3 Fs = lerp( Cspec0, make_float3( 1 ), FH );
			const float Gs = SmithGGX( NDotV, a ) * SmithGGX( NDotL, a );

			// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
			// and mix in diffuse retro-reflection based on roughness
			const float FL = SchlickFresnel( NDotL ), FV = SchlickFresnel( NDotV );
			const float Fd90 = 0.5 + 2.0f * LDotH * LDotH * a;
			const float Fd = lerp( 1.0f, Fd90, FL ) * lerp( 1.0f, Fd90, FV );

			// clearcoat (ior = 1.5 -> F0 = 0.04)
			const float Dr = GTR1( NDotH, lerp( .1, .001, CLEARCOATGLOSS ) );
			const float Fc = lerp( .04f, 1.0f, FH );
			const float Gr = SmithGGX( NDotL, .25 ) * SmithGGX( NDotV, .25 );

			brdf = INVPI * Fd * Cdlin * (1.0f - METALLIC) * (1.0f - SUBSURFACE) + Gs * Fs * Ds + CLEARCOAT * Gr * Fc * Dr;
		}
	}

	return lerp( brdf, bsdf, TRANSMISSION );
}

// generate an importance sampled BSDF direction
__device__ static void BSDFSample( const ShadingData& shadingData,
	const float3& T, const float3& B, const float3& N, const float3& wo, float3& wi, float& pdf, BSDFType& type, const float r3, const float r4 )
{
	if (r3 < TRANSMISSION)
	{
		// sample BSDF
		float F = Fr( dot( N, wo ), ETA );
		if (r4 < F) // sample reflectance or transmission based on Fresnel term
		{
			// sample reflection
			const float r1 = r3 / TRANSMISSION;
			const float r2 = r4 / F;
			const float cosThetaHalf = sqrtf( (1.0f - r2) / (1.0f + (sqr( ROUGHNESS ) - 1.0f) * r2) );
			const float sinThetaHalf = sqrtf( max( 0.0f, 1.0f - sqr( cosThetaHalf ) ) );
			float sinPhiHalf, cosPhiHalf;
			__sincosf( r1 * TWOPI, &sinPhiHalf, &cosPhiHalf );
			float3 half = T * (sinThetaHalf * cosPhiHalf) + B * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
			if (dot( half, wo ) <= 0.0f) half *= -1.0f; // ensure half angle in same hemisphere as wo
			type = eReflected;
			wi = reflect( wo * -1.0f, half );
		}
		else // sample transmission
		{
			pdf = 0;
			if (Refract( wo, N, ETA, wi )) type = eSpecular, pdf = (1.0f - F) * TRANSMISSION;
			return;
		}
	}
	else // sample BRDF
	{
		const float r1 = (r3 - TRANSMISSION) / (1 - TRANSMISSION);
		if (r4 < 0.5f)
		{
			// sample diffuse	
			const float r2 = r4 * 2;
			float3 d;
			if (r2 < SUBSURFACE)
			{
				const float r5 = r2 / SUBSURFACE;
				d = DiffuseReflectionUniform( r1, r5 ), type = eTransmitted, d.z *= -1.0f;
			}
			else
			{
				const float r5 = (r2 - SUBSURFACE) / (1 - SUBSURFACE);
				d = DiffuseReflectionCosWeighted( r1, r5 ), type = eReflected;
			}
			wi = T * d.x + B * d.y + N * d.z;
		}
		else
		{
			// sample specular
			const float r2 = (r4 - 0.5f) * 2.0f;
			const float cosThetaHalf = sqrtf( (1.0f - r2) / (1.0f + (sqr( ROUGHNESS ) - 1.0f) * r2) );
			const float sinThetaHalf = sqrtf( max( 0.0f, 1.0f - sqr( cosThetaHalf ) ) );
			float sinPhiHalf, cosPhiHalf;
			__sincosf( r1 * TWOPI, &sinPhiHalf, &cosPhiHalf );
			float3 half = T * (sinThetaHalf * cosPhiHalf) + B * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
			if (dot( half, wo ) <= 0.0f) half *= -1.0f; // ensure half angle in same hemisphere as wi
			wi = reflect( wo * -1.0f, half );
			type = eReflected;
		}
	}
	pdf = BSDFPdf( shadingData, N, wo, wi );
}

// ----------------------------------------------------------------

__device__ static float3 EvaluateBSDF( const ShadingData& shadingData, const float3& iN, const float3& T,
	const float3 wo, const float3 wi, float& pdf )
{
	const float3 bsdf = BSDFEval( shadingData, iN, wo, wi );
	pdf = BSDFPdf( shadingData, iN, wo, wi );
	return bsdf;
}

__device__ static float3 SampleBSDF( const ShadingData& shadingData, 
	const float3& iN, const float3& N, const float3& T, const float3& wo,
	const float r3, const float r4, float3& wi, float& pdf )
{
	BSDFType type;
	const float3 B = normalize( cross( T, iN ) );
	const float3 Tfinal = cross( B, iN );
	BSDFSample( shadingData, Tfinal, B, iN, wo, wi, pdf, type, r3, r4 );
	return BSDFEval( shadingData, iN, wo, wi );
}

// EOF