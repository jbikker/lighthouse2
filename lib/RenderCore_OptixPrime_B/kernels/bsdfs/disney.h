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

struct MatData
{
	float3 color;
	float3 absorption;
	float eta;
	float metallic;
	float subsurface;
	float specular;
	float roughness;
	float specularTint;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	float transmission;
};

__device__ static float3 _Local2World( const float3 V, const float3 T, const float3 B, const float3 N )
{
	return normalize( V.x * T + V.y * B + V.z * N );
}

__device__ static float3 _World2Local( const float3 V, const float3 T, const float3 B, const float3 N )
{
	return normalize( make_float3( dot( V, T ), dot( V, B ), dot( V, N ) ) );
}

__device__ static bool Refract( const float3 &wi, const float3 &n, const float eta, float3& wt )
{
	float cosThetaI = dot( n, wi );
	float sin2ThetaI = max( 0.0f, 1.0f - cosThetaI * cosThetaI );
	float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1) return false; // TIR
	float cosThetaT = sqrtf( 1.0f - sin2ThetaT );
	wt = eta * (wi * -1.0f) + (eta * cosThetaI - cosThetaT) * float3( n );
	return true;
}

__device__ static float SchlickFresnel( const float u )
{
	const float m = clamp( 1 - u, 0.0f, 1.0f );
	return float (m * m) * (m * m) * m;
}

__device__ static float GTR1( const float NDotH, const float a )
{
	if (a >= 1) return INVPI;
	const float a2 = a * a;
	const float t = 1 + (a2 - 1) * NDotH * NDotH;
	return (a2 - 1) / (PI * logf( a2 ) * t);
}

__device__ static float GTR2( const float NDotH, const float a )
{
	const float a2 = a * a;
	const float t = 1.0f + (a2 - 1.0f)*NDotH*NDotH;
	return a2 / (PI * t*t);
}

__device__ static float SmithGGX( const float NDotv, const float alphaG )
{
	const float a = alphaG * alphaG;
	const float b = NDotv * NDotv;
	return 1 / (NDotv + sqrtf( a + b - a * b ));
}

__device__ static float Fr( float VDotN, float etaI, float etaT )
{
	const float SinThetaT2 = sqr( etaI / etaT ) * (1.0f - VDotN * VDotN);
	if (SinThetaT2 > 1.0f) return 1.0f; // TIR
	const float LDotN = sqrtf( 1.0f - SinThetaT2 );
	// todo: reformulate to remove this division
	const float eta = etaT / etaI;
	const float r1 = (VDotN - eta * LDotN) / (VDotN + eta * LDotN);
	const float r2 = (LDotN - eta * VDotN) / (LDotN + eta * VDotN);
	return 0.5f * (sqr( r1 ) + sqr( r2 ));
}

__device__ static float3 SafeNormalize( const float3& a ) 
{
	float ls = dot( a, a );
	if (ls > 0.0f) return a * (1.0f / sqrtf( ls )); else return make_float3( 0 );
}

__device__ static float BSDFPdf( const MatData& mat, float etaI, float etaO, const float3& N, const float3& wo, const float3& wi )
{
	if (dot( wi, N ) <= 0.0f)
	{
		const float bsdfPdf = 0.0f;
		const float brdfPdf = INV2PI * mat.subsurface * 0.5f;
		return lerp( brdfPdf, bsdfPdf, mat.transmission );
	}
	else
	{
		float F = Fr( dot( N, wo ), etaI, etaO );
		const float a = max( 0.001f, mat.roughness );
		const float3 half = SafeNormalize( wi + wo );
		const float cosThetaHalf = abs( dot( half, N ) );
		const float pdfHalf = GTR2( cosThetaHalf, a ) * cosThetaHalf;
		// calculate pdf for each method given outgoing light vector
		float pdfSpec = 0.25f*pdfHalf / max( 1.e-6f, dot( wi, half ) );
		float pdfDiff = abs( dot( wi, N ) ) * INVPI * (1.0f - mat.subsurface);
		float bsdfPdf = pdfSpec * F;
		float brdfPdf = lerp( pdfDiff, pdfSpec, 0.5f );
		// weight pdfs equally
		return lerp( brdfPdf, bsdfPdf, mat.transmission );
	}
}

// evaluate the BSDF for a given pair of directions
__device__ static float3 BSDFEval( const MatData& mat, float etaI, float etaO, const float3& N, const float3& wo, const float3& wi )
{
	float NDotL = dot( N, wi );
	float NDotV = dot( N, wo );
	float3 H = normalize( wi + wo );
	float NDotH = dot( N, H );
	float LDotH = dot( wi, H );
	float3 Cdlin = mat.color;
	float Cdlum = .3f * Cdlin.x + .6f * Cdlin.y + .1f * Cdlin.z; // luminance approx.
	float3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : make_float3( 1.0f ); // normalize lum. to isolate hue+sat
	float3 Cspec0 = lerp( mat.specular * .08f * lerp( make_float3( 1.0f ), Ctint, mat.specularTint ), Cdlin, mat.metallic );
	float3 bsdf = make_float3( 0 );
	float3 brdf = make_float3( 0 );
	if (mat.transmission > 0.0f)
	{
		// evaluate BSDF
		if (NDotL <= 0)
		{
			// transmission Fresnel
			float F = Fr( NDotV, etaI, etaO );
			bsdf = make_float3( (1.0f - F) / abs( NDotL ) * (1.0f - mat.metallic) * mat.transmission );
		}
		else
		{
			// specular lobe
			float a = max( 0.001f, mat.roughness );
			float Ds = GTR2( NDotH, a );

			// Fresnel term with the microfacet normal
			float FH = Fr( LDotH, etaI, etaO );
			float3 Fs = lerp( Cspec0, make_float3( 1.0f ), FH );
			float roughg = a;
			float Gs = SmithGGX( NDotV, roughg ) * SmithGGX( NDotL, roughg );
			bsdf = (Gs * Ds) * Fs;
		}
	}
	if (mat.transmission < 1.0f)
	{
		// evaluate BRDF
		if (NDotL <= 0)
		{
			if (mat.subsurface > 0.0f)
			{
				// take sqrt to account for entry/exit of the ray through the medium
				// this ensures transmitted light corresponds to the diffuse model
				float3 s = make_float3( sqrtf( mat.color.x ), sqrtf( mat.color.y ), sqrtf( mat.color.z ) );
				float FL = SchlickFresnel( abs( NDotL ) ), FV = SchlickFresnel( NDotV );
				float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);
				brdf = INVPI * s*mat.subsurface * Fd * (1.0f - mat.metallic);
			}
		}
		else
		{
			// specular
			float a = max( 0.001f, mat.roughness );
			float Ds = GTR2( NDotH, a );

			// Fresnel term with the microfacet normal
			float FH = SchlickFresnel( LDotH );
			float3 Fs = lerp( Cspec0, make_float3( 1 ), FH );
			float roughg = a;
			float Gs = SmithGGX( NDotV, roughg ) * SmithGGX( NDotL, roughg );

			// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
			// and mix in diffuse retro-reflection based on roughness
			float FL = SchlickFresnel( NDotL ), FV = SchlickFresnel( NDotV );
			float Fd90 = 0.5 + 2.0f * LDotH*LDotH * mat.roughness;
			float Fd = lerp( 1.0f, Fd90, FL ) * lerp( 1.0f, Fd90, FV );

			// clearcoat (ior = 1.5 -> F0 = 0.04)
			float Dr = GTR1( NDotH, lerp( .1, .001, mat.clearcoatGloss ) );
			float Fc = lerp( .04f, 1.0f, FH );
			float Gr = SmithGGX( NDotL, .25 ) * SmithGGX( NDotV, .25 );

			brdf = INVPI * Fd * Cdlin * (1.0f - mat.metallic) * (1.0f - mat.subsurface) + Gs * Fs * Ds + mat.clearcoat * Gr * Fc * Dr;
		}
	}

	return lerp( brdf, bsdf, mat.transmission );
}

// generate an importance sampled BSDF direction
__device__ static void BSDFSample( const MatData& mat, float etaI, float etaO, const float3& T, const float3& B, const float3& N, const float3& wo, float3& wi, float& pdf, BSDFType& type, uint& seed )
{
	const float r1 = RandomFloat( seed ), r2 = RandomFloat( seed );
	if (RandomFloat( seed ) < mat.transmission)
	{
		// sample BSDF
		float F = Fr( dot( N, wo ), etaI, etaO );

		// sample reflectance or transmission based on Fresnel term
		if (RandomFloat( seed ) < F)
		{
			// sample specular
			const float a = max( 0.001f, mat.roughness );
			const float phiHalf = r1 * TWOPI;
			const float cosThetaHalf = sqrtf( (1.0f - r2) / (1.0f + (sqr( a ) - 1.0f) * r2) );
			const float sinThetaHalf = sqrtf( max( 0.0f, 1.0f - sqr( cosThetaHalf ) ) );
			const float sinPhiHalf = sinf( phiHalf );
			const float cosPhiHalf = cosf( phiHalf );
			float3 half = T * (sinThetaHalf * cosPhiHalf) + B * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;

			// ensure half angle in same hemisphere as wi
			if (dot( half, wo ) <= 0.0f) half *= -1.0f;

			type = eReflected;
			wi = 2.0f * dot( wo, half ) * half - wo;
		}
		else
		{
			// sample transmission
			float eta = etaI / etaO;

			if (Refract( wo, N, eta, wi ))
			{
				type = eSpecular, pdf = (1.0f - F) * mat.transmission;
				return;
			}
			else
			{
				pdf = 0.0f;
				return;
			}
		}
	}
	else
	{
		// sample brdf
		if (RandomFloat( seed ) < 0.5f)
		{
			// sample diffuse	
			if (RandomFloat( seed ) < mat.subsurface)
			{
				const float3 d = DiffuseReflectionUniform( r1, r2 );

				// negate z coordinate to sample inside the surface
				wi = T * d.x + B * d.y - N * d.z;
				type = eTransmitted;
			}
			else
			{
				const float3 d = DiffuseReflectionCosWeighted( r1, r2 );
				wi = T * d.x + B * d.y + N * d.z;
				type = eReflected;
			}
		}
		else
		{
			// sample specular
			const float a = max( 0.001f, mat.roughness );
			const float phiHalf = r1 * TWOPI;
			const float cosThetaHalf = sqrtf( (1.0f - r2) / (1.0f + (sqr( a ) - 1.0f) * r2) );
			const float sinThetaHalf = sqrtf( max( 0.0f, 1.0f - sqr( cosThetaHalf ) ) );
			const float sinPhiHalf = sinf( phiHalf );
			const float cosPhiHalf = cosf( phiHalf );
			float3 half = T * (sinThetaHalf * cosPhiHalf) + B * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
			if (dot( half, wo ) <= 0.0f) half *= -1.0f; // ensure half angle in same hemisphere as wi
			wi = 2.0f * dot( wo, half ) * half - wo;
			type = eReflected;
		}
	}
	pdf = BSDFPdf( mat, etaI, etaO, N, wo, wi );
}

// ----------------------------------------------------------------

__device__ static float3 EvaluateBSDF( const ShadingData& shadingData, const float3 iN, const float3 T,
	const float3 wo, const float3 wi, float& pdf )
{
	MatData material;
	const uint4 parameters = shadingData.parameters;
	const float scale = 1.0f / 256.0f;
	material.color = shadingData.color;
	material.absorption = shadingData.absorption;
	material.metallic = scale * (float)(parameters.x & 255);
	material.subsurface = scale * (float)((parameters.x >> 8) & 255);
	material.specular = scale * (float)((parameters.x >> 16) & 255);
	material.roughness = scale * (float)((parameters.x >> 24) & 255);
	material.specularTint = scale * (float)(parameters.y & 255);
	material.anisotropic = scale * (float)((parameters.y >> 8) & 255);
	material.sheen = scale * (float)((parameters.y >> 16) & 255);
	material.sheenTint = scale * (float)((parameters.y >> 24) & 255);
	material.clearcoat = scale * (float)(parameters.z & 255);
	material.clearcoatGloss = scale * (float)((parameters.z >> 8) & 255);
	material.transmission = scale * (float)((parameters.z >> 16) & 255);
	material.eta = 0.0f; // scale * (float)((parameters.z >> 16) & 255) * 2.0f;
	float3 bsdf = BSDFEval( material, 1, 1, iN, wo, wi );
	pdf = BSDFPdf( material, 1, 1, iN, wo, wi );
	return bsdf;
}

__device__ static float3 SampleBSDF( const ShadingData& shadingData, float3 iN, const float3 N, const float3 T, const float3 wo,
	uint& seed, float3& wi, float& pdf )
{
	MatData material;
	const uint4 parameters = shadingData.parameters;
	const float scale = 1.0f / 256.0f;
	material.color = shadingData.color;
	material.absorption = shadingData.absorption;
	material.metallic = scale * (float)(parameters.x & 255);
	material.subsurface = scale * (float)((parameters.x >> 8) & 255);
	material.specular = scale * (float)((parameters.x >> 16) & 255);
	material.roughness = scale * (float)((parameters.x >> 24) & 255);
	material.specularTint = scale * (float)(parameters.y & 255);
	material.anisotropic = scale * (float)((parameters.y >> 8) & 255);
	material.sheen = scale * (float)((parameters.y >> 16) & 255);
	material.sheenTint = scale * (float)((parameters.y >> 24) & 255);
	material.clearcoat = scale * (float)(parameters.z & 255);
	material.clearcoatGloss = scale * (float)((parameters.z >> 8) & 255);
	material.transmission = scale * (float)((parameters.z >> 16) & 255);
	material.eta = 0.0f; // scale * (float)((parameters.z >> 16) & 255) * 2.0f;
	BSDFType type;
	const float3 B = normalize( cross( T, iN ) );
	const float3 Tfinal = cross( B, iN );
	BSDFSample( material, 1, 1, Tfinal, B, iN, wo, wi, pdf, type, seed );
	pdf = BSDFPdf( material, 1, 1, iN, wo, wi );
	return BSDFEval( material, 1, 1, iN, wo, wi );
}