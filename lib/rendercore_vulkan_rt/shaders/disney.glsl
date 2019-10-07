/* disney.glsl - Copyright 2019 Utrecht University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

	   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef DISNEY_H
#define DISNEY_H

#include "structures.glsl"

#define BSDF_TYPE_REFLECTED 0
#define BSDF_TYPE_TRANSMITTED 1
#define BSDF_TYPE_SPECULAR 2


float lerp(float a, float b, float t)
{
  return a + t*(b-a);
}
vec2 lerp(vec2 a, vec2 b, float t)
{
  return a + t*(b-a);
}
vec3 lerp(vec3 a, vec3 b, float t)
{
  return a + t*(b-a);
}
vec4 lerp(vec4 a, vec4 b, float t)
{
  return a + t*(b-a);
}
float sqr( const float x ) { return x * x; }

bool Refract( const vec3 wi, const vec3 n, const float eta, inout vec3 wt )
{
	const float cosThetaI = dot( n, wi );
	const float sin2ThetaI = max( 0.0f, 1.0f - cosThetaI * cosThetaI );
	const float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1) return false; // TIR
	float cosThetaT = sqrt( 1.0f - sin2ThetaT );
	wt = eta * (wi * -1.0f) + (eta * cosThetaI - cosThetaT) * vec3( n );
	return true;
}

float SchlickFresnel( const float u )
{
	const float m = clamp( 1 - u, 0.0f, 1.0f );
	return float( m * m ) * (m * m) * m;
}

float GTR1( const float NDotH, const float a )
{
	if (a >= 1) return INVPI;
	const float a2 = a * a;
	const float t = 1 + (a2 - 1) * NDotH * NDotH;
	return (a2 - 1) / (PI * log( a2 ) * t);
}

float GTR2( const float NDotH, const float a )
{
	const float a2 = a * a;
	const float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
	return a2 / (PI * t * t);
}

float SmithGGX( const float NDotv, const float alphaG )
{
	const float a = alphaG * alphaG;
	const float b = NDotv * NDotv;
	return 1 / (NDotv + sqrt( a + b - a * b ));
}

float Fr( const float VDotN, const float eio )
{
	const float SinThetaT2 = sqr( eio ) * (1.0f - VDotN * VDotN);
	if (SinThetaT2 > 1.0f) return 1.0f; // TIR
	const float LDotN = sqrt( 1.0f - SinThetaT2 );
	// todo: reformulate to remove this division
	const float eta = 1.0f / eio;
	const float r1 = (VDotN - eta * LDotN) / (VDotN + eta * LDotN);
	const float r2 = (LDotN - eta * VDotN) / (LDotN + eta * VDotN);
	return 0.5f * (sqr( r1 ) + sqr( r2 ));
}

vec3 SafeNormalize( const vec3 a )
{
	const float ls = dot( a, a );
	if (ls > 0.0f) return a * (1.0f / sqrt( ls )); else return vec3( 0 );
}

float BSDFPdf( const ShadingData shadingData, const vec3 N, const vec3 wo, const vec3 wi )
{
	float bsdfPdf = 0.0f, brdfPdf;
	if (dot( wi, N ) <= 0.0f) brdfPdf = INV2PI * SUBSURFACE * 0.5f; else
	{
		const float F = Fr( dot( N, wo ), ETA );
		const vec3 halfway = SafeNormalize( wi + wo );
		const float cosThetaHalf = abs( dot( halfway, N ) );
		const float pdfHalf = GTR2( cosThetaHalf, ROUGHNESS ) * cosThetaHalf;
		// calculate pdf for each method given outgoing light vector
		const float pdfSpec = 0.25f * pdfHalf / max( 1.e-6f, dot( wi, halfway ) );
		const float pdfDiff = abs( dot( wi, N ) ) * INVPI * (1.0f - SUBSURFACE);
		bsdfPdf = pdfSpec * F;
		brdfPdf = lerp( pdfDiff, pdfSpec, 0.5f );
	}
	return lerp( brdfPdf, bsdfPdf, TRANSMISSION );
}

// evaluate the BSDF for a given pair of directions
vec3 BSDFEval( const ShadingData shadingData, const vec3 N, const vec3 wo, const vec3 wi )
{
	const float NDotL = dot( N, wi );
	const float NDotV = dot( N, wo );
	const vec3 H = normalize( wi + wo );
	const float NDotH = dot( N, H );
	const float LDotH = dot( wi, H );
	const vec3 Cdlin = shadingData.color_flags.xyz;
	const float Cdlum = .3f * Cdlin.x + .6f * Cdlin.y + .1f * Cdlin.z; // luminance approx.
	const vec3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : vec3( 1.0f ); // normalize lum. to isolate hue+sat
	const vec3 Cspec0 = lerp( SPECULAR * .08f * lerp( vec3( 1.0f ), Ctint, SPECTINT ), Cdlin, METALLIC );
	vec3 bsdf = vec3( 0 );
	vec3 brdf = vec3( 0 );
	if (TRANSMISSION > 0.0f)
	{
		// evaluate BSDF
		if (NDotL <= 0)
		{
			// transmission Fresnel
			const float F = Fr( NDotV, ETA );
			bsdf = vec3( (1.0f - F) / abs( NDotL ) * (1.0f - METALLIC) * TRANSMISSION );
		}
		else
		{
			// specular lobe
			const float a = ROUGHNESS;
			const float Ds = GTR2( NDotH, a );

			// Fresnel term with the microfacet normal
			const float FH = Fr( LDotH, ETA );
			const vec3 Fs = lerp( Cspec0, vec3( 1.0f ), FH );
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
				const vec3 s = vec3( sqrt( shadingData.color_flags.x ), sqrt( shadingData.color_flags.y ), sqrt( shadingData.color_flags.z ) );
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
			const vec3 Fs = lerp( Cspec0, vec3( 1 ), FH );
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
void BSDFSample( const ShadingData shadingData,
	const vec3 T, const vec3 B, const vec3 N, const vec3 wo, inout vec3 wi, inout float pdf, inout int type, const float r3, const float r4 )
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
			const float cosThetaHalf = sqrt( (1.0f - r2) / (1.0f + (sqr( ROUGHNESS ) - 1.0f) * r2) );
			const float sinThetaHalf = sqrt( max( 0.0f, 1.0f - sqr( cosThetaHalf ) ) );
			const float sinPhiHalf = sin(r1 * TWOPI);
			const float cosPhiHalf = cos(r1 * TWOPI);
			vec3 halfway = T * (sinThetaHalf * cosPhiHalf) + B * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
			if (dot( halfway, wo ) <= 0.0f) halfway *= -1.0f; // ensure half angle in same hemisphere as wo
			type = BSDF_TYPE_REFLECTED;
			wi = reflect( wo * -1.0f, halfway );
		}
		else // sample transmission
		{
			pdf = 0;
			if (Refract( wo, N, ETA, wi )) type = BSDF_TYPE_SPECULAR, pdf = (1.0f - F) * TRANSMISSION;
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
			vec3 d;
			if (r2 < SUBSURFACE)
			{
				const float r5 = r2 / SUBSURFACE;
				d = DiffuseReflectionUniform( r1, r5 ), type = BSDF_TYPE_TRANSMITTED, d.z *= -1.0f;
			}
			else
			{
				const float r5 = (r2 - SUBSURFACE) / (1 - SUBSURFACE);
				d = DiffuseReflectionCosWeighted( r1, r5 ), type = BSDF_TYPE_REFLECTED;
			}
			wi = T * d.x + B * d.y + N * d.z;
		}
		else
		{
			// sample specular
			const float r2 = (r4 - 0.5f) * 2.0f;
			const float cosThetaHalf = sqrt( (1.0f - r2) / (1.0f + (sqr( ROUGHNESS ) - 1.0f) * r2) );
			const float sinThetaHalf = sqrt( max( 0.0f, 1.0f - sqr( cosThetaHalf ) ) );
			const float sinPhiHalf = sin(r1 * TWOPI);
			const float cosPhiHalf = cos(r1 * TWOPI);
			vec3 halfway = T * (sinThetaHalf * cosPhiHalf) + B * (sinThetaHalf * sinPhiHalf) + N * cosThetaHalf;
			if (dot( halfway, wo ) <= 0.0f) halfway *= -1.0f; // ensure half angle in same hemisphere as wi
			wi = reflect( wo * -1.0f, halfway );
			type = BSDF_TYPE_REFLECTED;
		}
	}
	pdf = BSDFPdf( shadingData, N, wo, wi );
}

// ----------------------------------------------------------------

vec3 EvaluateBSDF( const ShadingData shadingData, const vec3 iN, const vec3 T, const vec3 wo, const vec3 wi, inout float pdf )
{
	const vec3 bsdf = BSDFEval( shadingData, iN, wo, wi );
	pdf = BSDFPdf( shadingData, iN, wo, wi );
	return bsdf;
}

vec3 SampleBSDF( const ShadingData shadingData, const vec3 iN, const vec3 N, const vec3 T, const vec3 wo,
	const float r3, const float r4, inout vec3 wi, inout float pdf )
{
	int type;
	const vec3 B = normalize( cross( T, iN ) );
	const vec3 Tfinal = cross( B, iN );
	BSDFSample( shadingData, Tfinal, B, iN, wo, wi, pdf, type, r3, r4 );
	return BSDFEval( shadingData, iN, wo, wi );
}

#endif