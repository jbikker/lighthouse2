/* lambert.h - Copyright 2019 Utrecht University

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

#include "compatibility.h"

// Lambert BSDF
// ----------------------------------------------------------------

LH2_DEVFUNC float Fr_L( float VDotN, float eio )
{
	if (VDotN < 0.0f)
	{
		eio = 1.0f / eio;
		VDotN = fabs( VDotN );
	}
	const float SinThetaT2 = sqr( eio ) * (1.0f - VDotN * VDotN);
	if (SinThetaT2 > 1.0f) return 1.0f; // TIR
	const float LDotN = sqrtf( 1.0f - SinThetaT2 );
	const float r1 = (VDotN - eio * LDotN) / (VDotN + eio * LDotN);
	const float r2 = (LDotN - eio * VDotN) / (LDotN + eio * VDotN);
	return 0.5f * (sqr( r1 ) + sqr( r2 ));
}

LH2_DEVFUNC bool Refract_L( const float3& wi, const float3& n, const float eta, float3& wt )
{
	const float cosThetaI = fabs( dot( n, wi ) );
	const float sin2ThetaI = max( 0.0f, 1.0f - cosThetaI * cosThetaI );
	const float sin2ThetaT = eta * eta * sin2ThetaI;
	if (sin2ThetaT >= 1) return false; // TIR
	const float cosThetaT = sqrtf( 1.0f - sin2ThetaT );
	wt = eta * (wi * -1.0f) + (eta * cosThetaI - cosThetaT) * float3( n );
	return true;
}

LH2_DEVFUNC float3 EvaluateBSDF( const ShadingData shadingData, const float3 iN, const float3 T,
	const float3 wo, const float3 wi, REFERENCE_OF( float ) pdf )
{
	pdf = fabs( dot( wi, iN ) ) * INVPI;
	return shadingData.color * INVPI;
}

LH2_DEVFUNC float3 SampleBSDF( const ShadingData shadingData, float3 iN, const float3 N, const float3 T, const float3 wo, const float distance,
	const float r3, const float r4, REFERENCE_OF( float3 ) wi, REFERENCE_OF( float ) pdf, REFERENCE_OF( bool ) specular, bool adjoint = false )
{
	specular = true, pdf = 1; // default
	float3 bsdf;
	if (r4 < TRANSMISSION)
	{
		// specular
		const float eio = 1 / ETA, F = Fr_L( dot( iN, wo ), eio );
		if (r3 < F)
		{
			wi = reflect( wo * -1.0f, iN );
			bsdf = shadingData.color * (1 / abs( dot( iN, wi ) ));
		}
		else
		{
			if (!Refract_L( wo, iN, eio, wi )) return make_float3( 0 );
			float ajointCorrection = 1.0f;
			if (adjoint) ajointCorrection = (eio * eio);
			float3 beer = make_float3( 1 );
			// beer.x = expf( -shadingData.transmittance.x * distance * 2.0f );
			// beer.y = expf( -shadingData.transmittance.y * distance * 2.0f );
			// beer.z = expf( -shadingData.transmittance.z * distance * 2.0f );
			return shadingData.color * beer * ajointCorrection * (1 / abs( dot( iN, wi ) ));
		}
	}
	else
	{
		// specular and diffuse
		float pReflect = 1 - ROUGHNESS; /* e.g. at ROUGHNESS = 0.1 we have a 90% chance of getting a specular reflection */
		if (r3 < pReflect)
		{
			// pure specular
			wi = reflect( wo * -1.0f, iN );
			bsdf = shadingData.color * (1.0f / abs( dot( iN, wi ) ));
		}
		else
		{
			const float r5 = (r3 - pReflect) / (1 - pReflect);			// renormalize and reuse
			const float r6 = (r4 - TRANSMISSION) / (1 - TRANSMISSION);	// renormalize and reuse
			wi = normalize( Tangent2World( DiffuseReflectionCosWeighted( r5, r6 ), iN ) );
			pdf = max( 0.0f, dot( wi, iN ) ) * INVPI;
			specular = false;
			bsdf = shadingData.color * INVPI;
		}
	}
	APPLYSAFENORMALS;
	return bsdf;
}

// EOF