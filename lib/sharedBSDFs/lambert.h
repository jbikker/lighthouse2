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

// for debugging: Lambert brdf
LH2_DEVFUNC float3 EvaluateBSDF( const ShadingData shadingData, const float3 iN, const float3 T,
	const float3 wo, const float3 wi, REFERENCE_OF(float) pdf )
{
	pdf = fabs( dot( wi, iN ) ) * INVPI;
	return shadingData.color * INVPI;
}

LH2_DEVFUNC float3 SampleBSDF( const ShadingData shadingData, float3 iN, const float3 N, const float3 T, const float3 wo,
	const float r3, const float r4, REFERENCE_OF(float3) wi, REFERENCE_OF(float) pdf )
{
	// specular and diffuse
	if (fabs( ROUGHNESS ) < 0.1f)
	{
		// pure specular
		wi = -reflect( wo, iN );
		pdf = 1;
		APPLYSAFENORMALS;
		return shadingData.color * (1.0f / abs( dot( iN, wi ) ));
	}
	else
	{
		wi = normalize( Tangent2World( DiffuseReflectionCosWeighted( r3, r4 ), iN ) );
		pdf = max( 0.0f, dot( wi, iN ) ) * INVPI;
		APPLYSAFENORMALS;
		return shadingData.color * INVPI;
	}
}

// EOF