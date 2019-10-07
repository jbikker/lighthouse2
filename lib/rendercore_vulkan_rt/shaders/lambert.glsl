/* lambert.glsl - Copyright 2019 Utrecht University

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

#ifndef LAMBERT_H
#define LAMBERT_H

#include "tools.glsl"
#include "structures.glsl"

vec3 EvaluateBSDF(
	const ShadingData shadingData, const vec3 iN, const vec3 T,
	const vec3 wo, const vec3 wi, inout float pdf)
{
	pdf = abs(dot(wi, iN)) * INVPI;
	return shadingData.color_flags.xyz * INVPI;
}

vec3 SampleBSDF(
	const ShadingData shadingData, const vec3 iN, const vec3 N, const vec3 T,
	const vec3 wo, const float r0, const float r1, inout vec3 wi, inout float pdf)
{
	if (abs(ROUGHNESS) < 0.1f)
	{
		wi = -reflect(wo, iN);
		pdf = 1.0f;
		if (dot( N, wi ) <= 0.0f) pdf = 0.0f;
		return shadingData.color_flags.xyz * (1.0f / (abs(dot(iN, wi))));
	}

	wi = normalize(Tangent2World(DiffuseReflectionUniform(r0, r1), iN));
	pdf = max(0.0f, dot(wi, iN)) * INVPI;
	if (dot( N, wi ) <= 0) pdf = 0;
	return shadingData.color_flags.xyz * INVPI;
}

#endif