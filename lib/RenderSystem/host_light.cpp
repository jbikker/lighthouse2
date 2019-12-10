/* host_light.cpp - Copyright 2019 Utrecht University

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

#include "rendersystem.h"

//  +-----------------------------------------------------------------------------+
//  |  HostAreaLight::HostAreaLight                                               |
//  |  Constructor. An area light is just a regular triangle in LH2, but we do    |
//  |  store some additional data:                                                |
//  |  - For efficient sampling, we store the vertices, normal and radiace;       |
//  |  - For MIS, we store the original triangle (idx and instance idx).    LH2'19|
//  +-----------------------------------------------------------------------------+
HostAreaLight::HostAreaLight( HostTri* origTri, int origIdx, int origInstance )
{
	triIdx = origIdx;
	instIdx = origInstance;
	vertex0 = origTri->vertex0;
	vertex1 = origTri->vertex1;
	vertex2 = origTri->vertex2;
	centre = 0.333333f * (vertex0 + vertex1 + vertex2);
	N = make_float3( origTri->Nx, origTri->Ny, origTri->Nz );
	const float a = length( vertex1 - vertex0 );
	const float b = length( vertex2 - vertex1 );
	const float c = length( vertex0 - vertex2 );
	const float s = (a + b + c) * 0.5f;
	area = sqrtf( s * (s - a) * (s - b) * (s - c) ); // Heron's formula
	radiance = HostScene::materials[origTri->material]->color();
	const float3 E = radiance * area;
	energy = E.x + E.y + E.z;
}

//  +-----------------------------------------------------------------------------+
//  |  HostAreaLight::ConvertToCoreLightTri                                       |
//  |  Prepare an area light for the core.                                  LH2'19|
//  +-----------------------------------------------------------------------------+
CoreLightTri HostAreaLight::ConvertToCoreLightTri()
{
	CoreLightTri light;
	light.triIdx = triIdx;
	light.instIdx = instIdx;
	light.vertex0 = vertex0;
	light.vertex1 = vertex1;
	light.vertex2 = vertex2;
	light.centre = centre;
	light.radiance = radiance;
	light.energy = energy;
	light.N = N;
	light.area = area;
	return light;
}

//  +-----------------------------------------------------------------------------+
//  |  HostPointLight::ConvertToCorePointLight                                    |
//  |  Prepare a point light for the core.                                  LH2'19|
//  +-----------------------------------------------------------------------------+
CorePointLight HostPointLight::ConvertToCorePointLight()
{
	CorePointLight light;
	light.position = position;
	light.radiance = radiance;
	light.energy = energy;
	return light;
}

//  +-----------------------------------------------------------------------------+
//  |  HostSpotLight::ConvertToCoreSpotLight                                      |
//  |  Prepare a spot light for the core.                                   LH2'19|
//  +-----------------------------------------------------------------------------+
CoreSpotLight HostSpotLight::ConvertToCoreSpotLight()
{
	CoreSpotLight light;
	light.position = position;
	light.radiance = radiance;
	light.direction = direction;
	light.cosInner = cosInner;
	light.cosOuter = cosOuter;
	return light;
}

//  +-----------------------------------------------------------------------------+
//  |  HostDirectionalLight::ConvertToCoreDirectionalLight                        |
//  |  Prepare a directional light for the core.                            LH2'19|
//  +-----------------------------------------------------------------------------+
CoreDirectionalLight HostDirectionalLight::ConvertToCoreDirectionalLight()
{
	CoreDirectionalLight light;
	light.radiance = radiance;
	light.direction = direction;
	return light;
}

// EOF