/* host_light.h - Copyright 2019 Utrecht University

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

#pragma once

namespace lighthouse2 {

//  +-----------------------------------------------------------------------------+
//  |  HostAreaLight                                                              |
//  |  Host-side light tri.                                                 LH2'19|
//  +-----------------------------------------------------------------------------+
class HostAreaLight
{
public:
	// constructor / destructor
	HostAreaLight() = default;
	HostAreaLight( HostTri* origTri, int origIdx, int origInstance );
	// methods
	CoreLightTri ConvertToCoreLightTri();
	// data members
	int triIdx = 0;								// the index of the triangle this ltri is based on
	int instIdx = 0;							// the instance to which this triangle belongs
	float3 vertex0 = make_float3( 0 );
	float3 vertex1 = make_float3( 0 );
	float3 vertex2 = make_float3( 0 );
	float3 centre = make_float3( 0 );
	float3 radiance = make_float3( 0 );
	float3 N = make_float3( 0, -1, 0 );
	float area = 0;
	float energy = 0;
	bool enabled = true;
	TRACKCHANGES;
};

//  +-----------------------------------------------------------------------------+
//  |  HostPointLight                                                             |
//  |  Host-side point light definition.                                    LH2'19|
//  +-----------------------------------------------------------------------------+
class HostPointLight
{
public:
	// constructor / destructor
	HostPointLight() = default;
	// methods
	CorePointLight ConvertToCorePointLight();
	// data members
	float3 position = make_float3( 0 );
	float energy = 0;
	float3 radiance = make_float3( 0 );
	int ID = 0;
	bool enabled = true;
	TRACKCHANGES;
};

//  +-----------------------------------------------------------------------------+
//  |  HostSpotLight                                                              |
//  |  Host-side spot light definition.                                     LH2'19|
//  +-----------------------------------------------------------------------------+
class HostSpotLight
{
public:
	// constructor / destructor
	HostSpotLight() = default;
	// methods
	CoreSpotLight ConvertToCoreSpotLight();
	// data members
	float3 position = make_float3( 0 );
	float cosInner = 0;
	float3 radiance = make_float3( 0 );
	float cosOuter = 0;
	float3 direction = make_float3( 0, -1, 0 );
	int ID = 0;
	bool enabled = true;
	TRACKCHANGES;
};

//  +-----------------------------------------------------------------------------+
//  |  HostDirectionalLight                                                       |
//  |  Host-side directional light definition.                              LH2'19|
//  +-----------------------------------------------------------------------------+
class HostDirectionalLight
{
public:
	// constructor / destructor
	HostDirectionalLight() = default;
	// methods
	CoreDirectionalLight ConvertToCoreDirectionalLight();
	// data members
	float3 direction = make_float3( 0, -1, 0 );
	float energy = 0;
	float3 radiance = make_float3( 0 );
	int ID = 0;
	bool enabled = true;
	TRACKCHANGES;
};

} // namespace lighthouse2

// EOF