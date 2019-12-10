/* host_skydome.h - Copyright 2019 Utrecht University

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

namespace lighthouse2
{

//  +-----------------------------------------------------------------------------+
//  |  HostSkyDome                                                                |
//  |  Stores data for a HDR sky dome.                                            |
//  |  Also implements calculation of the PDF/CDF for efficient sampling.   LH2'19|
//  +-----------------------------------------------------------------------------+
class HostSkyDome
{
public:
	// constructor / destructor
	HostSkyDome();
	~HostSkyDome();
	void Load();
	// public data members
	float3* pixels = nullptr;			// HDR texture data for sky dome
	int width = 0;						// width of the sky texture
	int height = 0;						// height of the sky texture
	float* cdf = nullptr;				// cdf for importance sampling
	float* pdf = nullptr;				// pdf for importance sampling
	float* columncdf = nullptr;			// column cdf for importance sampling
	mat4 worldToLight;					// for PBRT scenes; transform for skydome
	TRACKCHANGES;						// add Changed(), MarkAsDirty() methods, see system.h
};

} // namespace lighthouse2

// EOF