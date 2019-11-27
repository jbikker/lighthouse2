/* uniform_objects.cpp - Copyright 2019 Utrecht University

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

#include "core_settings.h"

lh2core::VulkanCamera::VulkanCamera( const ViewPyramid &view, int samplesTaken, int renderPhase )
{
	pos_lensSize = make_float4( view.pos.x, view.pos.y, view.pos.z, view.aperture );
	const float3 r = view.p2 - view.p1;
	const float3 u = view.p3 - view.p1;

	right_aperture = make_float4( r.x, r.y, r.z, view.aperture );
	up_spreadAngle = make_float4( u.x, u.y, u.z, view.spreadAngle );
	p1 = make_float4( view.p1.x, view.p1.y, view.p1.z, 1.0f );

	pass = samplesTaken;
	phase = renderPhase;
}

lh2core::VulkanFinalizeParams::VulkanFinalizeParams( const int w, const int h, int samplespp )
{
	this->scrwidth = w;
	this->scrheight = h;

	this->spp = samplespp;
	this->pixelValueScale = 1.0f / float( this->spp );
}