/* rt_shaders.rmiss - Copyright 2019 Utrecht University

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

#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_NV_ray_tracing : require

#include "../bindings.h"
#include "tools.glsl"

layout( location = 0 ) rayPayloadInNV vec4 hitData;

void main()
{
	hitData = vec4(0.0f, 0.0f, intBitsToFloat(NOHIT), -1.0f);
}