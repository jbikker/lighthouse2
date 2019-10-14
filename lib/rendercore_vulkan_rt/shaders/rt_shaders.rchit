/* rt_shaders.rchit - Copyright 2019 Utrecht University

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

layout( location = 0 ) rayPayloadInNV vec4 hitData;

hitAttributeNV vec2 attribs;

void main()
{
	const uint bary = uint(65535.0f * attribs.x) + (uint(65535.0f * attribs.y) << 16);
	hitData = vec4(uintBitsToFloat(bary), intBitsToFloat(int(gl_InstanceCustomIndexNV)), intBitsToFloat(int(gl_PrimitiveID)), gl_RayTmaxNV);
}