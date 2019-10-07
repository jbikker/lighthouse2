/* core_mesh.h - Copyright 2019 Utrecht University

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

#include <array>
#include <vulkan/vulkan.hpp>

namespace lh2core
{

class RenderCore;
class BottomLevelAS;
class CoreMesh
{
  public:
	CoreMesh( VulkanDevice device );
	~CoreMesh();

	void Cleanup();
	void SetGeometry( const float4 *vertexData, const int vertexCount, const int triCount, const CoreTri *tris, const uint *alphaFlags = 0 );

	VulkanCoreBuffer<CoreTri> *triangles = nullptr;
	BottomLevelAS *accelerationStructure = nullptr;

  private:
	VulkanDevice m_Device;
};

} // namespace lh2core