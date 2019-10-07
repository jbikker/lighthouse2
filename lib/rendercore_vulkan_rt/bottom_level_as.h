/* bottom_level_as.h - Copyright 2019 Utrecht University

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

namespace lh2core
{

class BottomLevelAS
{
  public:
	BottomLevelAS( VulkanDevice device, const float4 *vertices, uint32_t vertexCount, AccelerationStructureType type = FastestTrace );
	~BottomLevelAS();

	void Cleanup();
	void UpdateVertices( const float4 *vertices, uint32_t vertexCount );

	void Build();
	void Rebuild();

	uint64_t GetHandle();
	uint32_t GetVertexCount() const;

	bool CanUpdate() const { return uint( m_Flags & vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate ) > 0; }
	const vk::AccelerationStructureNV &GetAccelerationStructure() const { return m_Structure; }

  private:
	void Build( bool update );

	// Converts desired type to Vulkan build flags
	static vk::BuildAccelerationStructureFlagsNV TypeToFlags( AccelerationStructureType type )
	{
		switch ( type )
		{
		case ( FastestBuild ): return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastBuild;
		case ( FastRebuild ): return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastBuild | vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate;
		case ( FastestTrace ): return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace | vk::BuildAccelerationStructureFlagBitsNV::eAllowCompaction;
		case ( FastTrace ): return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace | vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate;
		default: return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace | vk::BuildAccelerationStructureFlagBitsNV::eAllowCompaction;
		}
	}

	VulkanDevice m_Device;
	vk::DeviceSize m_ResultSize, m_ScratchSize;
	AccelerationStructureType m_Type;
	vk::BuildAccelerationStructureFlagsNV m_Flags;
	vk::GeometryNV m_Geometry;
	vk::AccelerationStructureNV m_Structure;
	VulkanCoreBuffer<uint8_t> *m_Memory = nullptr;
	VulkanCoreBuffer<float4> *m_Vertices = nullptr;
};

} // namespace lh2core