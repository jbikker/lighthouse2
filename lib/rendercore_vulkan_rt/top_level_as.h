/* top_level_as.h - Copyright 2019 Utrecht University

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
class CoreMesh;
enum AccelerationStructureType // See: https://devblogs.nvidia.com/rtx-best-practices/ for more info
{
	FastestBuild, // Used for geometry like particles
	FastRebuild,  // Low level of detail objects unlikely to be hit, but need to updated frequently
	FastestTrace, // Best for static geometry, provides fastest trace possible
	FastTrace,	// Good compromise between fast tracing and build times, best for geometry like player character etc.
};

class TopLevelAS
{
  public:
	TopLevelAS( const VulkanDevice &device, AccelerationStructureType type = FastestTrace, uint32_t instanceCount = 32 );
	~TopLevelAS();

	void Cleanup();

	void UpdateInstances( const std::vector<GeometryInstance> &instances );
	void Build();
	void Rebuild();

	uint64_t GetHandle();
	uint32_t GetInstanceCount() const;

	bool CanUpdate() const { return uint( m_Flags & vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate ) > 0; }
	const vk::AccelerationStructureNV &GetAccelerationStructure() const { return m_Structure; }
	vk::WriteDescriptorSetAccelerationStructureNV GetDescriptorBufferInfo() const;

  private:
	void Build( bool update );

	// Converts desired type to vulkan build flags
	static vk::BuildAccelerationStructureFlagsNV TypeToFlags( AccelerationStructureType type )
	{
		// Different version than bottom level AS, top level acceleration structure do not allow for compaction
		switch ( type )
		{
		case ( FastestBuild ): return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastBuild;
		case ( FastRebuild ): return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastBuild | vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate;
		case ( FastestTrace ): return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace;
		case ( FastTrace ): return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace | vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate;
		default: return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace;
		}
	}

	VulkanDevice m_Device;
	uint32_t m_InstanceCnt = 0;
	vk::DeviceSize m_ResultSize{}, m_ScratchSize{};
	AccelerationStructureType m_Type{};
	vk::BuildAccelerationStructureFlagsNV m_Flags{};
	vk::AccelerationStructureNV m_Structure{};
	VulkanCoreBuffer<uint8_t> *m_Memory = nullptr;
	VulkanCoreBuffer<GeometryInstance> *m_InstanceBuffer = nullptr;
};
} // namespace lh2core