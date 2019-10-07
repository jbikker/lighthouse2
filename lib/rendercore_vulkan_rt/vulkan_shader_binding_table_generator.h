/* shader_binding_table_generator.h - Copyright 2019 Utrecht University

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

class VulkanShaderBindingTableGenerator
{
  public:
	// Add a ray generation program by name, with its list of data pointers or values according to
	// the layout of its root signature
	void AddRayGenerationProgram( uint32_t groupIdx, const std::vector<uchar> &inlineData );

	// Add a miss program by name, with its list of data pointers or values according to
	// the layout of its root signature
	void AddMissProgram( uint32_t groupIdx, const std::vector<uchar> &inlineData );

	// Add a hit group by name, with its list of data pointers or values according to
	// the layout of its root signature
	void AddHitGroup( uint32_t groupIdx, const std::vector<uchar> &inlineData );

	/// Compute the size of the SBT based on the set of programs and hit groups it contains
	vk::DeviceSize ComputeSBTSize( const vk::PhysicalDeviceRayTracingPropertiesNV &props );

	// Build the SBT and store it into sbtBuffer, which has to be preallocated on the upload heap.
	// Access to the ray tracing pipeline object is required to fetch program identifiers using their names
	void Generate( VulkanDevice &device, vk::Pipeline rtPipeline, VulkanCoreBuffer<uint8_t> *sbtBuffer );

	void Reset(); /// Reset the sets of programs and hit groups

	vk::DeviceSize GetRayGenSectionSize() const;
	// Get the size in bytes of one ray generation program entry in the SBT
	vk::DeviceSize GetRayGenEntrySize() const;

	vk::DeviceSize GetRayGenOffset() const;

	// Get the size in bytes of the SBT section dedicated to miss programs
	vk::DeviceSize GetMissSectionSize() const;
	// Get the size in bytes of one miss program entry in the SBT
	vk::DeviceSize GetMissEntrySize();

	vk::DeviceSize GetMissOffset() const;

	// Get the size in bytes of the SBT section dedicated to hit groups
	vk::DeviceSize GetHitGroupSectionSize() const;
	// Get the size in bytes of hit group entry in the SBT
	vk::DeviceSize GetHitGroupEntrySize() const;

	vk::DeviceSize GetHitGroupOffset() const;

  private:
	// Wrapper for SBT entries, each consisting of the name of the program and a list of values,
	// which can be either offsets or raw 32-bit constants
	struct SBTEntry
	{
		SBTEntry( uint32_t groupIdx, std::vector<uchar> inlineData );

		uint32_t m_GroupIdx;
		const std::vector<uchar> m_InlineData;
	};

	// For each entry, copy the shader identifier followed by its resource pointers and/or root
	// constants in outputData, with a stride in bytes of entrySize, and returns the size in bytes
	// actually written to outputData.
	vk::DeviceSize CopyShaderData( vk::Pipeline pipeline, uint8_t *outputData,
								   const std ::vector<SBTEntry> &shaders,
								   vk::DeviceSize entrySize,
								   const uint8_t *shaderHandleStorage );

	// Compute the size of the SBT entries for a set of entries, which is determined by the maximum
	// number of parameters of their root signature
	vk::DeviceSize GetEntrySize( const std::vector<SBTEntry> &entries );

	std::vector<SBTEntry> m_RayGen;   // Ray generation shader entries
	std::vector<SBTEntry> m_Miss;	 // Miss shader entries
	std::vector<SBTEntry> m_HitGroup; /// Hit group entries

	// For each category, the size of an entry in the SBT depends on the maximum number of resources
	// used by the shaders in that category.The helper computes those values automatically in
	// GetEntrySize()
	vk::DeviceSize m_RayGenEntrySize;
	vk::DeviceSize m_MissEntrySize;
	vk::DeviceSize m_HitGroupEntrySize;

	// The program names are translated into program identifiers.The size in bytes of an identifier
	// is provided by the device and is the same for all categories.
	vk::DeviceSize m_ProgIdSize;
	vk::DeviceSize m_SBTSize;
};

} // namespace lh2core