/* shader_binding_table_generator.cpp - Copyright 2019 Utrecht University

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

void VulkanShaderBindingTableGenerator::AddRayGenerationProgram( uint32_t groupIdx, const std::vector<uchar> &inlineData )
{
	m_RayGen.emplace_back( SBTEntry( groupIdx, inlineData ) );
}

void VulkanShaderBindingTableGenerator::AddMissProgram( uint32_t groupIdx, const std::vector<uchar> &inlineData )
{
	m_Miss.emplace_back( groupIdx, inlineData );
}

void VulkanShaderBindingTableGenerator::AddHitGroup( uint32_t groupIdx, const std::vector<uchar> &inlineData )
{
	m_HitGroup.emplace_back( groupIdx, inlineData );
}

vk::DeviceSize VulkanShaderBindingTableGenerator::ComputeSBTSize( const vk::PhysicalDeviceRayTracingPropertiesNV &props )
{
	// Size of a program identifier
	m_ProgIdSize = props.shaderGroupHandleSize;

	// Compute the entry size of each program type depending on the maximum number of parameters in each category
	m_RayGenEntrySize = GetEntrySize( m_RayGen );
	m_MissEntrySize = GetEntrySize( m_Miss );
	m_HitGroupEntrySize = GetEntrySize( m_HitGroup );

	m_SBTSize = m_RayGenEntrySize * (vk::DeviceSize)m_RayGen.size() + m_MissEntrySize * (vk::DeviceSize)m_Miss.size() +
				m_HitGroupEntrySize * (vk::DeviceSize)m_HitGroup.size();
	return m_SBTSize;
}

void VulkanShaderBindingTableGenerator::Generate( VulkanDevice &device, vk::Pipeline rtPipeline, VulkanCoreBuffer<uint8_t> *sbtBuffer )
{
	uint32_t groupCount = static_cast<uint32_t>( m_RayGen.size() ) + static_cast<uint32_t>( m_Miss.size() ) + static_cast<uint32_t>( m_HitGroup.size() );

	// Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
	auto shaderHandleStorage = std::vector<uint8_t>( groupCount * m_ProgIdSize );
	CheckVK( device.GetVkDevice().getRayTracingShaderGroupHandlesNV( rtPipeline, 0, groupCount,
																	 m_ProgIdSize * groupCount, shaderHandleStorage.data(),
																	 RenderCore::instance->dynamicDispatcher ) );
	std::vector<uint8_t> tempBuffer( m_SBTSize );
	auto data = tempBuffer.data();

	vk::DeviceSize offset = 0;

	// Copy ray generation SBT data
	offset = CopyShaderData( rtPipeline, data, m_RayGen, m_RayGenEntrySize, shaderHandleStorage.data() );
	data += offset;
	// Copy ray miss SBT data
	offset = CopyShaderData( rtPipeline, data, m_Miss, m_MissEntrySize, shaderHandleStorage.data() );
	data += offset;
	// Copy ray hit-groups SBT data
	offset = CopyShaderData( rtPipeline, data, m_HitGroup, m_HitGroupEntrySize, shaderHandleStorage.data() );

	// Unmap the SBT
	sbtBuffer->CopyToDevice( tempBuffer.data(), m_SBTSize );
}

void VulkanShaderBindingTableGenerator::Reset()
{
	m_RayGen.clear();
	m_Miss.clear();
	m_HitGroup.clear();

	m_RayGenEntrySize = 0;
	m_MissEntrySize = 0;
	m_HitGroupEntrySize = 0;
	m_ProgIdSize = 0;
}

vk::DeviceSize VulkanShaderBindingTableGenerator::GetRayGenSectionSize() const
{
	return m_RayGenEntrySize * static_cast<VkDeviceSize>( m_RayGen.size() );
}

vk::DeviceSize VulkanShaderBindingTableGenerator::GetRayGenEntrySize() const
{
	return m_RayGenEntrySize;
}

vk::DeviceSize VulkanShaderBindingTableGenerator::GetRayGenOffset() const
{
	return 0;
}

vk::DeviceSize VulkanShaderBindingTableGenerator::GetMissSectionSize() const
{
	return m_MissEntrySize * static_cast<vk::DeviceSize>( m_Miss.size() );
}

vk::DeviceSize VulkanShaderBindingTableGenerator::GetMissEntrySize()
{
	return m_MissEntrySize;
}

vk::DeviceSize VulkanShaderBindingTableGenerator::GetMissOffset() const
{
	return GetRayGenSectionSize();
}

vk::DeviceSize VulkanShaderBindingTableGenerator::GetHitGroupSectionSize() const
{
	return m_HitGroupEntrySize * static_cast<vk::DeviceSize>( m_HitGroup.size() );
}

vk::DeviceSize VulkanShaderBindingTableGenerator::GetHitGroupEntrySize() const
{
	return m_HitGroupEntrySize;
}

vk::DeviceSize VulkanShaderBindingTableGenerator::GetHitGroupOffset() const
{
	return GetRayGenSectionSize() + GetMissSectionSize();
}

vk::DeviceSize VulkanShaderBindingTableGenerator::CopyShaderData( vk::Pipeline pipeline, uint8_t *outputData, const std::vector<SBTEntry> &shaders, vk::DeviceSize entrySize, const uint8_t *shaderHandleStorage )
{
	uint8_t *pData = outputData;
	for ( const auto &shader : shaders )
	{
		// Copy the shader identifier that was previously obtained with vkGetRayTracingShaderGroupHandlesNV
		memcpy( pData, shaderHandleStorage + shader.m_GroupIdx * m_ProgIdSize, m_ProgIdSize );

		// Copy all its resources pointers or values in bulk
		if ( !shader.m_InlineData.empty() )
			memcpy( pData + m_ProgIdSize, shader.m_InlineData.data(), shader.m_InlineData.size() );

		pData += entrySize;
	}
	// Return the number of bytes actually written to the output buffer
	return static_cast<uint32_t>( shaders.size() ) * entrySize;
}

vk::DeviceSize VulkanShaderBindingTableGenerator::GetEntrySize( const std::vector<SBTEntry> &entries )
{
	// Find the maximum number of parameters used by a single entry
	size_t maxArgs = 0;
	for ( const auto &shader : entries )
	{
		maxArgs = std::max( maxArgs, shader.m_InlineData.size() );
	}
	// A SBT entry is made of a program ID and a set of 4-byte parameters (offsets or push constants)
	VkDeviceSize entrySize = m_ProgIdSize + static_cast<VkDeviceSize>( maxArgs );

	// The entries of the shader binding table must be 16-bytes-aligned
	entrySize = ( ( ( entrySize ) + ( 16 ) - 1 ) & ~( ( 16 ) - 1 ) );

	return entrySize;
}

lh2core::VulkanShaderBindingTableGenerator::SBTEntry::SBTEntry( uint32_t groupIdx, std::vector<uchar> inlineData )
	: m_GroupIdx( groupIdx ), m_InlineData( inlineData )
{
}
