/* vulkan_descriptor_set.cpp - Copyright 2019 Utrecht University

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

lh2core::VulkanDescriptorSet::VulkanDescriptorSet( const VulkanDevice &device )
	: m_Device( device ), m_Buffers( this ), m_Images( this ), m_AccelerationStructures( this )
{
}

void lh2core::VulkanDescriptorSet::Cleanup()
{
	if ( m_DescriptorSet ) m_Device->destroyDescriptorSetLayout( m_Layout );
	if ( m_Pool ) m_Device->destroyDescriptorPool( m_Pool );

	m_DescriptorSet = nullptr;
	m_Layout = nullptr;
	m_Pool = nullptr;

	m_Generated = false;
	m_Dirty = true;
	m_Bindings.clear();
	m_Buffers.Clear();
	m_Images.Clear();
	m_AccelerationStructures.Clear();
}

void lh2core::VulkanDescriptorSet::AddBinding( uint32_t binding, uint32_t descriptorCount, vk::DescriptorType type,
											   vk::ShaderStageFlags stage, vk::Sampler *sampler )
{
	FATALERROR_IF( m_Generated, "Cannot add bindings after descriptor set has been generated." );
	vk::DescriptorSetLayoutBinding b{};
	b.setBinding( binding );
	b.setDescriptorCount( descriptorCount );
	b.setDescriptorType( type );
	b.setPImmutableSamplers( sampler );
	b.setStageFlags( stage );

	FATALERROR_IF( m_Bindings.find( binding ) != m_Bindings.end(), "Binding collision at %i", binding );

	m_Bindings[binding] = b;
}

void lh2core::VulkanDescriptorSet::Finalize()
{
	assert( !m_Generated );
	GeneratePool();
	GenerateLayout();
	GenerateSet();
}

void lh2core::VulkanDescriptorSet::ClearBindings()
{
	assert( m_Generated );
	m_Buffers.Clear();
	m_Images.Clear();
	m_AccelerationStructures.Clear();
}

void lh2core::VulkanDescriptorSet::UpdateSetContents()
{
	assert( m_Generated );
	if ( !m_Dirty ) return;

	// For each resource type, set the actual pointers in the vk::WriteDescriptorSet structures, and
	// write the resulting structures into the descriptor set
	if ( !m_Buffers.writeDescs.empty() )
	{
		m_Buffers.SetPointers();
		m_Device->updateDescriptorSets( static_cast<uint32_t>( m_Buffers.writeDescs.size() ),
										m_Buffers.writeDescs.data(), 0, nullptr );
	}

	if ( !m_Images.writeDescs.empty() )
	{
		m_Images.SetPointers();
		m_Device->updateDescriptorSets( static_cast<uint32_t>( m_Images.writeDescs.size() ),
										m_Images.writeDescs.data(), 0, nullptr );
	}

	if ( !m_AccelerationStructures.writeDescs.empty() )
	{
		m_AccelerationStructures.SetPointers();
		m_Device->updateDescriptorSets( static_cast<uint32_t>( m_AccelerationStructures.writeDescs.size() ),
										m_AccelerationStructures.writeDescs.data(), 0, nullptr );
	}
}

void lh2core::VulkanDescriptorSet::Bind( uint32_t binding, const std::vector<vk::DescriptorBufferInfo> &bufferInfo )
{
	assert( m_Generated );
	m_Dirty = true;
	m_Buffers.Bind( binding, m_Bindings[binding].descriptorType, bufferInfo );
}

void lh2core::VulkanDescriptorSet::Bind( uint32_t binding, const std::vector<vk::DescriptorImageInfo> &imageInfo )
{
	assert( m_Generated );
	m_Dirty = true;
	m_Images.Bind( binding, m_Bindings[binding].descriptorType, imageInfo );
}

void VulkanDescriptorSet::Bind( uint32_t binding, const std::vector<vk::WriteDescriptorSetAccelerationStructureNV> &accelInfo )
{
	assert( m_Generated );
	m_Dirty = true;
	m_AccelerationStructures.Bind( binding, m_Bindings[binding].descriptorType, accelInfo );
}

void lh2core::VulkanDescriptorSet::GeneratePool()
{
	m_Generated = true;
	std::vector<vk::DescriptorPoolSize> counters;
	counters.reserve( m_Bindings.size() );

	for ( const auto &b : m_Bindings )
		counters.emplace_back( b.second.descriptorType, b.second.descriptorCount );

	vk::DescriptorPoolCreateInfo poolInfo{};
	poolInfo.setPoolSizeCount( static_cast<uint32_t>( counters.size() ) );
	poolInfo.setPPoolSizes( counters.data() );
	poolInfo.setMaxSets( 1 );

	m_Pool = m_Device->createDescriptorPool( poolInfo );
}

void lh2core::VulkanDescriptorSet::GenerateLayout()
{
	m_Generated = true;
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	bindings.reserve( m_Bindings.size() );

	for ( const auto &b : m_Bindings )
		bindings.push_back( b.second );

	vk::DescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.setBindingCount( (uint32_t)m_Bindings.size() );
	layoutInfo.setPBindings( bindings.data() );

	m_Layout = m_Device->createDescriptorSetLayout( layoutInfo, nullptr, RenderCore::instance->dynamicDispatcher );
}

void lh2core::VulkanDescriptorSet::GenerateSet()
{
	m_Generated = true;
	vk::DescriptorSetLayout layouts[] = {m_Layout};
	vk::DescriptorSetAllocateInfo allocInfo{};
	allocInfo.setPNext( nullptr );
	allocInfo.setDescriptorPool( m_Pool );
	allocInfo.setDescriptorSetCount( 1 );
	allocInfo.setPSetLayouts( layouts );

	m_Device->allocateDescriptorSets( &allocInfo, &m_DescriptorSet, RenderCore::instance->dynamicDispatcher );
}