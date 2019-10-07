/* vulkan_compute_pipeline.h - Copyright 2019 Utrecht University

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

lh2core::VulkanComputePipeline::VulkanComputePipeline( const VulkanDevice &device, const VulkanShader &computeShader )
	: m_Device( device )
{
	m_ShaderStage = computeShader.GetShaderStage( vk::ShaderStageFlagBits::eCompute );
}

lh2core::VulkanComputePipeline::~VulkanComputePipeline()
{
	Cleanup();
}

void lh2core::VulkanComputePipeline::AddPushConstant( vk::PushConstantRange pushConstant )
{
	assert( !m_Generated );
	m_PushConstants.emplace_back( pushConstant );
}

void lh2core::VulkanComputePipeline::AddDescriptorSet( const VulkanDescriptorSet *set )
{
	assert( !m_Generated );
	m_DescriptorSets.emplace_back( set );
}

void lh2core::VulkanComputePipeline::RecordPushConstant( vk::CommandBuffer &cmdBuffer, uint32_t idx, uint32_t sizeInBytes, void *data )
{
	assert( m_Generated );
	assert( m_PushConstants.size() > idx );
	assert( m_PushConstants.at( idx ).size <= sizeInBytes );

	const auto &pushConstant = m_PushConstants.at( idx );
	cmdBuffer.pushConstants( m_Layout, pushConstant.stageFlags, 0, sizeInBytes, data );
}

void lh2core::VulkanComputePipeline::RecordDispatchCommand( vk::CommandBuffer &cmdBuffer, uint32_t width, uint32_t height, uint32_t depth )
{
	assert( m_Generated );
	cmdBuffer.bindPipeline( vk::PipelineBindPoint::eCompute, m_Pipeline );
	cmdBuffer.bindDescriptorSets( vk::PipelineBindPoint::eCompute, m_Layout, 0, (uint32_t)m_DescriptorSets.size(), m_VkDescriptorSets.data(), 0, nullptr );
	cmdBuffer.dispatch( width, height, depth );
}

void lh2core::VulkanComputePipeline::Finalize()
{
	assert( !m_Generated );
	assert( !m_DescriptorSets.empty() );

	std::vector<vk::DescriptorSetLayout> layouts;
	if ( !m_DescriptorSets.empty() )
	{
		layouts.resize( m_DescriptorSets.size() );
		m_VkDescriptorSets.resize( m_DescriptorSets.size() );
		for ( size_t i = 0; i < m_DescriptorSets.size(); i++ )
		{
			layouts.at( i ) = m_DescriptorSets.at( i )->GetLayout();
			m_VkDescriptorSets.at( i ) = m_DescriptorSets.at( i )->GetSet();
		}
	}

	vk::PipelineLayoutCreateInfo computeLayoutCreateInfo;
	computeLayoutCreateInfo.pNext = nullptr;
	computeLayoutCreateInfo.flags = vk::PipelineLayoutCreateFlags();
	computeLayoutCreateInfo.setLayoutCount = uint32_t( layouts.size() );
	computeLayoutCreateInfo.pSetLayouts = layouts.empty() ? nullptr : layouts.data();

	if ( m_PushConstants.empty() )
	{
		computeLayoutCreateInfo.pushConstantRangeCount = 0;
		computeLayoutCreateInfo.pPushConstantRanges = nullptr;
	}
	else
	{
		computeLayoutCreateInfo.pushConstantRangeCount = uint32_t( m_PushConstants.size() );
		computeLayoutCreateInfo.pPushConstantRanges = m_PushConstants.data();
	}

	m_Layout = m_Device->createPipelineLayout( computeLayoutCreateInfo );

	const auto computeCreateInfo = vk::ComputePipelineCreateInfo( vk::PipelineCreateFlags(), m_ShaderStage, m_Layout, nullptr );
	m_Pipeline = m_Device->createComputePipeline( nullptr, computeCreateInfo );

	m_Generated = true;
}

void lh2core::VulkanComputePipeline::Cleanup()
{
	m_DescriptorSets.clear();
	m_VkDescriptorSets.clear();
	m_PushConstants.clear();

	if ( m_Pipeline ) m_Device->destroyPipeline( m_Pipeline );
	if ( m_Layout ) m_Device->destroyPipelineLayout( m_Layout );

	m_Pipeline = nullptr;
	m_Layout = nullptr;

	m_Generated = false;
}
