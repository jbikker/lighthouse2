/* vulkan_ray_trace_nv_pipeline.cpp - Copyright 2019 Utrecht University

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

lh2core::VulkanRayTraceNVPipeline::VulkanRayTraceNVPipeline( const VulkanDevice &device )
	: m_Device( device )
{
}

void lh2core::VulkanRayTraceNVPipeline::Cleanup()
{
	m_ShaderIndices.clear();
	m_DescriptorSets.clear();
	m_VkDescriptorSets.clear();
	m_ShaderStages.clear();
	m_ShaderGroups.clear();
	m_CurrentGroupIdx = 0;
	m_MaxRecursionDepth = 5;

	if ( m_Pipeline ) m_Device->destroyPipeline( m_Pipeline );
	if ( m_Layout ) m_Device->destroyPipelineLayout( m_Layout );
	if ( SBTBuffer ) delete SBTBuffer;

	m_Pipeline = nullptr;
	m_Layout = nullptr;
	SBTBuffer = nullptr;

	m_SBTGenerator = {};
	m_Generated = false;
}

uint32_t lh2core::VulkanRayTraceNVPipeline::AddEmptyHitGroup()
{
	assert( !m_Generated );
	vk::RayTracingShaderGroupCreateInfoNV groupInfo{};
	groupInfo.setPNext( nullptr );
	groupInfo.setType( vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup );
	groupInfo.setGeneralShader( VK_SHADER_UNUSED_NV );
	groupInfo.setClosestHitShader( VK_SHADER_UNUSED_NV );
	groupInfo.setAnyHitShader( VK_SHADER_UNUSED_NV );
	groupInfo.setIntersectionShader( VK_SHADER_UNUSED_NV );

	m_ShaderGroups.push_back( groupInfo );
	const auto idx = m_CurrentGroupIdx;
	m_CurrentGroupIdx++;
	m_ShaderIndices.emplace_back( std::make_pair( HITGROUP, idx ) );
	return idx;
}

uint32_t lh2core::VulkanRayTraceNVPipeline::AddHitGroup( const VulkanHitGroup &hitGroup )
{
	assert( !m_Generated );
	vk::RayTracingShaderGroupCreateInfoNV groupInfo{};
	groupInfo.setPNext( nullptr );
	groupInfo.setType( vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup );
	if ( hitGroup.generalShader )
	{
		vk::PipelineShaderStageCreateInfo stageCreate{};
		stageCreate.setPNext( nullptr );
		stageCreate.setStage( vk::ShaderStageFlagBits::eCallableNV );
		stageCreate.setModule( *hitGroup.generalShader );
		stageCreate.setPName( "main" );
		stageCreate.setFlags( vk::PipelineShaderStageCreateFlags() );
		stageCreate.setPSpecializationInfo( nullptr );

		m_ShaderStages.emplace_back( stageCreate );
		const auto shaderIdx = static_cast<uint32_t>( m_ShaderStages.size() - 1 );
		groupInfo.setGeneralShader( shaderIdx );
	}
	else
		groupInfo.setGeneralShader( VK_SHADER_UNUSED_NV );
	if ( hitGroup.closestHitShader )
	{
		vk::PipelineShaderStageCreateInfo stageCreate{};
		stageCreate.setPNext( nullptr );
		stageCreate.setStage( vk::ShaderStageFlagBits::eClosestHitNV );
		stageCreate.setModule( *hitGroup.closestHitShader );
		stageCreate.setPName( "main" );
		stageCreate.setFlags( vk::PipelineShaderStageCreateFlags() );
		stageCreate.setPSpecializationInfo( nullptr );

		m_ShaderStages.emplace_back( stageCreate );
		const auto shaderIdx = static_cast<uint32_t>( m_ShaderStages.size() - 1 );
		groupInfo.setClosestHitShader( shaderIdx );
	}
	else
		groupInfo.setClosestHitShader( VK_SHADER_UNUSED_NV );
	if ( hitGroup.anyHitShader )
	{
		vk::PipelineShaderStageCreateInfo stageCreate{};
		stageCreate.setPNext( nullptr );
		stageCreate.setStage( vk::ShaderStageFlagBits::eAnyHitNV );
		stageCreate.setModule( *hitGroup.anyHitShader );
		stageCreate.setPName( "main" );
		stageCreate.setFlags( vk::PipelineShaderStageCreateFlags() );
		stageCreate.setPSpecializationInfo( nullptr );

		m_ShaderStages.emplace_back( stageCreate );
		const auto shaderIdx = static_cast<uint32_t>( m_ShaderStages.size() - 1 );
		groupInfo.setAnyHitShader( shaderIdx );
	}
	else
		groupInfo.setAnyHitShader( VK_SHADER_UNUSED_NV );
	if ( hitGroup.intersectionShader )
	{
		vk::PipelineShaderStageCreateInfo stageCreate{};
		stageCreate.setPNext( nullptr );
		stageCreate.setStage( vk::ShaderStageFlagBits::eIntersectionNV );
		stageCreate.setModule( *hitGroup.intersectionShader );
		stageCreate.setPName( "main" );
		stageCreate.setFlags( vk::PipelineShaderStageCreateFlags() );
		stageCreate.setPSpecializationInfo( nullptr );

		m_ShaderStages.emplace_back( stageCreate );
		const auto shaderIdx = static_cast<uint32_t>( m_ShaderStages.size() - 1 );
		groupInfo.setIntersectionShader( shaderIdx );
	}
	else
		groupInfo.setIntersectionShader( VK_SHADER_UNUSED_NV );

	m_ShaderGroups.push_back( groupInfo );
	const auto idx = m_CurrentGroupIdx;
	m_CurrentGroupIdx++;
	m_ShaderIndices.emplace_back( std::make_pair( HITGROUP, idx ) );
	return idx;
}

uint32_t lh2core::VulkanRayTraceNVPipeline::AddRayGenShaderStage( vk::ShaderModule module )
{
	assert( !m_Generated );
	vk::PipelineShaderStageCreateInfo stageCreate{};
	stageCreate.setPNext( nullptr );
	stageCreate.setStage( vk::ShaderStageFlagBits::eRaygenNV );
	stageCreate.setModule( module );
	stageCreate.setPName( "main" );
	stageCreate.setFlags( vk::PipelineShaderStageCreateFlags() );
	stageCreate.setPSpecializationInfo( nullptr );

	m_ShaderStages.emplace_back( stageCreate );
	const auto shaderIdx = static_cast<uint32_t>( m_ShaderStages.size() - 1 );

	vk::RayTracingShaderGroupCreateInfoNV groupInfo{};
	groupInfo.setPNext( nullptr );
	groupInfo.setType( vk::RayTracingShaderGroupTypeNV::eGeneral );
	groupInfo.setGeneralShader( shaderIdx );
	groupInfo.setClosestHitShader( VK_SHADER_UNUSED_NV );
	groupInfo.setAnyHitShader( VK_SHADER_UNUSED_NV );
	groupInfo.setIntersectionShader( VK_SHADER_UNUSED_NV );
	m_ShaderGroups.emplace_back( groupInfo );

	m_ShaderIndices.emplace_back( std::make_pair( RAYGEN, m_CurrentGroupIdx ) );

	return m_CurrentGroupIdx++;
}

uint32_t lh2core::VulkanRayTraceNVPipeline::AddMissShaderStage( vk::ShaderModule module )
{
	vk::PipelineShaderStageCreateInfo stageCreate{};
	stageCreate.setPNext( nullptr );
	stageCreate.setStage( vk::ShaderStageFlagBits::eMissNV );
	stageCreate.setModule( module );
	stageCreate.setPName( "main" );
	stageCreate.setFlags( vk::PipelineShaderStageCreateFlags() );
	stageCreate.setPSpecializationInfo( nullptr );

	m_ShaderStages.emplace_back( stageCreate );
	const auto shaderIdx = static_cast<uint32_t>( m_ShaderStages.size() - 1 );

	vk::RayTracingShaderGroupCreateInfoNV groupInfo{};
	groupInfo.setPNext( nullptr );
	groupInfo.setType( vk::RayTracingShaderGroupTypeNV::eGeneral );
	groupInfo.setGeneralShader( shaderIdx );
	groupInfo.setClosestHitShader( VK_SHADER_UNUSED_NV );
	groupInfo.setAnyHitShader( VK_SHADER_UNUSED_NV );
	groupInfo.setIntersectionShader( VK_SHADER_UNUSED_NV );
	m_ShaderGroups.emplace_back( groupInfo );

	m_ShaderIndices.emplace_back( std::make_pair( MISS, m_CurrentGroupIdx ) );

	return m_CurrentGroupIdx++;
}

void lh2core::VulkanRayTraceNVPipeline::SetMaxRecursionDepth( uint32_t maxDepth )
{
	m_MaxRecursionDepth = maxDepth;
}

void lh2core::VulkanRayTraceNVPipeline::AddPushConstant( vk::PushConstantRange pushConstant )
{
	m_PushConstants.emplace_back( pushConstant );
}

void lh2core::VulkanRayTraceNVPipeline::AddDescriptorSet( const VulkanDescriptorSet *set )
{
	m_DescriptorSets.emplace_back( set );
}

void lh2core::VulkanRayTraceNVPipeline::Finalize()
{
	assert( !m_Generated );

	std::vector<vk::DescriptorSetLayout> layouts;
	vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
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

	pipelineLayoutCreateInfo.setPNext( nullptr );
	pipelineLayoutCreateInfo.setFlags( vk::PipelineLayoutCreateFlags() );
	pipelineLayoutCreateInfo.setSetLayoutCount( uint32_t( layouts.size() ) );
	pipelineLayoutCreateInfo.setPSetLayouts( layouts.empty() ? nullptr : layouts.data() );

	if ( !m_PushConstants.empty() )
	{
		pipelineLayoutCreateInfo.setPushConstantRangeCount( uint32_t( m_PushConstants.size() ) );
		pipelineLayoutCreateInfo.setPPushConstantRanges( m_PushConstants.data() );
	}
	else
	{
		pipelineLayoutCreateInfo.setPushConstantRangeCount( 0 );
		pipelineLayoutCreateInfo.setPPushConstantRanges( nullptr );
	}

	CheckVK( m_Device->createPipelineLayout( &pipelineLayoutCreateInfo, nullptr, &m_Layout ) );

	vk::RayTracingPipelineCreateInfoNV rayPipelineInfo{};
	rayPipelineInfo.setPNext( nullptr );
	rayPipelineInfo.setFlags( vk::PipelineCreateFlags() );
	rayPipelineInfo.setStageCount( (uint32_t)m_ShaderStages.size() );
	rayPipelineInfo.setPStages( m_ShaderStages.data() );
	rayPipelineInfo.setGroupCount( (uint32_t)m_ShaderGroups.size() );
	rayPipelineInfo.setPGroups( m_ShaderGroups.data() );
	rayPipelineInfo.setMaxRecursionDepth( 1 );
	rayPipelineInfo.setLayout( m_Layout );
	rayPipelineInfo.setBasePipelineHandle( nullptr );
	rayPipelineInfo.setBasePipelineIndex( 0 );

	CheckVK( m_Device->createRayTracingPipelinesNV( nullptr, 1, &rayPipelineInfo, nullptr,
													&m_Pipeline, RenderCore::instance->dynamicDispatcher ) );

	for ( const auto &shader : m_ShaderIndices )
	{
		const auto type = shader.first;
		const auto index = shader.second;

		switch ( type )
		{
		case ( RAYGEN ):
			m_SBTGenerator.AddRayGenerationProgram( index, {} );
			break;
		case ( MISS ):
			m_SBTGenerator.AddMissProgram( index, {} );
			break;
		case ( HITGROUP ):
			m_SBTGenerator.AddHitGroup( index, {} );
			break;
		}
	}

	vk::PhysicalDeviceProperties2 props;
	vk::PhysicalDeviceRayTracingPropertiesNV rtProperties;
	props.pNext = &rtProperties;
	m_Device.GetPhysicalDevice().getProperties2( &props, RenderCore::instance->dynamicDispatcher );
	const auto sbtSize = m_SBTGenerator.ComputeSBTSize( rtProperties );
	SBTBuffer = new VulkanCoreBuffer<uint8_t>( m_Device, sbtSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
											   vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferDst );
	m_SBTGenerator.Generate( m_Device, m_Pipeline, SBTBuffer );

	m_Generated = true;
}

void lh2core::VulkanRayTraceNVPipeline::RecordPushConstant( vk::CommandBuffer &cmdBuffer, uint32_t idx, uint32_t sizeInBytes, void *data )
{
	assert( m_Generated );
	assert( m_PushConstants.size() > idx );
	assert( m_PushConstants.at( idx ).size <= sizeInBytes );

	const auto &pushConstant = m_PushConstants.at( idx );
	cmdBuffer.pushConstants( m_Layout, pushConstant.stageFlags, 0, sizeInBytes, data );
}

void lh2core::VulkanRayTraceNVPipeline::RecordTraceCommand( vk::CommandBuffer &cmdBuffer, uint32_t width, uint32_t height, uint32_t depth )
{
	assert( m_Generated );

	// Setup pipeline
	cmdBuffer.bindPipeline( vk::PipelineBindPoint::eRayTracingNV, m_Pipeline, RenderCore::instance->dynamicDispatcher );
	if ( !m_DescriptorSets.empty() ) cmdBuffer.bindDescriptorSets( vk::PipelineBindPoint::eRayTracingNV, m_Layout, 0, uint32_t( m_VkDescriptorSets.size() ), m_VkDescriptorSets.data(), 0, nullptr );

	const vk::Buffer shaderBindingTableBuffer = *SBTBuffer;
	const auto rayGenOffset = m_SBTGenerator.GetRayGenOffset();
	const auto missOffset = m_SBTGenerator.GetMissOffset();
	const auto hitOffset = m_SBTGenerator.GetHitGroupOffset();
	const auto rayGenSize = m_SBTGenerator.GetRayGenSectionSize();
	const auto missSize = m_SBTGenerator.GetMissSectionSize();
	const auto hitSize = m_SBTGenerator.GetHitGroupSectionSize();

	// Intersect rays
	cmdBuffer.traceRaysNV( shaderBindingTableBuffer, rayGenOffset,
						   shaderBindingTableBuffer, missOffset, missSize,
						   shaderBindingTableBuffer, hitOffset, hitSize,
						   nullptr, 0, 0,
						   width, height, depth, RenderCore::instance->dynamicDispatcher );
}
