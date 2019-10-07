/* vulkan_ray_trace_nv_pipeline.h - Copyright 2019 Utrecht University

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
struct VulkanHitGroup
{
	VulkanHitGroup( VulkanShader *general = nullptr, VulkanShader *closestHit = nullptr, VulkanShader *anyHit = nullptr, VulkanShader *intersection = nullptr )
		: generalShader( general ), closestHitShader( closestHit ), anyHitShader( anyHit ), intersectionShader( intersection )
	{
	}

	const VulkanShader *generalShader = nullptr;
	const VulkanShader *closestHitShader = nullptr;
	const VulkanShader *anyHitShader = nullptr;
	const VulkanShader *intersectionShader = nullptr;
};

class VulkanRayTraceNVPipeline
{
  public:
	VulkanRayTraceNVPipeline( const VulkanDevice &device );
	~VulkanRayTraceNVPipeline()
	{
		Cleanup();
	}

	void Cleanup();
	// Helper function for use-cases like shadow-rays, any shader type requires at least 1 hit group to be usable
	uint32_t AddEmptyHitGroup();
	uint32_t AddHitGroup( const VulkanHitGroup &hitGroup );
	uint32_t AddRayGenShaderStage( vk::ShaderModule module );
	uint32_t AddMissShaderStage( vk::ShaderModule module );
	void SetMaxRecursionDepth( uint32_t maxDepth );
	void AddPushConstant( vk::PushConstantRange pushConstant );
	void AddDescriptorSet( const VulkanDescriptorSet *set );

	void Finalize();

	void RecordPushConstant(vk::CommandBuffer& cmdBuffer, uint32_t idx, uint32_t sizeInBytes, void* data);
	void RecordTraceCommand( vk::CommandBuffer &cmdBuffer, uint32_t width, uint32_t height = 1, uint32_t depth = 1 );

	operator vk::Pipeline() const
	{
		assert( m_Generated );
		return m_Pipeline;
	}
	operator vk::PipelineLayout() const
	{
		assert( m_Generated );
		return m_Layout;
	}

  private:
	enum ShaderType
	{
		RAYGEN,
		MISS,
		HITGROUP
	};

	bool m_Generated = false;
	VulkanDevice m_Device;
	std::vector<std::pair<ShaderType, uint32_t>> m_ShaderIndices;
	std::vector<const VulkanDescriptorSet *> m_DescriptorSets;
	std::vector<vk::DescriptorSet> m_VkDescriptorSets;
	std::vector<vk::PipelineShaderStageCreateInfo> m_ShaderStages;
	std::vector<vk::RayTracingShaderGroupCreateInfoNV> m_ShaderGroups;
	uint32_t m_CurrentGroupIdx = 0;
	uint32_t m_MaxRecursionDepth = 5;

	// Pipeline
	vk::Pipeline m_Pipeline = nullptr;
	vk::PipelineLayout m_Layout = nullptr;
	VulkanCoreBuffer<uint8_t> *SBTBuffer = nullptr;

	std::vector<vk::PushConstantRange> m_PushConstants;

	VulkanShaderBindingTableGenerator m_SBTGenerator;
};
} // namespace lh2core
