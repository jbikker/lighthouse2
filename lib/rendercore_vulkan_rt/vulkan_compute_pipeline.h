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

#pragma once

namespace lh2core
{
class VulkanComputePipeline
{
  public:
	VulkanComputePipeline( const VulkanDevice &device, const VulkanShader &computeShader );
	~VulkanComputePipeline();

	void AddPushConstant( vk::PushConstantRange pushConstant );
	void AddDescriptorSet( const VulkanDescriptorSet *set );

	void RecordPushConstant( vk::CommandBuffer &cmdBuffer, uint32_t idx, uint32_t sizeInBytes, void *data );
	void RecordDispatchCommand( vk::CommandBuffer &cmdBuffer, uint32_t width, uint32_t height = 1, uint32_t depth = 1 );

	void Finalize();

	void Cleanup();

  private:
	bool m_Generated = false;
	VulkanDevice m_Device;
	vk::PipelineShaderStageCreateInfo m_ShaderStage;
	std::vector<const VulkanDescriptorSet *> m_DescriptorSets;
	std::vector<vk::DescriptorSet> m_VkDescriptorSets;
	std::vector<vk::PushConstantRange> m_PushConstants;
	vk::Pipeline m_Pipeline = nullptr;
	vk::PipelineLayout m_Layout = nullptr;
};
} // namespace lh2core