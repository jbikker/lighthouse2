/* vulkan_descriptor_set.h - Copyright 2019 Utrecht University

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
#include <vulkan/vulkan.hpp>

namespace lh2core
{

class VulkanDescriptorSet
{
  public:
	VulkanDescriptorSet( const VulkanDevice &device );
	~VulkanDescriptorSet()
	{
		Cleanup();
	}

	void Cleanup();

	void AddBinding( uint32_t binding, uint32_t descriptorCount,
					 vk::DescriptorType type, vk::ShaderStageFlags stage, vk::Sampler *sampler = nullptr );

	void Finalize();

	template <typename T, /* Type of the descriptor info, such as vk::DescriptorBufferInfo*/ uint32_t offset /* Offset in the vk::WriteDescriptorSet structure */>
	struct WriteInfo
	{
		WriteInfo( VulkanDescriptorSet *set ) : descriptorSet( set ) {}

		VulkanDescriptorSet *descriptorSet;

		std::map<uint32_t, uint32_t> bindingIndices;
		std::vector<vk::WriteDescriptorSet> writeDescs;
		std::vector<std::vector<T>> descContents;

		~WriteInfo()
		{
			Clear();
		}

		void Clear()
		{
			bindingIndices.clear();
			writeDescs.clear();
			for ( auto &v : descContents )
				v.clear();
			descContents.clear();
		}

		void SetPointers()
		{
			for ( size_t i = 0; i < writeDescs.size(); i++ )
			{
				T **dest = reinterpret_cast<T **>( reinterpret_cast<uint8_t *>( &writeDescs[i] ) + offset );
				*dest = descContents[i].data();
			}
		}

		void Bind( uint32_t binding, vk::DescriptorType type, const std::vector<T> &info )
		{
			// Initialize the descriptor write, keeping all the resource pointers to NULL since they will
			// be set by SetPointers once all resources have been bound
			vk::WriteDescriptorSet descriptorWrite = {};
			descriptorWrite.dstSet = *descriptorSet;
			descriptorWrite.dstBinding = binding;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = type;
			descriptorWrite.descriptorCount = static_cast<uint32_t>( info.size() );
			descriptorWrite.pBufferInfo = nullptr;
			descriptorWrite.pImageInfo = nullptr;
			descriptorWrite.pTexelBufferView = nullptr;
			descriptorWrite.pNext = nullptr;

			if ( bindingIndices.find( binding ) != bindingIndices.end() ) // Binding already had a value, replace it
			{
				const uint32_t index = bindingIndices[binding];
				writeDescs[index] = descriptorWrite;
				descContents[index] = info;
			}
			else // Add the write descriptor and resource info for later actual binding
			{
				bindingIndices[binding] = static_cast<uint32_t>( writeDescs.size() );
				writeDescs.push_back( descriptorWrite );
				descContents.push_back( info );
			}
		}
	};

	// Bind a buffer
	void Bind( uint32_t binding, const std::vector<vk::DescriptorBufferInfo> &bufferInfo );
	// Bind an image
	void Bind( uint32_t binding, const std::vector<vk::DescriptorImageInfo> &imageInfo );
	// Bind an acceleration structure
	void Bind( uint32_t binding, const std::vector<vk::WriteDescriptorSetAccelerationStructureNV> &accelInfo );

	// Clear currently bound objects of descriptor set
	void ClearBindings();

	// Actually write the binding info into the descriptor set
	void UpdateSetContents();

	bool IsDirty() const { return m_Dirty; }

	operator vk::DescriptorSet() const { return m_DescriptorSet; }
	operator vk::DescriptorSetLayout() const { return m_Layout; }
	operator vk::DescriptorPool() const { return m_Pool; }

	operator vk::DescriptorSet *() { return &m_DescriptorSet; }
	operator vk::DescriptorSetLayout *() { return &m_Layout; }
	operator vk::DescriptorPool *() { return &m_Pool; }

	operator const vk::DescriptorSet *() const { return &m_DescriptorSet; }
	operator const vk::DescriptorSetLayout *() const { return &m_Layout; }
	operator const vk::DescriptorPool *() const { return &m_Pool; }

	vk::DescriptorSet GetSet() const { return m_DescriptorSet; }
	vk::DescriptorSetLayout GetLayout() const { return m_Layout; }
	vk::DescriptorPool GetPool() const { return m_Pool; }

  private:
	void GeneratePool();
	void GenerateLayout();
	void GenerateSet();

	VulkanDevice m_Device;
	vk::DescriptorSet m_DescriptorSet;
	vk::DescriptorSetLayout m_Layout;
	vk::DescriptorPool m_Pool;

	bool m_Dirty = false;
	bool m_Generated = false;

	std::unordered_map<uint32_t, vk::DescriptorSetLayoutBinding> m_Bindings;
	WriteInfo<vk::DescriptorBufferInfo, offsetof( VkWriteDescriptorSet, pBufferInfo )> m_Buffers;
	WriteInfo<vk::DescriptorImageInfo, offsetof( VkWriteDescriptorSet, pImageInfo )> m_Images;
	WriteInfo<vk::WriteDescriptorSetAccelerationStructureNV, offsetof( VkWriteDescriptorSet, pNext )> m_AccelerationStructures;
};

} // namespace lh2core