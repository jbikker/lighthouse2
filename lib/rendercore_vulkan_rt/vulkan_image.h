/* vulkan_image.h - Copyright 2019 Utrecht University

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
class VulkanImage
{
  public:
	VulkanImage( const VulkanDevice &device, vk::ImageType type, vk::Format format, vk::Extent3D extent,
				 vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags memProps );
	~VulkanImage();

	void Cleanup();

	template <typename T>
	bool SetData( const std::vector<T> &data, uint32_t width, uint32_t height )
	{
		return SetData( data.data(), width, height, sizeof( T ) );
	}
	bool SetData( const void *data, uint32_t width, uint32_t height, uint32_t stride );
	bool CreateImageView( vk::ImageViewType viewType, vk::Format format, vk::ImageSubresourceRange subresourceRange );
	bool CreateSampler( vk::Filter magFilter, vk::Filter minFilter, vk::SamplerMipmapMode mipmapMode, vk::SamplerAddressMode addressMode );
	void TransitionToLayout( vk::ImageLayout layout, vk::AccessFlags dstAccessMask, vk::CommandBuffer cmdBuffer = nullptr );
	vk::DescriptorImageInfo GetDescriptorImageInfo() const;

	vk::Extent3D GetExtent() const { return m_Extent; }
	vk::Image GetImage() const { return m_Image; }
	vk::ImageView GetImageView() const { return m_ImageView; }
	vk::Sampler GetSampler() const { return m_Sampler; }
	vk::DeviceMemory GetMemory() const { return m_Memory; }

	operator vk::Image() const { return m_Image; }
	operator vk::ImageView() const { return m_ImageView; }
	operator vk::Sampler() const { return m_Sampler; }

  private:
	vk::ImageLayout m_CurLayout = vk::ImageLayout::eUndefined;
	VulkanDevice m_Device;
	vk::Extent3D m_Extent;
	vk::Image m_Image = nullptr;
	vk::DeviceMemory m_Memory = nullptr;
	vk::ImageView m_ImageView = nullptr;
	vk::Sampler m_Sampler = nullptr;
};
} // namespace lh2core