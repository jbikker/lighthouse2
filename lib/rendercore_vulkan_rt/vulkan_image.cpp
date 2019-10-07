/* vulkan_image.cpp - Copyright 2019 Utrecht University

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

VulkanImage::VulkanImage( const VulkanDevice &dev, vk::ImageType type, vk::Format format, vk::Extent3D extent,
						  vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags memProps )
	: m_Device( dev ), m_Extent( extent )
{
	vk::ImageCreateInfo imageCreateInfo{};
	imageCreateInfo.setPNext( nullptr );
	imageCreateInfo.setFlags( vk::ImageCreateFlags() );
	imageCreateInfo.setImageType( type );
	imageCreateInfo.setFormat( format );
	imageCreateInfo.setExtent( m_Extent );
	imageCreateInfo.setMipLevels( 1 );
	imageCreateInfo.setArrayLayers( 1 );
	imageCreateInfo.setSamples( vk::SampleCountFlagBits::e1 );
	imageCreateInfo.setTiling( tiling );
	imageCreateInfo.setUsage( usage );
	imageCreateInfo.setSharingMode( vk::SharingMode::eExclusive );
	imageCreateInfo.setQueueFamilyIndexCount( 0 );
	imageCreateInfo.setPQueueFamilyIndices( nullptr );
	imageCreateInfo.setInitialLayout( vk::ImageLayout::eUndefined );

	m_Image = m_Device->createImage( imageCreateInfo );
	if ( !m_Image )
		FATALERROR( "Could not create image." );

	const auto memoryRequirements = m_Device->getImageMemoryRequirements( m_Image );
	vk::MemoryAllocateInfo memoryAllocateInfo{};
	memoryAllocateInfo.setPNext( nullptr );
	memoryAllocateInfo.setAllocationSize( memoryRequirements.size );
	memoryAllocateInfo.setMemoryTypeIndex( m_Device.GetMemoryType( memoryRequirements, memProps ) );
	m_Memory = m_Device->allocateMemory( memoryAllocateInfo );

	if ( !m_Memory )
		FATALERROR( "Could not allocate memory for image." );

	m_Device->bindImageMemory( m_Image, m_Memory, 0 );
}

VulkanImage::~VulkanImage()
{
	Cleanup();
}

void VulkanImage::Cleanup()
{
	vk::Device device = m_Device.GetVkDevice();

	if ( m_Sampler )
	{
		device.destroySampler( m_Sampler );
		m_Sampler = nullptr;
	}
	if ( m_ImageView )
	{
		device.destroyImageView( m_ImageView );
		m_ImageView = nullptr;
	}
	if ( m_Image )
	{
		device.destroyImage( m_Image );
		m_Image = nullptr;
	}
	if ( m_Memory )
	{
		device.freeMemory( m_Memory );
		m_Memory = nullptr;
	}
}

bool VulkanImage::SetData( const void *data, uint32_t width, uint32_t height, uint32_t stride )
{
	vk::DeviceSize imageSize = width * height * stride;

	auto stagingBuffer = VulkanCoreBuffer<uint8_t>( m_Device, imageSize, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
													vk::BufferUsageFlagBits::eTransferSrc );
	stagingBuffer.CopyToDevice( data, imageSize );
	auto cmdBuffer = m_Device.CreateOneTimeCmdBuffer();

	vk::ImageMemoryBarrier barrier{};
	barrier.setPNext( nullptr );
	barrier.setSrcAccessMask( vk::AccessFlags() );
	barrier.setDstAccessMask( vk::AccessFlagBits::eTransferWrite );
	barrier.setOldLayout( vk::ImageLayout::eUndefined );
	barrier.setNewLayout( vk::ImageLayout::eTransferDstOptimal );
	barrier.setSrcQueueFamilyIndex( VK_QUEUE_FAMILY_IGNORED );
	barrier.setDstQueueFamilyIndex( VK_QUEUE_FAMILY_IGNORED );
	barrier.setImage( m_Image );
	barrier.setSubresourceRange( {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} );

	cmdBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
								vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier );

	vk::BufferImageCopy region{};
	region.setBufferOffset( 0 );
	region.setBufferRowLength( 0 );
	region.setBufferImageHeight( 0 );
	region.setImageSubresource( vk::ImageSubresourceLayers( vk::ImageAspectFlagBits::eColor, 0, 0, 1 ) );
	region.setImageOffset( {0, 0, 0} );
	region.setImageExtent( {width, height, 1} );

	cmdBuffer->copyBufferToImage( stagingBuffer, m_Image, vk::ImageLayout::eTransferDstOptimal, 1, &region );

	barrier.setSrcAccessMask( vk::AccessFlagBits::eTransferWrite );
	barrier.setDstAccessMask( vk::AccessFlagBits::eShaderRead );
	barrier.setOldLayout( vk::ImageLayout::eTransferDstOptimal );
	barrier.setNewLayout( vk::ImageLayout::eShaderReadOnlyOptimal );

	cmdBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlags(), 0, nullptr, 0, nullptr, 0, &barrier );

	m_CurLayout = vk::ImageLayout::eTransferDstOptimal;

	auto queue = m_Device.GetTransferQueue();
	cmdBuffer.Submit( queue, true );
	return true;
}

bool VulkanImage::CreateImageView( vk::ImageViewType viewType, vk::Format format, vk::ImageSubresourceRange subresourceRange )
{
	vk::ImageViewCreateInfo imageViewCreateInfo{};
	imageViewCreateInfo.setPNext( nullptr );
	imageViewCreateInfo.setViewType( viewType );
	imageViewCreateInfo.setFormat( format );
	imageViewCreateInfo.setSubresourceRange( subresourceRange );
	imageViewCreateInfo.setImage( m_Image );
	imageViewCreateInfo.setFlags( vk::ImageViewCreateFlags() );
	imageViewCreateInfo.setComponents( {vk::ComponentSwizzle::eR,
										vk::ComponentSwizzle::eG,
										vk::ComponentSwizzle::eB,
										vk::ComponentSwizzle::eA} );

	m_ImageView = m_Device->createImageView( imageViewCreateInfo );
	if ( !m_ImageView )
		FATALERROR( "Could not create image view." );
	return true;
}

bool VulkanImage::CreateSampler( vk::Filter magFilter, vk::Filter minFilter, vk::SamplerMipmapMode mipmapMode, vk::SamplerAddressMode addressMode )
{
	vk::SamplerCreateInfo samplerCreateInfo{};
	samplerCreateInfo.setPNext( nullptr );
	samplerCreateInfo.setFlags( vk::SamplerCreateFlags() );
	samplerCreateInfo.setMagFilter( magFilter );
	samplerCreateInfo.setMinFilter( minFilter );
	samplerCreateInfo.setMipmapMode( mipmapMode );
	samplerCreateInfo.setAddressModeU( addressMode );
	samplerCreateInfo.setAddressModeV( addressMode );
	samplerCreateInfo.setAddressModeW( addressMode );
	samplerCreateInfo.setMipLodBias( 0.0f );
	samplerCreateInfo.setAnisotropyEnable( false );
	samplerCreateInfo.setMaxAnisotropy( 1 );
	samplerCreateInfo.setCompareEnable( false );
	samplerCreateInfo.setCompareOp( vk::CompareOp::eAlways );
	samplerCreateInfo.setMinLod( 0.0f );
	samplerCreateInfo.setMaxLod( 0.0f );
	samplerCreateInfo.setBorderColor( vk::BorderColor::eIntOpaqueBlack );
	samplerCreateInfo.setUnnormalizedCoordinates( false );

	m_Sampler = m_Device->createSampler( samplerCreateInfo );
	if ( !m_Sampler ) FATALERROR( "Could not create sampler." );

	return true;
}

void lh2core::VulkanImage::TransitionToLayout( vk::ImageLayout dstLayout, vk::AccessFlags dstAccessMask, vk::CommandBuffer cmdBuffer )
{
	vk::ImageMemoryBarrier barrier{};
	barrier.setPNext( nullptr );
	barrier.setSrcAccessMask( vk::AccessFlags() );
	barrier.setDstAccessMask( dstAccessMask );
	barrier.setOldLayout( m_CurLayout );
	barrier.setNewLayout( dstLayout );
	barrier.setSrcQueueFamilyIndex( VK_QUEUE_FAMILY_IGNORED );
	barrier.setDstQueueFamilyIndex( VK_QUEUE_FAMILY_IGNORED );
	barrier.setImage( m_Image );
	barrier.setSubresourceRange( {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1} );

	if ( !cmdBuffer )
	{
		auto queue = m_Device.GetGraphicsQueue();
		auto commandBuffer = m_Device.CreateOneTimeCmdBuffer();
		commandBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
										vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier );
		commandBuffer.Submit( queue, true );
	}
	else
	{
		cmdBuffer.pipelineBarrier( vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
								   vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &barrier );
	}

	m_CurLayout = dstLayout;
}

vk::DescriptorImageInfo VulkanImage::GetDescriptorImageInfo() const
{
	vk::DescriptorImageInfo info{};
	info.setImageLayout( m_CurLayout );
	info.setImageView( m_ImageView );
	info.setSampler( m_Sampler );
	return info;
}
