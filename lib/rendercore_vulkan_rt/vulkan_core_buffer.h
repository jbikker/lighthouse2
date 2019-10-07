/* vulkan_core_buffer.h - Copyright 2019 Utrecht University

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
enum AllocLocation
{
	ON_DEVICE = 1,
	ON_HOST = 2
};

template <typename T>
class VulkanCoreBuffer
{
  public:
	VulkanCoreBuffer( const VulkanDevice &device, vk::DeviceSize elementCount, vk::MemoryPropertyFlags memFlags, vk::BufferUsageFlags usageFlags, uint location = ON_DEVICE )
		: m_Device( device ), m_Elements( elementCount ), m_MemFlags( memFlags ), m_UsageFlags( usageFlags ), m_Flags( location )
	{
		vk::Device vkDevice = device.GetVkDevice();

		vk::BufferCreateInfo createInfo{};
		createInfo.setPNext( nullptr );
		createInfo.setSize( elementCount * sizeof( T ) );
		createInfo.setUsage( usageFlags );
		createInfo.setSharingMode( vk::SharingMode::eExclusive );
		createInfo.setQueueFamilyIndexCount( 0u );
		createInfo.setPQueueFamilyIndices( nullptr );

		m_Buffer = vkDevice.createBuffer( createInfo );
		const vk::MemoryRequirements memReqs = vkDevice.getBufferMemoryRequirements( m_Buffer );

		vk::MemoryAllocateInfo memoryAllocateInfo{};
		memoryAllocateInfo.setPNext( nullptr );
		memoryAllocateInfo.setAllocationSize( memReqs.size );
		memoryAllocateInfo.setMemoryTypeIndex( device.GetMemoryType( memReqs, memFlags ) );

		m_Memory = vkDevice.allocateMemory( memoryAllocateInfo );

		vkDevice.bindBufferMemory( m_Buffer, m_Memory, 0 );

		if ( location & ON_HOST ) m_HostBuffer = new T[elementCount];
	}
	~VulkanCoreBuffer() { Cleanup(); }

	void Cleanup()
	{
		if ( m_HostBuffer ) delete[] m_HostBuffer;
		if ( m_Buffer ) m_Device->destroyBuffer( m_Buffer );
		if ( m_Memory ) m_Device->freeMemory( m_Memory );

		m_Flags = 0;
		m_HostBuffer = nullptr;
		m_Buffer = nullptr;
		m_Memory = nullptr;
		m_Elements = 0;
	}

	void CopyToDevice()
	{
		CopyToDevice( m_HostBuffer, GetSize() );
	}

	void CopyToDevice( const void *storage, vk::DeviceSize size = 0 )
	{
		assert( size <= ( m_Elements * sizeof( T ) ) );

		if ( CanMap() )
		{
			void *memory = Map();
			memcpy( memory, storage, size );
			Unmap();
		}
		else
		{
			auto stagingBuffer = VulkanCoreBuffer<uint8_t>( m_Device, size, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
															vk::BufferUsageFlagBits::eTransferSrc );
			memcpy( stagingBuffer.Map(), storage, size );
			stagingBuffer.Unmap();
			stagingBuffer.CopyTo( this );
		}
	}

	void CopyToHost()
	{
		assert( m_Flags & ON_HOST );
		assert( m_HostBuffer != nullptr );

		CopyToHost( m_HostBuffer );
	}

	void CopyToHost( void *storage )
	{
		if ( CanMap() )
		{
			void *memory = Map();
			memcpy( storage, memory, m_Elements * sizeof( T ) );
			Unmap();
		}
		else
		{
			assert( m_UsageFlags & vk::BufferUsageFlagBits::eTransferSrc );
			auto stagingBuffer = VulkanCoreBuffer<uint8_t>( m_Device, GetSize(), vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
															vk::BufferUsageFlagBits::eTransferDst );
			this->CopyTo( &stagingBuffer );
			memcpy( storage, stagingBuffer.Map(), stagingBuffer.GetSize() );
			stagingBuffer.Unmap();
		}
	}

	template <typename B>
	void CopyTo( VulkanCoreBuffer<B> *buffer )
	{
		assert( m_UsageFlags & vk::BufferUsageFlagBits::eTransferSrc );
		assert( buffer->GetBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst );

		auto cmdBuffer = m_Device.CreateOneTimeCmdBuffer();
		vk::BufferCopy copyRegion = vk::BufferCopy( 0, 0, m_Elements * sizeof( T ) );
		cmdBuffer->copyBuffer( m_Buffer, *buffer, 1, &copyRegion );
		
		auto transferQueue = m_Device.GetTransferQueue();
		cmdBuffer.Submit( transferQueue, true );
	}

	T *Map()
	{
		assert( CanMap() );
		void *memory = m_Device->mapMemory( m_Memory, 0, m_Elements * sizeof( T ) );
		assert( memory );
		return (T *)( memory );
	}

	void Unmap()
	{
		assert( CanMap() );
		m_Device->unmapMemory( m_Memory );
	}

	vk::DescriptorBufferInfo GetDescriptorBufferInfo( vk::DeviceSize offset = 0, vk::DeviceSize range = 0 ) const
	{
		vk::DescriptorBufferInfo info{};
		info.setBuffer( m_Buffer );
		info.setOffset( offset );
		info.setRange( range != 0 ? range : GetSize() );
		return info;
	}

	operator vk::Buffer() const { return m_Buffer; }
	operator vk::DeviceMemory() const { return m_Memory; }
	operator vk::Buffer *() { return &m_Buffer; }
	operator vk::DeviceMemory *() { return &m_Memory; }
	operator const vk::Buffer *() const { return &m_Buffer; }
	operator const vk::DeviceMemory *() const { return &m_Memory; }
	operator vk::DescriptorBufferInfo() const { return GetDescriptorBufferInfo( 0, 0 ); }

	constexpr bool CanMap() const
	{
		return ( m_MemFlags & vk::MemoryPropertyFlagBits::eHostVisible ) &&
			   ( m_MemFlags & vk::MemoryPropertyFlagBits::eHostCoherent );
	}

	T *GetHostBuffer() { return m_HostBuffer; }
	vk::DeviceSize GetElementCount() const { return m_Elements; }
	vk::DeviceSize GetSize() const { return m_Elements * sizeof( T ); }
	vk::MemoryPropertyFlags GetMemoryProperties() const { return m_MemFlags; }
	vk::BufferUsageFlags GetBufferUsageFlags() const { return m_UsageFlags; }

  private:
	uint m_Flags;
	T *m_HostBuffer = nullptr;
	VulkanDevice m_Device;
	vk::Buffer m_Buffer = nullptr;
	vk::DeviceMemory m_Memory = nullptr;
	vk::DeviceSize m_Elements = 0;
	vk::MemoryPropertyFlags m_MemFlags;
	vk::BufferUsageFlags m_UsageFlags;
};

template <typename T, typename B>
static void RecordCopyCommand( VulkanCoreBuffer<B> *target, VulkanCoreBuffer<T> *source, vk::CommandBuffer &cmdBuffer )
{
	assert( target->GetBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst );
	assert( source->GetBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferSrc );

	const vk::DeviceSize copySize = std::min( target->GetSize(), source->GetSize() );
	vk::BufferCopy copyRegion{};
	copyRegion.setSrcOffset( 0 );
	copyRegion.setDstOffset( 0 );
	copyRegion.setSize( copySize );
	cmdBuffer.copyBuffer( *source, *target, 1, &copyRegion );
}
} // namespace lh2core