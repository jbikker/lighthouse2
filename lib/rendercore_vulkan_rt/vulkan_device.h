/* vulkan_device.h - Copyright 2019 Utrecht University

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

#include <optional>

namespace lh2core
{

struct QueueFamilyIndices
{
	QueueFamilyIndices() = default;
	QueueFamilyIndices( vk::PhysicalDevice device, std::optional<vk::SurfaceKHR> surface = std::nullopt );
	bool IsComplete() const;

	bool hasSurface = false;
	std::optional<uint32_t> graphicsIdx;
	std::optional<uint32_t> computeIdx;
	std::optional<uint32_t> transferIdx;
	std::optional<uint32_t> presentIdx;
};

class OneTimeCommandBuffer;

/*
 * This class is a wrapper around vk::Device with some helper functions.
 * The object is reference counted to make using a Vulkan device easier.
 * It also allows objects to keep a relatively cheap reference as this object's
 *  members are stored in a reference counted pointer.
 */
class VulkanDevice
{
  public:
	VulkanDevice() = default;
	VulkanDevice( const VulkanDevice &rhs );
	VulkanDevice( vk::PhysicalDevice physicalDevice, const std::vector<const char *> &extensions, std::optional<vk::SurfaceKHR> surface = std::nullopt );
	~VulkanDevice() = default;

	const QueueFamilyIndices &GetQueueIndices() const { return m_Members->m_Indices; }
	vk::Queue &GetGraphicsQueue() { return m_Members->m_GraphicsQueue; }
	vk::Queue &GetComputeQueue() { return m_Members->m_ComputeQueue; }
	vk::Queue &GetTransferQueue() { return m_Members->m_TransferQueue; }
	vk::Queue &GetPresentQueue() { return m_Members->m_PresentQueue; }
	vk::Device GetVkDevice() const { return m_Members->m_VkDevice; }
	vk::PhysicalDevice GetPhysicalDevice() const { return m_Members->m_PhysicalDevice; }
	vk::PhysicalDeviceMemoryProperties GetMemoryProperties() const { return m_Members->m_MemProps; }
	vk::CommandPool GetCommandPool() const { return m_Members->m_CommandPool; }
	uint32_t GetMemoryType( const vk::MemoryRequirements &memReqs, vk::MemoryPropertyFlags memProps ) const;
	void Cleanup();

	vk::CommandBuffer CreateCommandBuffer( vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary );
	std::vector<vk::CommandBuffer> CreateCommandBuffers( uint32_t count, vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary );
	OneTimeCommandBuffer CreateOneTimeCmdBuffer( vk::CommandBufferLevel = vk::CommandBufferLevel::ePrimary );
	void SubmitCommandBuffer( vk::CommandBuffer cmdBuffer, vk::Queue queue, vk::Fence fence = nullptr, vk::PipelineStageFlags waitStageMask = {}, uint32_t waitSemaphoreCount = 0,
							  vk::Semaphore *waitSemaphores = nullptr, uint32_t signalSemaphoreCount = 0, vk ::Semaphore *signalSemaphores = nullptr );
	void SubmitCommandBuffers( uint32_t cmdBufferCount, vk::CommandBuffer *cmdBuffer, vk::Queue queue, vk::Fence fence = nullptr,
							   vk::PipelineStageFlags waitStageMask = {}, uint32_t waitSemaphoreCount = 0, vk::Semaphore *waitSemaphores = nullptr,
							   uint32_t signalSemaphoreCount = 0, vk::Semaphore *signalSemaphores = nullptr );
	void FreeCommandBuffer( vk::CommandBuffer cmdBuffer );
	void FreeCommandBuffers( const std::vector<vk::CommandBuffer> &cmdBuffers );
	void WaitIdle() const;

	vk::Device *operator->() { return &m_Members->m_VkDevice; }

	static std::optional<vk::PhysicalDevice> PickDeviceWithExtensions( vk::Instance &instance, const std::vector<const char *> &extensions,
																	   std::optional<vk::SurfaceKHR> surface = std::nullopt );

	static unsigned int RateDevice( const vk::PhysicalDevice &pDevice, std::optional<vk::SurfaceKHR> surface = std::nullopt );

	operator VkDevice() { return m_Members->m_VkDevice; }
	operator VkPhysicalDevice() { return m_Members->m_PhysicalDevice; }
	operator vk::Device() { return m_Members->m_VkDevice; }
	operator vk::PhysicalDevice() { return m_Members->m_PhysicalDevice; }
	operator bool() { return m_Members != nullptr && m_Members->m_VkDevice; }

  private:
	struct DeviceMembers
	{
		DeviceMembers( vk::PhysicalDevice &pDevice, std::optional<vk::SurfaceKHR> &surface ) : m_Indices( pDevice, surface ), m_PhysicalDevice( pDevice )
		{
		}

		~DeviceMembers()
		{
			Cleanup();
		}

		void Cleanup()
		{
			if ( m_VkDevice )
			{
				m_VkDevice.waitIdle();
				m_VkDevice.destroyCommandPool( m_CommandPool );
				m_VkDevice.destroy();
				m_VkDevice = nullptr;
			}
		}

		QueueFamilyIndices m_Indices;
		vk::CommandPool m_CommandPool;
		vk::PhysicalDevice m_PhysicalDevice;
		vk::PhysicalDeviceMemoryProperties m_MemProps;
		vk::Device m_VkDevice;
		vk::Queue m_GraphicsQueue;
		vk::Queue m_ComputeQueue;
		vk::Queue m_TransferQueue;
		vk::Queue m_PresentQueue;
	};

	std::shared_ptr<DeviceMembers> m_Members;
};

class OneTimeCommandBuffer
{
  public:
	OneTimeCommandBuffer( const VulkanDevice &device, vk::CommandBuffer cmdBuffer )
	{
		m_Recording = false;
		m_Device = device;
		m_CmdBuffer = cmdBuffer;
		Begin();
	}

	~OneTimeCommandBuffer()
	{
		Cleanup();
	}

	void Cleanup()
	{
		if ( m_CmdBuffer ) m_Device.FreeCommandBuffer( m_CmdBuffer );
	}

	void Begin()
	{
		assert( !m_Recording );
		m_Recording = true;
		m_CmdBuffer.begin( vk::CommandBufferBeginInfo( vk::CommandBufferUsageFlagBits::eOneTimeSubmit ) );
	}

	void End()
	{
		assert( m_Recording );
		m_CmdBuffer.end();
		m_Recording = false;
	}

	void Submit( vk::Queue queue, bool wait = true )
	{
		if ( m_Recording ) End(), m_Recording = false;
		m_Device.SubmitCommandBuffer( m_CmdBuffer, queue );
		if ( wait ) queue.waitIdle();
	}

	vk::CommandBuffer *operator->() { return &m_CmdBuffer; }
	operator vk::CommandBuffer() { return m_CmdBuffer; }
	operator vk::CommandBuffer &() { return m_CmdBuffer; }
	operator vk::CommandBuffer *() { return &m_CmdBuffer; }

  private:
	bool m_Recording;
	VulkanDevice m_Device;
	vk::CommandBuffer m_CmdBuffer;
};

} // namespace lh2core