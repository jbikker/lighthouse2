/* vulkan_gl_texture_interop.h - Copyright 2019 Utrecht University

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
class VulkanGLTextureInterop
{
  public:
	VulkanGLTextureInterop( const VulkanDevice &device, uint32_t width, uint32_t height );
	~VulkanGLTextureInterop();

	void RecordTransitionToVulkan( vk::CommandBuffer &cmdBuffer );
	void RecordTransitionToGL( vk::CommandBuffer &cmdBuffer );

	void TransitionImageToInitialState( vk::CommandBuffer &cmdBuffer, vk::Queue &queue );
	void Cleanup();

	vk::Image GetImage() const { return m_Image; }
	vk::DeviceMemory GetMemory() const { return m_Memory; }
	vk::DeviceSize GetBufferSize() const { return m_Width * m_Height * 4 * sizeof( float ); }
	uint32_t GetWidth() const { return m_Width; }
	uint32_t GetHeight() const { return m_Height; }
	uint32_t GetID() const { return m_TexID; }
	static std::vector<const char *> GetRequiredExtensions();
	void Resize( uint32_t width, uint32_t height, bool deleteOldGLTexture = false );

	operator vk::Image() const { return m_Image; }

  private:
	vk::Image m_Image = nullptr;
	vk::DeviceMemory m_Memory = nullptr;
	VulkanDevice m_Device;
	vk::DeviceSize m_BufferSize = 0;
	uint32_t m_TexID = 0;
	uint32_t m_GLMemoryObj = 0;
	uint32_t m_Width = 0, m_Height = 0;
};
} // namespace lh2core