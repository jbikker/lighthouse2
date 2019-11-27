/* uniform_objects.h - Copyright 2019 Utrecht University

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
template <typename T>
class UniformObject
{
  public:
	static_assert( sizeof( T ) == 4 || ( sizeof( T ) % 8 ) == 0 ); // Make sure object is either at least 4 bytes big or 8 byte aligned
	UniformObject( VulkanDevice device, vk::BufferUsageFlagBits usage = vk::BufferUsageFlagBits(), vk::DeviceSize count = 1 )
		: m_Buffer( device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal,
					usage | vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, ON_HOST | ON_DEVICE )
	{
	}
	~UniformObject() { Cleanup(); }

	void CopyToDevice() { m_Buffer.CopyToDevice(); }
	void UpdateData( const T *data, uint32_t index = 0, uint32_t count = 1 )
	{
		assert( index + count < m_Buffer.GetElementCount() );
		memcpy( m_Buffer.GetHostBuffer() + index, data, count * sizeof( T ) );
	}
	T *GetData() { return m_Buffer.GetHostBuffer(); }
	T *ReadDataFromDevice()
	{
		m_Buffer.CopyToHost();
		return m_Buffer.GetHostBuffer();
	}

	void Cleanup() { m_Buffer.Cleanup(); }

	vk::Buffer GetVkBuffer() const { return m_Buffer.GetVkBuffer(); }
	vk::DeviceMemory GetVkMemory() const { return m_Buffer.GetVkMemory(); }
	VulkanCoreBuffer<T> &GetBuffer() { return m_Buffer; }
	vk::DescriptorBufferInfo GetDescriptorBufferInfo() const { return m_Buffer.GetDescriptorBufferInfo(); }

  private:
	VulkanCoreBuffer<T> m_Buffer;
};

/**
 * Uniform objects need to be at least 8-byte aligned thus using float3 instead of float4 is not an option.
 */
struct VulkanCamera
{
	VulkanCamera() = default;
	VulkanCamera( const ViewPyramid &view, int samplesTaken, int renderPhase );

	float4 pos_lensSize;
	float4 right_aperture;
	float4 up_spreadAngle;
	float4 p1;
	int pass, phase;
	int scrwidth, scrheight;
};

struct VulkanFinalizeParams
{
	VulkanFinalizeParams() = default;
	VulkanFinalizeParams( const int w, const int h, int samplespp );

	uint scrwidth;
	uint scrheight;
	uint spp;
	uint idummy = 0;
	float pixelValueScale;
	float fdummy0 = 0.0f, fdummy1 = 0.0f, fdummy2 = 0.0f;
};

using UniformCamera = UniformObject<VulkanCamera>;
using UniformFinalizeParams = UniformObject<VulkanFinalizeParams>;

} // namespace lh2core