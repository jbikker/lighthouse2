/* top_level_as.cpp - Copyright 2019 Utrecht University

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

lh2core::TopLevelAS::TopLevelAS( const VulkanDevice &dev, AccelerationStructureType type, uint32_t instanceCount )
	: m_Device( dev ), m_InstanceCnt( instanceCount ), m_Type( type ), m_Flags( TypeToFlags( type ) )
{
	m_InstanceBuffer = new VulkanCoreBuffer<GeometryInstance>( m_Device, instanceCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
															   vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferDst );

	vk::AccelerationStructureInfoNV accelerationStructureInfo{};
	accelerationStructureInfo.pNext = nullptr;
	accelerationStructureInfo.type = vk::AccelerationStructureTypeNV::eTopLevel;
	accelerationStructureInfo.flags = m_Flags;
	accelerationStructureInfo.instanceCount = instanceCount;
	accelerationStructureInfo.geometryCount = 0;
	accelerationStructureInfo.pGeometries = nullptr;

	vk::AccelerationStructureCreateInfoNV accelerationStructureCreateInfo{};
	accelerationStructureCreateInfo.pNext = nullptr;
	accelerationStructureCreateInfo.info = accelerationStructureInfo;
	accelerationStructureCreateInfo.compactedSize = 0;

	CheckVK( m_Device->createAccelerationStructureNV( &accelerationStructureCreateInfo, nullptr, &m_Structure, RenderCore::instance->dynamicDispatcher ) );

	// Create a descriptor for the memory requirements, and provide the acceleration structure descriptor
	vk::AccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo;
	memoryRequirementsInfo.pNext = nullptr;
	memoryRequirementsInfo.accelerationStructure = m_Structure;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eObject;

	vk::MemoryRequirements2 memoryRequirements;
	m_Device->getAccelerationStructureMemoryRequirementsNV( &memoryRequirementsInfo, &memoryRequirements, RenderCore::instance->dynamicDispatcher );

	// Size of the resulting AS
	m_ResultSize = memoryRequirements.memoryRequirements.size;

	// Get the largest scratch size requirement
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eBuildScratch;
	m_Device->getAccelerationStructureMemoryRequirementsNV( &memoryRequirementsInfo, &memoryRequirements, RenderCore::instance->dynamicDispatcher );
	m_ScratchSize = memoryRequirements.memoryRequirements.size;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eUpdateScratch;
	m_Device->getAccelerationStructureMemoryRequirementsNV( &memoryRequirementsInfo, &memoryRequirements, RenderCore::instance->dynamicDispatcher );
	m_ScratchSize = std::max( m_ScratchSize, memoryRequirements.memoryRequirements.size );

	// Create result memory
	m_Memory = new VulkanCoreBuffer<uint8_t>( m_Device, m_ResultSize, vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible,
											  vk::BufferUsageFlagBits::eRayTracingNV );
	// Bind the acceleration structure descriptor to the actual memory that will contain it
	vk::BindAccelerationStructureMemoryInfoNV bindInfo;
	bindInfo.pNext = nullptr;
	bindInfo.accelerationStructure = m_Structure;
	bindInfo.memory = *m_Memory;
	bindInfo.memoryOffset = 0;
	bindInfo.deviceIndexCount = 0;
	bindInfo.pDeviceIndices = nullptr;
	CheckVK( m_Device->bindAccelerationStructureMemoryNV( 1, &bindInfo, RenderCore::instance->dynamicDispatcher ) );
}

lh2core::TopLevelAS::~TopLevelAS()
{
	Cleanup();
}

void lh2core::TopLevelAS::Cleanup()
{
	if ( m_Structure ) m_Device->destroyAccelerationStructureNV( m_Structure, nullptr, RenderCore::instance->dynamicDispatcher );
	if ( m_Memory ) delete m_Memory;
	if ( m_InstanceBuffer ) delete m_InstanceBuffer;
	m_Structure = nullptr;
	m_Memory = nullptr;
	m_InstanceBuffer = nullptr;
}

vk::WriteDescriptorSetAccelerationStructureNV lh2core::TopLevelAS::GetDescriptorBufferInfo() const
{
	return vk::WriteDescriptorSetAccelerationStructureNV( 1, &m_Structure );
}

void lh2core::TopLevelAS::Build( bool update )
{
	// Build the acceleration structure and store it in the result memory
	vk::AccelerationStructureInfoNV buildInfo = {vk::AccelerationStructureTypeNV::eTopLevel, m_Flags, m_InstanceCnt, 0, nullptr};
	auto scratchBuffer = VulkanCoreBuffer<uint8_t>( m_Device, m_ScratchSize, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eRayTracingNV );

	auto commandBuffer = m_Device.CreateOneTimeCmdBuffer();
	commandBuffer->buildAccelerationStructureNV( &buildInfo, *m_InstanceBuffer, 0, update, m_Structure, update ? m_Structure : nullptr,
												scratchBuffer, 0, RenderCore::instance->dynamicDispatcher );

	// Ensure that the build will be finished before using the AS using a barrier
	vk::MemoryBarrier memoryBarrier = {vk::AccessFlagBits::eAccelerationStructureWriteNV | vk::AccessFlagBits::eAccelerationStructureReadNV,
									   vk::AccessFlagBits::eAccelerationStructureReadNV};
	commandBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eAccelerationStructureBuildNV, vk::PipelineStageFlagBits::eRayTracingShaderNV,
								   vk::DependencyFlags(), 1, &memoryBarrier, 0, nullptr, 0, nullptr );

	auto computeQueue = m_Device.GetComputeQueue();
	commandBuffer.Submit( computeQueue, true );
}

void lh2core::TopLevelAS::UpdateInstances( const std::vector<GeometryInstance> &instances )
{
	assert( instances.size() <= m_InstanceCnt );
	m_InstanceBuffer->CopyToDevice( instances.data(), instances.size() * sizeof( GeometryInstance ) );
}

void lh2core::TopLevelAS::Build()
{
	Build( false );
}

void lh2core::TopLevelAS::Rebuild()
{
	Build( true );
}

uint64_t lh2core::TopLevelAS::GetHandle()
{
	uint64_t handle = 0;
	m_Device->getAccelerationStructureHandleNV( m_Structure, sizeof( uint64_t ), &handle, RenderCore::instance->dynamicDispatcher );
	assert( handle );
	return handle;
}

uint32_t lh2core::TopLevelAS::GetInstanceCount() const
{
	return m_InstanceCnt;
}
