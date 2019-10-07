/* bottom_level_as.cpp - Copyright 2019 Utrecht University

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

BottomLevelAS::BottomLevelAS( VulkanDevice device, const float4 *vertices, uint32_t vertexCount, AccelerationStructureType type )
	: m_Device( device ), m_Type( type )
{
	assert( vertexCount > 0 );
	m_Vertices = new VulkanCoreBuffer<float4>( m_Device, vertexCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
											   vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferDst );
	m_Vertices->CopyToDevice( vertices, vertexCount * sizeof( float4 ) );
	m_Flags = TypeToFlags( type );

	m_Geometry.pNext = nullptr;
	m_Geometry.geometryType = vk::GeometryTypeNV::eTriangles;
	m_Geometry.geometry.triangles.pNext = nullptr;
	m_Geometry.geometry.triangles.vertexData = *m_Vertices;
	m_Geometry.geometry.triangles.vertexOffset = 0;
	m_Geometry.geometry.triangles.vertexCount = vertexCount;
	m_Geometry.geometry.triangles.vertexStride = sizeof( float4 );
	m_Geometry.geometry.triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;

	m_Geometry.geometry.triangles.indexData = nullptr;
	m_Geometry.geometry.triangles.indexOffset = 0;
	m_Geometry.geometry.triangles.indexCount = 0;
	m_Geometry.geometry.triangles.indexType = vk::IndexType::eNoneNV;
	m_Geometry.geometry.triangles.transformData = nullptr;
	m_Geometry.geometry.triangles.transformOffset = 0;
	m_Geometry.flags = vk::GeometryFlagBitsNV::eOpaque;

	// Create the descriptor of the acceleration structure, which contains the number of geometry descriptors it will contain
	vk::AccelerationStructureInfoNV accelerationStructureInfo = {vk::AccelerationStructureTypeNV::eBottomLevel, m_Flags, 0, 1, &m_Geometry};

	vk::AccelerationStructureCreateInfoNV accelerationStructureCreateInfo = {0, accelerationStructureInfo};

	CheckVK( device->createAccelerationStructureNV( &accelerationStructureCreateInfo, nullptr, &m_Structure, RenderCore::instance->dynamicDispatcher ) );

	// Create a descriptor for the memory requirements, and provide the acceleration structure descriptor
	vk::AccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo;
	memoryRequirementsInfo.pNext = nullptr;
	memoryRequirementsInfo.accelerationStructure = m_Structure;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eObject;

	vk::MemoryRequirements2 memoryRequirements;
	device->getAccelerationStructureMemoryRequirementsNV( &memoryRequirementsInfo, &memoryRequirements, RenderCore::instance->dynamicDispatcher );

	// Size of the resulting AS
	m_ResultSize = memoryRequirements.memoryRequirements.size;

	// Get the largest scratch size requirement
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eBuildScratch;
	device->getAccelerationStructureMemoryRequirementsNV( &memoryRequirementsInfo, &memoryRequirements, RenderCore::instance->dynamicDispatcher );
	m_ScratchSize = memoryRequirements.memoryRequirements.size;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eUpdateScratch;
	device->getAccelerationStructureMemoryRequirementsNV( &memoryRequirementsInfo, &memoryRequirements, RenderCore::instance->dynamicDispatcher );
	m_ScratchSize = std::max( m_ScratchSize, memoryRequirements.memoryRequirements.size );

	// Create result memory
	m_Memory = new VulkanCoreBuffer<uint8_t>( m_Device, m_ResultSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
											  vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferSrc );

	// Bind the acceleration structure descriptor to the actual memory that will contain it
	vk::BindAccelerationStructureMemoryInfoNV bindInfo = {m_Structure, *m_Memory, 0, 0, nullptr};
	CheckVK( device->bindAccelerationStructureMemoryNV( 1, &bindInfo, RenderCore::instance->dynamicDispatcher ) );
}

BottomLevelAS::~BottomLevelAS()
{
	Cleanup();
}

void lh2core::BottomLevelAS::Cleanup()
{
	if ( m_Structure ) m_Device->destroyAccelerationStructureNV( m_Structure, nullptr, RenderCore::instance->dynamicDispatcher );
	if ( m_Vertices ) delete m_Vertices;
	if ( m_Memory ) delete m_Memory;
	m_Structure = nullptr;
	m_Vertices = nullptr;
	m_Memory = nullptr;
}

void BottomLevelAS::UpdateVertices( const float4 *vertices, uint32_t vertexCount )
{
	assert( m_Vertices->GetElementCount() == vertexCount );
	m_Vertices->CopyToDevice( vertices, vertexCount * sizeof( float4 ) );
}

void lh2core::BottomLevelAS::Build()
{
	Build( false );
}

void lh2core::BottomLevelAS::Rebuild()
{
	Build( true );
}

uint64_t BottomLevelAS::GetHandle()
{
	uint64_t handle = 0;
	m_Device->getAccelerationStructureHandleNV( m_Structure, sizeof( uint64_t ), &handle, RenderCore::instance->dynamicDispatcher );
	assert( handle );
	return handle;
}

uint32_t lh2core::BottomLevelAS::GetVertexCount() const
{
	return m_Vertices->GetElementCount();
}

void lh2core::BottomLevelAS::Build( bool update )
{
	assert( m_Vertices->GetElementCount() > 0 );

	// Create temporary scratch buffer
	auto scratchBuffer = VulkanCoreBuffer<uint8_t>( m_Device, m_ScratchSize, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eRayTracingNV );

	// Build the actual bottom-level AS
	vk::AccelerationStructureInfoNV buildInfo = {vk::AccelerationStructureTypeNV::eBottomLevel, m_Flags, 0, 1, &m_Geometry};

	// Submit build command
	auto commandBuffer = m_Device.CreateOneTimeCmdBuffer();
	auto computeQueue = m_Device.GetComputeQueue();

	if ( m_Flags & vk::BuildAccelerationStructureFlagBitsNV::eAllowCompaction )
	{
		commandBuffer->buildAccelerationStructureNV( &buildInfo, nullptr, 0, update, m_Structure,
													 update ? m_Structure : nullptr, scratchBuffer,
													 0, RenderCore::instance->dynamicDispatcher );
		// Create memory barrier for building AS to make sure it can only be used when ready
		vk::MemoryBarrier memoryBarrier = {vk::AccessFlagBits::eAccelerationStructureWriteNV | vk::AccessFlagBits::eAccelerationStructureReadNV,
										   vk::AccessFlagBits::eAccelerationStructureReadNV};
		commandBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eAccelerationStructureBuildNV,
										vk::PipelineStageFlagBits::eRayTracingShaderNV, vk::DependencyFlags(), 1, &memoryBarrier,
										0, nullptr, 0, nullptr );

		// Create query pool to get compacted AS size
		vk::QueryPoolCreateInfo queryPoolCreateInfo = vk::QueryPoolCreateInfo( vk::QueryPoolCreateFlags(), vk::QueryType::eAccelerationStructureCompactedSizeNV, 1 );
		vk::QueryPool queryPool = m_Device->createQueryPool( queryPoolCreateInfo );

		// Query for compacted size
		commandBuffer->resetQueryPool( queryPool, 0, 1 );
		commandBuffer->beginQuery( queryPool, 0, vk::QueryControlFlags() );
		commandBuffer->writeAccelerationStructuresPropertiesNV( 1, &m_Structure, vk::QueryType::eAccelerationStructureCompactedSizeNV,
																queryPool, 0, RenderCore::instance->dynamicDispatcher );
		commandBuffer->endQuery( queryPool, 0 );
		commandBuffer.Submit( computeQueue, true );

		uint32_t size;
		CheckVK( m_Device->getQueryPoolResults( queryPool, 0, 1, sizeof( uint32_t ),
												&size, sizeof( uint32_t ), vk::QueryResultFlagBits::eWait ) );

		if ( size > 0 ) // Only compact if the queried result returns a valid size value
		{
			buildInfo.geometryCount = 0; // Must be zero for compacted AS
			buildInfo.pGeometries = nullptr;
			vk::AccelerationStructureCreateInfoNV accelerationStructureCreateInfo = {0, buildInfo};
			accelerationStructureCreateInfo.compactedSize = size;
			// Create AS handle
			vk::AccelerationStructureNV compactedAS;
			CheckVK( m_Device->createAccelerationStructureNV( &accelerationStructureCreateInfo, nullptr, &compactedAS, RenderCore::instance->dynamicDispatcher ) );
			// Get new memory requirements
			vk::AccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {vk::AccelerationStructureMemoryRequirementsTypeNV::eObject, compactedAS};
			vk::MemoryRequirements2 memoryRequirements;
			m_Device->getAccelerationStructureMemoryRequirementsNV( &memoryRequirementsInfo, &memoryRequirements, RenderCore::instance->dynamicDispatcher );
			// Create new, smaller buffer for compacted AS
			auto newMemory = new VulkanCoreBuffer<uint8_t>( m_Device, memoryRequirements.memoryRequirements.size, vk::MemoryPropertyFlagBits::eDeviceLocal,
															vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferDst );
			// Bind the acceleration structure descriptor to the memory that will contain it
			vk::BindAccelerationStructureMemoryInfoNV bindInfo = {compactedAS, *newMemory, 0, 0, nullptr};
			CheckVK( m_Device->bindAccelerationStructureMemoryNV( 1, &bindInfo, RenderCore::instance->dynamicDispatcher ) );
			// Submit copy & compact command to command buffer
			commandBuffer.Begin();
			commandBuffer->copyAccelerationStructureNV( compactedAS, m_Structure, vk::CopyAccelerationStructureModeNV::eCompact, RenderCore::instance->dynamicDispatcher );
			commandBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eAccelerationStructureBuildNV, vk::PipelineStageFlagBits::eRayTracingShaderNV,
										   vk::DependencyFlags(), 1, &memoryBarrier, 0, nullptr, 0, nullptr );
			commandBuffer.Submit( computeQueue, true );

			// Cleanup
			m_Device->destroyQueryPool( queryPool );
			m_Device->destroyAccelerationStructureNV( m_Structure, nullptr, RenderCore::instance->dynamicDispatcher );
			delete m_Memory;

			// Assign new AS to this object
			m_Memory = newMemory;
			m_Structure = compactedAS;
		}
	}
	else
	{
		commandBuffer->buildAccelerationStructureNV( &buildInfo, nullptr, 0, update, m_Structure, update ? m_Structure : nullptr,
													scratchBuffer, 0, RenderCore::instance->dynamicDispatcher );
		// Create memory barrier for building AS to make sure it can only be used when ready
		vk::MemoryBarrier memoryBarrier = {vk::AccessFlagBits::eAccelerationStructureWriteNV | vk::AccessFlagBits::eAccelerationStructureReadNV,
										   vk::AccessFlagBits::eAccelerationStructureReadNV};
		commandBuffer->pipelineBarrier( vk::PipelineStageFlagBits::eAccelerationStructureBuildNV,
									   vk::PipelineStageFlagBits::eRayTracingShaderNV, vk::DependencyFlags(), 1, &memoryBarrier,
									   0, nullptr, 0, nullptr );
		auto computeQueue = m_Device.GetComputeQueue();
		commandBuffer.Submit( computeQueue, true );
	}
}
