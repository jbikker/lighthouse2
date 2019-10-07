/* core_mesh.cpp - Copyright 2019 Utrecht University

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

namespace lh2core
{

CoreMesh::CoreMesh( VulkanDevice device )
	: m_Device( device )
{
	triangles = new VulkanCoreBuffer<CoreTri>( m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal,
											   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
}

CoreMesh::~CoreMesh()
{
	Cleanup();
}

void CoreMesh::SetGeometry( const float4 *vertexData, const int vertexCount, const int triCount, const CoreTri *tris, const uint *alphaFlags )
{
	const bool sameTriCount = triangles && ( triangles->GetSize() / sizeof( CoreTri ) == triCount );

	if ( !sameTriCount )
	{
		delete triangles;
		triangles = new VulkanCoreBuffer<CoreTri>( m_Device, triCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
												   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst );
	}

	triangles->CopyToDevice( tris, triCount * sizeof( CoreTri ) );

	if ( accelerationStructure != nullptr )
	{
		if ( accelerationStructure->CanUpdate() && sameTriCount ) // Same data count, rebuild acceleration structure
		{
			accelerationStructure->UpdateVertices( vertexData, vertexCount );
			accelerationStructure->Rebuild();
		}
		else
		{
			delete accelerationStructure;
			accelerationStructure = nullptr;
			accelerationStructure = new BottomLevelAS( m_Device, vertexData, vertexCount, FastTrace ); // Create new, update able acceleration structure
			accelerationStructure->Build();
		}
	}
	else
	{
		accelerationStructure = new BottomLevelAS( m_Device, vertexData, vertexCount, FastestTrace ); // Create initial acceleration structure
		accelerationStructure->Build();
	}

	assert( accelerationStructure );
}

void CoreMesh::Cleanup()
{
	if ( accelerationStructure ) delete accelerationStructure, accelerationStructure = nullptr;
	if ( triangles ) delete triangles, triangles = nullptr;
}

} // namespace lh2core