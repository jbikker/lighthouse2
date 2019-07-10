/* gpu_mesh.cpp - Copyright 2019 Utrecht University

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

Program CoreMesh::attribProgram = 0;
RenderCore* CoreMesh::renderCore = 0;

//  +-----------------------------------------------------------------------------+
//  |  CoreMesh::~CoreMesh                                                        |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
CoreMesh::~CoreMesh()
{
	delete triangles;
	buffers.positions4->destroy();
	geometryTriangles->destroy();
}

//  +-----------------------------------------------------------------------------+
//  |  CoreMesh::SetGeometry                                                      |
//  |  Set the geometry data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void CoreMesh::SetGeometry( const float4* vertexData, const int vertexCount, const int triCount, const CoreTri* tris, const uint* alphaFlags )
{
	// copy triangle data to GPU
	delete triangles;
	triangles = new CoreBuffer<CoreTri4>( triCount, ON_DEVICE, tris );
	buffers.triangleCount = triCount;
	buffers.vertexCount = triCount * 3;
	// create OptiX geometry buffers
	buffers.positions4 = RenderCore::context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, buffers.vertexCount );
	float4* positions4 = reinterpret_cast<float4*>(buffers.positions4->map());
	for (int i = 0; i < buffers.vertexCount; i++) positions4[i] = vertexData[i];
	buffers.positions4->unmap();
	// instantiate geometry descriptor
	geometryTriangles = RenderCore::context->createGeometryTriangles();
	geometryTriangles->setPrimitiveCount( buffers.triangleCount );
	geometryTriangles->setVertices( buffers.vertexCount, buffers.positions4, 0, 16, RT_FORMAT_FLOAT3 );
	geometryTriangles->setBuildFlags( RTgeometrybuildflags( 0 ) );
	if (attribProgram) geometryTriangles->setAttributeProgram( attribProgram );
	geometryTriangles->validate();
}

// EOF