/* gpu_mesh.h - Copyright 2019 Utrecht University

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

namespace lh2core {

//  +-----------------------------------------------------------------------------+
//  |  CoreMesh                                                                   |
//  |  Container for geometry data. Actual data resides on device:                |
//  |  - indicesDesc and verticesDesc describe on-device OptiX buffers;           |
//  |  - triangles contains the fully equiped triangle data.                LH2'19|
//  +-----------------------------------------------------------------------------+
class RenderCore;
class CoreMesh
{
public:
	struct MeshBuffers
	{
		optix::Buffer positions4;
		int triangleCount = 0;
		int vertexCount = 0;
	};
	// constructor / destructor
	CoreMesh() = default;
	CoreMesh::~CoreMesh();
	// methods
	void SetGeometry( const float4* vertexData, const int vertexCount, const int triCount, const CoreTri* tris, const uint* alphaFlags = 0 );
	// data
	CoreBuffer<CoreTri4>* triangles = 0;		// original triangle data, as received from RenderSystem
	MeshBuffers buffers;					// OptiX geometry data buffers
	optix::GeometryTriangles geometryTriangles = 0; // OptiX geometryTriangles descriptor
	// context, for global access
	static optix::Program attribProgram;	// Optix triangle attribute program
	static RenderCore* renderCore;			// for access to material list, in case of alpha mapped triangles
};

//  +-----------------------------------------------------------------------------+
//  |  CoreInstance                                                               |
//  |  Stores the data for a scene graph object.                            LH2'19|
//  +-----------------------------------------------------------------------------+
class CoreInstance
{
public:
	// constructor / destructor
	CoreInstance() = default;
	// data
	int mesh = 0;									// ID of the mesh used for this instance
	optix::GeometryGroup geometryGroup = 0;			// minimum OptiX scene: GeometryGroup, referencing
	optix::Transform transform = 0;					// a Transform, which has as a child a
	optix::GeometryInstance geometryInstance = 0;	// GeometryInstance, which in turn references Geometry.
};

} // namespace lh2core

// EOF