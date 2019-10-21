/* core_mesh.h - Copyright 2019 Utrecht University

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
	// constructor / destructor
	CoreMesh() = default;
	~CoreMesh();
	// methods
	void SetGeometry( const float4* vertexData, const int vertexCount, const int triCount, const CoreTri* tris, const uint* alphaFlags = 0 );
	// data
	int triangleCount = 0;					// number of triangles in the mesh
	CoreBuffer<float4>* positions4 = 0;		// vertex data for intersection
	CoreBuffer<CoreTri4>* triangles = 0;	// original triangle data, as received from RenderSystem, for shading
	CoreBuffer<uchar>* buildTemp = 0;		// reusable temporary buffer for Optix BVH construction
	CoreBuffer<uchar>* buildBuffer = 0;		// reusable target buffer for Optix BVH construction
	// aceleration structure
	uint32_t inputFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT /* handled in CUDA shading code instead */ };
	OptixBuildInput buildInput;				// acceleration structure build parameters
	OptixAccelBuildOptions buildOptions;	// acceleration structure build options
	OptixAccelBufferSizes buildSizes;		// buffer sizes for acceleration structure construction
	OptixTraversableHandle gasHandle;		// handle to the mesh BVH
	CUdeviceptr gasData;					// acceleration structure data
	// global access
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
	int mesh = 0;							// ID of the mesh used for this instance
	OptixInstance instance;
	float transform[12];					// rigid transform of the instance
};

} // namespace lh2core

// EOF