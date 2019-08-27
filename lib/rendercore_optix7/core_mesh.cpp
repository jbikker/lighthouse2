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

RenderCore* CoreMesh::renderCore = 0;

template<typename T> T roundUp( T x, T y ) { return ((x + y - 1) / y) * y; }

//  +-----------------------------------------------------------------------------+
//  |  CoreMesh::~CoreMesh                                                        |
//  |  Destructor.                                                          LH2'19|
//  +-----------------------------------------------------------------------------+
CoreMesh::~CoreMesh()
{
	delete triangles;
	delete positions4;
}

//  +-----------------------------------------------------------------------------+
//  |  CoreMesh::SetGeometry                                                      |
//  |  Set the geometry data.                                               LH2'19|
//  +-----------------------------------------------------------------------------+
void CoreMesh::SetGeometry( const float4* vertexData, const int vertexCount, const int triCount, const CoreTri* tris, const uint* alphaFlags )
{
	// copy triangle data to GPU
	delete triangles;
	delete positions4;
	triangleCount = triCount;
	triangles = new CoreBuffer<CoreTri4>( triCount, ON_DEVICE, tris );
	positions4 = new CoreBuffer<float4>( triangleCount * 3, ON_DEVICE, vertexData );
	// prepare acceleration structure build parameters
	buildInput = {};
	buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
	buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
	buildInput.triangleArray.vertexStrideInBytes = sizeof( float4 );
	buildInput.triangleArray.numVertices = triangleCount * 3;
	buildInput.triangleArray.vertexBuffers = (CUdeviceptr*)positions4->DevPtrPtr();
	buildInput.triangleArray.flags = inputFlags;
	buildInput.triangleArray.numSbtRecords = 1;
	// set acceleration structure build options
	buildOptions = {};
	buildOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	// determine buffer sizes for the acceleration structure
	CHK_OPTIX( optixAccelComputeMemoryUsage( RenderCore::optixContext, &buildOptions, &buildInput, 1, &buildSizes ) );
	uint compactedSizeOffset = roundUp<uint>( (uint)buildSizes.outputSizeInBytes, 8 );
	CoreBuffer<uchar> temp( buildSizes.tempSizeInBytes, ON_DEVICE );
	CoreBuffer<uchar>* output = new CoreBuffer<uchar>( compactedSizeOffset + 8, ON_DEVICE );
	OptixAccelEmitDesc emitProperty = {};
	emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
	emitProperty.result = (CUdeviceptr)((char*)output->DevPtr() + compactedSizeOffset);
	CHK_OPTIX( optixAccelBuild( RenderCore::optixContext, 0, &buildOptions, &buildInput, 1,
		(CUdeviceptr)temp.DevPtr(), buildSizes.tempSizeInBytes, (CUdeviceptr)output->DevPtr(),
		buildSizes.outputSizeInBytes, &gasHandle, &emitProperty, 1 ) );
	// compact
	size_t compacted_gas_size;
	cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost );
	if (compacted_gas_size < buildSizes.outputSizeInBytes)
	{
		CoreBuffer<uchar>* compacted = new CoreBuffer<uchar>( compacted_gas_size, ON_DEVICE );
		gasData = (CUdeviceptr)compacted->DevPtr();
		CHK_OPTIX( optixAccelCompact( RenderCore::optixContext, 0, gasHandle, gasData, compacted_gas_size, &gasHandle ) );
		delete output;
	}
	else gasData = (CUdeviceptr)output->DevPtr();
}

// EOF