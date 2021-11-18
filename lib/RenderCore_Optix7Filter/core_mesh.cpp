/* core_mesh.cpp - Copyright 2019/2021 Utrecht University

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
//  |  Set the geometry data for the accstruc.                              LH2'19|
//  +-----------------------------------------------------------------------------+
void CoreMesh::SetGeometry( const float4* vertexData, const int vertexCount, const int triCount, const CoreTri* tris )
{
	// allocate for the first frame, reallocate when the triangle data grows
	bool reallocate = false;
	if (triangles == 0) reallocate = true; else if (triCount > triangles->GetSize()) reallocate = true;
	// BVH compaction is done for the first frame only.
	// If we get here a second time we will assume this is an animation and compaction is not worthwhile.
	allowCompaction = (triangles == 0);
	// allocate and copy triangle data to GPU
	triangleCount = triCount;
	if (reallocate)
	{
		delete triangles;
		delete positions4;
		triangles = new CoreBuffer<CoreTri4>( triCount, ON_HOST | ON_DEVICE | STAGED, tris, POLICY_COPY_SOURCE );
		positions4 = new CoreBuffer<float4>( triangleCount * 3, ON_HOST | ON_DEVICE | STAGED, vertexData, POLICY_COPY_SOURCE );
		stageMemcpy( triangles->DevPtr(), triangles->HostPtr(), triangles->GetSizeInBytes() );
		stageMemcpy( positions4->DevPtr(), positions4->HostPtr(), positions4->GetSizeInBytes() );
	}
	else
	{
		triangles->SetHostData( (CoreTri4*)tris );
		positions4->SetHostData( (float4*)vertexData );
		stageMemcpy( triangles->DevPtr(), triangles->HostPtr(), triangles->GetSizeInBytes() );
		stageMemcpy( positions4->DevPtr(), positions4->HostPtr(), positions4->GetSizeInBytes() );
	}
	accstrucNeedsUpdate = true;
}

//  +-----------------------------------------------------------------------------+
//  |  CoreMesh::UpdateAccstruc                                                   |
//  |  Update the Optix BVH for modified geometry.                          LH2'20|
//  +-----------------------------------------------------------------------------+
void CoreMesh::UpdateAccstruc()
{
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
	// NOTE: compacting the first time geometry is handed works well for static geometry but
	// crashes the bird animation, for unknown reasons. Disabled for now.
	buildOptions = {};
	buildOptions.buildFlags = /* (allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0) | */ OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
	buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
	// determine buffer sizes for the acceleration structure
	CHK_OPTIX( optixAccelComputeMemoryUsage( RenderCore::optixContext, &buildOptions, &buildInput, 1, &buildSizes ) );
	uint compactedSizeOffset = roundUp<uint>( (uint)buildSizes.outputSizeInBytes, 8 );
	// (re)allocate when needed
	if (buildTemp == 0 || (size_t)buildTemp->GetSize() < buildSizes.tempSizeInBytes)
	{
		delete buildTemp;
		buildTemp = new CoreBuffer<uchar>( buildSizes.tempSizeInBytes, ON_DEVICE );
	}
	if (buildBuffer == 0 || buildBuffer->GetSize() < compactedSizeOffset)
	{
		delete buildBuffer;
		buildBuffer = new CoreBuffer<uchar>( compactedSizeOffset + 8, ON_DEVICE );
	}
	// build
	if (0) /* see note on compaction above */ // (allowCompaction)
	{
		// build with compaction
		OptixAccelEmitDesc emitProperty = {};
		emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
		emitProperty.result = (CUdeviceptr)((char*)buildBuffer->DevPtr() + compactedSizeOffset);
		CHK_OPTIX( optixAccelBuild( RenderCore::optixContext, 0, &buildOptions, &buildInput, 1,
			(CUdeviceptr)buildTemp->DevPtr(), buildSizes.tempSizeInBytes, (CUdeviceptr)buildBuffer->DevPtr(),
			buildSizes.outputSizeInBytes, &gasHandle, &emitProperty, 1 ) );
		size_t compacted_gas_size;
		cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof( size_t ), cudaMemcpyDeviceToHost );
		if (compacted_gas_size < buildSizes.outputSizeInBytes)
		{
			CoreBuffer<uchar>* compacted = new CoreBuffer<uchar>( compacted_gas_size, ON_DEVICE );
			gasData = (CUdeviceptr)compacted->DevPtr();
			CHK_OPTIX( optixAccelCompact( RenderCore::optixContext, 0, gasHandle, gasData, compacted_gas_size, &gasHandle ) );
			delete buildBuffer;
			buildBuffer = compacted;
		}
		else gasData = (CUdeviceptr)buildBuffer->DevPtr();
	}
	else
	{
		// build without compaction
		CHK_OPTIX( optixAccelBuild( RenderCore::optixContext, 0, &buildOptions, &buildInput, 1,
			(CUdeviceptr)buildTemp->DevPtr(), buildSizes.tempSizeInBytes, (CUdeviceptr)buildBuffer->DevPtr(),
			buildSizes.outputSizeInBytes, &gasHandle, 0, 0 ) );
		gasData = (CUdeviceptr)buildBuffer->DevPtr();
	}
	accstrucNeedsUpdate = false;
}

// EOF